# mindmap_utils.py
"""
Pipeline tạo mind map dựa trên kỹ thuật iterative prompting:

1. Root generation: chọn chủ đề trung tâm bằng cách lọc noise và nhắc lệnh riêng.
2. Iterative expansion: lần lượt mở rộng từng nút lá, mô hình tự quyết định tiếp tục hay dừng.
3. Critics (factuality → local structure → global structure): ba lần rà soát độc lập
   giúp đối chiếu dẫn chứng, cụ thể hóa nhánh lá và cân bằng bố cục toàn cục giống mục lục.

Mỗi bước đều vận hành hoàn toàn bằng JSON để dễ parse/verify.
"""

import re
import json
import ast
from collections import deque
from ollama_utils import run_ollama_chat, SLM_MODEL


MAX_SEGMENTS_FOR_MINDMAP = 24
MAX_CHARS_FOR_MINDMAP = 8000
MAX_EXPANSION_CALLS_BASE = 18
MIN_ROOT_CHILDREN = 4
MIN_INNER_CHILDREN = 2
CONTEXT_SEGMENTS_PER_NODE = 8
CRITIC_SEGMENTS = 12
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# CMGN (Coreference-Guided Mind-Map Generation) helpers
CORE_GRAPH_SENTENCE_LIMIT = 48
CORE_GRAPH_SENTENCE_MAX_CHARS = 280
CORE_GRAPH_EDGE_LIMIT = 64
CORE_GRAPH_CLUSTER_LIMIT = 24

def _is_noise_topic(name: str, noise_terms: set[str] | None = None) -> bool:
    if not name:
        return True
    lowered = name.strip().lower()
    if not lowered:
        return True
    for term in noise_terms or ():
        if term and term in lowered:
            return True
    return False


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _prepare_mindmap_chunks(chunks: list[str]) -> list[str]:
    """Clean & limit chunks for prompting while preserving order."""
    prepared: list[str] = []
    total_chars = 0

    for chunk in chunks or []:
        if not chunk:
            continue
        normalized = re.sub(r"\s+", " ", str(chunk)).strip()
        if not normalized:
            continue

        next_len = total_chars + len(normalized)
        if prepared and (len(prepared) >= MAX_SEGMENTS_FOR_MINDMAP or next_len > MAX_CHARS_FOR_MINDMAP):
            break

        prepared.append(normalized)
        total_chars = next_len

    return prepared


def _escape_inner_quotes(body: str) -> str:
    """Best-effort escape for unescaped quotes inside JSON string values."""
    result: list[str] = []
    inside_string = False
    escape = False
    closers = {",", "}", "]", ":", " ", "\n", "\r", "\t"}
    length = len(body)

    for idx, ch in enumerate(body):
        next_char = body[idx + 1] if idx + 1 < length else ""

        if ch == "\"" and not escape:
            if inside_string:
                if next_char in closers or next_char == "":
                    inside_string = False
                    result.append(ch)
                else:
                    result.append("\\\"")
                continue
            else:
                inside_string = True
                result.append(ch)
                continue

        if ch == "\\" and not escape:
            escape = True
        else:
            escape = False

        result.append(ch)

    return "".join(result)


def _insert_missing_commas(body: str) -> str:
    """Try to insert commas between adjacent JSON objects/arrays when omitted."""
    if not body:
        return body

    chars: list[str] = []
    inside_string = False
    escape = False
    length = len(body)
    i = 0

    while i < length:
        ch = body[i]
        chars.append(ch)

        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == "\"":
            inside_string = not inside_string
        elif not inside_string and ch in {"}", "]"}:
            j = i + 1
            while j < length and body[j].isspace():
                j += 1
            if j < length:
                next_char = body[j]
                next_lower = body[j:j + 5].lower()
                starts_value = (
                    next_char in {"{", "[", "\"", "-"}
                    or next_char.isdigit()
                    or next_lower.startswith("true")
                    or next_lower.startswith("false")
                    or next_lower.startswith("null")
                )
                if starts_value:
                    whitespace_segment = body[i + 1:j]
                    if "," not in whitespace_segment:
                        k = len(chars) - 2
                        while k >= 0 and chars[k].isspace():
                            k -= 1
                        prev_char = chars[k] if k >= 0 else ""
                        if prev_char not in {"{", "[", ",", ":"}:
                            chars.append(",")

        i += 1

    return "".join(chars)


def _remove_invalid_control_chars(body: str) -> str:
    if not body:
        return body
    return CONTROL_CHAR_RE.sub(" ", body)


def _extract_sentences_from_segments(segments: list[str], limit: int = CORE_GRAPH_SENTENCE_LIMIT) -> list[dict]:
    if not segments:
        return []

    joined = " ".join(str(seg).strip() for seg in segments if seg).strip()
    if not joined:
        return []

    # Split sentences; include Vietnamese punctuation.
    raw_sentences = re.split(r"(?<=[\.!?。？！])\s+", joined)
    sentences: list[dict] = []
    seen: set[str] = set()

    for idx, sentence in enumerate(raw_sentences, start=1):
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned:
            continue
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        sentences.append({
            "id": f"S{len(sentences) + 1}",
            "text": cleaned[:CORE_GRAPH_SENTENCE_MAX_CHARS].strip()
        })
        if len(sentences) >= limit:
            break

    return sentences


def _sanitize_coreference_graph(obj, default_sentences: list[dict]) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Coreference graph không phải JSON object")

    sentences_input = default_sentences or []
    by_id = {item.get("id"): item.get("text", "") for item in sentences_input if item.get("id")}

    sanitized_sentences: list[dict] = []
    provided_sentences = obj.get("sentences")
    if isinstance(provided_sentences, list) and provided_sentences:
        for item in provided_sentences:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or item.get("sentenceId") or "").strip()
            text = str(item.get("text") or "").strip()
            if not sid:
                continue
            if not text and sid in by_id:
                text = by_id[sid]
            if not text:
                continue
            entities = item.get("entities")
            if not isinstance(entities, list):
                entities = []
            entities = [str(e).strip() for e in entities if e]
            importance = item.get("importance")
            try:
                importance_val = float(importance)
            except (TypeError, ValueError):
                importance_val = None
            sanitized_sentences.append({
                "id": sid,
                "text": text,
                "entities": entities,
                "importance": importance_val,
            })
            by_id[sid] = text

    if not sanitized_sentences:
        sanitized_sentences = []
        for item in sentences_input:
            sid = item.get("id")
            if not sid:
                continue
            sanitized_sentences.append({
                "id": sid,
                "text": item.get("text", ""),
                "entities": [],
                "importance": None,
            })

    valid_ids = {item["id"] for item in sanitized_sentences if item.get("id")}

    sanitized_clusters: list[dict] = []
    for cluster in obj.get("clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        entity = str(cluster.get("entity") or cluster.get("label") or "").strip()
        mentions = cluster.get("mentions") or cluster.get("sentenceIds") or cluster.get("nodes")
        if not isinstance(mentions, list):
            continue
        filtered_mentions = []
        for mention in mentions:
            mention_id = str(mention).strip()
            if mention_id in valid_ids and mention_id not in filtered_mentions:
                filtered_mentions.append(mention_id)
        if not filtered_mentions:
            continue
        note = str(cluster.get("note") or cluster.get("description") or "").strip()
        sanitized_clusters.append({
            "entity": entity or ", ".join(filtered_mentions[:2]),
            "mentions": filtered_mentions,
            "note": note,
        })
        if len(sanitized_clusters) >= CORE_GRAPH_CLUSTER_LIMIT:
            break

    sanitized_edges: list[dict] = []
    for edge in obj.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or edge.get("from") or edge.get("start") or "").strip()
        target = str(edge.get("target") or edge.get("to") or edge.get("end") or "").strip()
        if source not in valid_ids or target not in valid_ids or source == target:
            continue
        relation = str(edge.get("relation") or edge.get("reason") or edge.get("label") or "").strip()
        sanitized_edges.append({
            "source": source,
            "target": target,
            "relation": relation,
        })
        if len(sanitized_edges) >= CORE_GRAPH_EDGE_LIMIT:
            break

    root_candidates = obj.get("rootCandidates") or obj.get("roots") or obj.get("focus")
    sanitized_roots: list[str] = []
    if isinstance(root_candidates, list):
        for rc in root_candidates:
            rc_id = str(rc).strip()
            if rc_id in valid_ids and rc_id not in sanitized_roots:
                sanitized_roots.append(rc_id)

    if not sanitized_roots and sanitized_sentences:
        sanitized_roots.append(sanitized_sentences[0]["id"])

    return {
        "sentences": sanitized_sentences,
        "clusters": sanitized_clusters,
        "edges": sanitized_edges,
        "rootCandidates": sanitized_roots,
    }


def _generate_coreference_graph(sentences: list[dict], model: str | None) -> dict:
    if not sentences:
        raise ValueError("Không có câu để dựng đồ thị coreference")

    listing = "\n".join(f"{item['id']}: {item['text']}" for item in sentences)
    system_prompt = "\n".join([
        "Bạn là bộ trích xuất Coreference Graph cho mạng CMGN.",
        "Nhiệm vụ: dựa trên các câu đã đánh số, hãy xác định thực thể được nhắc lại và cấu trúc liên kết logic.",
        "Trả về DUY NHẤT một block ```json``` với cấu trúc:",
        "{",
        "  \"sentences\": [{\"id\": \"S1\", \"text\": \"...\", \"entities\": [], \"importance\": 0.8}],",
        "  \"clusters\": [{\"entity\": \"...\", \"mentions\": [\"S1\", \"S3\"], \"note\": \"...\"}],",
        "  \"edges\": [{\"source\": \"S1\", \"target\": \"S3\", \"relation\": \"share entity X\"}],",
        "  \"rootCandidates\": [\"S1\", \"S2\"]",
        "}",
        "Ghi chú:",
        "- Giữ nguyên ID câu như đã cho (S1, S2, ...).",
        "- entities là danh sách ngắn các thực thể hoặc khái niệm chính xuất hiện trong câu.",
        "- clusters nhóm các câu cùng thực thể đồng tham chiếu.",
        "- edges mô tả quan hệ chi phối (ví dụ: cùng thực thể, giải thích, nguyên nhân).",
        "- rootCandidates ưu tiên tối đa 3 câu thể hiện chủ đề trung tâm.",
        "- Toàn bộ entity, note, relation phải ghi bằng tiếng Việt tự nhiên khi có thể.",
        "Chỉ trả về JSON, không thêm lời giải thích khác.",
    ])

    user_prompt = "\n".join([
        "Các câu từ tài liệu (giữ nguyên ID):",
        listing,
        "Hãy dựng đồ thị tham chiếu đồng ngữ theo hướng dẫn CMGN.",
    ])

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
    graph_obj = extract_json_tree(raw)
    return _sanitize_coreference_graph(graph_obj, sentences)


def _summarize_coreference_graph(graph: dict) -> str:
    if not isinstance(graph, dict):
        return "-"

    lines: list[str] = []
    sentences = graph.get("sentences", []) or []
    if sentences:
        lines.append("Câu & thực thể:")
        for sentence in sentences:
            sent_id = sentence.get("id")
            text = sentence.get("text", "")
            entities = ", ".join(sentence.get("entities") or [])
            if entities:
                lines.append(f"- {sent_id}: {text} (entities: {entities})")
            else:
                lines.append(f"- {sent_id}: {text}")

    clusters = graph.get("clusters", []) or []
    if clusters:
        lines.append("\nCụm đồng tham chiếu:")
        for cluster in clusters:
            entity = cluster.get("entity", "")
            mentions = ", ".join(cluster.get("mentions") or [])
            note = cluster.get("note")
            if note:
                lines.append(f"- {entity}: {mentions} ({note})")
            else:
                lines.append(f"- {entity}: {mentions}")

    edges = graph.get("edges", []) or []
    if edges:
        lines.append("\nQuan hệ chính:")
        for edge in edges:
            rel = edge.get("relation") or "liên kết"
            lines.append(f"- {edge.get('source')} → {edge.get('target')}: {rel}")

    roots = graph.get("rootCandidates", []) or []
    if roots:
        lines.append("\nGợi ý root: " + ", ".join(roots))

    return "\n".join(lines) if lines else "-"


def _generate_mindmap_from_coreference_graph(
    graph: dict,
    content_segments: list[str],
    noise_terms: set[str],
    model: str | None,
) -> dict:
    summary = _summarize_coreference_graph(graph)
    bullet_block = "\n".join(f"- {seg}" for seg in content_segments[:MAX_SEGMENTS_FOR_MINDMAP])

    root_hint = ", ".join(graph.get("rootCandidates", []) or [])

    system_prompt = "\n".join([
        "Bạn là Coreference-Guided Mind-Map Generation Network (CMGN).",
        "Sử dụng đồ thị tham chiếu đồng ngữ đã cho để tạo mindmap logic, giữ đúng JSON hợp lệ.",
        "Nguyên tắc:",
        "1) Root dựa trên các câu rootCandidates (ưu tiên câu chứa chủ đề trung tâm).",
        "2) Các nhánh cấp 1 gộp theo cụm coreference hoặc quan hệ ngữ nghĩa dài hạn.",
        "3) Các nhánh con triển khai chi tiết dựa trên câu liên kết qua edges và clusters.",
        "4) detail (nếu có) <= 20 từ, chứa citation dạng [S1], [S2-S4] thể hiện câu tham chiếu.",
        "5) Tuyệt đối không tạo thông tin ngoài nội dung đã cho, tránh metadata hành chính.",
        "6) Trả về DUY NHẤT block ```json``` dạng {name, detail?, children}.",
        "7) Tất cả tên node và detail phải bằng tiếng Việt tự nhiên, có thể giữ nguyên thuật ngữ chuyên môn cần thiết.",
    ])

    user_prompt = "\n".join([
        "Thông tin nguồn (đã lọc):",
        bullet_block or "-",
        "\nĐồ thị coreference (CMGN):",
        summary,
        "\nGợi ý root candidates: " + (root_hint or "(không)"),
        "Hãy xuất mindmap hoàn chỉnh, cân đối 4-7 nhánh cấp 1 nếu có đủ nội dung.",
    ])

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
    tree_obj = extract_json_tree(raw)
    sanitized_tree = _sanitize_tree(tree_obj, noise_terms)
    if not sanitized_tree.get("children"):
        raise ValueError("Mindmap CMGN rỗng")
    return sanitized_tree


def _context_for_path(path: list[str], content_segments: list[str], limit: int = CONTEXT_SEGMENTS_PER_NODE) -> list[str]:
    if not content_segments:
        return []

    keywords: set[str] = set()
    for name in path:
        for token in re.findall(r"[\wÀ-ỹ']+", name or "", flags=re.I):
            token_clean = token.lower()
            if len(token_clean) >= 4:
                keywords.add(token_clean)

    matched: list[str] = []
    if keywords:
        for segment in content_segments:
            lowered = segment.lower()
            if any(keyword in lowered for keyword in keywords):
                if segment not in matched:
                    matched.append(segment)
            if len(matched) >= limit:
                break

    if not matched:
        matched = content_segments[:limit]
    elif len(matched) < limit:
        for segment in content_segments:
            if segment in matched:
                continue
            matched.append(segment)
            if len(matched) >= limit:
                break

    return matched[:limit]


def _filter_segments_by_noise(segments: list[str], noise_terms: set[str] | None) -> list[str]:
    if not segments or not noise_terms:
        return segments or []

    filtered: list[str] = []
    for segment in segments:
        lowered = segment.lower()
        if any(term in lowered for term in noise_terms if term):
            continue
        filtered.append(segment)

    return filtered or segments


def _detect_noise_terms(content_segments: list[str], model: str | None) -> set[str]:
    sample = [seg for seg in content_segments if seg]
    sample = sample[:10]
    if not sample:
        return set()

    bullet_block = "\n".join(f"- {item}" for item in sample)
    system_prompt = "\n".join([
        "Bạn là bộ phân loại metadata của tài liệu.",
        "Hãy xác định những cụm từ ngắn biểu thị thông tin hành chính (ví dụ: giảng viên, nhận xét, ngày tháng, điểm số, ký tên, địa điểm).",
        "Chỉ trả về JSON hợp lệ dạng {\"noise\": [\"cụm 1\", ...]}.",
        "Không liệt kê các chủ đề học thuật, chỉ chọn metadata/administrative.",
    ])

    user_prompt = "\n".join([
        "Các đoạn trích tiêu biểu:",
        bullet_block,
        "Liệt kê tối đa 12 cụm cần loại bỏ; nếu không có thì trả về danh sách rỗng.",
    ])

    try:
        raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
        parsed = extract_json_tree(raw)
    except Exception:
        parsed = None

    candidates: list[str] = []
    if isinstance(parsed, dict):
        noise = parsed.get("noise") or parsed.get("metadata") or parsed.get("admin")
        if isinstance(noise, (list, tuple)):
            candidates.extend(str(item) for item in noise)
    elif isinstance(parsed, list):
        candidates.extend(str(item) for item in parsed)

    cleaned = {candidate.strip().lower() for candidate in candidates if candidate and len(candidate.strip()) >= 3}
    return {item for item in cleaned if item}


def _estimate_depth(content_segments: list[str]) -> int | None:
    if not content_segments:
        return 3

    word_counts = [len(segment.split()) for segment in content_segments if segment]
    total_words = sum(word_counts)
    if total_words <= 0:
        return 3

    avg_words = total_words / max(len(word_counts), 1)
    unique_segments = len({segment.strip().lower() for segment in content_segments if segment})

    if total_words < 250:
        return 4
    if total_words < 750:
        return 5
    if total_words < 1500:
        return 6

    # Nếu nội dung rất lớn, cho phép độ sâu linh hoạt (None = không giới hạn cứng)
    if avg_words > 160 or unique_segments > 18:
        return None

    return 7


def _estimate_expansion_budget(content_segments: list[str]) -> int:
    if not content_segments:
        return MAX_EXPANSION_CALLS_BASE

    base = MAX_EXPANSION_CALLS_BASE
    scaled = len(content_segments) * 3
    word_total = sum(len(segment.split()) for segment in content_segments if segment)

    if word_total > 1500:
        scaled += 12
    if word_total > 2500:
        scaled += 18

    return max(base, min(120, scaled))


def _literal_eval_json(body: str):
    """Fallback parser using ast.literal_eval for JSON-like strings."""
    try:
        safe = re.sub(r"\btrue\b", "True", body, flags=re.I)
        safe = re.sub(r"\bfalse\b", "False", safe, flags=re.I)
        safe = re.sub(r"\bnull\b", "None", safe, flags=re.I)
        return ast.literal_eval(safe)
    except Exception:
        return None


def _fallback_root_from_segments(segments: list[str], noise_terms: set[str] | None = None) -> str:
    for segment in segments or []:
        snippet = segment.strip()
        if not snippet:
            continue
        candidate = re.split(r"[\.:\-–|]", snippet)[0].strip()
        if len(candidate) < 4:
            continue
        if _is_noise_topic(candidate, noise_terms):
            continue
        return candidate
    return "Mind Map"


def extract_json_tree(raw: str) -> dict:
    """
    Tách khối JSON tree nested từ response.
    Có nhiều tầng fallback khi SLM sinh thêm text.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response")

    # 1) Ưu tiên block ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)```", raw, flags=re.I)
    body = m.group(1).strip() if m else raw.strip()

    # 2) Lấy block {...} lớn nhất
    m2 = re.search(r"(\{[\s\S]*\})", body, flags=re.S)
    body = m2.group(1) if m2 else body
    body = _remove_invalid_control_chars(body)

    if not body.strip():
        raise ValueError("Empty JSON body")

    # 3) Cố parse JSON
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Thử escape các dấu ngoặc kép chưa được escape trong nội dung
        escaped_body = _escape_inner_quotes(body)
        escaped_body = _remove_invalid_control_chars(escaped_body)
        try:
            return json.loads(escaped_body)
        except json.JSONDecodeError:
            body = escaped_body

        # Thử chèn dấu phẩy bị thiếu giữa các object liền kề
        fixed_commas = _insert_missing_commas(body)
        fixed_commas = _remove_invalid_control_chars(fixed_commas)
        if fixed_commas != body:
            try:
                return json.loads(fixed_commas)
            except json.JSONDecodeError:
                body = fixed_commas

        # Thử literal eval (cho trường hợp dùng dấu nháy đơn, hoặc thiếu dấu phẩy nhỏ)
        literal_obj = _literal_eval_json(body)
        if literal_obj is not None:
            return literal_obj

        # Nếu vẫn fail thì clean comment, markdown, bullet
        lines = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(("*", "-", "#")):
                continue
            if s.lower().startswith(("note:", "warning:", "gợi ý", "**")):
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = _remove_invalid_control_chars(cleaned)
        return json.loads(cleaned)


def _generate_root_topic(content_segments: list[str], noise_terms: set[str], model: str | None) -> str:
    sample = content_segments[:8]
    bullet_block = "\n".join(f"- {item}" for item in sample)
    noise_display = ", ".join(sorted(noise_terms)) if noise_terms else "không"

    system_prompt = "\n".join([
        "Bạn là chuyên gia tạo sơ đồ tư duy với phương pháp iterative prompting.",
        "Giai đoạn hiện tại: xác định nút gốc duy nhất đại diện cho chủ đề học thuật chính.",
        "BỎ QUA mọi thông tin hành chính (giảng viên, nhận xét, điểm số, ngày tháng, địa điểm).",
        "Luôn trả lời bằng tiếng Việt, kể cả khi nội dung nguồn pha trộn ngôn ngữ.",
        "Chỉ trả về JSON hợp lệ dạng {\"root\": \"...\", \"alternatives\": [..]} (alternatives tùy chọn)."
    ])

    user_prompt = "\n".join([
        "Tóm tắt nội dung chỉ để chọn root (không tạo branches ở bước này):",
        bullet_block,
        f"Các cụm cần tránh: {noise_display}",
        "Root phải mô tả chính xác chủ đề học thuật trung tâm của tài liệu." ,
        "Không dùng các cụm liên quan đến chấm điểm, giảng viên, hay metadata hành chính." ,
    ])

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
    try:
        root_obj = extract_json_tree(raw)
    except Exception as exc:
        raise ValueError(f"Không thể trích JSON root: {exc}")

    candidates: list[str] = []
    if isinstance(root_obj, dict):
        for key in ("root", "name", "title"):
            value = root_obj.get(key)
            if value:
                candidates.append(str(value).strip())
        alt = root_obj.get("alternatives")
        if isinstance(alt, (list, tuple)):
            candidates.extend(str(item).strip() for item in alt if item)
    elif isinstance(root_obj, list):
        candidates.extend(str(item).strip() for item in root_obj if item)

    for candidate in candidates:
        if not candidate:
            continue
        if _is_noise_topic(candidate, noise_terms):
            continue
        return candidate

    return _fallback_root_from_segments(content_segments, noise_terms)


def _expand_leaf_node(
    path: list[str],
    node: dict,
    content_segments: list[str],
    blocked_names: set[str],
    depth: int,
    noise_terms: set[str],
    max_depth: int | None,
    model: str | None,
) -> dict:
    path_str = " > ".join(path)
    current_children = [str(child.get("name", "")).strip() for child in node.get("children", []) or [] if child]
    blocked_display = sorted(name for name in (blocked_names or set()) if name)

    min_children = MIN_ROOT_CHILDREN if depth == 0 else MIN_INNER_CHILDREN
    max_children = max(3, 6 - min(depth, 3))
    range_hint = f"{min_children}-{max_children}"
    depth_hint = max_depth if max_depth is not None else "linh hoạt"

    context = content_segments or []
    if not context:
        context = [""]
    bullet_context = "\n".join(f"- {segment}" for segment in context)

    instructions = [
        "Bạn đang ở pha mở rộng của kỹ thuật iterative prompting.",
        "Mục tiêu: mở rộng nút hiện tại bằng các chủ đề con cụ thể, được hỗ trợ bởi nội dung nguồn.",
        "BẮT BUỘC tránh lặp lại những nhánh đã có hoặc các cụm hành chính (giảng viên, điểm, ngày tháng, địa điểm).",
        "Tên nhánh dài 2-8 từ, tập trung vào khái niệm học thuật; detail (nếu cần) tối đa 16 từ.",
        "Chỉ thêm detail khi nó bổ sung bối cảnh; bỏ field detail nếu không cần.",
        "Nếu không còn nội dung phù hợp, trả về {\"expand\": false, \"children\": []}.",
        "Luôn trả về JSON hợp lệ duy nhất với khóa expand (boolean) và children (list).",
        "Toàn bộ tên nhánh và detail phải viết bằng tiếng Việt tự nhiên, không dịch sang tiếng Anh.",
    ]
    system_prompt = "\n".join(instructions)

    user_lines = [
        f"Đường dẫn nút: {path_str or 'Root'}",
        f"Độ sâu hiện tại: {depth}",
        f"Giới hạn độ sâu gợi ý: {depth_hint}",
        f"Nhánh đã có: {', '.join(current_children) if current_children else 'Chưa có'}",
        f"Tên cần tránh (toàn cục): {', '.join(blocked_display[:15]) if blocked_display else 'Không'}",
        f"Số nhánh cần đề xuất: khoảng {range_hint} (có thể linh hoạt nếu nội dung hạn chế)",
        "Nguồn nội dung liên quan:",
        bullet_context,
        "Đầu ra mẫu: {\"expand\": true, \"children\": [{\"name\": \"Khái niệm chính\", \"detail\": \"Mô tả ngắn\"}]}"
    ]
    user_prompt = "\n".join(user_lines)

    attempts = 2
    last_error: Exception | None = None
    result = None

    for attempt in range(attempts):
        system_prompt_current = system_prompt
        user_prompt_current = user_prompt
        if attempt and last_error:
            system_prompt_current += (
                "\nLưu ý: phản hồi trước không phải JSON hợp lệ ("
                + str(last_error)
                + "). Chỉ trả về JSON duy nhất theo mẫu đã nêu."
            )
            user_prompt_current += "\n\n⚠️ Bổ sung: JSON lần trước lỗi, hãy trả về đúng schema {\"expand\": bool, \"children\": [...]}"

        raw = run_ollama_chat(system_prompt_current, user_prompt_current, model=model or SLM_MODEL)
        try:
            result = extract_json_tree(raw)
            break
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                result = None
            else:
                continue

    if result is None:
        print(f"⚠️ Không thể mở rộng nhánh {path_str}: {last_error}")
        return {"expand": False, "children": []}

    if not isinstance(result, dict):
        print(f"⚠️ Phản hồi mở rộng không phải JSON object ({path_str}): {result}")
        return {"expand": False, "children": []}

    expand_flag = _to_bool(result.get("expand"), default=True)
    children = result.get("children")
    if not isinstance(children, list):
        children = []

    return {"expand": expand_flag, "children": children}


def _sanitize_node(node, noise_terms: set[str] | None = None, allow_noise: bool = False):
    if not isinstance(node, dict):
        return None

    name = str(node.get("name", "")).strip()
    if not name:
        name = "Untitled"

    if not allow_noise and _is_noise_topic(name, noise_terms):
        return None

    detail = node.get("detail")
    if detail is not None:
        detail = str(detail).strip()
        if not detail:
            detail = None

    children = []
    for child in node.get("children", []) or []:
        sanitized_child = _sanitize_node(child, noise_terms)
        if sanitized_child:
            children.append(sanitized_child)

    def _merge_child_lists(existing: list[dict], new: list[dict]) -> list[dict]:
        if not existing:
            return list(new or [])
        index = {c.get("name", "").strip().lower(): c for c in existing if isinstance(c, dict)}
        for candidate in new or []:
            if not isinstance(candidate, dict):
                continue
            key = candidate.get("name", "").strip().lower()
            if not key:
                continue
            target = index.get(key)
            if target:
                if candidate.get("detail") and not target.get("detail"):
                    target["detail"] = candidate["detail"]
                merged_children = _merge_child_lists(target.get("children", []), candidate.get("children", []))
                target["children"] = merged_children
            else:
                index[key] = candidate
                existing.append(candidate)
        return existing

    deduped = []
    seen = {}
    for child in children:
        key = child.get("name", "").strip().lower()
        if not key:
            continue
        existing_child = seen.get(key)
        if existing_child:
            if child.get("detail") and not existing_child.get("detail"):
                existing_child["detail"] = child["detail"]
            merged_children = _merge_child_lists(existing_child.get("children", []), child.get("children", []))
            existing_child["children"] = merged_children
        else:
            seen[key] = child
            deduped.append(child)
    children = deduped

    if name.startswith("TC") and not children:
        return None

    if not allow_noise and not children and not detail and len(name) <= 3:
        return None

    sanitized = {"name": name, "children": children}
    if detail:
        sanitized["detail"] = detail
    return sanitized


def _sanitize_tree(obj, noise_terms: set[str] | None = None):
    if isinstance(obj, dict):
        root = _sanitize_node(obj, noise_terms, allow_noise=True) or {"name": "Mind Map", "children": []}
        if not root.get("name"):
            root["name"] = "Mind Map"
        if not isinstance(root.get("children"), list):
            root["children"] = []
        return root
    if isinstance(obj, list):
        children = []
        for item in obj:
            sanitized = _sanitize_node(item, noise_terms)
            if sanitized:
                children.append(sanitized)
        return {"name": "Mind Map", "children": children}
    return {"name": "Mind Map", "children": []}


def _count_nodes(node):
    if not isinstance(node, dict):
        return 0
    total = 1
    for child in node.get("children", []):
        total += _count_nodes(child)
    return total


def _needs_enrichment(tree: dict, content_segments: list[str]) -> bool:
    top_children = tree.get("children", [])
    topic_count = len(top_children)
    if topic_count >= 4:
        return False
    if len(content_segments) <= 4:
        return False
    if topic_count == 0:
        return True
    avg_children = sum(len(child.get("children", [])) for child in top_children) / max(topic_count, 1)
    if avg_children < 1 and len(content_segments) > 6:
        return True
    if len(content_segments) >= 8:
        target_nodes = min(30, max(10, int(len(content_segments) * 0.8)))
        if _count_nodes(tree) < target_nodes:
            return True
    return False


def _expand_tree(tree: dict, bullet_block: str, model: str | None, noise_terms: set[str] | None = None):
    try:
        current = json.dumps(tree, ensure_ascii=False)
    except TypeError:
        current = str(tree)

    system_prompt = "\n".join([
        "Bạn là AI mindmap chuyên nghiệp. Dựa trên mindmap hiện có, mở rộng thành phiên bản đầy đủ hơn và giữ đúng JSON hợp lệ.",
        "- Bảo toàn khung chính nhưng có thể đổi tên cho rõ và bổ sung các nhánh còn thiếu.",
        "- Không thêm thông tin hành chính (tên trường, họ tên, ngày tháng...) trừ khi là ý trọng tâm.",
        "- Không tạo node trùng lặp; mỗi node phải có 'name' và chỉ thêm 'detail' ngắn khi cần.",
        "- TẤT CẢ tiêu đề và detail trả về phải bằng tiếng Việt, bám sát ngôn ngữ tài liệu.",
        "- Trả về DUY NHẤT một block ```json ...``` với cấu trúc mindmap hoàn chỉnh."
    ])
    user_prompt = "\n".join([
        "Mindmap hiện tại (JSON gốc):",
        "```json",
        current,
        "```",
        "Các ý liệu chi tiết (đã lọc theo nội dung):",
        bullet_block,
        "Hãy trả về mindmap đã mở rộng theo phong cách NotebookLM trong block ```json``` duy nhất, tất cả nhãn/detail bằng tiếng Việt."
    ])


    try:
        raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
        expanded_obj = extract_json_tree(raw)
        return _sanitize_tree(expanded_obj, noise_terms)
    except Exception as e:
        print(f"⚠️ Mindmap enrichment failed: {e}")
        return None


def _build_mindmap_single_shot(content_segments: list[str], noise_terms: set[str], model: str | None) -> dict:
    system_prompt = "\n".join([
        "Bạn là AI mindmap chuyên nghiệp. Trả về DUY NHẤT JSON tree nested (```json ...```), bảo đảm JSON hợp lệ.",
        "- Đặt root theo chủ đề trọng tâm, không giữ nguyên các tiêu đề hành chính/bìa.",
        "- Phân tích nội dung và xác định số lượng nhánh linh hoạt theo phong cách Google NotebookLM.",
        "- Bỏ qua thông tin hành chính (tên trường, họ tên, ngày tháng, mục chấm điểm...) trừ khi nó là nội dung chính.",
        "- TẤT CẢ tiêu đề và detail phải bằng tiếng Việt, ưu tiên cùng ngôn ngữ với tài liệu.",
        "- Cấu trúc gợi ý: Root → Chủ đề → Nhánh con → Chi tiết (độ sâu ≤ 4 nếu cần).",
        "- Mỗi node phải có trường name; detail chỉ dùng cho mô tả ngắn (dùng dấu ' thay vì \" khi trích dẫn).",
        "- Không lặp node cùng tên; mỗi nhánh đại diện một ý riêng biệt.",
        "- Các nhánh logic và cân đối: thường 4-7 chủ đề chính, mỗi chủ đề 2-5 nhánh phụ, nhưng linh hoạt theo nội dung.",
        "- Ví dụ JSON: {\"name\":\"Chủ đề chính\",\"children\":[{\"name\":\"Chủ đề phụ\",\"children\":[{\"name\":\"Ý chính\",\"detail\":\"Mô tả\"}]}]}"
    ])
    bullet_block = "\n".join(f"- {item}" for item in content_segments)
    base_user_prompt = "\n".join([
        "Sinh mindmap phong cách NotebookLM từ các ý dưới đây, giữ đúng trình tự logic.",
        "Gom nhóm các ý liên quan thành chủ đề lớn rồi phân rã thành nhánh phụ và chi tiết rõ ràng.",
        "Bỏ qua phần bìa, tiêu đề hành chính, họ tên, chữ ký, ngày tháng nếu không liên quan nội dung.",
        "BẮT BUỘC dùng tiếng Việt tự nhiên cho mọi tiêu đề và detail (giữ thuật ngữ chuyên môn khi cần).",
        "Dữ liệu tham khảo:",
        bullet_block
    ])

    last_error = None
    tree_obj = None

    for attempt in range(2):
        system_prompt_final = system_prompt
        if attempt and last_error:
            system_prompt_final += (
                "\nLưu ý: phản hồi trước không phải JSON hợp lệ ("
                + str(last_error)
                + "). Chỉ trả về block ```json ...``` chứa mindmap hợp lệ, không thêm text khác."
            )

        raw = run_ollama_chat(system_prompt_final, base_user_prompt, model=model or SLM_MODEL)
        try:
            tree_obj = extract_json_tree(raw)
            break
        except Exception as err:
            last_error = err
            if attempt == 1:
                raise err

    tree = _sanitize_tree(tree_obj, noise_terms)

    if _needs_enrichment(tree, content_segments):
        enriched = _expand_tree(tree, bullet_block, model, noise_terms)
        if enriched:
            tree = enriched

    return tree


def _build_mindmap_iterative(content_segments: list[str], noise_terms: set[str], model: str | None) -> dict:
    root_name = _generate_root_topic(content_segments, noise_terms, model)
    if not root_name:
        raise ValueError("Không xác định được root topic")

    root = {"name": root_name, "children": []}
    used_titles = {root_name.strip().lower()}
    queue: deque[dict] = deque([{"node": root, "path": [root_name], "depth": 0}])
    expansion_attempts: dict[tuple[str, ...], int] = {}
    steps = 0
    max_depth = _estimate_depth(content_segments)
    max_steps = _estimate_expansion_budget(content_segments)

    while queue and steps < max_steps:
        current = queue.popleft()
        node = current["node"]
        depth = current["depth"]
        path = current["path"]
        path_key = tuple(path)

        if max_depth is not None and depth >= max_depth:
            continue

        expansion_attempts[path_key] = expansion_attempts.get(path_key, 0) + 1
        context_segments = _context_for_path(path, content_segments)

        blocked_names = set(used_titles)
        for child in node.get("children", []) or []:
            title = str(child.get("name", "")).strip().lower()
            if title:
                blocked_names.add(title)

        expand_result = _expand_leaf_node(
            path,
            node,
            context_segments,
            blocked_names,
            depth,
            noise_terms,
            max_depth,
            model,
        )
        raw_children = expand_result.get("children", [])
        should_expand = _to_bool(expand_result.get("expand"), default=True)

        valid_children: list[dict] = []
        for raw_child in raw_children:
            sanitized = _sanitize_node(raw_child, noise_terms)
            if not sanitized:
                continue
            child_name = sanitized.get("name", "").strip()
            if not child_name:
                continue
            lowered = child_name.lower()
            if lowered in blocked_names or _is_noise_topic(child_name, noise_terms):
                continue
            blocked_names.add(lowered)
            used_titles.add(lowered)
            valid_children.append(sanitized)

        if valid_children:
            node.setdefault("children", []).extend(valid_children)

        total_children = len(node.get("children", []) or [])
        min_children = MIN_ROOT_CHILDREN if depth == 0 else MIN_INNER_CHILDREN
        max_children = max(3, 6 - min(depth, 3))
        expand_targets = list(valid_children)
        if len(expand_targets) > max_children:
            expand_targets = expand_targets[:max_children]

        if should_expand and expand_targets and (max_depth is None or depth + 1 <= max_depth):
            for child in expand_targets:
                queue.append({
                    "node": child,
                    "path": path + [child.get("name", "")],
                    "depth": depth + 1,
                })

        if (
            should_expand
            and total_children < min_children
            and expansion_attempts[path_key] < 3
            and (max_depth is None or depth < max_depth)
        ):
            queue.append({"node": node, "path": path, "depth": depth})

        steps += 1

    sanitized = _sanitize_tree(root, noise_terms)
    if not sanitized.get("children"):
        raise ValueError("Iterative builder trả về cây rỗng")
    return sanitized


def _format_segments_for_prompt(segments: list[str], limit: int = CRITIC_SEGMENTS) -> str:
    if not segments:
        return "-"
    lines = []
    for idx, segment in enumerate(segments[:limit]):
        snippet = segment.strip()
        if not snippet:
            continue
        lines.append(f"[{idx + 1}] {snippet}")
    return "\n".join(lines) if lines else "-"


def _run_critic(system_prompt: str, user_prompt: str, noise_terms: set[str] | None, model: str | None) -> dict | None:
    attempts = 2
    last_error: Exception | None = None

    for attempt in range(attempts):
        system_prompt_current = system_prompt
        user_prompt_current = user_prompt
        if attempt and last_error:
            system_prompt_current += (
                "\nLưu ý: phản hồi trước không phải JSON hợp lệ ("
                + str(last_error)
                + "). Chỉ trả về duy nhất block ```json``` với schema {name, detail?, children} và toàn bộ nội dung bằng tiếng Việt."
            )
            user_prompt_current += "\n\n⚠️ JSON lần trước lỗi, hãy trả về đúng block ```json``` duy nhất cho mindmap (tiếng Việt 100%)."

        raw = run_ollama_chat(system_prompt_current, user_prompt_current, model=model or SLM_MODEL)
        try:
            candidate = extract_json_tree(raw)
            return _sanitize_tree(candidate, noise_terms)
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise



def _apply_factuality_critic(tree: dict, content_segments: list[str], noise_terms: set[str], model: str | None) -> dict | None:
    if not tree or not tree.get("children"):
        return tree

    tree_dump = json.dumps(tree, ensure_ascii=False, indent=2)
    context_block = _format_segments_for_prompt(content_segments)

    system_prompt = "\n".join([
        "Bạn là factuality critic cho mind map sinh bởi kỹ thuật iterative prompting.",
        "Mục tiêu: chỉ giữ lại các nhánh có bằng chứng trong văn bản, hoặc hợp nhất/gỡ bỏ phần không được hỗ trợ.",
        "- Mỗi đường dẫn root→leaf cần được hỗ trợ bởi ít nhất một câu trong danh sách trích đoạn.",
        "- Nếu một nhánh thiếu dẫn chứng, hãy xóa hoặc gộp vào nhánh phù hợp khác.",
        "- Viết citation ngay trong trường detail của nút lá theo định dạng [1], [2-3] tương ứng với chỉ số câu (1-based).",
        "- Không tạo thông tin mới không xuất hiện trong văn bản.",
        "- Giữ toàn bộ tiêu đề và detail bằng tiếng Việt nhất quán; chỉ giữ nguyên thuật ngữ chuyên ngành khi cần.",
        "Chỉ trả về DUY NHẤT một block JSON hợp lệ cho mind map đã chỉnh sửa (schema {name, detail?, children}).",
    ])

    user_prompt = "\n".join([
        "Mind map hiện tại:",
        "```json",
        tree_dump,
        "```",
        "Các đoạn văn bản tham chiếu:",
        context_block,
        "Hãy rà soát factuality, giữ nguyên cấu trúc tối đa nhưng loại bỏ/điều chỉnh nhánh thiếu dẫn chứng.",
    ])

    try:
        return _run_critic(system_prompt, user_prompt, noise_terms, model)
    except Exception as exc:
        print(f"⚠️ Factuality critic bỏ qua do lỗi: {exc}")
        return tree


def _apply_local_structure_critic(tree: dict, content_segments: list[str], noise_terms: set[str], model: str | None) -> dict | None:
    if not tree or not tree.get("children"):
        return tree

    tree_dump = json.dumps(tree, ensure_ascii=False, indent=2)
    context_block = _format_segments_for_prompt(content_segments)

    system_prompt = "\n".join([
        "Bạn là local-structure critic cho mind map.",
        "Yêu cầu: đảm bảo mỗi đường dẫn root→leaf kết thúc ở một khái niệm cụ thể, tránh trùng lặp với tiêu đề cha.",
        "- Nếu tiêu đề lá quá chung chung (ví dụ trùng với cha, hoặc chỉ là 'Giới thiệu'), hãy đổi tên cho cụ thể hoặc hợp nhất.",
        "- Có thể thêm một lớp con mới nếu cần để đạt tới ý cụ thể, nhưng giữ độ sâu ≤ 4.",
        "- detail (nếu có) phải ngắn gọn ≤ 16 từ và có thể tái sử dụng citation hiện có.",
        "- Không thay đổi ý nghĩa những nhánh đã qua factuality (không được bịa thêm nội dung mới).",
        "- Bảo đảm mọi nhãn node/detail ở kết quả vẫn bằng tiếng Việt tự nhiên, không quay lại tiếng Anh.",
        "Đầu ra bắt buộc: DUY NHẤT một block JSON hợp lệ với schema {name, detail?, children}.",
    ])

    user_prompt = "\n".join([
        "Mind map sau bước factuality:",
        "```json",
        tree_dump,
        "```",
        "Các đoạn văn bản tham chiếu (dùng để kiểm tra mức độ cụ thể):",
        context_block,
        "Chuẩn hóa cấu trúc cục bộ theo yêu cầu trên, ưu tiên giữ nguyên tên khi đã đủ cụ thể.",
    ])

    try:
        return _run_critic(system_prompt, user_prompt, noise_terms, model)
    except Exception as exc:
        print(f"⚠️ Local structure critic bỏ qua do lỗi: {exc}")
        return tree


def _apply_global_structure_critic(tree: dict, content_segments: list[str], noise_terms: set[str], model: str | None) -> dict | None:
    if not tree or not tree.get("children"):
        return tree

    tree_dump = json.dumps(tree, ensure_ascii=False, indent=2)
    context_block = _format_segments_for_prompt(content_segments)

    system_prompt = "\n".join([
        "Bạn là global-structure critic cho mind map.",
        "Trước khi chỉnh sửa, hãy hình dung mind map dưới dạng mục lục (ToC) để kiểm tra cấp độ trừu tượng.",
        "- Tối ưu số nhánh cấp 1 khoảng 4-7 (linh hoạt theo nội dung).",
        "- Đảm bảo phân nhóm logic, cân đối số nhánh con giữa các ngành chính.",
        "- Có thể đổi tên node cấp 1 cho rõ ràng hoặc tái phân bổ nhánh con, nhưng không được thêm nội dung ngoài văn bản.",
        "- Nếu tồn tại nhánh riêng lẻ yếu (ít con, trùng chủ đề), hãy hợp nhất vào nhánh phù hợp hơn.",
        "- Giữ nguyên citation trong detail nếu đã có.",
        "- Kết quả cuối cùng phải dùng tiếng Việt tự nhiên cho mọi tiêu đề, detail, kể cả khi tài liệu gốc chứa tiếng Anh.",
        "Đầu ra bắt buộc: DUY NHẤT một block JSON hợp lệ (schema {name, detail?, children}).",
    ])

    user_prompt = "\n".join([
        "Mind map sau bước local structure:",
        "```json",
        tree_dump,
        "```",
        "Các đoạn văn bản tham chiếu (để cân nhắc bố cục):",
        context_block,
        "Tối ưu cấu trúc tổng thể nhưng tránh tạo node mới không có trong nội dung.",
    ])

    try:
        return _run_critic(system_prompt, user_prompt, noise_terms, model)
    except Exception as exc:
        print(f"⚠️ Global structure critic bỏ qua do lỗi: {exc}")
        return tree


def _apply_mindmap_critics(tree: dict, content_segments: list[str], noise_terms: set[str], model: str | None) -> dict:
    if not tree or not isinstance(tree, dict):
        return tree
    if not tree.get("children"):
        return tree

    critics = (
        _apply_factuality_critic,
        _apply_local_structure_critic,
        _apply_global_structure_critic,
    )

    refined = tree
    for critic in critics:
        updated = critic(refined, content_segments, noise_terms, model)
        if isinstance(updated, dict) and updated.get("children"):
            refined = updated

    return refined


def get_nested_mindmap(chunks: list[str], model: str = None) -> dict:
    """Sinh mindmap nested với iterative prompting, fallback single-shot."""
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        raise ValueError("Không có dữ liệu nguồn để tạo mindmap")

    noise_terms = _detect_noise_terms(prepared_chunks, model)
    filtered_chunks = _filter_segments_by_noise(prepared_chunks, noise_terms)
    if not filtered_chunks:
        filtered_chunks = prepared_chunks

    builders = (
        ("iterative", _build_mindmap_iterative),
        ("single_shot", _build_mindmap_single_shot),
    )
    last_error: Exception | None = None

    for label, builder in builders:
        try:
            tree = builder(filtered_chunks, noise_terms, model)
            if tree and tree.get("children"):
                return _apply_mindmap_critics(tree, filtered_chunks, noise_terms, model)
        except Exception as exc:
            print(f"⚠️ Mindmap builder {label} failed: {exc}")
            last_error = exc

    if last_error:
        raise last_error

    return {"name": "Mind Map", "children": []}


def get_main_branches(chunks: list[str], model: str = None) -> list[str]:
    """
    Fallback: Lấy ra 3-5 mục chính từ nội dung khi JSON tree lỗi.
    """
    system_prompt = (
        "Bạn là AI tạo mind map. BẮT BUỘC trả về DUY NHẤT một JSON list (```json ...```).\n"
        "Ví dụ: ```json\n[\"Mục 1\",\"Mục 2\"]\n```"
    )
    user_prompt = "Liệt kê 3-5 mục chính (ngắn gọn 2-4 từ) từ nội dung:\n\n" + "\n\n".join(chunks[:6])

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)

    m = re.search(r"(\[.*?\])", raw or "", flags=re.S)
    try:
        return json.loads(m.group(1)) if m else []
    except Exception:
        return []


def flatten_mindmap(tree) -> list[dict]:
    """
    Flatten cây mindmap thành list node {id, parent, title}.
    """
    flat_nodes = []

    def dfs(node, parent=None, index=0):
        if not isinstance(node, dict):
            return
        title = (node.get("name", "") or "").strip() or "Untitled"
        node_id = "root" if parent is None else f"{parent}-{index}"
        flat_nodes.append({"id": node_id, "parent": parent, "title": title})
        for idx, child in enumerate(node.get("children", [])):
            dfs(child, node_id, idx)

    dfs(tree)
    return flat_nodes


def generate_mindmap_flat(chunks: list[str], model: str = None) -> list[dict]:
    """
    Sinh mindmap dạng phẳng (flat nodes).
    Có fallback sang main branches nếu JSON nested lỗi.
    """
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        return [
            {"id": "root", "parent": None, "title": "Mind Map"},
            {"id": "root-0", "parent": "root", "title": "Không có dữ liệu"}
        ]

    try:
        tree = get_nested_mindmap(prepared_chunks, model=model)
    except Exception as e:
        print(f"⚠️ JSON lỗi trong generate_mindmap: {e} — fallback sang main branches")
        mains = get_main_branches(prepared_chunks, model=model)
        tree = {"name": "Mind Map", "children": [{"name": m, "children": []} for m in mains]}

    return flatten_mindmap(tree)


def generate_mindmap_cmgn(chunks: list[str], model: str = None) -> list[dict]:
    """Sinh mindmap theo phương pháp CMGN (Coreference-Guided)."""
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        return [
            {"id": "root", "parent": None, "title": "Mind Map"},
            {"id": "root-0", "parent": "root", "title": "Không có dữ liệu"}
        ]

    noise_terms = _detect_noise_terms(prepared_chunks, model)
    filtered_chunks = _filter_segments_by_noise(prepared_chunks, noise_terms) or prepared_chunks

    sentences = _extract_sentences_from_segments(filtered_chunks)
    if not sentences:
        print("⚠️ CMGN: không tạo được danh sách câu, fallback sang generate_mindmap_flat")
        return generate_mindmap_flat(filtered_chunks, model=model)

    try:
        coref_graph = _generate_coreference_graph(sentences, model)
    except Exception as exc:
        print(f"⚠️ CMGN: lỗi dựng coreference graph ({exc}), fallback sang generate_mindmap_flat")
        return generate_mindmap_flat(filtered_chunks, model=model)

    try:
        tree = _generate_mindmap_from_coreference_graph(coref_graph, filtered_chunks, noise_terms, model)
    except Exception as exc:
        print(f"⚠️ CMGN: lỗi sinh mindmap từ graph ({exc}), fallback nested builder")
        try:
            tree = get_nested_mindmap(filtered_chunks, model=model)
        except Exception as nested_exc:
            print(f"⚠️ CMGN fallback nested cũng lỗi: {nested_exc}")
            mains = get_main_branches(filtered_chunks, model=model)
            tree = {"name": "Mind Map", "children": [{"name": m, "children": []} for m in mains]}

    refined_tree = _apply_mindmap_critics(tree, filtered_chunks, noise_terms, model)
    return flatten_mindmap(refined_tree)
