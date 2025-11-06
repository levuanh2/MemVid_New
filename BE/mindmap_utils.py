# mindmap_utils.py
import re
import json
import ast
from collections import deque
from ollama_utils import run_ollama_chat, SLM_MODEL


MAX_SEGMENTS_FOR_MINDMAP = 24
MAX_CHARS_FOR_MINDMAP = 8000
MAX_MINDMAP_DEPTH = 3
MAX_EXPANSION_CALLS = 18
MIN_ROOT_CHILDREN = 4
MIN_INNER_CHILDREN = 2
CONTEXT_SEGMENTS_PER_NODE = 8

ADMIN_SKIP_PATTERNS = [
    r"\bnhận\s*xét\b",
    r"\bgiảng\s*viên\b",
    r"\bgvhd\b",
    r"\bbằng\s*số\b",
    r"\bbằng\s*chữ\b",
    r"\bchấm\s*điểm\b",
    r"\btp\.?\s*h\.??c\.??m\b",
    r"\btháng\b",
    r"\bngày\b",
    r"\bký\s*(?:tên|duyệt)\b",
    r"\bmục\s*lục\b",
    r"\bdanh\s*mục\b",
]

ADMIN_TOPIC_PATTERNS = [
    r"\bnhận\s*xét\b",
    r"\bgiảng\s*viên\b",
    r"\bgvhd\b",
    r"\bbằng\s*(?:số|chữ)\b",
    r"\bchấm\s*điểm\b",
    r"\btp\.?\s*h\.??c\.??m\b",
    r"\btháng\b",
    r"\bngày\b",
]


def _is_noise_topic(name: str) -> bool:
    if not name:
        return True
    lowered = name.strip().lower()
    if not lowered:
        return True
    return any(re.search(pattern, lowered, flags=re.I) for pattern in ADMIN_TOPIC_PATTERNS)


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
    filtered: list[str] = []
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

    if not prepared:
        return []

    for item in prepared:
        if any(re.search(pattern, item, flags=re.I) for pattern in ADMIN_SKIP_PATTERNS):
            continue
        filtered.append(item)

    return filtered or prepared


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


def _literal_eval_json(body: str):
    """Fallback parser using ast.literal_eval for JSON-like strings."""
    try:
        safe = re.sub(r"\btrue\b", "True", body, flags=re.I)
        safe = re.sub(r"\bfalse\b", "False", safe, flags=re.I)
        safe = re.sub(r"\bnull\b", "None", safe, flags=re.I)
        return ast.literal_eval(safe)
    except Exception:
        return None


def _fallback_root_from_segments(segments: list[str]) -> str:
    for segment in segments or []:
        snippet = segment.strip()
        if not snippet:
            continue
        candidate = re.split(r"[\.:\-–|]", snippet)[0].strip()
        if len(candidate) < 4:
            continue
        if _is_noise_topic(candidate):
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

    if not body.strip():
        raise ValueError("Empty JSON body")

    # 3) Cố parse JSON
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Thử escape các dấu ngoặc kép chưa được escape trong nội dung
        escaped_body = _escape_inner_quotes(body)
        try:
            return json.loads(escaped_body)
        except json.JSONDecodeError:
            body = escaped_body

        # Thử chèn dấu phẩy bị thiếu giữa các object liền kề
        fixed_commas = _insert_missing_commas(body)
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
        return json.loads(cleaned)


def _generate_root_topic(content_segments: list[str], model: str | None) -> str:
    sample = content_segments[:8]
    bullet_block = "\n".join(f"- {item}" for item in sample)

    system_prompt = "\n".join([
        "Bạn là chuyên gia tạo sơ đồ tư duy với phương pháp iterative prompting.",
        "Giai đoạn hiện tại: xác định nút gốc duy nhất đại diện cho chủ đề học thuật chính.",
        "BỎ QUA mọi thông tin hành chính (giảng viên, nhận xét, điểm số, ngày tháng, địa điểm).",
        "Chỉ trả về JSON hợp lệ dạng {\"root\": \"...\", \"alternatives\": [..]} (alternatives tùy chọn)."
    ])

    user_prompt = "\n".join([
        "Tóm tắt nội dung chỉ để chọn root (không tạo branches ở bước này):",
        bullet_block,
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
        if _is_noise_topic(candidate):
            continue
        return candidate

    return _fallback_root_from_segments(content_segments)


def _expand_leaf_node(
    path: list[str],
    node: dict,
    content_segments: list[str],
    blocked_names: set[str],
    depth: int,
    model: str | None,
) -> dict:
    path_str = " > ".join(path)
    current_children = [str(child.get("name", "")).strip() for child in node.get("children", []) or [] if child]
    blocked_display = sorted(name for name in (blocked_names or set()) if name)

    min_children = MIN_ROOT_CHILDREN if depth == 0 else MIN_INNER_CHILDREN
    max_children = 6 if depth == 0 else 5 if depth == 1 else 4
    range_hint = f"{min_children}-{max_children}"

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
    ]
    system_prompt = "\n".join(instructions)

    user_lines = [
        f"Đường dẫn nút: {path_str or 'Root'}",
        f"Độ sâu hiện tại: {depth}",
        f"Nhánh đã có: {', '.join(current_children) if current_children else 'Chưa có'}",
        f"Tên cần tránh (toàn cục): {', '.join(blocked_display[:15]) if blocked_display else 'Không'}",
        f"Số nhánh cần đề xuất: khoảng {range_hint} (có thể linh hoạt nếu nội dung hạn chế)",
        "Nguồn nội dung liên quan:",
        bullet_context,
        "Đầu ra mẫu: {\"expand\": true, \"children\": [{\"name\": \"Khái niệm chính\", \"detail\": \"Mô tả ngắn\"}]}"
    ]
    user_prompt = "\n".join(user_lines)

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
    try:
        result = extract_json_tree(raw)
    except Exception as exc:
        raise ValueError(f"Không thể mở rộng nhánh {path_str}: {exc}")

    if not isinstance(result, dict):
        raise ValueError(f"Phản hồi mở rộng không phải JSON object: {result}")

    expand_flag = _to_bool(result.get("expand"), default=True)
    children = result.get("children")
    if not isinstance(children, list):
        children = []

    return {"expand": expand_flag, "children": children}


def _sanitize_node(node):
    if not isinstance(node, dict):
        return None

    name = str(node.get("name", "")).strip()
    if not name:
        name = "Untitled"

    detail = node.get("detail")
    if detail is not None:
        detail = str(detail).strip()
        if not detail:
            detail = None

    children = []
    for child in node.get("children", []) or []:
        sanitized_child = _sanitize_node(child)
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

    if not children and not detail and len(name) <= 3:
        return None

    sanitized = {"name": name, "children": children}
    if detail:
        sanitized["detail"] = detail
    return sanitized


def _sanitize_tree(obj):
    if isinstance(obj, dict):
        root = _sanitize_node(obj) or {"name": "Mind Map", "children": []}
        if not root.get("name"):
            root["name"] = "Mind Map"
        if not isinstance(root.get("children"), list):
            root["children"] = []
        return root
    if isinstance(obj, list):
        children = []
        for item in obj:
            sanitized = _sanitize_node(item)
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


def _expand_tree(tree: dict, bullet_block: str, model: str | None):
    try:
        current = json.dumps(tree, ensure_ascii=False)
    except TypeError:
        current = str(tree)

    system_prompt = "\n".join([
        "Bạn là AI mindmap chuyên nghiệp. Dựa trên mindmap hiện có, mở rộng thành phiên bản đầy đủ hơn và giữ đúng JSON hợp lệ.",
        "- Bảo toàn khung chính nhưng có thể đổi tên cho rõ và bổ sung các nhánh còn thiếu.",
        "- Không thêm thông tin hành chính (tên trường, họ tên, ngày tháng...) trừ khi là ý trọng tâm.",
        "- Không tạo node trùng lặp; mỗi node phải có 'name' và chỉ thêm 'detail' ngắn khi cần.",
        "- Trả về DUY NHẤT một block ```json ...``` với cấu trúc mindmap hoàn chỉnh."
    ])
    user_prompt = "\n".join([
        "Mindmap hiện tại:",
        "```json",
        current,
        "```",
        "Các ý liệu chi tiết (đã lọc theo nội dung):",
        bullet_block,
        "Hãy trả về mindmap đã mở rộng theo phong cách NotebookLM trong block ```json``` duy nhất."
    ])


    try:
        raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
        expanded_obj = extract_json_tree(raw)
        return _sanitize_tree(expanded_obj)
    except Exception as e:
        print(f"⚠️ Mindmap enrichment failed: {e}")
        return None


def _build_mindmap_single_shot(prepared_chunks: list[str], model: str | None) -> dict:
    system_prompt = "\n".join([
        "Bạn là AI mindmap chuyên nghiệp. Trả về DUY NHẤT JSON tree nested (```json ...```), bảo đảm JSON hợp lệ.",
        "- Đặt root theo chủ đề trọng tâm, không giữ nguyên các tiêu đề hành chính/bìa.",
        "- Phân tích nội dung và xác định số lượng nhánh linh hoạt theo phong cách Google NotebookLM.",
        "- Bỏ qua thông tin hành chính (tên trường, họ tên, ngày tháng, mục chấm điểm...) trừ khi nó là nội dung chính.",
        "- Cấu trúc gợi ý: Root → Chủ đề → Nhánh con → Chi tiết (độ sâu ≤ 4 nếu cần).",
        "- Mỗi node phải có trường name; detail chỉ dùng cho mô tả ngắn (dùng dấu ' thay vì \" khi trích dẫn).",
        "- Không lặp node cùng tên; mỗi nhánh đại diện một ý riêng biệt.",
        "- Các nhánh logic và cân đối: thường 4-7 chủ đề chính, mỗi chủ đề 2-5 nhánh phụ, nhưng linh hoạt theo nội dung.",
        "- Ví dụ JSON: {\"name\":\"Chủ đề chính\",\"children\":[{\"name\":\"Chủ đề phụ\",\"children\":[{\"name\":\"Ý chính\",\"detail\":\"Mô tả\"}]}]}"
    ])
    bullet_block = "\n".join(f"- {item}" for item in prepared_chunks)
    base_user_prompt = "\n".join([
        "Sinh mindmap phong cách NotebookLM từ các ý dưới đây, giữ đúng trình tự logic.",
        "Gom nhóm các ý liên quan thành chủ đề lớn rồi phân rã thành nhánh phụ và chi tiết rõ ràng.",
        "Bỏ qua phần bìa, tiêu đề hành chính, họ tên, chữ ký, ngày tháng nếu không liên quan nội dung.",
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

    tree = _sanitize_tree(tree_obj)

    if _needs_enrichment(tree, prepared_chunks):
        enriched = _expand_tree(tree, bullet_block, model)
        if enriched:
            tree = enriched

    return tree


def _build_mindmap_iterative(prepared_chunks: list[str], model: str | None) -> dict:
    root_name = _generate_root_topic(prepared_chunks, model)
    if not root_name:
        raise ValueError("Không xác định được root topic")

    root = {"name": root_name, "children": []}
    used_titles = {root_name.strip().lower()}
    queue: deque[dict] = deque([{"node": root, "path": [root_name], "depth": 0}])
    expansion_attempts: dict[tuple[str, ...], int] = {}
    steps = 0

    while queue and steps < MAX_EXPANSION_CALLS:
        current = queue.popleft()
        node = current["node"]
        depth = current["depth"]
        path = current["path"]
        path_key = tuple(path)

        if depth >= MAX_MINDMAP_DEPTH:
            continue

        expansion_attempts[path_key] = expansion_attempts.get(path_key, 0) + 1
        context_segments = _context_for_path(path, prepared_chunks)

        blocked_names = set(used_titles)
        for child in node.get("children", []) or []:
            title = str(child.get("name", "")).strip().lower()
            if title:
                blocked_names.add(title)

        expand_result = _expand_leaf_node(path, node, context_segments, blocked_names, depth, model)
        raw_children = expand_result.get("children", [])
        should_expand = _to_bool(expand_result.get("expand"), default=True)

        valid_children: list[dict] = []
        for raw_child in raw_children:
            sanitized = _sanitize_node(raw_child)
            if not sanitized:
                continue
            child_name = sanitized.get("name", "").strip()
            if not child_name:
                continue
            lowered = child_name.lower()
            if lowered in blocked_names or _is_noise_topic(child_name):
                continue
            blocked_names.add(lowered)
            used_titles.add(lowered)
            valid_children.append(sanitized)

        if valid_children:
            node.setdefault("children", []).extend(valid_children)

        total_children = len(node.get("children", []) or [])
        min_children = MIN_ROOT_CHILDREN if depth == 0 else MIN_INNER_CHILDREN

        if should_expand and valid_children and depth + 1 < MAX_MINDMAP_DEPTH:
            for child in valid_children:
                queue.append({
                    "node": child,
                    "path": path + [child.get("name", "")],
                    "depth": depth + 1,
                })

        if should_expand and total_children < min_children and expansion_attempts[path_key] < 3:
            queue.append({"node": node, "path": path, "depth": depth})

        steps += 1

    sanitized = _sanitize_tree(root)
    if not sanitized.get("children"):
        raise ValueError("Iterative builder trả về cây rỗng")
    return sanitized


def get_nested_mindmap(chunks: list[str], model: str = None) -> dict:
    """Sinh mindmap nested với iterative prompting, fallback single-shot."""
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        raise ValueError("Không có dữ liệu nguồn để tạo mindmap")

    builders = (_build_mindmap_iterative, _build_mindmap_single_shot)
    last_error: Exception | None = None

    for builder in builders:
        try:
            tree = builder(prepared_chunks, model)
            if tree and tree.get("children"):
                return tree
        except Exception as exc:
            print(f"⚠️ Mindmap builder {builder.__name__} failed: {exc}")
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
