# mindmap_utils.py
import re
import json
import ast
from ollama_utils import run_ollama_chat, SLM_MODEL


MAX_SEGMENTS_FOR_MINDMAP = 24
MAX_CHARS_FOR_MINDMAP = 8000
ADMIN_KEYWORDS = {
    "họ và tên",
    "giảng viên",
    "gvhd",
    "trường đại học",
    "viện đào tạo",
    "tp.hcm",
    "th.s",
    "ths",
    "thạc sĩ",
    "chấm điểm",
    "ký tên",
    "mssv",
    "lớp",
}


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
    closers = {",", "}", "]", " ", "\n", "\r", "\t"}
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


def _literal_eval_json(body: str):
    """Fallback parser using ast.literal_eval for JSON-like strings."""
    try:
        safe = re.sub(r"\btrue\b", "True", body, flags=re.I)
        safe = re.sub(r"\bfalse\b", "False", safe, flags=re.I)
        safe = re.sub(r"\bnull\b", "None", safe, flags=re.I)
        return ast.literal_eval(safe)
    except Exception:
        return None


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

    lower_name = name.lower()

    if name.startswith("TC") and not children:
        return None

    if not children and not detail and len(name) <= 3:
        return None

    if not children and any(keyword in lower_name for keyword in ADMIN_KEYWORDS):
        return None

    if detail:
        lowered_detail = detail.lower()
        if any(keyword in lowered_detail for keyword in ADMIN_KEYWORDS) and not children:
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


def get_nested_mindmap(chunks: list[str], model: str = None) -> dict:
    """
    Gọi SLM để tạo nested mind map JSON có logic chặt chẽ.
    """
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        raise ValueError("Không có dữ liệu nguồn để tạo mindmap")

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
