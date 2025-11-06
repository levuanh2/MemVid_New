# mindmap_utils.py
import re
import json
from ollama_utils import run_ollama_chat, SLM_MODEL


MAX_SEGMENTS_FOR_MINDMAP = 24
MAX_CHARS_FOR_MINDMAP = 8000


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
    closers = {",", "}", "]", " ", "\n", "\r", "\t", ""}
    length = len(body)

    for idx, ch in enumerate(body):
        next_char = body[idx + 1] if idx + 1 < length else ""

        if ch == "\"" and not escape:
            if inside_string:
                if next_char in closers:
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


def get_nested_mindmap(chunks: list[str], model: str = None) -> dict:
    """
    Gọi SLM để tạo nested mind map JSON có logic chặt chẽ.
    """
    prepared_chunks = _prepare_mindmap_chunks(chunks)
    if not prepared_chunks:
        raise ValueError("Không có dữ liệu nguồn để tạo mindmap")

    system_prompt = (
        "Bạn là AI mindmap chuyên nghiệp. Trả về DUY NHẤT JSON tree nested (```json ...```), bảo đảm JSON hợp lệ.\n"
        "- Phân tích nội dung và tự xác định số lượng nhánh phù hợp (không cố định).\n"
        "- Cấu trúc gợi ý: Root → Chủ đề → Nhánh con → Chi tiết (độ sâu ≤ 4 nếu cần).\n"
        "- node bắt buộc có trường name; có thể thêm detail mô tả ngắn gọn khi hữu ích.\n"
        "- Không dùng dấu ngoặc kép chưa escape trong nội dung; nếu cần trích dẫn hãy dùng dấu '.\n"
        "- Tên node ngắn gọn (2-5 từ), ưu tiên cùng ngôn ngữ với tài liệu.\n"
        "- Ví dụ JSON: {\"name\":\"Root\",\"children\":[{\"name\":\"Chủ đề\",\"children\":[{\"name\":\"Nhánh con\",\"detail\":\"Mô tả\"}]}]}"
    )
    bullet_block = "\n".join(f"- {item}" for item in prepared_chunks)
    user_prompt = (
        "Sinh mindmap liền mạch từ các ý dưới đây (theo đúng thứ tự được cung cấp).\n"
        "Gom các ý liên quan thành chủ đề chính rồi chia tiếp thành các nhánh phụ logic.\n"
        "Dữ liệu tham khảo:\n"
        f"{bullet_block}"
    )

    raw = run_ollama_chat(system_prompt, user_prompt, model=model or SLM_MODEL)
    tree = extract_json_tree(raw)

    # Post-clean: loại node TC lạc lõng
    def clean_logic(node):
        if isinstance(node, dict):
            name = node.get("name", "")
            children = node.get("children", [])
            if name.startswith("TC") and not children:
                return None
            node["children"] = [c for c in (clean_logic(ch) for ch in children) if c]
            return node
        return None

    return clean_logic(tree) or {"name": "Mind Map", "children": []}


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
