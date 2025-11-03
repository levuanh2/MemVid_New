# mindmap_utils.py
import re
import json
import uuid
from ollama_utils import run_ollama_chat, SLM_MODEL


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
        # Nếu fail thì clean comment, markdown, bullet
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
    system_prompt = (
        "Bạn là AI mindmap chuyên nghiệp. Trả về DUY NHẤT JSON tree nested (```json ...```).\n"
        "- Phân tích nội dung để xác định 3-5 chủ đề chính.\n"
        "- Cấu trúc: Root → Theme → Sub → Detail (depth ≤ 3).\n"
        "- Mỗi level có 2-4 children, tránh liệt kê dàn trải.\n"
        "- Tên node ngắn gọn (2-4 từ), tiếng Việt nếu nội dung VI.\n"
        "- Ví dụ JSON: {\"name\":\"Root\",\"children\":[{\"name\":\"Theme1\",\"children\":[{\"name\":\"Sub1\",\"children\":[]}]}]}"
    )
    user_prompt = "Sinh mindmap từ các đoạn sau:\n\n" + "\n".join(chunks[:6])

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

    def dfs(node, parent=None):
        if not isinstance(node, dict):
            return
        node_id = str(uuid.uuid4())
        flat_nodes.append({"id": node_id, "parent": parent, "title": node.get("name", "")})
        for child in node.get("children", []):
            dfs(child, node_id)

    dfs(tree)
    return flat_nodes


def generate_mindmap_flat(chunks: list[str], model: str = None) -> list[dict]:
    """
    Sinh mindmap dạng phẳng (flat nodes).
    Có fallback sang main branches nếu JSON nested lỗi.
    """
    try:
        tree = get_nested_mindmap(chunks, model=model)
    except Exception as e:
        print(f"⚠️ JSON lỗi trong generate_mindmap: {e} — fallback sang main branches")
        mains = get_main_branches(chunks, model=model)
        tree = {"name": "Mind Map", "children": [{"name": m, "children": []} for m in mains]}

    return flatten_mindmap(tree)
