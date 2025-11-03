# ollama_utils.py
from ollama import Client
import traceback
from typing import List
from langdetect import detect

# Default SLM model (thay đổi theo model bạn cài trên Ollama)
SLM_MODEL = 'gemma2:2b'  
OLLAMA_HOST = "http://localhost:11434"
def _safe_chat(messages: list[dict], model: str = None) -> str:
    """
    Hàm gọi Ollama an toàn (low-level).
    messages: list of {"role":..., "content":...}
    model: override model (nếu None sẽ dùng SLM_MODEL).
    Trả về raw text (hoặc chuỗi lỗi để debug).
    """
    model = model or SLM_MODEL
    try:
        cli = Client(host=OLLAMA_HOST)
        resp = cli.chat(model=model, messages=messages)
        raw = resp.get("message", {}).get("content", "")
        if raw is None:
            raw = ""
        raw = raw.strip()

        print("=== RAW OLLAMA RESPONSE (first 1500 chars) ===")
        print(raw[:1500])
        print("=== END RAW ===")

        if not raw:
            return "⚠️ LLM trả về rỗng."
        return raw
    except Exception:
        traceback.print_exc()
        return "⚠️ Lỗi khi gọi Ollama."


def run_ollama_chat(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """
    Wrapper high-level: truyền system + user, có thể override model.
    SLM-friendly: dùng model nhỏ theo SLM_MODEL nếu không override.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _safe_chat(messages, model=model)


def summarize_whole_document(text: str, model: str = None) -> str:
    """Tóm tắt toàn bộ tài liệu bằng ngôn ngữ phù hợp."""
    try:
        lang = detect(text)
    except:
        lang = "vi"

    if lang == "vi":
        system_prompt = (
            "Bạn là trợ lý AI chuyên nghiệp, tóm tắt tài liệu ngắn gọn bằng tiếng Việt.\n"
            "- Trình bày chủ đề, mục tiêu, và 3-6 ý chính.\n"
            "- Tránh liệt kê dài dòng.\n"
            "- Viết súc tích, 3-6 câu.\n"
        )
    elif lang.startswith("zh"):
        system_prompt = "你是专业助手，请用中文简洁总结主要内容，3-6句。"
    else:
        system_prompt = "You are a concise assistant. Summarize the document in 3-6 sentences."

    return _safe_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ], model=model)


def summarize_results(query: str, chunks: List[str], model: str = None) -> str:
    """
    Trả lời câu hỏi dựa trên các đoạn văn đã lưu.
    SLM: ép model chỉ dùng nội dung cung cấp, trả lời ngắn.
    """
    try:
        lang = detect(query)
    except:
        lang = "vi"

    if lang == "vi":
        system_prompt = (
            "Bạn là trợ lý AI. Trả lời ngắn gọn bằng tiếng Việt, chỉ dùng thông tin cung cấp.\n"
            "- Nếu không đủ dữ liệu, nói rõ.\n"
            "- Tối đa 4 câu.\n"
        )
    elif lang.startswith("zh"):
        system_prompt = "你是AI助手。请用中文简洁回答，仅基于提供内容，若信息不足请说明。"
    else:
        system_prompt = "You are an AI assistant. Answer concisely using only provided content. If insufficient, say so."

    sources = "\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    user_msg = f"Relevant content:\n{sources}\n\nQuestion: {query}"

    return _safe_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ], model=model)
