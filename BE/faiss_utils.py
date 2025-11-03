import os
import json
from datetime import datetime
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Đường dẫn index và metadata
INDEX_PATH = "index/index.faiss"
META_PATH = "index/index.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model 1 lần (tối ưu tốc độ)
_model = SentenceTransformer(MODEL_NAME)


# ===== Helpers =====
def _load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_index(dim: int):
    """Luôn dùng IndexIDMap để giữ id"""
    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)
        # Nếu index đang không phải IDMap thì rebuild lại
        if not isinstance(idx, faiss.IndexIDMap):
            base = faiss.IndexFlatL2(dim)
            new_idx = faiss.IndexIDMap(base)
            xb = idx.reconstruct_n(0, idx.ntotal)
            ids = np.arange(idx.ntotal, dtype="int64")
            new_idx.add_with_ids(xb, ids)
            return new_idx
        return idx
    else:
        base = faiss.IndexFlatL2(dim)
        return faiss.IndexIDMap(base)

# ===== Public API =====
def append_to_index(chunks: List[str], video_name: str = ""):
    
    if not chunks:
        return

    embeds = _model.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeds.shape[1]

    meta = _load_meta()

    # Tìm id tiếp theo
    existing_ids = [int(k) for k in meta.keys()] if meta else []
    next_id = max(existing_ids) + 1 if existing_ids else 0
    ids = np.arange(next_id, next_id + len(chunks), dtype="int64")

    # Load hoặc tạo index
    idx = _load_index(dim)
    idx.add_with_ids(embeds, ids)
    faiss.write_index(idx, INDEX_PATH)

    # Cập nhật metadata
    now = datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        meta[str(int(ids[i]))] = {
            "text": chunk,
            "video": video_name,
            "timestamp": now,
        }
    _save_meta(meta)


def search_index(query: str, k: int = 5) -> List[str]:
    """Tìm kiếm và trả về list text chunks"""
    if not os.path.exists(INDEX_PATH):
        return []

    qv = _model.encode([query], convert_to_numpy=True).astype("float32")
    idx = faiss.read_index(INDEX_PATH)
    D, I = idx.search(qv, k)

    meta = _load_meta()
    results = []
    for iid in I[0]:
        if iid == -1:
            continue
        key = str(int(iid))
        if key in meta:
            results.append(meta[key]["text"])
    return results


def delete_source_from_index(video_name: str):
    """Xóa tất cả chunks liên quan tới 1 video"""
    meta = _load_meta()
    keep_meta = {k: v for k, v in meta.items() if v["video"] != video_name}
    _save_meta(keep_meta)

    # Rebuild FAISS từ metadata còn lại
    if not keep_meta:
        # clear index
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        return

    texts = []
    ids = []
    for k, v in keep_meta.items():
        ids.append(int(k))
        texts.append(v["text"])

    embeds = _model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = embeds.shape[1]

    base = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIDMap(base)
    idx.add_with_ids(embeds, np.array(ids, dtype="int64"))
    faiss.write_index(idx, INDEX_PATH)
