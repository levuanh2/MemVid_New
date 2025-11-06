import os
import unicodedata
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ingest_utils import extract_text, split_text
from video_utils import generate_qr_frames, save_qr_frames_to_video
from faiss_utils import append_to_index, search_index, delete_source_from_index, MODEL_NAME
from ollama_utils import summarize_whole_document, summarize_results, SLM_MODEL
from mindmap_utils import get_main_branches, generate_mindmap_flat, generate_mindmap_cmgn

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"]
)


BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = 'videos'
INPUT_DIR = 'input_docs'
MINDMAPS_PATH = BASE_DIR / 'mindmaps.json'
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


@app.get('/')
def home():
    return 'MemvidX API is running.'


def _load_mindmaps() -> list[dict]:
    if not MINDMAPS_PATH.exists():
        return []
    try:
        with open(MINDMAPS_PATH, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        print(f"⚠️ Không thể đọc mindmaps.json: {exc}")
    return []


def _save_mindmaps(records: list[dict]) -> None:
    try:
        tmp_path = MINDMAPS_PATH.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp_path.replace(MINDMAPS_PATH)
    except Exception as exc:
        print(f"⚠️ Không thể lưu mindmaps.json: {exc}")


def _append_mindmap(record: dict) -> None:
    records = _load_mindmaps()
    records.insert(0, record)
    _save_mindmaps(records)


def _mindmap_response(record: dict) -> dict:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "nodes": nodes,
        "sources": record.get("sources", []),
        "createdAt": record.get("createdAt"),
        "strategy": record.get("strategy") or "iterative",
    }


# -------------------------
# 📤 Process raw text
# -------------------------
@app.post('/process-doc')
def process_doc():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    chunks = split_text(text)
    frames = generate_qr_frames(chunks)
    video_path = save_qr_frames_to_video(frames)
    append_to_index(chunks, Path(video_path).name)
    return jsonify({'video_path': video_path})


# -------------------------
# 📤 Upload single file
# -------------------------
@app.post('/upload-file')
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Missing file'}), 400

    save_path = os.path.join(INPUT_DIR, file.filename)
    file.save(save_path)

    text = extract_text(save_path)
    if not text.strip():
        return jsonify({'error': 'Cannot read file content'}), 400

    chunks = split_text(text)
    frames = generate_qr_frames(chunks)
    video_path = save_qr_frames_to_video(frames, prefix=file.filename.replace('.', '_'))
    append_to_index(chunks, Path(video_path).name)

    return jsonify({'video_path': video_path, 'message': 'File processed and index built'})


# -------------------------
# 📤 Upload multiple files
# -------------------------
@app.post('/upload-multiple')
def upload_multiple():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Missing files'}), 400

    results = []
    for file in files:
        save_path = os.path.join(INPUT_DIR, file.filename)
        file.save(save_path)

        text = extract_text(save_path)
        if not text.strip():
            results.append({'file': file.filename, 'error': 'Cannot read content'})
            continue

        chunks = split_text(text)
        frames = generate_qr_frames(chunks)
        video_path = save_qr_frames_to_video(frames, prefix=file.filename.replace('.', '_'))
        append_to_index(chunks, Path(video_path).name)

        results.append({
            'file': file.filename,
            'video_path': video_path,
            'message': 'OK'
        })

    return jsonify({'results': results})


@app.get('/list-indexed')
def list_indexed():
    try:
        with open('index/index.json', encoding='utf-8') as f:
            meta = json.load(f)

        video_map = {}
        for item in meta.values():
            video = unicodedata.normalize('NFKD', item.get('video', '').strip()).replace('\u00a0', ' ')
            if not video or video.lower() == 'unknown':
                continue
            video_name = Path(video).name
            text = item.get('text', '')
            video_map.setdefault(video_name, []).append(text)

        sources = []
        for video, chunks in video_map.items():
            sources.append({
                'video': Path(video).stem,  # FE expects stem
                'chunks': chunks,
                'num_chunks': len(chunks)
            })

        return jsonify({'sources': sources})
    except Exception as e:
        return jsonify({'error': str(e), 'sources': []})


# -------------------------
# 🎥 Serve video
# -------------------------
@app.get('/videos/<name>')
def serve_video(name):
    return send_from_directory(VIDEOS_DIR, name)


# -------------------------
# 🔍 Query
# -------------------------
@app.post('/query')
def query():
    q = request.json.get('q') or request.json.get('question') or ''
    selected_sources = request.json.get('sources') or []

    if not q.strip():
        return jsonify({'error': 'Missing query'}), 400

    all_chunks = search_index(q)
    chunks_with_file = []

    try:
        with open('index/index.json', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        return jsonify({'error': 'No index metadata found', 'detail': str(e)}), 500

    # Normalize video names in meta
    meta_norm = {}
    for k, m in meta.items():
        video_raw = m.get('video', '').strip()
        video_name = Path(video_raw).name
        video_stem = unicodedata.normalize('NFKD', Path(video_name).stem).replace('\u00a0', ' ').lower()
        meta_norm[k] = {
            'text': m['text'],
            'video_stem': video_stem
        }

    # Normalize selected sources (stem)
    selected_norm = set()
    for s in selected_sources:
        try:
            selected_norm.add(
                unicodedata.normalize('NFKD', Path(s).stem).replace('\u00a0', ' ').lower()
            )
        except Exception as e:
            print("⚠️ Lỗi normalize source:", s, e)

    # Match chunks by exact text match returned by search_index
    for chunk in all_chunks:
        for k, m_norm in meta_norm.items():
            if m_norm['text'] == chunk:
                if not selected_sources or m_norm['video_stem'] in selected_norm:
                    chunks_with_file.append(f"[FILE: {m_norm['video_stem']}]\n{chunk}")
                break

    # Fallback: if no matches found, assemble from selected sources or all meta
    if not chunks_with_file:
        if selected_sources:
            for m_norm in meta_norm.values():
                if m_norm['video_stem'] in selected_norm: 
                    chunks_with_file.append(f"[FILE: {m_norm['video_stem']}]\n{m_norm['text']}")
        else:
            for m_norm in meta_norm.values():
                chunks_with_file.append(f"[FILE: {m_norm['video_stem']}]\n{m_norm['text']}")

    if not chunks_with_file:
        return jsonify({'answer': "Không tìm thấy dữ liệu phù hợp trong file đã chọn."})

    answer = summarize_results(q, chunks_with_file, model=SLM_MODEL)
    return jsonify({'answer': answer})

# -------------------------
# 📝 Summarize file
# -------------------------
@app.post('/summarize-file')
def summarize_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Missing file'}), 400

    save_path = os.path.join(INPUT_DIR, file.filename)
    file.save(save_path)

    text = extract_text(save_path)
    if not text.strip():
        return jsonify({'error': 'Cannot read file content'}), 400

    summary = summarize_whole_document(text)
    return jsonify({'summary': summary})


# -------------------------
# 🗑️ Delete source
# -------------------------
@app.post('/delete-source')
def delete_source():
    data = request.json or {}
    video_name = data.get('video', '')

    if not video_name:
        return jsonify({'error': 'Missing video name'}), 400

    # FE gửi stem -> normalize
    video_stem = unicodedata.normalize('NFKD', video_name.strip()).replace('\u00a0', ' ').replace('.mp4', '').lower()

    meta_path = Path('index/index.json')
    if not meta_path.exists():
        return jsonify({'error': 'No index metadata found'}), 404

    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

        # Tìm danh sách stored video names có stem khớp
        stored_names = set()
        for v in meta.values():
            stored_video = unicodedata.normalize('NFKD', v.get('video', '').strip()).replace('\u00a0', ' ')
            if Path(stored_video).stem.lower() == video_stem:
                stored_names.add(stored_video)

        if not stored_names:
            return jsonify({'message': 'No matching source found', 'removed': 0})

        removed_total = 0
        # Gọi delete_source_from_index cho từng stored name (faiss_utils sẽ rebuild index)
        for stored in stored_names:
            delete_source_from_index(stored)
            # count removed in meta by checking previous entries (best-effort)
            removed_total += sum(1 for v in meta.values() if Path(unicodedata.normalize('NFKD', v.get('video', '').strip()).replace('\u00a0',' ')).stem.lower() == video_stem)

        # Xóa file video vật lý (match by stem)
        for f in Path(VIDEOS_DIR).glob(f"{video_stem}*"):
            try:
                f.unlink()
            except Exception as e:
                print("⚠️ Could not delete video file:", f, e)

        return jsonify({'message': 'Deleted', 'removed': removed_total})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# -------------------------
@app.post("/generate-mindmap")
def generate_mindmap():
    try:
        data = request.json or {}
        sources = data.get("sources", [])
        if not sources:
            return jsonify({"error": "No sources selected"}), 400

        strategy_requested = (data.get("strategy") or data.get("mode") or data.get("method") or "iterative").strip().lower()

        root_title = Path(sources[0]).stem if sources else "Mind Map"

        with open("index/index.json", encoding="utf-8") as f:
            meta = json.load(f)

        normalized_sources = [
            unicodedata.normalize('NFKD', s.strip()).replace('\u00a0', '').replace('.mp4', '').lower()
            for s in sources
        ]

        # chọn chunks thuộc sources được chọn
        chunks = [
            m["text"]
            for m in meta.values()
            if unicodedata.normalize('NFKD', m.get("video", "").strip()).replace('\u00a0', '').replace('.mp4', '').lower()
            in normalized_sources
        ]

        if not chunks:
            # Fallback tree basic để modal không trắng
            flat_nodes = [
                {"id": "root", "parent": None, "title": root_title},
                {"id": "root-0", "parent": "root", "title": "No content available"}
            ]
            strategy_used = strategy_requested if strategy_requested in {"cmgn", "semantic", "coreference"} else "iterative"
        else:
            if strategy_requested in {"cmgn", "semantic", "coreference"}:
                try:
                    flat_nodes = generate_mindmap_cmgn(chunks, model=SLM_MODEL)
                    strategy_used = "cmgn"
                except Exception as exc:
                    print(f"⚠️ generate_mindmap: CMGN strategy lỗi ({exc}), fallback iterative")
                    flat_nodes = generate_mindmap_flat(chunks, model=SLM_MODEL)
                    strategy_used = "iterative"
            else:
                flat_nodes = generate_mindmap_flat(chunks, model=SLM_MODEL)
                strategy_used = "iterative"

        # Ép root_title nếu cần
        if flat_nodes:
            root_node = next((n for n in flat_nodes if n.get("parent") is None), flat_nodes[0])
            root_node["title"] = root_title or root_node.get("title") or "Mind Map"

        mindmap_record = {
            "id": str(uuid.uuid4()),
            "title": root_title,
            "nodes": flat_nodes,
            "sources": sources,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "strategy": strategy_used,
        }

        _append_mindmap(mindmap_record)

        return jsonify(_mindmap_response(mindmap_record))

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.get('/mindmaps')
def list_mindmaps():
    records = _load_mindmaps()
    return jsonify({"mindmaps": [_mindmap_response(r) for r in records]})


@app.delete('/mindmaps/<string:mindmap_id>')
def delete_mindmap(mindmap_id: str):
    records = _load_mindmaps()
    new_records = [r for r in records if r.get("id") != mindmap_id]
    if len(new_records) == len(records):
        return jsonify({"error": "Mind map not found"}), 404
    _save_mindmaps(new_records)
    return jsonify({"message": "Deleted"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
