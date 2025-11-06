import { useState, useEffect, useCallback } from "react";
import { FiMoreVertical } from "react-icons/fi";
import MindMapModal from "./MindMapModal";

const API_BASE = "http://localhost:5000";

export default function SidebarRight({ selectedSources }) {
  const [mindMaps, setMindMaps] = useState([]);
  const [showModalMap, setShowModalMap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  const fetchMindMaps = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/mindmaps`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = Array.isArray(data?.mindmaps) ? data.mindmaps : [];
      setMindMaps(list);
    } catch (err) {
      console.error("Mind map fetch error:", err);
      setMindMaps([]);
    } finally {
      setInitialLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMindMaps();
  }, [fetchMindMaps]);

  const handleGenerateMindMap = async () => {
    console.log("handleGenerateMindMap called with:", selectedSources);
    if (!selectedSources || selectedSources.length === 0) {
      alert("Vui lòng chọn ít nhất một file để tạo Mind Map!");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/generate-mindmap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sources: selectedSources, q: "tóm tắt tài liệu" }),
      });
      console.log("▶️ POST /generate-mindmap status:", res.status);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      console.log("▶️ MindMap response data:", data);
      if (data.error) throw new Error(data.error);

      const record = {
        id: data.id || Date.now().toString(),
        title: data.title || "Mind Map mới",
        nodes: Array.isArray(data.nodes) ? data.nodes : [],
        sources: Array.isArray(data.sources) ? data.sources : selectedSources,
        createdAt: data.createdAt || new Date().toISOString(),
      };

      setMindMaps(prev => [record, ...prev.filter(item => item.id !== record.id)]);
      await fetchMindMaps();
    } catch (err) {
      console.error("Mind Map Error:", err);
      alert("Không tạo được Mind Map, kiểm tra console!");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMap = async (id) => {
    if (!window.confirm("Xóa mind map này?")) return;
    try {
      const res = await fetch(`${API_BASE}/mindmaps/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setMindMaps(prev => prev.filter(m => m.id !== id));
      await fetchMindMaps();
    } catch (err) {
      console.error("Mind Map delete error:", err);
      alert("Không xóa được mind map, xem console!");
    }
  };

  const formatTimeAgo = (isoDate) => {
    if (!isoDate) return "Không xác định";
    const timestamp = new Date(isoDate).getTime();
    if (Number.isNaN(timestamp)) return "Không xác định";
    const diff = (Date.now() - timestamp) / 1000;
    if (diff < 60) return `${Math.floor(diff)} giây trước`;
    if (diff < 3600) return `${Math.floor(diff / 60)} phút trước`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} giờ trước`;
    return `${Math.floor(diff / 86400)} ngày trước`;
  };

  return (
    <div className="flex flex-col h-full border-l bg-white">
      {/* Nút chức năng */}
      <div className="p-4 grid grid-cols-2 gap-2">
        <button className="bg-blue-100 p-2 rounded hover:bg-blue-200">Audio Overview</button>
        <button className="bg-green-100 p-2 rounded hover:bg-green-200">Video Overview</button>
        <button
          onClick={handleGenerateMindMap}
          className="bg-pink-100 p-2 rounded hover:bg-pink-200 flex items-center justify-center"
        >
          {loading ? "Đang tạo..." : "Mind Map"}
        </button>
        <button className="bg-yellow-100 p-2 rounded hover:bg-yellow-200">Reports</button>
      </div>

      {/* Danh sách mind map */}
      <div className="flex-1 overflow-auto p-2 space-y-2 border-t">
        {initialLoading && (
          <div className="text-sm text-gray-500">Đang tải mind map...</div>
        )}
        {!initialLoading && mindMaps.length === 0 && (
          <div className="text-sm text-gray-500">Chưa có mind map nào. Hãy tạo mới!</div>
        )}
        {mindMaps.map((map) => (
          <div
            key={map.id}
            onClick={() => setShowModalMap(map)}
            className="flex items-start justify-between border rounded p-2 cursor-pointer hover:bg-gray-50"
          >
            <div>
              <div className="font-semibold text-sm">{map.title}</div>
              <div className="text-xs text-gray-500">
                {(map.sources?.length || 0)} nguồn · {formatTimeAgo(map.createdAt)}
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteMap(map.id);
              }}
              className="p-1 rounded hover:bg-gray-100"
            >
              <FiMoreVertical size={16} />
            </button>
          </div>
        ))}
      </div>

      {/* Modal mind map */}
      {showModalMap && (
        <MindMapModal data={showModalMap} onClose={() => setShowModalMap(null)} />
      )}
    </div>
  );
}
