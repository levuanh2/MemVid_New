import { useState, useEffect } from "react";
import { FiMoreVertical } from "react-icons/fi";
import MindMapModal from "./MindMapModal";

export default function SidebarRight({ selectedSources }) {
  const [mindMaps, setMindMaps] = useState([]);
  const [showModalMap, setShowModalMap] = useState(null);
  const [loading, setLoading] = useState(false);

  // Load từ localStorage khi mở trang
  // Load từ localStorage khi mở trang
useEffect(() => {
  try {
    const saved = localStorage.getItem("mindMaps");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) {  // Validate array
        setMindMaps(parsed);
      } else {
        console.warn("Invalid mindMaps format, resetting to empty");
        localStorage.removeItem("mindMaps");  // Clean invalid
        setMindMaps([]);
      }
    } else {
      setMindMaps([]);  // Default empty nếu no saved
    }
  } catch (err) {
    console.error("localStorage parse error:", err);
    localStorage.removeItem("mindMaps");  // Clean corrupted
    setMindMaps([]);  // Fallback empty
  }
}, []);

  // Lưu vào localStorage khi mindMaps thay đổi
  useEffect(() => {
    localStorage.setItem("mindMaps", JSON.stringify(mindMaps));
  }, [mindMaps]);

  const handleGenerateMindMap = async () => {
  console.log("handleGenerateMindMap called with:", selectedSources);
  if (!selectedSources || selectedSources.length === 0) {
    alert("Vui lòng chọn ít nhất một file để tạo Mind Map!");
    return;
  }
  setLoading(true);
  try {
    const res = await fetch("http://localhost:5000/generate-mindmap", {  // Fix: Full URL, bỏ /api
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sources: selectedSources, q: "tóm tắt tài liệu" }),
    });
    console.log("▶️ POST /generate-mindmap status:", res.status);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    console.log("▶️ MindMap response data:", data);
    if (data.error) throw new Error(data.error);

    // Lưu vào danh sách
    const newMap = {
      id: Date.now().toString(),
      title: data.title || "Mind Map mới",
      nodes: data.nodes || [],
      sources: selectedSources,
      createdAt: new Date().toISOString()
    };
    setMindMaps(prev => [newMap, ...prev]);
  } catch (err) {
    console.error("Mind Map Error:", err);
    alert("Không tạo được Mind Map, kiểm tra console!");
  } finally {
    setLoading(false);
  }
};
  const handleDeleteMap = (id) => {
    if (window.confirm("Xóa mind map này?")) {
      setMindMaps(prev => prev.filter(m => m.id !== id));
    }
  };

  const formatTimeAgo = (isoDate) => {
    const diff = (Date.now() - new Date(isoDate).getTime()) / 1000;
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
        {mindMaps.map((map) => (
          <div
            key={map.id}
            onClick={() => setShowModalMap(map)}
            className="flex items-start justify-between border rounded p-2 cursor-pointer hover:bg-gray-50"
          >
            <div>
              <div className="font-semibold text-sm">{map.title}</div>
              <div className="text-xs text-gray-500">
                {map.sources.length} nguồn · {formatTimeAgo(map.createdAt)}
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); handleDeleteMap(map.id); }}
              className="p-1 rounded hover:bg-gray-100"
            >
              <FiMoreVertical size={16} />
            </button>
          </div>
        ))}
      </div>

      {/* Modal mind map */}
      {showModalMap && (
        <MindMapModal
          data={showModalMap}
          onClose={() => setShowModalMap(null)}
        />
      )}
    </div>
  );
}
