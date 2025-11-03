import { useState, useEffect, useRef } from "react";
import { FiMoreVertical } from "react-icons/fi";

export default function SidebarLeft({ selectedSources, setSelectedSources }) {
  const [sources, setSources] = useState([]);
  const [menuOpen, setMenuOpen] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [deletingFile, setDeletingFile] = useState(null);
  const fileInputRef = useRef(null);

  const fetchSources = () => {
    fetch("http://localhost:5000/list-indexed")
      .then((res) => res.json())
      .then((data) => {
        const newSources = data.sources || [];
        setSources(newSources);
        setSelectedSources((prev) =>
          prev.filter((p) => newSources.some((s) => s.video === p))
        );
      })
      .catch((err) => console.error(err));
  };

  useEffect(() => {
    fetchSources();
  }, []);

  const handleAddFiles = async (e) => {
    const files = e.target.files;
    if (!files.length) return;

    const formData = new FormData();
    for (let file of files) {
      formData.append("files", file);
    }

    setUploading(true);
    try {
      await fetch("http://localhost:5000/upload-multiple", {
        method: "POST",
        body: formData,
      });
      fetchSources();
    } catch (err) {
      console.error(err);
    }
    setUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedSources(sources.map((s) => s.video));
    } else {
      setSelectedSources([]);
    }
  };

  const toggleSelect = (video) => {
    setSelectedSources((prev) =>
      prev.includes(video) ? prev.filter((v) => v !== video) : [...prev, video]
    );
  };

  const handleDeleteSource = async (video) => {
    setDeletingFile(video);
    setMenuOpen(null);
    try {
      await fetch("http://localhost:5000/delete-source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video }),
      });
      fetchSources();
    } catch (err) {
      console.error(err);
    }
    setDeletingFile(null);
  };

  // 👉 helper format tên file
  const formatFileName = (videoPath) => {
    const rawName = videoPath.split("/").pop().replace(/\.mp4$/, "");
    const parts = rawName.split("_");

    const timePart = parts[parts.length - 1];
    let displayTime = "";
    if (timePart && timePart.length === 6) {
      displayTime = `${timePart.slice(0, 2)}:${timePart.slice(2, 4)}:${timePart.slice(4, 6)}`;
    }

    return parts.slice(0, -1).join("_") + (displayTime ? ` (${displayTime})` : "");
  };

  return (
    <div className="p-4 flex flex-col h-full relative">
      <h2 className="text-lg font-semibold mb-4">Sources</h2>

      {/* Select all */}
      <label className="flex items-center space-x-2 mb-2">
        <input
          type="checkbox"
          checked={
            selectedSources.length === sources.length && sources.length > 0
          }
          onChange={(e) => handleSelectAll(e.target.checked)}
        />
        <span className="text-sm">Chọn tất cả</span>
      </label>

      {/* Add button */}
      <label
        className={`px-3 py-1 rounded mb-4 text-center cursor-pointer flex items-center justify-center space-x-2 ${uploading ? "bg-gray-400" : "bg-blue-500 text-white"
          }`}
      >
        {uploading ? (
          <>
            <svg
              className="animate-spin h-4 w-4 text-white"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8z"
              />
            </svg>
            <span>Uploading...</span>
          </>
        ) : (
          <span>+ Thêm</span>
        )}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleAddFiles}
          disabled={uploading}
        />

      </label>

      {/* Sources list */}
      <div className="flex-1 overflow-auto space-y-2">
        {sources.map((src, idx) => {
          const displayName = formatFileName(src.video);
          const isDeleting = deletingFile === src.video;

          return (
            <div
              key={idx}
              className={`p-2 border rounded flex justify-between items-center hover:bg-gray-100 transition ${isDeleting ? "opacity-50" : ""
                }`}
            >
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={selectedSources.includes(src.video)}
                  onChange={() => toggleSelect(src.video)}
                  disabled={isDeleting}
                />
                <div className="max-w-[120px] truncate" title={displayName}>
                  {displayName}
                  <div className="text-xs text-gray-500">
                    {src.num_chunks} chunks
                  </div>
                </div>
              </div>

              {/* Menu */}
              <div className="relative">
                {isDeleting ? (
                  <svg
                    className="animate-spin h-5 w-5 text-gray-500"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8z"
                    />
                  </svg>
                ) : (
                  <>
                    <button
                      onClick={() =>
                        setMenuOpen(menuOpen === idx ? null : idx)
                      }
                      className="p-1 hover:bg-gray-200 rounded"
                    >
                      <FiMoreVertical />
                    </button>

                    {menuOpen === idx && (
                      <div className="absolute right-0 top-6 bg-white border rounded shadow text-sm z-10">
                        <button
                          onClick={() => handleDeleteSource(src.video)}
                          className="block px-4 py-2 hover:bg-red-100 text-red-500 w-full text-left"
                        >
                          Delete
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
