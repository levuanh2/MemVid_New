import SidebarLeft from "./SidebarLeft";
import ChatArea from "./ChatArea";
import SidebarRight from "./SidebarRight";

export default function MainLayout({ selectedSources, setSelectedSources }) {
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar trái */}
      <div className="w-1/5 border-r bg-white">
        <SidebarLeft
          selectedSources={selectedSources}
          setSelectedSources={setSelectedSources}
        />
      </div>

      {/* Chat Area */}
      <div className="flex flex-1 flex-col min-h-0">
        <ChatArea selectedSources={selectedSources} />
      </div>

      {/* Sidebar phải */}
      <div className="w-1/4 border-l bg-white">
        <SidebarRight selectedSources={selectedSources} />
      </div>
    </div>
  );
}
