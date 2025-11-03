import { useState } from "react";
import MainLayout from "./components/Layout/MainLayout";

export default function App() {
  // Danh sách file đang được chọn để chat
  const [selectedSources, setSelectedSources] = useState([]);

  return (
    <MainLayout
      selectedSources={selectedSources}
      setSelectedSources={setSelectedSources}
    />
  );
}
