import { useState, useLayoutEffect, useCallback, useMemo } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useReactFlow,
  ReactFlowProvider
} from "reactflow";
import "reactflow/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";
import { Handle, Position } from "reactflow";

const elk = new ELK();

function MindMapContent({ data, onClose }) {
  const reactFlowInstance = useReactFlow();
  const [innerNodes, setInnerNodes] = useState([]);
  const [innerEdges, setInnerEdges] = useState([]);
  const [collapsed, setCollapsed] = useState({});
  const [shapeType, setShapeType] = useState("rectangle");

  const shapeConfig = useMemo(() => {
    switch (shapeType) {
      case "circle":
        return { width: 160, height: 160, innerSize: 120 };
      case "diamond":
        return { width: 180, height: 180, innerSize: 130 };
      default:
        return { width: 220, height: 110, innerSize: null };
    }
  }, [shapeType]);

  // Fix: Toggle updater nested, compute newCollapsed sync, no dep loop
  const toggleCollapse = useCallback((id) => {
    setCollapsed((prev) => {
      const newCollapsed = { ...prev, [id]: !prev[id] };
      // Update node data với new value (sync trong updater)
      setInnerNodes((prevNodes) =>
        prevNodes.map((node) =>
          node.id === id
            ? { ...node, data: { ...node.data, collapsed: newCollapsed[id] } }
            : node
        )
      );
      return newCollapsed;
    });
  }, []);  // No dep, tránh loop

  // Helper isVisible: Check ancestor collapsed (ẩn subtree)
  const isVisible = useCallback((nodeId) => {
    let currentId = nodeId;
    while (currentId && currentId !== 'root') {
      const parentNode = innerNodes.find((n) => n.id === currentId);
      const parentId = parentNode?.parent;
      if (parentId && collapsed[parentId]) return false;  // Ancestor collapsed → hide
      currentId = parentId;
    }
    return true;
  }, [innerNodes, collapsed]);

  const getLayoutedElements = useCallback(async (flatData, direction = 'RIGHT') => {
    if (!Array.isArray(flatData) || flatData.length === 0) {
      console.warn("getLayoutedElements: Invalid flatData");
      return { nodes: [], edges: [] };
    }

    const nodeWidth = shapeConfig.width;
    const nodeHeight = shapeConfig.height;

    // Build nodes với initial collapsed từ state
    const nodeListWithData = flatData.map((node) => ({
      id: node.id || `node-${Math.random().toString(36).substr(2, 9)}`,
      parent: node.parent,
      title: node.title,
      width: nodeWidth,
      height: nodeHeight,
      style: {
        width: nodeWidth,
        height: nodeHeight,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
      data: {
        label: node.title,
        hasChildren: flatData.some((child) => child.parent === node.id),
        collapsed: collapsed[node.id] ?? false,  // Initial sync state
        onToggle: () => toggleCollapse(node.id),
        isRoot: !node.parent,
        shapeType,
      },
      type: 'custom',
      targetPosition: direction === 'RIGHT' ? Position.Left : Position.Top,
      sourcePosition: direction === 'RIGHT' ? Position.Right : Position.Bottom,
    }));

    const edgeList = [];
    flatData.forEach((node) => {
      if (node.parent) {
        edgeList.push({
          id: `e-${node.parent}-${node.id}`,
          source: node.parent,
          target: node.id,
          type: 'smoothstep',
          style: { stroke: '#1f2937', strokeWidth: 2.2 },
        });
      }
    });

    // Add root nếu missing (giữ nguyên)
    const rootId = flatData.find((n) => !n.parent)?.id || 'root';
    if (!nodeListWithData.find((n) => n.id === rootId)) {
      const rootNode = {
        id: rootId,
        width: nodeWidth,
        height: nodeHeight,
        style: {
          width: nodeWidth,
          height: nodeHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
        data: {
          label: data.title || 'Mind Map',
          hasChildren: flatData.length > 0,
          collapsed: false,
          onToggle: () => { },
          isRoot: true,
          shapeType,
        },
        type: 'custom',
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
      };
      nodeListWithData.unshift(rootNode);
    }

    // Fallback dummy (giữ nguyên)
    if (nodeListWithData.length <= 1) {
      console.log("Fallback: Adding dummy nodes");
      const dummyId1 = 'dummy-1', dummyId2 = 'dummy-2';
      nodeListWithData.push({
        id: dummyId1,
        width: nodeWidth,
        height: nodeHeight,
        style: {
          width: nodeWidth,
          height: nodeHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
        data: { label: 'No data - Check sources', hasChildren: false, isRoot: false, shapeType },
        type: 'custom',
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
      });
      nodeListWithData.push({
        id: dummyId2,
        width: nodeWidth,
        height: nodeHeight,
        style: {
          width: nodeWidth,
          height: nodeHeight,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
        data: { label: 'Backend errors?', hasChildren: false, isRoot: false, shapeType },
        type: 'custom',
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
      });
      edgeList.push({ id: `e-${rootId}-${dummyId1}`, source: rootId, target: dummyId1, type: 'smoothstep', style: { stroke: '#1f2937', strokeWidth: 2.2 } });
      edgeList.push({ id: `e-${rootId}-${dummyId2}`, source: rootId, target: dummyId2, type: 'smoothstep', style: { stroke: '#1f2937', strokeWidth: 2.2 } });
    }

    console.log(`Building layout for ${nodeListWithData.length} nodes, ${edgeList.length} edges`);

    const graph = {
      id: 'root',
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': direction,
        'elk.layered.spacing.nodeNodeBetweenLayers': '120',
        'elk.spacing.nodeNode': '80',
        'elk.spacing.edgeNode': '40',
        'elk.spacing.edgeEdge': '50',
        'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
        'elk.alignment': 'CENTER',
        'elk.hierarchyHandling': 'SEPARATE',
      },
      children: nodeListWithData.map((node) => ({
        id: node.id,
        width: node.width || nodeWidth,
        height: node.height || nodeHeight,
      })),
      edges: edgeList.map((edge) => ({
        id: edge.id,
        sources: [edge.source],
        targets: [edge.target],
      })),
    };

    try {
      const layoutedGraph = await elk.layout(graph);
      console.log('ELK layout success:', layoutedGraph);

      const layoutedNodes = layoutedGraph.children.map((node) => {
        const originalNode = nodeListWithData.find((n) => n.id === node.id);
        return {
          ...originalNode,
          position: { x: (node.x || 0) + 50, y: (node.y || 0) + 50 },
        };
      });

      const layoutedEdges = layoutedGraph.edges.map((edge) => {
        const originalEdge = edgeList.find((e) => e.id === edge.id);
        return { ...originalEdge };
      });

      return { nodes: layoutedNodes, edges: layoutedEdges };
    } catch (err) {
      console.error('ELK layout failed:', err);
      // Manual fallback (giữ nguyên)
      const levels = {};
      nodeListWithData.forEach((node) => {
        let depth = 0;
        let currentId = node.id;
        while (flatData.find((n) => n.id === currentId && n.parent)) {
          depth++;
          currentId = flatData.find((n) => n.id === currentId)?.parent;
        }
        levels[depth] = levels[depth] || [];
        levels[depth].push(node);
      });

      const manualNodes = nodeListWithData.map((node) => {
        const depthKey = Object.keys(levels).find((d) => levels[d].includes(node)) || 0;
        const depth = parseInt(depthKey);
        const levelIndex = levels[depth].indexOf(node);
        return {
          ...node,
          position: { x: depth * 250, y: levelIndex * (nodeHeight + 40) },
        };
      });

      return { nodes: manualNodes, edges: edgeList };
    }
  }, [collapsed, data.title, shapeConfig, shapeType, toggleCollapse]);  // Depend collapsed để rebuild initial data

  useLayoutEffect(() => {
    if (!data?.nodes || data.nodes.length === 0) {
      console.warn("No data.nodes, skipping layout");
      return;
    }

    console.log('MindMap data received:', data);
    getLayoutedElements(data.nodes, 'RIGHT').then(({ nodes, edges }) => {
      console.log('Set nodes after layout:', nodes);
      setInnerNodes(nodes);
      setInnerEdges(edges);
      requestAnimationFrame(() => {
        reactFlowInstance.fitView({ padding: 0.2, minZoom: 0.05, includeHiddenNodes: true });
      });
    });
  }, [data, getLayoutedElements, reactFlowInstance]);

  // Filter visible nodes/edges (ẩn subtree full)
  const visibleNodes = useMemo(() => innerNodes.filter((node) => isVisible(node.id)), [innerNodes, isVisible]);
  const filteredEdges = useMemo(() => {
    return innerEdges.filter((edge) => isVisible(edge.target));  // Edge chỉ nếu target visible
  }, [innerEdges, isVisible]);

  const memoizedNodes = useMemo(() => visibleNodes, [visibleNodes]);
  const memoizedEdges = useMemo(() => filteredEdges, [filteredEdges]);

  const nodeTypes = useMemo(() => {
    const palette = ['#22c55e', '#38bdf8', '#f97316', '#a855f7', '#facc15', '#ec4899', '#14b8a6'];
    const getColorIndex = (nodeId) => {
      if (!nodeId) return 0;
      const hashValue = Array.from(nodeId).reduce((acc, char, idx) => acc + char.charCodeAt(0) * (idx + 1), 0);
      return Math.abs(hashValue) % palette.length;
    };

    return {
      custom: ({ data, id }) => {
        const { label, hasChildren, collapsed, onToggle, isRoot } = data;
        const nodeColor = isRoot ? '#6366f1' : palette[getColorIndex(id)];

        const wrapperStyle = {
          width: shapeConfig.width,
          height: shapeConfig.height,
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: shapeType === 'rectangle' ? '0' : '6px',
          overflow: 'visible',
        };

        const baseTextStyle = {
          color: '#fff',
          fontWeight: 600,
          textAlign: 'center',
          wordBreak: 'break-word',
          lineHeight: 1.35,
          fontSize: '0.95rem',
        };

        let renderedShape;
        if (shapeType === 'circle') {
          const circleSize = Math.min(shapeConfig.innerSize ?? shapeConfig.width - 20, shapeConfig.width - 20);
          renderedShape = (
            <div
              style={{
                width: circleSize,
                height: circleSize,
                borderRadius: '50%',
                background: nodeColor,
                border: '2px solid rgba(15, 23, 42, 0.16)',
                boxShadow: '0 10px 20px rgba(15, 23, 42, 0.18)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '16px',
              }}
            >
              <span style={baseTextStyle}>{label}</span>
            </div>
          );
        } else if (shapeType === 'diamond') {
          const diamondSize = shapeConfig.innerSize ?? shapeConfig.width - 30;
          renderedShape = (
            <div
              style={{
                width: diamondSize,
                height: diamondSize,
                transform: 'rotate(45deg)',
                background: nodeColor,
                borderRadius: 18,
                border: '2px solid rgba(15, 23, 42, 0.16)',
                boxShadow: '0 10px 20px rgba(15, 23, 42, 0.18)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <span
                style={{
                  ...baseTextStyle,
                  transform: 'rotate(-45deg)',
                  width: '70%',
                  display: 'inline-block',
                }}
              >
                {label}
              </span>
            </div>
          );
        } else {
          renderedShape = (
            <div
              style={{
                width: '100%',
                height: '100%',
                background: nodeColor,
                borderRadius: 18,
                padding: '18px 22px',
                border: '2px solid rgba(15, 23, 42, 0.16)',
                boxShadow: '0 10px 20px rgba(15, 23, 42, 0.18)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <span style={baseTextStyle}>{label}</span>
            </div>
          );
        }

        return (
          <div style={wrapperStyle}>
            <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
            {renderedShape}
            {hasChildren && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggle();
                }}
                aria-label={collapsed ? 'Mở rộng nhánh con' : 'Thu gọn nhánh con'}
                style={{
                  position: 'absolute',
                  top: '50%',
                  right: shapeType === 'rectangle' ? -20 : -26,
                  transform: 'translateY(-50%)',
                  border: '1px solid rgba(148, 163, 184, 0.6)',
                  borderRadius: '9999px',
                  width: 34,
                  height: 34,
                  background: '#fff',
                  cursor: 'pointer',
                  fontSize: 16,
                  fontWeight: 700,
                  color: '#1e293b',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 4px 12px rgba(15, 23, 42, 0.18)',
                }}
              >
                {collapsed ? '+' : '-'}
              </button>
            )}
            <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
          </div>
        );
      },
    };
  }, [shapeConfig, shapeType]);

  return (
    <div className="fixed inset-0 bg-white flex flex-col z-50" style={{ height: '100vh' }}>
      <div className="flex flex-wrap items-center justify-between gap-3 bg-gray-100 p-3 border-b">
        <div className="flex items-center gap-4 flex-wrap">
          <h3 className="text-lg font-semibold">{data?.title || 'Mind Map'}</h3>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <span>Kiểu nút:</span>
            <select
              value={shapeType}
              onChange={(e) => setShapeType(e.target.value)}
              className="border border-slate-300 rounded px-2 py-1 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400"
            >
              <option value="rectangle">Hình chữ nhật</option>
              <option value="circle">Hình tròn</option>
              <option value="diamond">Hình thoi</option>
            </select>
          </label>
        </div>
        <button onClick={onClose} className="px-3 py-1 bg-red-500 text-white rounded">Đóng</button>
      </div>
      <div className="flex-1 relative">
        <ReactFlow
          nodes={memoizedNodes}
          edges={memoizedEdges}
          fitView={false}
          nodesDraggable={false}
          nodeTypes={nodeTypes}
          minZoom={0.05}
          maxZoom={2}
          zoomOnScroll
          panOnScroll
          style={{ height: '100%', width: '100%' }}
        >
          <MiniMap zoomable pannable />
          <Controls />
          <Background variant="dots" gap={12} size={1} color="#e2e8f0" />
        </ReactFlow>
        {memoizedNodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            No data - Check console
          </div>
        )}
      </div>
    </div>
  );
}

export default function MindMapModal({ data, onClose }) {
  return (
    <ReactFlowProvider>
      <MindMapContent data={data} onClose={onClose} />
    </ReactFlowProvider>
  );
}
