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

    // Build nodes với initial collapsed từ state
    const nodeListWithData = flatData.map((node) => ({
      id: node.id || `node-${Math.random().toString(36).substr(2, 9)}`,
      parent: node.parent,
      title: node.title,
      data: {
        label: node.title,
        hasChildren: flatData.some((child) => child.parent === node.id),
        collapsed: collapsed[node.id] ?? false,  // Initial sync state
        onToggle: () => toggleCollapse(node.id),
        isRoot: !node.parent,
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
          style: { stroke: '#000000', strokeWidth: 2 },
        });
      }
    });

    // Add root nếu missing (giữ nguyên)
    const rootId = flatData.find((n) => !n.parent)?.id || 'root';
    if (!nodeListWithData.find((n) => n.id === rootId)) {
      const rootNode = {
        id: rootId,
        data: { label: data.title || 'Mind Map', hasChildren: flatData.length > 0, collapsed: false, onToggle: () => { }, isRoot: true },
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
        data: { label: 'No data - Check sources', hasChildren: false, isRoot: false },
        type: 'custom',
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
      });
      nodeListWithData.push({
        id: dummyId2,
        data: { label: 'Backend errors?', hasChildren: false, isRoot: false },
        type: 'custom',
        targetPosition: Position.Left,
        sourcePosition: Position.Right,
      });
      edgeList.push({ id: `e-${rootId}-${dummyId1}`, source: rootId, target: dummyId1, type: 'smoothstep' });
      edgeList.push({ id: `e-${rootId}-${dummyId2}`, source: rootId, target: dummyId2, type: 'smoothstep' });
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
        width: 150,
        height: 50,
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

      const manualNodes = nodeListWithData.map((node, i) => {
        const depthKey = Object.keys(levels).find((d) => levels[d].includes(node)) || 0;
        const depth = parseInt(depthKey);
        const levelIndex = levels[depth].indexOf(node);
        return {
          ...node,
          position: { x: depth * 250, y: levelIndex * 80 },
        };
      });

      return { nodes: manualNodes, edges: edgeList };
    }
  }, [collapsed, data.title]);  // Depend collapsed để rebuild initial data

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

  const nodeTypes = useMemo(() => ({
    custom: ({ data, id }) => {
      const { label, hasChildren, collapsed, onToggle, isRoot } = data;
      const colors = ['#eab308', '#a78bfa', '#4ade80', '#f472b6', '#60a5fa'];
      const colorIdx = parseInt(id.split('-')[0]) % colors.length || 0;
      return (
        <div
          style={{
            padding: '6px 12px',
            borderRadius: 6,
            background: isRoot ? '#d946ef' : colors[colorIdx],
            color: '#fff',
            border: '1px solid #000',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
            minWidth: 120,
          }}
        >
          <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
          <span>{label}</span>
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggle();
              }}
              style={{
                position: 'absolute',
                top: '50%',
                right: -15,
                transform: 'translateY(-50%)',
                border: '1px solid #ccc',
                borderRadius: '50%',
                width: 30,
                height: 30,
                background: '#fff',
                cursor: 'pointer',
                fontSize: 14,
                fontWeight: 'bold',
                lineHeight: '18px',
                textAlign: 'center',
                color: '#000',
              }}
            >
              {collapsed ? '+' : '-'}
            </button>
          )}
          <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
        </div>
      );
    },
  }), []);

  return (
    <div className="fixed inset-0 bg-white flex flex-col z-50" style={{ height: '100vh' }}>
      <div className="flex items-center justify-between bg-gray-100 p-3 border-b">
        <h3 className="text-lg font-semibold">{data?.title || 'Mind Map'}</h3>
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
