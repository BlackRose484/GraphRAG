"""
GraphVisualizer — Tạo đồ thị tương tác từ graph_data bằng pyvis.

Sử dụng:
    from ui.graph_visualizer import build_graph_html
    html = build_graph_html(nodes, triplets, subgraph)
    st.components.v1.html(html, height=600, scrolling=False)

Màu sắc node:
    Play      → cam   (#f5a442)
    Character → xanh dương (#4a90d9)
    Actor     → xanh lá   (#27ae60)
    Scene     → tím        (#9b59b6)
    Unknown   → xám        (#95a5a6)
"""
from __future__ import annotations

from typing import Any

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False


# ── Color palette ─────────────────────────────────────────────────────────────

_NODE_STYLE: dict[str, dict[str, str]] = {
    "play":           {"color": "#e8811a", "shape": "box",          "label_prefix": "📖 "},
    "character":      {"color": "#2e7dd6", "shape": "dot",          "label_prefix": "🎭 "},
    "actor":          {"color": "#1a9c4e", "shape": "diamond",      "label_prefix": "🎤 "},
    "scene":          {"color": "#7b3fa0", "shape": "ellipse",      "label_prefix": "🎬 "},
    "version":        {"color": "#5bafd6", "shape": "triangleDown", "label_prefix": "🔖 "},
    "roleassignment": {"color": "#c0c0c0", "shape": "square",       "label_prefix": "📌 "},
    "appearance":     {"color": "#f0c040", "shape": "star",         "label_prefix": "⭐ "},
    "unknown":        {"color": "#aaaaaa", "shape": "dot",          "label_prefix": ""},
}

# Relationship type → short Vietnamese label
_REL_LABELS: dict[str, str] = {
    "HAS_CHARACTER":  "có nhân vật",
    "HAS_SCENE":      "có cảnh",
    "HAS_VERSION":    "có phiên bản",
    "FOR_CHARACTER":  "cho nhân vật",
    "PERFORMED_BY":   "diễn bởi",
    "IN_VERSION":     "thuộc phiên bản",
    "HAS_APPEARANCE": "xuất hiện",
    "PLAYED_BY":      "diễn bởi",
}

_MAX_NODES    = 120   # cap to keep rendering fast
_MAX_TRIPLETS = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_type(node: dict[str, Any]) -> str:
    """Infer the node type from its property dict."""
    if "title" in node:
        return "play"
    if "charName" in node or "charGender" in node:
        return "character"
    if "actorName" in node:
        return "actor"
    if "sceneName" in node or "sceneSummary" in node:
        return "scene"
    if "versionId" in node:
        return "version"
    # RoleAssignment has no user-visible name — identify by absence of others
    if not any(k in node for k in ("title", "charName", "actorName", "sceneName", "versionId")):
        return "roleassignment"
    return "unknown"


def _node_label(node: dict[str, Any], ntype: str) -> str:
    """Return the display label for a node."""
    prefix = _NODE_STYLE.get(ntype, _NODE_STYLE["unknown"])["label_prefix"]
    if ntype == "roleassignment":
        return f"{prefix}Role"
    if ntype == "version":
        return f"{prefix}{node.get('versionId', '?')}"
    name = (
        node.get("charName")
        or node.get("actorName")
        or node.get("title")
        or node.get("sceneName")
        or node.get("name")
        or "?"
    )
    return f"{prefix}{name}"


def _unique_id(node: dict[str, Any]) -> str:
    """Derive a stable unique ID from a node dict."""
    return (
        node.get("charName")
        or node.get("actorName")
        or node.get("title")
        or node.get("sceneName")
        or node.get("name")
        or str(id(node))
    )


def _node_tooltip(node: dict[str, Any], ntype: str) -> str:
    """HTML tooltip shown on hover."""
    lines = [f"<b>Loại:</b> {ntype.title()}"]
    for k, v in node.items():
        if v:
            lines.append(f"<b>{k}:</b> {v}")
    return "<br>".join(lines)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph_html(
    nodes: list[dict[str, Any]],
    triplets: list[Any],
    subgraph: dict[str, Any] | None = None,
    height: int = 600,
    physics: bool = True,
) -> str:
    """Convert retrieval data into an interactive pyvis HTML string.

    Args:
        nodes:    List of node property dicts from GraphRetriever.
        triplets: List of (subject, rel_type, object) tuples.
        subgraph: Optional subgraph dict {nodes, relationships}.
        height:   Canvas height in pixels.
        physics:  Enable force-directed physics simulation.

    Returns:
        HTML string ready for ``st.components.v1.html()``.
        Returns an error message if pyvis is not installed.
    """
    if not _PYVIS_AVAILABLE:
        return "<p style='color:red'>pyvis chưa được cài đặt. Chạy: pip install pyvis</p>"

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1a1a1a",
        directed=True,
        notebook=False,
    )

    # Physics config — Barnes-Hut, smoother layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "font": { "size": 10, "color": "#333333", "align": "middle" }
      },
      "nodes": {
        "font": { "size": 13, "bold": true, "color": "#111111" },
        "borderWidth": 2,
        "borderWidthSelected": 4
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    added_node_ids: set[str] = set()
    added_edges: set[tuple[str, str]] = set()

    def add_node(node: dict[str, Any]) -> str | None:
        nid = _unique_id(node)
        if not nid or nid in added_node_ids:
            return nid
        if len(added_node_ids) >= _MAX_NODES:
            return None
        ntype  = _node_type(node)
        style  = _NODE_STYLE[ntype]
        net.add_node(
            nid,
            label=_node_label(node, ntype),
            color=style["color"],
            shape=style["shape"],
            title=_node_tooltip(node, ntype),
            size=22 if ntype == "play" else 16,
        )
        added_node_ids.add(nid)
        return nid

    def add_edge(src: str, dst: str, rel: str) -> None:
        key = (src, dst, rel)
        if key in added_edges:
            return
        added_edges.add(key)
        label = _REL_LABELS.get(rel, rel)
        edge_color = "#666666"
        if rel == "PERFORMED_BY":
            edge_color = "#27ae60"
        elif rel in ("HAS_CHARACTER",):
            edge_color = "#4a90d9"
        elif rel in ("HAS_SCENE",):
            edge_color = "#9b59b6"
        net.add_edge(src, dst, label=label, color=edge_color, width=1.5)

    # ── Add nodes from flat node list ─────────────────────────────────────────
    for node in nodes[:_MAX_NODES]:
        add_node(node)

    # ── Add nodes & edges from subgraph ──────────────────────────────────────
    if subgraph:
        for node in subgraph.get("nodes", [])[:_MAX_NODES]:
            add_node(node)
        for rel in subgraph.get("relationships", [])[:_MAX_TRIPLETS]:
            start = rel.get("start", {})
            end   = rel.get("end", {})
            rtype = rel.get("type", "")
            sid = add_node(start)
            eid = add_node(end)
            if sid and eid:
                add_edge(sid, eid, rtype)

    # ── Add edges from triplets ───────────────────────────────────────────────
    for triplet in triplets[:_MAX_TRIPLETS]:
        if not (isinstance(triplet, (list, tuple)) and len(triplet) >= 3):
            continue
        subj, rel, obj = str(triplet[0]), str(triplet[1]), str(triplet[2])

        # Auto-create minimal placeholder nodes if not yet in graph
        if subj not in added_node_ids and len(added_node_ids) < _MAX_NODES:
            net.add_node(subj, label=subj, color="#95a5a6", size=14)
            added_node_ids.add(subj)
        if obj not in added_node_ids and len(added_node_ids) < _MAX_NODES:
            net.add_node(obj, label=obj, color="#95a5a6", size=14)
            added_node_ids.add(obj)

        if subj in added_node_ids and obj in added_node_ids:
            add_edge(subj, obj, rel)

    # ── Legend overlay ────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:absolute;top:12px;left:12px;background:rgba(0,0,0,0.6);
         padding:8px 12px;border-radius:8px;font-size:12px;color:#eee;z-index:100">
      <b>Legend</b><br>
      <span style="color:#f5a442">■</span> Vở chèo &nbsp;
      <span style="color:#4a90d9">●</span> Nhân vật &nbsp;
      <span style="color:#27ae60">◆</span> Diễn viên &nbsp;
      <span style="color:#9b59b6">●</span> Trích đoạn
    </div>
    """

    html = net.generate_html()
    # Inject legend before </body>
    html = html.replace("</body>", legend_html + "</body>")
    return html


def is_available() -> bool:
    """Return True if pyvis is installed and usable."""
    return _PYVIS_AVAILABLE
