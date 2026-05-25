"""GraphFormatConverter — convert graph_data to multiple text representations."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.constants import FormatKey, RelType, NodeProp, Limit
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Synthetic relation name used by the retriever for derived edges. Kept here
# instead of constants because it's a display-only string, not a Neo4j type.
_DERIVED_PERFORMS_IN = "PERFORMS_IN"

_REL_VERB: Dict[str, str] = {
    RelType.PERFORMED_BY:        "được thể hiện bởi diễn viên",
    RelType.HAS_CHARACTER:       "có nhân vật",
    RelType.HAS_SCENE:           "có cảnh",
    RelType.FOR_CHARACTER:       "là vai diễn cho nhân vật",
    RelType.IN_VERSION:          "thuộc phiên bản",
    RelType.HAS_VERSION:         "có phiên bản",
    RelType.HAS_APPEARANCE:      "có lượt xuất hiện",
    RelType.EXPRESS:             "biểu lộ trạng thái cảm xúc",
    RelType.IS_WEAR_BY:          "được mặc bởi diễn viên",
    RelType.REPRESENT:           "thể hiện cảm xúc",
    RelType.HAS_RELATION:        "có quan hệ với",
    RelType.COLLABORATES_WITH:   "cộng tác với",
    _DERIVED_PERFORMS_IN:        "tham gia trình diễn vở",
}

_REL_SUBJ_PREFIX: Dict[str, str] = {
    RelType.PERFORMED_BY:        "Nhân vật",
    RelType.HAS_CHARACTER:       "Vở chèo",
    RelType.HAS_SCENE:           "Vở chèo",
    RelType.FOR_CHARACTER:       "Vai diễn này",
    RelType.IN_VERSION:          "Vai diễn này",
    RelType.HAS_VERSION:         "Trích đoạn",
    RelType.HAS_APPEARANCE:      "Vai diễn này",
    RelType.EXPRESS:             "Diễn viên",
    RelType.IS_WEAR_BY:          "Trang phục",
    RelType.REPRESENT:           "Trang phục",
    RelType.HAS_RELATION:        "Nhân vật",
    RelType.COLLABORATES_WITH:   "Diễn viên",
    _DERIVED_PERFORMS_IN:        "Diễn viên",
}


def _primary_name(node: Dict[str, Any]) -> str:
    """Return the most human-readable display name, falling back through the
    property hierarchy so nodes without v1 name fields still display via label."""
    return (
        node.get(NodeProp.CHAR_NAME)
        or node.get(NodeProp.ACTOR_NAME)
        or node.get(NodeProp.TITLE)
        or node.get(NodeProp.SCENE_NAME)
        or node.get(NodeProp.LABEL)
        or node.get(NodeProp.ID)
        or "Unknown"
    )


class GraphFormatConverter:
    """Convert ``graph_data`` dicts from ``GraphRetriever`` into text formats."""

    @staticmethod
    def to_adjacency_table(graph_data: Dict) -> str:
        triplets: List[Tuple] = graph_data.get("triplets", [])

        if not triplets:
            return "Không có quan hệ nào được tìm thấy."

        lines = [
            f"Tìm thấy {len(triplets)} quan hệ:\n",
            "| STT | Subject (Chủ thể) | Relation (Quan hệ) | Object (Đối tượng) |",
            "|-----|-------------------|--------------------|--------------------|",
        ]
        for idx, (subj, rel, obj) in enumerate(triplets, 1):
            lines.append(f"| {idx} | {subj} | {rel} | {obj} |")

        return "\n".join(lines)

    @staticmethod
    def to_natural_language(graph_data: Dict) -> str:
        nodes:    List[Dict]  = graph_data.get("nodes", [])
        triplets: List[Tuple] = graph_data.get("triplets", [])
        subgraph: Dict        = graph_data.get("subgraph", {})

        sentences: List[str] = []

        for node in nodes:
            if NodeProp.CHAR_NAME in node:
                name      = node[NodeProp.CHAR_NAME]
                gender    = node.get(NodeProp.CHAR_GENDER, "")
                role_type = node.get(NodeProp.ROLE_TYPE, "")
                sub_type  = node.get(NodeProp.SUB_TYPE, "")
                attrs: list[str] = []
                if role_type:
                    attrs.append(
                        f"loại {role_type}"
                        + (f" - {sub_type}" if sub_type else "")
                    )
                if gender:
                    attrs.append(f"giới tính {gender}")
                detail = f" ({', '.join(attrs)})" if attrs else ""
                sentences.append(f"Nhân vật {name}{detail}.")
            elif NodeProp.ACTOR_NAME in node:
                sentences.append(f"Diễn viên {node[NodeProp.ACTOR_NAME]}.")
            elif NodeProp.TITLE in node:
                sentences.append(f"Vở kịch {node[NodeProp.TITLE]}.")
            elif NodeProp.SCENE_NAME in node:
                name    = node[NodeProp.SCENE_NAME]
                summary = node.get(NodeProp.SCENE_SUMMARY, "")
                if summary:
                    short = (
                        summary[: Limit.SCENE_SUMMARY_MAX_LEN] + "..."
                        if len(summary) > Limit.SCENE_SUMMARY_MAX_LEN
                        else summary
                    )
                    sentences.append(f"Trích đoạn '{name}': {short}")
                else:
                    sentences.append(f"Trích đoạn '{name}'.")
            elif str(node.get(NodeProp.ID, "")).startswith("costume_"):
                label   = node.get(NodeProp.LABEL, node.get(NodeProp.ID, ""))
                comment = node.get(NodeProp.COMMENT, "")
                if comment:
                    short = (
                        comment[: Limit.SCENE_SUMMARY_MAX_LEN] + "..."
                        if len(comment) > Limit.SCENE_SUMMARY_MAX_LEN
                        else comment
                    )
                    sentences.append(f"{label}: {short}")
                else:
                    sentences.append(f"{label}.")
            elif str(node.get(NodeProp.ID, "")).startswith("face_"):
                label   = node.get(NodeProp.LABEL, node.get(NodeProp.ID, ""))
                comment = node.get(NodeProp.COMMENT, "")
                if comment:
                    sentences.append(f"{label}: {comment}")
                else:
                    sentences.append(f"{label}.")
            elif NodeProp.EMOTION in node:
                emo = node.get(NodeProp.EMOTION, "")
                sub = node.get(NodeProp.SUBTITLE, "")
                if emo and emo not in ("...", "Other"):
                    if sub:
                        sentences.append(
                            f"Lượt xuất hiện với cảm xúc '{emo}', lời thoại: \"{sub}\"."
                        )
                    else:
                        sentences.append(f"Lượt xuất hiện với cảm xúc '{emo}'.")

        for subj, rel, obj in triplets:
            verb   = _REL_VERB.get(rel, rel)
            prefix = _REL_SUBJ_PREFIX.get(rel, "")
            sentences.append(
                f"{prefix} {subj} {verb} {obj}.".strip()
                if prefix else f"{subj} {verb} {obj}."
            )

        for rel_item in subgraph.get("relationships", []):
            rel_type   = rel_item.get("type", "")
            start_name = _primary_name(rel_item.get("start", {}))
            end_name   = _primary_name(rel_item.get("end", {}))
            verb       = _REL_VERB.get(rel_type, rel_type)
            prefix     = _REL_SUBJ_PREFIX.get(rel_type, "")
            sentences.append(
                f"{prefix} {start_name} {verb} {end_name}.".strip()
                if prefix else f"{start_name} {verb} {end_name}."
            )

        return " ".join(sentences) if sentences else "Không có dữ liệu."

    @staticmethod
    def to_code_like(graph_data: Dict) -> str:
        try:
            return json.dumps(graph_data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as exc:
            logger.warning("to_code_like serialization error: %s", exc)
            return str(graph_data)

    @staticmethod
    def to_node_sequence(graph_data: Dict) -> str:
        paths: List[List[str]] = graph_data.get("paths", [])

        if not paths:
            return "Không có đường đi nào."

        return "\n".join(
            " -> ".join(str(step) for step in path) for path in paths
        )

    @staticmethod
    def convert_all(graph_data: Dict) -> Dict[str, str]:
        return {
            FormatKey.ADJACENCY_TABLE:  GraphFormatConverter.to_adjacency_table(graph_data),
            FormatKey.NATURAL_LANGUAGE: GraphFormatConverter.to_natural_language(graph_data),
            FormatKey.CODE_LIKE:        GraphFormatConverter.to_code_like(graph_data),
            FormatKey.NODE_SEQUENCE:    GraphFormatConverter.to_node_sequence(graph_data),
        }

    @staticmethod
    def convert_selected(
        graph_data: Dict,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Convert only the requested formats. ``None`` → all formats."""
        _dispatch = {
            FormatKey.ADJACENCY_TABLE:   GraphFormatConverter.to_adjacency_table,
            FormatKey.NATURAL_LANGUAGE:  GraphFormatConverter.to_natural_language,
            FormatKey.CODE_LIKE:         GraphFormatConverter.to_code_like,
            FormatKey.NODE_SEQUENCE:     GraphFormatConverter.to_node_sequence,
        }
        result: Dict[str, str] = {}
        for key in (formats or FormatKey.ALL):
            fn = _dispatch.get(key)
            if fn:
                result[key] = fn(graph_data)
            else:
                logger.warning("convert_selected: unknown format key '%s'", key)
        return result

    @staticmethod
    def extract_key_facts(graph_data: Dict) -> str:
        """Structured Vietnamese bullet-points injected into MID_GENERATION as {key_facts}."""
        nodes:    List[Dict]  = graph_data.get("nodes", [])
        triplets: List[Tuple] = graph_data.get("triplets", [])
        subgraph: Dict        = graph_data.get("subgraph", {})

        all_nodes = nodes + subgraph.get("nodes", [])
        facts: List[str] = []

        characters = sorted({n[NodeProp.CHAR_NAME]  for n in all_nodes if NodeProp.CHAR_NAME  in n})
        actors     = sorted({n[NodeProp.ACTOR_NAME] for n in all_nodes if NodeProp.ACTOR_NAME in n})
        plays      = sorted({n[NodeProp.TITLE]       for n in all_nodes if NodeProp.TITLE      in n})
        scenes     = sorted({n[NodeProp.SCENE_NAME]  for n in all_nodes if NodeProp.SCENE_NAME in n})

        if characters:
            facts.append(f"Nhân vật: {', '.join(characters)}")
        if actors:
            facts.append(f"Diễn viên: {', '.join(actors)}")
        if plays:
            facts.append(f"Vở kịch: {', '.join(plays)}")
        if scenes:
            facts.append(f"Trích đoạn: {', '.join(scenes)}")

        cap = Limit.FALLBACK_NAMES_COUNT * 2
        performed_by = [(s, o) for s, r, o in triplets if r == RelType.PERFORMED_BY]
        if performed_by:
            facts.append("Quan hệ diễn xuất:")
            for char, actor in performed_by[:cap]:
                facts.append(f"  - {char} ← {actor}")

        has_scene = [(s, o) for s, r, o in triplets if r == RelType.HAS_SCENE]
        if has_scene:
            facts.append("Trích đoạn trong vở:")
            for play, scene in has_scene[:cap]:
                facts.append(f"  - {play} → {scene}")

        express = [(s, o) for s, r, o in triplets if r == RelType.EXPRESS]
        if express:
            facts.append("Cảm xúc diễn viên biểu lộ:")
            for actor, emo in express[:cap]:
                facts.append(f"  - {actor} → {emo}")

        is_wear_by = [(s, o) for s, r, o in triplets if r == RelType.IS_WEAR_BY]
        if is_wear_by:
            facts.append("Trang phục — diễn viên:")
            for costume, actor in is_wear_by[:cap]:
                facts.append(f"  - {costume} ← {actor}")

        return "\n".join(facts) if facts else "Không có thông tin quan trọng."
