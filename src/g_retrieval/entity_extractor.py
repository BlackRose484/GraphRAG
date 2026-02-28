"""
Entity extraction from user queries.

Uses a two-stage approach:
  1. Regex pattern matching for capitalised Vietnamese proper nouns (fast, no LLM).
  2. LLM extraction using the ENTITY_EXTRACT prompt for structured JSON results.

Results from both stages are merged and de-duplicated before returning.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from src.constants.constant import EntityType
from src.constants.promt_engineer import ENTITY_EXTRACT
from src.core.base import BaseModel
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Regex that matches Vietnamese capitalised proper nouns (single or multi-word).
_VN_PROPER_NOUN = re.compile(
    r"\b[A-ZĐÂĂÊÔƠƯ][a-zđâăêôơưáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+"
    r"(?:\s+[A-ZĐÂĂÊÔƠƯ][a-zđâăêôơưáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+)*\b"
)

_EMPTY_ENTITIES: dict[str, list[str]] = {
    EntityType.CHARACTERS: [],
    EntityType.ACTORS:     [],
    EntityType.PLAYS:      [],
    EntityType.SCENES:     [],
}


class ExtractedEntities(TypedDict):
    characters: list[str]
    actors: list[str]
    plays: list[str]
    scenes: list[str]


class EntityExtractor(BaseModel):
    """Extract named entities from a Cheo-related query."""

    def __init__(self) -> None:
        super().__init__()
        _logger.info("EntityExtractor initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, query: str) -> ExtractedEntities:
        """Extract entities from *query*.

        Returns:
            Merged dict with lists for characters / actors / plays / scenes.
        """
        # Stage 1: regex patterns → go into characters as a safe default
        pattern_names = _VN_PROPER_NOUN.findall(query)

        entities: ExtractedEntities = {
            EntityType.CHARACTERS: list(pattern_names),
            EntityType.ACTORS:     [],
            EntityType.PLAYS:      [],
            EntityType.SCENES:     [],
        }

        # Stage 2: LLM
        try:
            llm_entities = self._extract_by_llm(query)
            for key in EntityType.ALL:
                merged = list({*entities[key], *llm_entities.get(key, [])})
                entities[key] = merged  # type: ignore[literal-required]
            total = sum(len(v) for v in entities.values())
            _logger.info("Entity extraction complete — %d entities found", total)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("LLM entity extraction failed: %s", exc)

        return entities

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_by_llm(self, query: str) -> dict[str, list[str]]:
        prompt = ENTITY_EXTRACT.format(query=query)
        raw = self.safe_generate(prompt).strip()

        # Strip ```json ... ``` fences if present
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                raw = m.group(1)

        # Also handle a bare JSON object without fences
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        if m2:
            raw = m2.group(0)

        try:
            parsed = json.loads(raw)
            # Normalise: ensure all keys exist and values are lists of strings
            result: dict[str, list[str]] = {}
            for key in EntityType.ALL:
                val = parsed.get(key, [])
                result[key] = [str(v) for v in val] if isinstance(val, list) else []
            return result
        except json.JSONDecodeError as exc:
            _logger.error("Failed to parse LLM entity JSON: %s | raw=%r", exc, raw[:200])
            return dict(_EMPTY_ENTITIES)
