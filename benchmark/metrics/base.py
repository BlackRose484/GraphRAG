"""
Base classes for all benchmark metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ── Metric groups ─────────────────────────────────────────────────────────────

class MetricGroup(str, Enum):
    """Logical category for each metric."""
    IR    = "IR"      # Information Retrieval
    NLG   = "NLG"     # Natural Language Generation
    EXACT = "Exact"   # Exact / keyword match
    RAGAS = "RAGAS"   # RAGAS LLM-based metrics


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """
    Single metric evaluation result.

    Attributes:
        name:     Metric name (e.g. ``"Precision@5"``).
        group:    Which :class:`MetricGroup` this belongs to.
        value:    Numeric score, typically in [0, 1].
        metadata: Optional dict with intermediate computation details.
        error:    Set when computation failed; ``value`` will be ``None``.
    """
    name:     str
    group:    MetricGroup
    value:    Optional[float]
    metadata: Dict[str, Any]  = field(default_factory=dict)
    error:    Optional[str]   = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def display(self) -> str:
        if not self.ok:
            return f"{self.name}: ERROR ({self.error})"
        return f"{self.name}: {self.value:.4f}"


# ── Abstract base ─────────────────────────────────────────────────────────────

class MetricBase(ABC):
    """
    Abstract base class every metric must implement.

    Subclasses define:
    - :attr:`name`               — display string
    - :attr:`group`              — :class:`MetricGroup`
    - :attr:`requires_llm`       — whether an LLM call is needed
    - :attr:`requires_ground_truth` — whether labeled ground-truth is needed
    - :meth:`evaluate`           — the computation itself
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def group(self) -> MetricGroup: ...

    @property
    @abstractmethod
    def requires_llm(self) -> bool: ...

    @property
    @abstractmethod
    def requires_ground_truth(self) -> bool: ...

    @abstractmethod
    def evaluate(self, **kwargs) -> float:
        """
        Compute and return the metric value in [0, 1].

        Each concrete class documents its own ``**kwargs``.
        """

    def safe_evaluate(self, **kwargs) -> MetricResult:
        """Wrap :meth:`evaluate` in a try/except, returning a :class:`MetricResult`."""
        try:
            value = self.evaluate(**kwargs)
            return MetricResult(name=self.name, group=self.group, value=value)
        except Exception as exc:
            return MetricResult(
                name=self.name, group=self.group,
                value=None, error=str(exc),
            )
