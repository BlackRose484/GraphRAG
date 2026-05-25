"""Base classes for all benchmark metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MetricGroup(str, Enum):
    IR    = "IR"
    NLG   = "NLG"
    EXACT = "Exact"
    RAGAS = "RAGAS"


@dataclass
class MetricResult:
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


class MetricBase(ABC):
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
    def evaluate(self, **kwargs) -> float: ...

    def safe_evaluate(self, **kwargs) -> MetricResult:
        try:
            value = self.evaluate(**kwargs)
            return MetricResult(name=self.name, group=self.group, value=value)
        except Exception as exc:
            return MetricResult(
                name=self.name, group=self.group,
                value=None, error=str(exc),
            )
