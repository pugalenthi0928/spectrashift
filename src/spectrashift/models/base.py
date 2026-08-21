from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from spectrashift.contracts import HyperspectralCube


@dataclass(frozen=True)
class ModelDescriptor:
    family: str
    revision: str
    adaptation: str
    checkpoint: str | None
    evidence_grade: str


class ModelAdapter(Protocol):
    descriptor: ModelDescriptor

    def predict(self, cube: HyperspectralCube) -> np.ndarray:
        """Return a ``(height, width, classes)`` score or probability array."""

