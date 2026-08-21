from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HyperspectralCube:
    """Validated in-memory representation of one hyperspectral scene.

    Reflectance uses ``(height, width, bands)`` and labels, when present, use
    ``(height, width, classes)``. Wavelengths are expressed in nanometres.
    """

    scene_id: str
    reflectance: np.ndarray
    wavelengths_nm: np.ndarray
    valid_mask: np.ndarray
    labels: np.ndarray | None = None
    class_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reflectance = np.asarray(self.reflectance, dtype=np.float64)
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, dtype=np.float64)
        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.uint8)
        self.validate()

    @property
    def height(self) -> int:
        return int(self.reflectance.shape[0])

    @property
    def width(self) -> int:
        return int(self.reflectance.shape[1])

    @property
    def bands(self) -> int:
        return int(self.reflectance.shape[2])

    def validate(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if self.reflectance.ndim != 3:
            raise ValueError("reflectance must have shape (height, width, bands)")
        if self.wavelengths_nm.ndim != 1:
            raise ValueError("wavelengths_nm must be one-dimensional")
        if self.bands != self.wavelengths_nm.size:
            raise ValueError("wavelength count must match the reflectance band dimension")
        if self.valid_mask.shape != (self.height, self.width):
            raise ValueError("valid_mask must match the spatial dimensions")
        if not np.all(np.isfinite(self.wavelengths_nm)):
            raise ValueError("wavelengths must be finite")
        if np.any(np.diff(self.wavelengths_nm) <= 0):
            raise ValueError("wavelengths must be strictly increasing")
        if np.any(self.wavelengths_nm <= 0):
            raise ValueError("wavelengths must be positive")
        if not np.all(np.isfinite(self.reflectance[self.valid_mask])):
            raise ValueError("valid reflectance pixels must contain only finite values")
        if self.labels is not None:
            if self.labels.ndim != 3:
                raise ValueError("labels must have shape (height, width, classes)")
            if self.labels.shape[:2] != (self.height, self.width):
                raise ValueError("label spatial dimensions must match reflectance")
            if self.class_names and self.labels.shape[2] != len(self.class_names):
                raise ValueError("class_names must match the label class dimension")
            if np.any((self.labels != 0) & (self.labels != 1)):
                raise ValueError("labels must be binary multi-hot values")

    def valid_spectra(self) -> np.ndarray:
        """Return a ``(valid_pixels, bands)`` view of valid spectra."""

        return self.reflectance[self.valid_mask]

