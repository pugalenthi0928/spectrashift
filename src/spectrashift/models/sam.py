from __future__ import annotations

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.models.base import ModelDescriptor
from spectrashift.spectra import angles_to_similarity, spectral_angles


class SpectralAngleMapper:
    """Spectral Angle Mapper baseline against aligned reference spectra."""

    def __init__(self, reference_spectra: np.ndarray, *, scale: float = 24.0) -> None:
        self.reference_spectra = np.asarray(reference_spectra, dtype=np.float64)
        self.scale = scale
        self.descriptor = ModelDescriptor(
            family="SpectralAngleMapper",
            revision="spectrashift-0.1.0",
            adaptation="none",
            checkpoint=None,
            evidence_grade="mechanism-tested",
        )

    def predict(self, cube: HyperspectralCube) -> np.ndarray:
        angles = spectral_angles(cube.reflectance, self.reference_spectra)
        scores = angles_to_similarity(angles, scale=self.scale)
        scores[~cube.valid_mask] = 0.0
        return scores

