from __future__ import annotations

import numpy as np

from spectrashift.contracts import HyperspectralCube


SYNTHETIC_CLASSES = (
    "synthetic_indicator_alpha",
    "synthetic_indicator_beta",
    "synthetic_indicator_gamma",
)


def _absorption_spectrum(
    wavelengths_nm: np.ndarray,
    centers_nm: tuple[float, ...],
    *,
    baseline: float,
) -> np.ndarray:
    spectrum = np.full_like(wavelengths_nm, baseline, dtype=np.float64)
    for index, center in enumerate(centers_nm):
        width = 35.0 + 12.0 * index
        depth = 0.18 - 0.025 * index
        spectrum -= depth * np.exp(-0.5 * ((wavelengths_nm - center) / width) ** 2)
    spectrum += 0.025 * np.sin(wavelengths_nm / 145.0)
    return np.clip(spectrum, 0.04, 0.95)


def make_synthetic_cube(
    *,
    seed: int = 20260822,
    height: int = 64,
    width: int = 64,
    bands: int = 285,
) -> tuple[HyperspectralCube, np.ndarray]:
    """Create a deterministic software fixture, not scientifically valid spectra."""

    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 2500.0, bands)
    background = 0.34 + 0.035 * np.cos(wavelengths / 210.0)
    references = np.stack(
        [
            _absorption_spectrum(wavelengths, (690.0, 960.0), baseline=0.48),
            _absorption_spectrum(wavelengths, (860.0, 1730.0), baseline=0.52),
            _absorption_spectrum(wavelengths, (1400.0, 2200.0), baseline=0.56),
        ]
    )

    reflectance = np.broadcast_to(background, (height, width, bands)).copy()
    reflectance += rng.normal(0.0, 0.006, size=reflectance.shape)
    labels = np.zeros((height, width, len(SYNTHETIC_CLASSES)), dtype=np.uint8)
    row_grid, col_grid = np.ogrid[:height, :width]
    regions = (
        ((19, 18), (11, 14)),
        ((43, 19), (10, 12)),
        ((34, 47), (13, 9)),
    )
    for class_index, ((row_center, col_center), (row_radius, col_radius)) in enumerate(regions):
        mask = (
            ((row_grid - row_center) / row_radius) ** 2
            + ((col_grid - col_center) / col_radius) ** 2
            <= 1.0
        )
        labels[..., class_index][mask] = 1
        mixture = 0.88 * references[class_index] + 0.12 * background
        reflectance[mask] = mixture + rng.normal(0.0, 0.004, size=(int(mask.sum()), bands))

    valid_mask = np.ones((height, width), dtype=bool)
    valid_mask[:4, :11] = False
    reflectance[~valid_mask] = 0.0
    cube = HyperspectralCube(
        scene_id="synthetic-fixture-v1",
        reflectance=np.clip(reflectance, 0.0, 1.0),
        wavelengths_nm=wavelengths,
        valid_mask=valid_mask,
        labels=labels,
        class_names=SYNTHETIC_CLASSES,
        metadata={
            "evidence_grade": "synthetic-demonstration",
            "scientific_use": False,
            "seed": seed,
        },
    )
    return cube, references

