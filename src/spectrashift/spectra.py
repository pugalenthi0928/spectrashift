from __future__ import annotations

import numpy as np


def interpolate_wavelengths(
    spectra: np.ndarray,
    source_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Interpolate the last spectral axis onto a target wavelength grid."""

    spectra = np.asarray(spectra, dtype=np.float64)
    source = np.asarray(source_wavelengths_nm, dtype=np.float64)
    target = np.asarray(target_wavelengths_nm, dtype=np.float64)
    if spectra.shape[-1] != source.size:
        raise ValueError("spectral axis does not match source wavelengths")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("source and target wavelengths must be strictly increasing")
    if not allow_extrapolation and (target[0] < source[0] or target[-1] > source[-1]):
        raise ValueError("target wavelengths fall outside the source range")

    flat = spectra.reshape(-1, spectra.shape[-1])
    result = np.empty((flat.shape[0], target.size), dtype=np.float64)
    for row_index, row in enumerate(flat):
        result[row_index] = np.interp(target, source, row)
    return result.reshape(*spectra.shape[:-1], target.size)


def spectral_angles(
    reflectance: np.ndarray,
    reference_spectra: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return spectral angle in radians for every pixel/reference pair."""

    reflectance = np.asarray(reflectance)
    references = np.asarray(reference_spectra)
    dtype = np.result_type(reflectance.dtype, references.dtype, np.float32)
    reflectance = reflectance.astype(dtype, copy=False)
    references = references.astype(dtype, copy=False)
    if reflectance.ndim != 3:
        raise ValueError("reflectance must have shape (height, width, bands)")
    if references.ndim != 2 or references.shape[1] != reflectance.shape[2]:
        raise ValueError("reference_spectra must have shape (classes, bands)")

    pixels = reflectance.reshape(-1, reflectance.shape[-1])
    pixel_norm = np.linalg.norm(pixels, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(references, axis=1, keepdims=True).T
    denominator = np.maximum(pixel_norm * reference_norm, epsilon)
    cosine = (pixels @ references.T) / denominator
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angles.reshape(reflectance.shape[0], reflectance.shape[1], references.shape[0])


def angles_to_similarity(angles: np.ndarray, *, scale: float = 24.0) -> np.ndarray:
    """Convert small spectral angles to bounded similarity-like scores.

    This monotonic transform is not a calibrated probability. Calibration must be
    measured separately on a validation set.
    """

    angles = np.asarray(angles, dtype=np.float64)
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.exp(-scale * np.clip(angles, 0.0, np.pi))
