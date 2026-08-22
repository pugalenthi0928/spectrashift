from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.models.base import ModelDescriptor


@dataclass(frozen=True)
class PcaLogisticConfig:
    components: int = 32
    regularization_c: float = 1.0
    max_iter: int = 500
    prediction_chunk_pixels: int = 65536
    seed: int = 20260822


class PcaLogisticBaseline:
    """Leakage-safe PCA plus one-vs-rest logistic regression baseline."""

    def __init__(self, config: PcaLogisticConfig | None = None) -> None:
        self.config = config or PcaLogisticConfig()
        self.descriptor = ModelDescriptor(
            family="PCA+LogisticRegression",
            revision="spectrashift-0.2.0",
            adaptation="supervised-baseline",
            checkpoint=None,
            evidence_grade="mechanism-tested",
        )
        self._scaler: Any | None = None
        self._pca: Any | None = None
        self._classifiers: list[Any] = []
        self._bands: int | None = None

    def fit(self, spectra: np.ndarray, labels: np.ndarray) -> PcaLogisticBaseline:
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - exercised without ml extras
            raise RuntimeError("scikit-learn is required; install spectrashift[ml]") from exc

        spectra = np.asarray(spectra, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint8)
        if spectra.ndim != 2 or labels.ndim != 2:
            raise ValueError("spectra and labels must be two-dimensional")
        if spectra.shape[0] != labels.shape[0]:
            raise ValueError("spectra and labels must contain the same samples")
        if spectra.shape[0] < 4:
            raise ValueError("at least four training spectra are required")
        if np.any(~np.isfinite(spectra)):
            raise ValueError("training spectra must be finite")
        if np.any((labels != 0) & (labels != 1)):
            raise ValueError("training labels must be binary")

        components = min(self.config.components, spectra.shape[1], spectra.shape[0] - 1)
        if components < 1:
            raise ValueError("PCA configuration produces zero components")
        self._scaler = StandardScaler(copy=True)
        scaled = self._scaler.fit_transform(spectra)
        self._pca = PCA(
            n_components=components,
            svd_solver="randomized" if components < min(scaled.shape) else "auto",
            random_state=self.config.seed,
        )
        features = self._pca.fit_transform(scaled)
        self._classifiers = []
        for class_index in range(labels.shape[1]):
            target = labels[:, class_index]
            if np.unique(target).size != 2:
                raise ValueError(
                    f"class {class_index} needs both positive and negative training samples"
                )
            classifier = LogisticRegression(
                C=self.config.regularization_c,
                class_weight="balanced",
                max_iter=self.config.max_iter,
                random_state=self.config.seed + class_index,
                solver="lbfgs",
            )
            classifier.fit(features, target)
            self._classifiers.append(classifier)
        self._bands = spectra.shape[1]
        return self

    def predict_spectra(self, spectra: np.ndarray) -> np.ndarray:
        if self._scaler is None or self._pca is None or not self._classifiers:
            raise RuntimeError("model must be fitted before prediction")
        spectra = np.asarray(spectra, dtype=np.float32)
        if spectra.ndim != 2 or spectra.shape[1] != self._bands:
            raise ValueError(f"spectra must have shape (samples, {self._bands})")
        probabilities = np.empty((spectra.shape[0], len(self._classifiers)), dtype=np.float32)
        step = self.config.prediction_chunk_pixels
        for start in range(0, spectra.shape[0], step):
            stop = min(start + step, spectra.shape[0])
            features = self._pca.transform(self._scaler.transform(spectra[start:stop]))
            for class_index, classifier in enumerate(self._classifiers):
                probabilities[start:stop, class_index] = classifier.predict_proba(features)[:, 1]
        return probabilities

    def predict(self, cube: HyperspectralCube) -> np.ndarray:
        if self._bands is not None and cube.bands != self._bands:
            raise ValueError(f"model expects {self._bands} bands, received {cube.bands}")
        flat = cube.reflectance.reshape(-1, cube.bands)
        probabilities = self.predict_spectra(flat).reshape(
            cube.height, cube.width, len(self._classifiers)
        )
        probabilities[~cube.valid_mask] = 0.0
        return probabilities

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)
