import unittest

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.models.base import ModelDescriptor
from spectrashift.models.dofa import (
    DofaFrozenProbe,
    FrozenProbeConfig,
    pool_patch_labels,
)


class FakeDenseEncoder:
    descriptor = ModelDescriptor(
        family="fake-dense-encoder",
        revision="test",
        adaptation="frozen-features+linear-probe",
        checkpoint=None,
        evidence_grade="mechanism-tested",
    )

    def __init__(self) -> None:
        self.fitted = False

    def fit_normalization(self, cubes: list[HyperspectralCube]) -> None:
        self.fitted = bool(cubes)

    def extract(self, cube: HyperspectralCube) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("normalization not fitted")
        output = np.zeros((4, 4, cube.bands), dtype=np.float32)
        for row in range(4):
            for column in range(4):
                region = cube.reflectance[
                    row * 2 : (row + 1) * 2,
                    column * 2 : (column + 1) * 2,
                ]
                output[row, column] = region.mean(axis=(0, 1))
        return output

    def configuration(self) -> dict[str, object]:
        return {"kind": "unit-test-fake"}


class DofaProbeTests(unittest.TestCase):
    def _cube(self, scene_id: str, offset: float = 0.0) -> HyperspectralCube:
        labels = np.zeros((8, 8, 3), dtype=np.uint8)
        labels[:4, :4, 0] = 1
        labels[:4, 4:, 1] = 1
        labels[4:, :4, 2] = 1
        reflectance = labels.astype(np.float32) * 0.8 + 0.1 + offset
        return HyperspectralCube(
            scene_id=scene_id,
            reflectance=reflectance,
            wavelengths_nm=np.array([500.0, 700.0, 900.0]),
            valid_mask=np.ones((8, 8), dtype=bool),
            labels=labels,
            class_names=("a", "b", "c"),
        )

    def test_pools_patch_labels_and_validity(self) -> None:
        cube = self._cube("pool")
        pooled, valid, share = pool_patch_labels(
            cube,
            (4, 4),
            positive_fraction=0.25,
            minimum_valid_fraction=0.8,
        )
        self.assertEqual(pooled.shape, (4, 4, 3))
        self.assertTrue(np.all(valid))
        self.assertAlmostEqual(float(share[0, 0, 0]), 1.0)
        self.assertEqual(int(pooled[0, 0, 0]), 1)

    def test_frozen_probe_returns_pixel_aligned_probabilities(self) -> None:
        model = DofaFrozenProbe(
            FakeDenseEncoder(),
            FrozenProbeConfig(positive_patch_fraction=0.25, seed=3),
        ).fit([self._cube("train-a"), self._cube("train-b", offset=0.01)])
        cube = self._cube("held-out", offset=0.02)
        probabilities = model.predict(cube)
        self.assertEqual(probabilities.shape, cube.labels.shape)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertGreater(probabilities[:4, :4, 0].mean(), probabilities[4:, 4:, 0].mean())
        configuration = model.configuration()
        self.assertEqual(configuration["feature_dimension"], 3)
        self.assertEqual(configuration["trainable_parameters"], 12)


if __name__ == "__main__":
    unittest.main()
