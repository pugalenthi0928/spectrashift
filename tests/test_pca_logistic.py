import unittest

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.models.pca_logistic import PcaLogisticBaseline, PcaLogisticConfig


class PcaLogisticTests(unittest.TestCase):
    def test_fits_and_predicts_multilabel_probabilities(self) -> None:
        rng = np.random.default_rng(12)
        spectra = rng.normal(size=(240, 8)).astype(np.float32)
        labels = np.column_stack(
            [spectra[:, 0] > 0, spectra[:, 1] > 0, spectra[:, 2] > 0]
        ).astype(np.uint8)
        model = PcaLogisticBaseline(PcaLogisticConfig(components=6, seed=12)).fit(
            spectra, labels
        )
        cube = HyperspectralCube(
            scene_id="held-out",
            reflectance=spectra[:24].reshape(4, 6, 8),
            wavelengths_nm=np.arange(8) * 10 + 400,
            valid_mask=np.ones((4, 6), dtype=bool),
            labels=labels[:24].reshape(4, 6, 3),
            class_names=("a", "b", "c"),
        )
        probabilities = model.predict(cube)
        self.assertEqual(probabilities.shape, (4, 6, 3))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))


if __name__ == "__main__":
    unittest.main()
