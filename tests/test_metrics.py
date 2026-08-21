import unittest

import numpy as np

from spectrashift.metrics import brier_score, expected_calibration_error, multilabel_report


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        probabilities = labels.astype(float)
        report = multilabel_report(labels, probabilities)
        self.assertAlmostEqual(report["macro_f1"], 1.0)
        self.assertAlmostEqual(report["mean_iou"], 1.0)
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)
        self.assertAlmostEqual(brier_score(labels, probabilities), 0.0)


if __name__ == "__main__":
    unittest.main()

