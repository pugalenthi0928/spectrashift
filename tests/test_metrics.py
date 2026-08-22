import unittest

import numpy as np

from spectrashift.metrics import (
    brier_score,
    expected_calibration_error,
    multilabel_report,
    optimize_f1_thresholds,
)


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        probabilities = labels.astype(float)
        report = multilabel_report(labels, probabilities)
        self.assertAlmostEqual(report["macro_f1"], 1.0)
        self.assertAlmostEqual(report["mean_iou"], 1.0)
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)
        self.assertAlmostEqual(brier_score(labels, probabilities), 0.0)

    def test_validation_thresholds_are_selected_per_class(self) -> None:
        labels = np.array([[0, 0], [0, 1], [1, 1], [1, 1]], dtype=np.uint8)
        probabilities = np.array(
            [[0.1, 0.2], [0.4, 0.45], [0.55, 0.6], [0.7, 0.8]], dtype=float
        )
        thresholds = optimize_f1_thresholds(
            labels, probabilities, candidates=np.array([0.3, 0.5, 0.7])
        )
        self.assertEqual(thresholds.shape, (2,))
        report = multilabel_report(labels, probabilities, threshold=thresholds)
        self.assertEqual(len(report["threshold"]), 2)


if __name__ == "__main__":
    unittest.main()
