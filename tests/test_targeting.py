import unittest

import numpy as np

from spectrashift.targeting import extract_target_cards


class TargetingTests(unittest.TestCase):
    def test_extracts_and_ranks_connected_targets(self) -> None:
        probabilities = np.zeros((12, 12, 1), dtype=float)
        probabilities[1:4, 1:4, 0] = 0.9
        probabilities[7:11, 7:11, 0] = 0.75
        cards = extract_target_cards(
            probabilities,
            ("indicator",),
            valid_mask=np.ones((12, 12), dtype=bool),
            min_pixels=4,
        )
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].rank, 1)
        self.assertGreater(cards[0].score, cards[1].score)

    def test_uses_scene_identity_and_per_class_thresholds(self) -> None:
        probabilities = np.zeros((8, 8, 2), dtype=float)
        probabilities[1:4, 1:4, 0] = 0.7
        probabilities[4:7, 4:7, 1] = 0.8
        cards = extract_target_cards(
            probabilities,
            ("a", "b"),
            valid_mask=np.ones((8, 8), dtype=bool),
            threshold=np.array([0.6, 0.75]),
            min_pixels=4,
            scene_id="real-tile-01",
        )
        self.assertEqual({card.scene_id for card in cards}, {"real-tile-01"})
        self.assertTrue(all(card.target_id.startswith("real-tile-01:") for card in cards))


if __name__ == "__main__":
    unittest.main()
