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


if __name__ == "__main__":
    unittest.main()

