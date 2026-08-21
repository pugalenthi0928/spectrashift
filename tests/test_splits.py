import unittest

from spectrashift.splits import SceneRecord, assert_group_disjoint, grouped_split


class SplitTests(unittest.TestCase):
    def test_grouped_split_is_disjoint_and_deterministic(self) -> None:
        records = [
            SceneRecord(sample_id=f"{group}-{index}", scene_group=group)
            for group in ("a", "b", "c", "d", "e", "f")
            for index in range(3)
        ]
        first = grouped_split(records, seed=9)
        second = grouped_split(records, seed=9)
        self.assertEqual(first, second)
        assert_group_disjoint(first)

    def test_overlap_is_rejected(self) -> None:
        record = SceneRecord(sample_id="one", scene_group="same")
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_group_disjoint({"train": [record], "test": [record]})


if __name__ == "__main__":
    unittest.main()

