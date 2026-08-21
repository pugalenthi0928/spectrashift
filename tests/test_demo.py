import tempfile
import unittest
from pathlib import Path

from spectrashift.cli import run_demo


class DemoTests(unittest.TestCase):
    def test_demo_emits_evidence_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_demo(output, seed=7)
            self.assertEqual(summary["evidence_grade"], "synthetic-demonstration")
            self.assertGreaterEqual(summary["targets"], 1)
            for name in (
                "run_manifest.json",
                "metrics.json",
                "target_cards.json",
                "synthetic_outputs.npz",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()

