import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectrashift.benchmark import run_oxhyper_benchmark
from spectrashift.contracts import HyperspectralCube
from spectrashift.data.oxhyper import OXHYPER_CLASS_NAMES, OxHyperRecord


class RealBenchmarkHarnessTests(unittest.TestCase):
    def _record(self, tile_id: str, split: str) -> OxHyperRecord:
        return OxHyperRecord(
            tile_id=tile_id,
            split=split,
            source_group=f"group-{tile_id}",
            cube_path=f"{tile_id}/C",
            header_path=f"{tile_id}/C.hdr",
            label_path=f"{tile_id}/minerals3ghk.tif",
            info_path=None,
            cube_size_bytes=1,
            label_size_bytes=1,
        )

    def _cube(self, record: OxHyperRecord) -> HyperspectralCube:
        seed = sum(record.tile_id.encode())
        rng = np.random.default_rng(seed)
        wavelengths = np.linspace(400, 900, 12)
        references = np.vstack(
            [
                0.3 + 0.15 * np.sin(wavelengths / 80),
                0.35 + 0.12 * np.cos(wavelengths / 90),
                0.4 + 0.10 * np.sin(wavelengths / 110 + 1.0),
            ]
        )
        reflectance = np.full((18, 18, 12), 0.25, dtype=np.float32)
        labels = np.zeros((18, 18, 3), dtype=np.uint8)
        regions = [
            (slice(1, 7), slice(1, 7)),
            (slice(7, 13), slice(2, 8)),
            (slice(9, 16), slice(10, 17)),
        ]
        for class_index, region in enumerate(regions):
            labels[region + (class_index,)] = 1
            reflectance[region] = references[class_index]
        reflectance += rng.normal(0, 0.01, reflectance.shape).astype(np.float32)
        return HyperspectralCube(
            scene_id=record.tile_id,
            reflectance=reflectance,
            wavelengths_nm=wavelengths,
            valid_mask=np.ones((18, 18), dtype=bool),
            labels=labels,
            class_names=OXHYPER_CLASS_NAMES,
            metadata={"source_group": record.source_group, "split": record.split},
        )

    def test_runner_preserves_complete_evidence_bundle(self) -> None:
        records = [
            self._record("train-1", "train"),
            self._record("validation-1", "validation"),
            self._record("test-1", "test"),
        ]
        manifest = {
            "schema_version": 1,
            "dataset": {"name": "test fixture"},
            "evidence_grade_on_success": "mechanism-tested",
            "records": [record.to_dict() for record in records],
            "limitations": ["Test fixture only."],
        }
        by_id = {record.tile_id: self._cube(record) for record in records}

        def loader(_root: str | Path, record: OxHyperRecord) -> HyperspectralCube:
            return by_id[record.tile_id]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "pilot.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "artifacts"
            summary = run_oxhyper_benchmark(
                manifest_path,
                root,
                output,
                model_name="prototype-sam",
                max_train_pixels_per_tile=200,
                seed=4,
                loader=loader,
            )
            self.assertEqual(summary["evidence_grade"], "mechanism-tested")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertGreater(metrics["efficiency"]["process_peak_rss_mb"], 0.0)
            for name in (
                "run_manifest.json",
                "metrics.json",
                "thresholds.json",
                "target_cards.json",
                "failure_cases.json",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file())
            self.assertTrue((output / "predictions" / "test-1.npz").is_file())


if __name__ == "__main__":
    unittest.main()
