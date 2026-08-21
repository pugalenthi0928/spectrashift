import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectrashift.data.oxhyper import (
    OxHyperRecord,
    assert_no_source_group_leakage,
    build_pilot_manifest,
    discover_oxhyper_records,
    infer_source_group,
    parse_envi_wavelengths,
)


class OxHyperTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        split_tiles = {
            "train_minerals.csv": ["areaA_captureA_00", "areaA_captureA_01"],
            "val_minerals.csv": ["areaB_captureB_00"],
            "test_minerals.csv": ["areaC_captureC_00"],
        }
        for filename, tiles in split_tiles.items():
            with (root / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "event_id"])
                writer.writeheader()
                for index, tile in enumerate(tiles):
                    writer.writerow({"id": index, "event_id": tile})
                    directory = root / tile
                    directory.mkdir(exist_ok=True)
                    (directory / "C").write_bytes(b"cube")
                    (directory / "C.hdr").write_text(
                        "ENVI\nwavelength units = Nanometers\nwavelength = {400, 500, 600}\n",
                        encoding="utf-8",
                    )
                    (directory / "minerals3ghk.tif").write_bytes(b"labels")

    def test_discovers_and_selects_leakage_safe_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            records = discover_oxhyper_records(root)
            self.assertEqual(len(records), 4)
            output = root / "pilot.json"
            manifest = build_pilot_manifest(
                root,
                output,
                groups_per_split={"train": 1, "validation": 1, "test": 1},
                tiles_per_group=1,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(len(manifest["records"]), 3)
            self.assertEqual({row["split"] for row in manifest["records"]}, {
                "train", "validation", "test"
            })

    def test_source_group_removes_only_tile_suffix(self) -> None:
        self.assertEqual(infer_source_group("areaA_captureA_03"), "areaA_captureA")

    def test_rejects_cross_split_source_group(self) -> None:
        base = {
            "cube_path": "tile/C",
            "header_path": "tile/C.hdr",
            "label_path": "tile/minerals3ghk.tif",
            "info_path": None,
            "cube_size_bytes": 1,
            "label_size_bytes": 1,
        }
        records = [
            OxHyperRecord(tile_id="a_00", split="train", source_group="same", **base),
            OxHyperRecord(tile_id="b_00", split="test", source_group="same", **base),
        ]
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_no_source_group_leakage(records)

    def test_parses_micrometre_envi_wavelengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "C.hdr"
            path.write_text(
                "ENVI\nwavelength units = Micrometers\nwavelength = {0.4, 0.5, 0.6}\n",
                encoding="utf-8",
            )
            np.testing.assert_allclose(parse_envi_wavelengths(path), [400, 500, 600])


if __name__ == "__main__":
    unittest.main()
