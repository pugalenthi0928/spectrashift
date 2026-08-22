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
    load_raw_oxhyper_labels,
    parse_envi_wavelengths,
    resolve_oxhyper_wavelengths,
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
            self.assertEqual(
                manifest["selection"]["selected_groups_per_split"],
                {"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(
                manifest["selection"]["selected_tiles_per_split"],
                {"train": 1, "validation": 1, "test": 1},
            )

    def test_auto_detects_mini_split_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            renames = {
                "train_minerals.csv": "train_minerals_10.csv",
                "val_minerals.csv": "val_minerals_10.csv",
                "test_minerals.csv": "test_minerals_10.csv",
            }
            for source, target in renames.items():
                (root / source).rename(root / target)
            self.assertEqual(len(discover_oxhyper_records(root)), 4)

    def test_manifest_preserves_leakage_from_complete_published_split_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            for filename, tile in (
                ("train_minerals.csv", "published_leak_00"),
                ("test_minerals.csv", "published_leak_01"),
            ):
                with (root / filename).open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["id", "event_id"])
                    writer.writerow({"id": 99, "event_id": tile})
            manifest = build_pilot_manifest(
                root,
                root / "pilot.json",
                groups_per_split={"train": 1, "validation": 1, "test": 1},
                tiles_per_group=1,
            )
            self.assertEqual(
                manifest["selection"]["excluded_leaking_groups"]["published_leak"],
                ["test", "train"],
            )

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

    def test_uses_published_emit_fallback_for_generic_285_band_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "C.hdr"
            path.write_text("ENVI\nbands = 285\n", encoding="utf-8")
            wavelengths, source = resolve_oxhyper_wavelengths(
                path, [f"Band {index}" for index in range(1, 286)]
            )
            self.assertEqual(wavelengths.size, 285)
            self.assertAlmostEqual(wavelengths[0], 381.00558)
            self.assertAlmostEqual(wavelengths[-1], 2492.9238)
            self.assertEqual(source, "HyperspectralViTs-published-285-band-fallback")

    def test_prefers_numeric_raster_band_descriptions_when_header_has_no_wavelengths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "C.hdr"
            path.write_text("ENVI\nbands = 3\n", encoding="utf-8")
            wavelengths, source = resolve_oxhyper_wavelengths(
                path, ["400.5 (400.5)", "500.5 (500.5)", "600.5 (600.5)"]
            )
            np.testing.assert_allclose(wavelengths, [400.5, 500.5, 600.5])
            self.assertEqual(source, "raster-band-descriptions")

    def test_loads_validated_headerless_mini_labels_as_little_endian_bsq(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minerals3ghk.tif"
            expected = np.zeros((3, 4, 5), dtype="<u2")
            expected[0, 0, 0] = 1
            expected[1, 1, 1] = 1
            expected[2, 2, 2] = 1
            path.write_bytes(expected.tobytes())
            loaded = load_raw_oxhyper_labels(path, height=4, width=5)
            np.testing.assert_array_equal(loaded, expected.astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
