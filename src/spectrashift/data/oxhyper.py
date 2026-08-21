from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.evidence import write_json

OXHYPER_CLASS_NAMES = ("goethite", "hematite", "kaolinite")
OXHYPER_MINI_REPOSITORY = "previtus/OxHyperMinerals_MINI"
OXHYPER_MINI_REVISION = "0b58274"
DEFAULT_SPLIT_FILES = {
    "train": "train_minerals.csv",
    "validation": "val_minerals.csv",
    "test": "test_minerals.csv",
}


@dataclass(frozen=True)
class OxHyperRecord:
    tile_id: str
    split: str
    source_group: str
    cube_path: str
    header_path: str
    label_path: str
    info_path: str | None
    cube_size_bytes: int
    label_size_bytes: int
    cube_sha256: str | None = None
    label_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OxHyperRecord:
        return cls(**value)


def infer_source_group(tile_id: str) -> str:
    """Infer the parent source capture while keeping adjacent tiles together."""

    tile_id = tile_id.strip().strip("/")
    if not tile_id:
        raise ValueError("tile_id must be non-empty")
    parts = tile_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return tile_id


def _normalise_split(value: str) -> str:
    value = value.strip().lower()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    value = aliases.get(value, value)
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {value!r}")
    return value


def _row_tile_id(row: dict[str, str]) -> str:
    for key in ("event_id", "folder", "tile_id", "sample_id", "id"):
        value = (row.get(key) or "").strip()
        if value:
            return Path(value).name
    raise ValueError(f"split row has no usable tile identifier: {sorted(row)}")


def _read_split_map(root: Path, split_files: dict[str, str]) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split_name, filename in split_files.items():
        path = root / filename
        if not path.is_file():
            continue
        split = _normalise_split(split_name)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"split file has no header: {path}")
            for row in reader:
                tile_id = _row_tile_id(row)
                row_split = row.get("split")
                resolved = _normalise_split(row_split) if row_split else split
                previous = split_map.get(tile_id)
                if previous is not None and previous != resolved:
                    raise ValueError(
                        f"tile {tile_id!r} is assigned to both {previous} and {resolved}"
                    )
                split_map[tile_id] = resolved
    return split_map


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_oxhyper_records(
    dataset_root: str | Path,
    *,
    split_files: dict[str, str] | None = None,
    hash_files: bool = False,
) -> list[OxHyperRecord]:
    """Discover complete OxHyperMinerals tiles without silently accepting partial data."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    split_files = split_files or DEFAULT_SPLIT_FILES
    split_map = _read_split_map(root, split_files)
    if not split_map:
        raise FileNotFoundError(
            "no OxHyperMinerals split CSVs found; expected train_minerals.csv, "
            "val_minerals.csv, and test_minerals.csv"
        )

    records: list[OxHyperRecord] = []
    incomplete: list[str] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        tile_id = directory.name
        if tile_id not in split_map:
            continue
        cube_path = directory / "C"
        header_path = directory / "C.hdr"
        label_path = directory / "minerals3ghk.tif"
        missing = [path.name for path in (cube_path, header_path, label_path) if not path.is_file()]
        if missing:
            incomplete.append(f"{tile_id}: {', '.join(missing)}")
            continue
        info_path = directory / "info.json"
        records.append(
            OxHyperRecord(
                tile_id=tile_id,
                split=split_map[tile_id],
                source_group=infer_source_group(tile_id),
                cube_path=str(cube_path.relative_to(root)),
                header_path=str(header_path.relative_to(root)),
                label_path=str(label_path.relative_to(root)),
                info_path=str(info_path.relative_to(root)) if info_path.is_file() else None,
                cube_size_bytes=cube_path.stat().st_size,
                label_size_bytes=label_path.stat().st_size,
                cube_sha256=_sha256_file(cube_path) if hash_files else None,
                label_sha256=_sha256_file(label_path) if hash_files else None,
            )
        )
    if incomplete:
        details = "; ".join(incomplete[:5])
        raise FileNotFoundError(f"incomplete OxHyperMinerals tiles detected: {details}")
    if not records:
        raise FileNotFoundError("no complete OxHyperMinerals tiles matched the split CSVs")
    return records


def group_split_overlaps(records: Iterable[OxHyperRecord]) -> dict[str, tuple[str, ...]]:
    assignments: dict[str, set[str]] = {}
    for record in records:
        assignments.setdefault(record.source_group, set()).add(record.split)
    return {
        group: tuple(sorted(splits))
        for group, splits in assignments.items()
        if len(splits) > 1
    }


def assert_no_source_group_leakage(records: Iterable[OxHyperRecord]) -> None:
    overlaps = group_split_overlaps(records)
    if overlaps:
        preview = ", ".join(
            f"{group}={list(splits)}" for group, splits in list(sorted(overlaps.items()))[:5]
        )
        raise ValueError(f"source-group leakage across dataset splits: {preview}")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_scene_safe_pilot(
    records: Iterable[OxHyperRecord],
    *,
    groups_per_split: dict[str, int] | None = None,
    tiles_per_group: int = 2,
    seed: int = 20260822,
) -> tuple[list[OxHyperRecord], dict[str, tuple[str, ...]]]:
    """Select whole source groups and remove groups crossing published splits."""

    if tiles_per_group < 1:
        raise ValueError("tiles_per_group must be positive")
    groups_per_split = groups_per_split or {"train": 4, "validation": 2, "test": 2}
    resolved_limits = {_normalise_split(k): int(v) for k, v in groups_per_split.items()}
    if set(resolved_limits) != {"train", "validation", "test"}:
        raise ValueError("groups_per_split must define train, validation, and test")
    if any(value < 1 for value in resolved_limits.values()):
        raise ValueError("each split must request at least one source group")

    records = list(records)
    overlaps = group_split_overlaps(records)
    safe = [record for record in records if record.source_group not in overlaps]
    selected: list[OxHyperRecord] = []
    for split in ("train", "validation", "test"):
        split_records = [record for record in safe if record.split == split]
        by_group: dict[str, list[OxHyperRecord]] = {}
        for record in split_records:
            by_group.setdefault(record.source_group, []).append(record)
        ordered_groups = sorted(by_group, key=lambda group: _stable_key(seed, group))
        chosen_groups = ordered_groups[: resolved_limits[split]]
        if not chosen_groups:
            raise ValueError(
                f"no leakage-safe source groups remain for {split}; inspect the published split"
            )
        for group in chosen_groups:
            ordered_tiles = sorted(
                by_group[group], key=lambda record: _stable_key(seed, record.tile_id)
            )
            selected.extend(ordered_tiles[:tiles_per_group])
    assert_no_source_group_leakage(selected)
    return sorted(selected, key=lambda record: (record.split, record.tile_id)), overlaps


def build_pilot_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    dataset_repository: str = OXHYPER_MINI_REPOSITORY,
    dataset_revision: str = OXHYPER_MINI_REVISION,
    groups_per_split: dict[str, int] | None = None,
    tiles_per_group: int = 2,
    seed: int = 20260822,
    hash_files: bool = False,
) -> dict[str, Any]:
    records = discover_oxhyper_records(dataset_root, hash_files=hash_files)
    selected, excluded = select_scene_safe_pilot(
        records,
        groups_per_split=groups_per_split,
        tiles_per_group=tiles_per_group,
        seed=seed,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "OxHyperMinerals",
            "repository": dataset_repository,
            "revision": dataset_revision,
            "sensor": "NASA EMIT",
            "spectral_bands": 285,
            "class_names": list(OXHYPER_CLASS_NAMES),
            "label_provenance": "EMIT L2B-derived experimental pseudo-labels",
            "license": "CC-BY-4.0 according to the dataset card",
        },
        "selection": {
            "seed": seed,
            "tiles_per_group": tiles_per_group,
            "groups_per_split": groups_per_split
            or {"train": 4, "validation": 2, "test": 2},
            "excluded_leaking_groups": {
                group: list(splits) for group, splits in sorted(excluded.items())
            },
        },
        "evidence_grade_on_success": "public-benchmark-observed",
        "records": [record.to_dict() for record in selected],
        "limitations": [
            "Labels are experimental EMIT L2B-derived pseudo-labels, not field truth.",
            "A small pilot estimates pipeline behavior and does not represent the full dataset.",
            "Surface mineral indicators do not establish a subsurface economic deposit.",
        ],
    }
    write_json(output_path, manifest)
    return manifest


def load_pilot_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported pilot manifest schema")
    records = [OxHyperRecord.from_dict(value) for value in manifest.get("records", [])]
    if not records:
        raise ValueError("pilot manifest contains no records")
    assert_no_source_group_leakage(records)
    present = {record.split for record in records}
    if present != {"train", "validation", "test"}:
        raise ValueError(f"pilot manifest must contain all three splits; found {sorted(present)}")
    return manifest


def parse_envi_wavelengths(header_path: str | Path) -> np.ndarray:
    text = Path(header_path).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"ENVI header has no wavelength field: {header_path}")
    values = [part.strip() for part in match.group(1).replace("\n", " ").split(",")]
    wavelengths = np.asarray([float(value) for value in values if value], dtype=np.float64)
    units_match = re.search(r"wavelength\s+units\s*=\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    units = units_match.group(1).strip().strip("{}").lower() if units_match else "nanometers"
    if "micro" in units or units in {"um", "µm"}:
        wavelengths *= 1000.0
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)):
        raise ValueError(f"ENVI header contains invalid wavelengths: {header_path}")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"ENVI wavelengths are not strictly increasing: {header_path}")
    return wavelengths


def verify_record_integrity(
    dataset_root: str | Path,
    record: OxHyperRecord,
    *,
    require_hashes: bool,
) -> None:
    root = Path(dataset_root).expanduser().resolve()
    cube_path = root / record.cube_path
    label_path = root / record.label_path
    for path, expected_size in (
        (cube_path, record.cube_size_bytes),
        (label_path, record.label_size_bytes),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"manifest size mismatch for {path}")
    if require_hashes and (record.cube_sha256 is None or record.label_sha256 is None):
        raise ValueError(
            "public-benchmark-observed runs require SHA-256 hashes; rebuild the pilot "
            "manifest with --hash-files"
        )
    if record.cube_sha256 is not None and _sha256_file(cube_path) != record.cube_sha256:
        raise ValueError(f"manifest SHA-256 mismatch for {cube_path}")
    if record.label_sha256 is not None and _sha256_file(label_path) != record.label_sha256:
        raise ValueError(f"manifest SHA-256 mismatch for {label_path}")


def load_oxhyper_cube(
    dataset_root: str | Path,
    record: OxHyperRecord,
) -> HyperspectralCube:
    """Load one real OxHyperMinerals tile using optional geospatial dependencies."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised in an environment with geo extras
        raise RuntimeError("rasterio is required; install spectrashift[geo]") from exc

    root = Path(dataset_root).expanduser().resolve()
    cube_path = root / record.cube_path
    header_path = root / record.header_path
    label_path = root / record.label_path
    wavelengths = parse_envi_wavelengths(header_path)

    with rasterio.open(cube_path) as source:
        reflectance_chw = source.read(out_dtype="float32")
        cube_mask = np.all(source.read_masks() > 0, axis=0)
        transform = tuple(source.transform)
        crs = source.crs.to_string() if source.crs else None
        nodata = source.nodata
    if reflectance_chw.shape[0] != wavelengths.size:
        raise ValueError(
            f"{record.tile_id} has {reflectance_chw.shape[0]} bands but "
            f"{wavelengths.size} wavelengths"
        )
    reflectance = np.moveaxis(reflectance_chw, 0, -1)
    finite = np.all(np.isfinite(reflectance), axis=-1)
    valid_mask = cube_mask & finite
    if nodata is not None and np.isfinite(nodata):
        valid_mask &= np.all(reflectance != nodata, axis=-1)
    reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=0.0, neginf=0.0)

    with rasterio.open(label_path) as source:
        labels_chw = source.read()
        label_mask = np.all(source.read_masks() > 0, axis=0)
    if labels_chw.shape[0] != len(OXHYPER_CLASS_NAMES):
        raise ValueError(
            f"{record.tile_id} label raster must contain three bands in "
            "goethite, hematite, kaolinite order"
        )
    labels = np.moveaxis((labels_chw > 0).astype(np.uint8), 0, -1)
    valid_mask &= label_mask
    if labels.shape[:2] != reflectance.shape[:2]:
        raise ValueError(f"{record.tile_id} cube and label spatial dimensions differ")

    return HyperspectralCube(
        scene_id=record.tile_id,
        reflectance=reflectance,
        wavelengths_nm=wavelengths,
        valid_mask=valid_mask,
        labels=labels,
        class_names=OXHYPER_CLASS_NAMES,
        metadata={
            "dataset": "OxHyperMinerals",
            "split": record.split,
            "source_group": record.source_group,
            "transform": transform,
            "crs": crs,
            "label_provenance": "EMIT L2B-derived experimental pseudo-labels",
        },
    )


def download_oxhyper_mini(
    output: str | Path, *, revision: str = OXHYPER_MINI_REVISION
) -> Path:
    """Download the authors' public development subset without embedding credentials."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional network dependency
        raise RuntimeError("huggingface_hub is required; install spectrashift[geo]") from exc
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=OXHYPER_MINI_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        local_dir=output,
    )
    return output
