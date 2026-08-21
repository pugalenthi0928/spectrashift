from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.data.oxhyper import (
    OXHYPER_CLASS_NAMES,
    OxHyperRecord,
    load_oxhyper_cube,
    load_pilot_manifest,
    verify_record_integrity,
)
from spectrashift.evidence import RunManifest, sha256_value, write_json
from spectrashift.metrics import (
    brier_score,
    expected_calibration_error,
    multilabel_report,
    optimize_f1_thresholds,
    selective_risk_curve,
)
from spectrashift.models.pca_logistic import PcaLogisticBaseline, PcaLogisticConfig
from spectrashift.models.sam import SpectralAngleMapper
from spectrashift.targeting import extract_target_cards

CubeLoader = Callable[[str | Path, OxHyperRecord], HyperspectralCube]


def _process_peak_rss_mb() -> float | None:
    """Return process peak resident memory using the platform's ru_maxrss units."""

    try:
        import resource
    except ImportError:  # pragma: no cover - unavailable on Windows
        return None
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def sample_labeled_pixels(
    cube: HyperspectralCube,
    *,
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a mix of rare positive pixels and scene-wide background pixels."""

    if cube.labels is None:
        raise ValueError("training cube has no labels")
    if max_pixels < len(cube.class_names) * 2:
        raise ValueError("max_pixels is too small for class-aware sampling")
    rng = np.random.default_rng(seed)
    valid_indices = np.flatnonzero(cube.valid_mask.reshape(-1))
    if valid_indices.size == 0:
        raise ValueError(f"{cube.scene_id} has no valid pixels")
    flat_labels = cube.labels.reshape(-1, cube.labels.shape[-1])
    selected: set[int] = set()
    positive_quota = max(1, max_pixels // (2 * flat_labels.shape[1]))
    for class_index in range(flat_labels.shape[1]):
        positives = valid_indices[flat_labels[valid_indices, class_index] == 1]
        if positives.size:
            chosen = rng.choice(positives, size=min(positive_quota, positives.size), replace=False)
            selected.update(int(value) for value in chosen)
    remaining = max_pixels - len(selected)
    if remaining > 0:
        candidates = np.asarray(
            [value for value in valid_indices if int(value) not in selected], dtype=np.int64
        )
        if candidates.size:
            chosen = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
            selected.update(int(value) for value in chosen)
    indices = np.asarray(sorted(selected), dtype=np.int64)
    flat_spectra = cube.reflectance.reshape(-1, cube.bands)
    return flat_spectra[indices].astype(np.float32), flat_labels[indices]


def _fit_model(
    model_name: str,
    spectra: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> Any:
    if model_name == "pca-logistic":
        return PcaLogisticBaseline(PcaLogisticConfig(seed=seed)).fit(spectra, labels)
    if model_name == "prototype-sam":
        references = []
        for class_index in range(labels.shape[1]):
            positives = spectra[labels[:, class_index] == 1]
            if positives.size == 0:
                raise ValueError(f"class {class_index} has no positive training spectra")
            references.append(np.median(positives, axis=0))
        return SpectralAngleMapper(np.asarray(references), scale=24.0)
    raise ValueError(f"unsupported benchmark model: {model_name}")


def _records(manifest: dict[str, Any], split: str) -> list[OxHyperRecord]:
    return [
        OxHyperRecord.from_dict(value)
        for value in manifest["records"]
        if value["split"] == split
    ]


def _load_predictions(
    records: Iterable[OxHyperRecord],
    *,
    dataset_root: str | Path,
    loader: CubeLoader,
    model: Any,
) -> tuple[list[tuple[OxHyperRecord, HyperspectralCube, np.ndarray]], list[float]]:
    outputs: list[tuple[OxHyperRecord, HyperspectralCube, np.ndarray]] = []
    latency_ms: list[float] = []
    for record in records:
        cube = loader(dataset_root, record)
        if cube.labels is None:
            raise ValueError(f"{record.tile_id} has no labels")
        start = time.perf_counter()
        probabilities = np.asarray(model.predict(cube), dtype=np.float32)
        latency_ms.append((time.perf_counter() - start) * 1000.0)
        if probabilities.shape != cube.labels.shape:
            raise ValueError(f"{record.tile_id} prediction shape does not match labels")
        if np.any(~np.isfinite(probabilities)):
            raise ValueError(f"{record.tile_id} produced non-finite probabilities")
        outputs.append((record, cube, probabilities))
    return outputs, latency_ms


def _stack_valid(
    outputs: Iterable[tuple[OxHyperRecord, HyperspectralCube, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for _, cube, scores in outputs:
        if cube.labels is None:
            raise ValueError("evaluation cube has no labels")
        labels.append(cube.labels[cube.valid_mask])
        probabilities.append(scores[cube.valid_mask])
    if not labels:
        raise ValueError("evaluation split contains no records")
    return np.concatenate(labels), np.concatenate(probabilities)


def _label_profile(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2 or labels.shape[1] != len(OXHYPER_CLASS_NAMES):
        raise ValueError("label profile expects (pixels, three classes)")
    supports = labels.sum(axis=0, dtype=np.int64)
    total = int(labels.shape[0])
    return {
        "pixels": total,
        "class_names": list(OXHYPER_CLASS_NAMES),
        "positive_pixels": supports.astype(int).tolist(),
        "negative_pixels": (total - supports).astype(int).tolist(),
        "prevalence": (supports / max(total, 1)).astype(float).tolist(),
    }


def _failure_cases(
    cube: HyperspectralCube,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    *,
    limit_per_kind: int = 5,
) -> list[dict[str, Any]]:
    if cube.labels is None:
        return []
    failures: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(cube.class_names):
        truth = cube.labels[..., class_index].astype(bool)
        predicted = probabilities[..., class_index] >= thresholds[class_index]
        for kind, mask, order_sign in (
            ("false_positive", predicted & ~truth & cube.valid_mask, -1.0),
            ("false_negative", ~predicted & truth & cube.valid_mask, 1.0),
        ):
            rows, cols = np.where(mask)
            if rows.size == 0:
                continue
            scores = probabilities[rows, cols, class_index]
            order = np.argsort(order_sign * scores, kind="stable")[:limit_per_kind]
            for index in order:
                row = int(rows[index])
                col = int(cols[index])
                failures.append(
                    {
                        "scene_id": cube.scene_id,
                        "source_group": cube.metadata.get("source_group"),
                        "class_name": class_name,
                        "kind": kind,
                        "row": row,
                        "col": col,
                        "probability": float(probabilities[row, col, class_index]),
                        "threshold": float(thresholds[class_index]),
                        "spectrum": cube.reflectance[row, col].astype(float).tolist(),
                    }
                )
    return failures


def run_oxhyper_benchmark(
    manifest_path: str | Path,
    dataset_root: str | Path,
    output: str | Path,
    *,
    model_name: str = "pca-logistic",
    max_train_pixels_per_tile: int = 20000,
    seed: int = 20260822,
    loader: CubeLoader = load_oxhyper_cube,
) -> dict[str, Any]:
    """Run a locked train/validation/test pilot and preserve its evidence bundle."""

    manifest_path = Path(manifest_path)
    pilot = load_pilot_manifest(manifest_path)
    records = [OxHyperRecord.from_dict(value) for value in pilot.get("records", [])]
    if {record.split for record in records} != {"train", "validation", "test"}:
        raise ValueError("manifest must contain train, validation, and test records")
    evidence_grade = pilot.get("evidence_grade_on_success", "mechanism-tested")
    if evidence_grade == "public-benchmark-observed":
        for record in records:
            verify_record_integrity(dataset_root, record, require_hashes=True)

    train_spectra: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    train_start = time.perf_counter()
    for index, record in enumerate(_records(pilot, "train")):
        cube = loader(dataset_root, record)
        spectra, labels = sample_labeled_pixels(
            cube,
            max_pixels=max_train_pixels_per_tile,
            seed=seed + index,
        )
        train_spectra.append(spectra)
        train_labels.append(labels)
    spectra = np.concatenate(train_spectra)
    labels = np.concatenate(train_labels)
    training_profile = _label_profile(labels)
    model = _fit_model(model_name, spectra, labels, seed=seed)
    train_seconds = time.perf_counter() - train_start

    validation_outputs, validation_latency = _load_predictions(
        _records(pilot, "validation"), dataset_root=dataset_root, loader=loader, model=model
    )
    validation_labels, validation_probabilities = _stack_valid(validation_outputs)
    validation_profile = _label_profile(validation_labels)
    threshold_candidates = np.linspace(0.05, 0.95, 19)
    thresholds = optimize_f1_thresholds(
        validation_labels,
        validation_probabilities,
        candidates=threshold_candidates,
    )
    del validation_outputs, validation_labels, validation_probabilities

    test_outputs, test_latency = _load_predictions(
        _records(pilot, "test"), dataset_root=dataset_root, loader=loader, model=model
    )
    test_labels, test_probabilities = _stack_valid(test_outputs)
    test_profile = _label_profile(test_labels)
    report = multilabel_report(test_labels, test_probabilities, threshold=thresholds)
    report["expected_calibration_error"] = expected_calibration_error(
        test_labels, test_probabilities
    )
    report["brier_score"] = brier_score(test_labels, test_probabilities)
    report["selective_risk_curve"] = selective_risk_curve(test_labels, test_probabilities)
    report["data_profile"] = {
        "training_sample": {
            **training_profile,
            "note": "Class-aware sampled pixels; prevalence is intentionally not natural.",
        },
        "validation": validation_profile,
        "test": test_profile,
        "tiles": {
            split: len(_records(pilot, split))
            for split in ("train", "validation", "test")
        },
    }
    report["threshold_selection"] = {
        "split": "validation",
        "objective": "per-class F1",
        "candidates": threshold_candidates.tolist(),
        "selected": thresholds.tolist(),
        "at_lower_search_bound": np.isclose(
            thresholds, threshold_candidates[0]
        ).tolist(),
        "at_upper_search_bound": np.isclose(
            thresholds, threshold_candidates[-1]
        ).tolist(),
    }
    report["efficiency"] = {
        "training_seconds": train_seconds,
        "validation_tile_latency_ms": validation_latency,
        "test_tile_latency_ms": test_latency,
        "mean_test_tile_latency_ms": float(np.mean(test_latency)),
        "sampled_training_pixels": int(spectra.shape[0]),
        "process_peak_rss_mb": _process_peak_rss_mb(),
    }
    for class_name, row in zip(OXHYPER_CLASS_NAMES, report["per_class"], strict=True):
        row["class_name"] = class_name

    output = Path(output)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    all_cards: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for _, cube, probabilities in test_outputs:
        np.savez_compressed(
            predictions_dir / f"{cube.scene_id}.npz",
            wavelengths_nm=cube.wavelengths_nm,
            valid_mask=cube.valid_mask,
            labels=cube.labels,
            probabilities=probabilities,
            thresholds=thresholds,
        )
        cards = extract_target_cards(
            probabilities,
            cube.class_names,
            valid_mask=cube.valid_mask,
            threshold=thresholds,
            min_pixels=16,
            evidence_grade=evidence_grade,
            scene_id=cube.scene_id,
            max_targets=25,
        )
        all_cards.extend(card.to_dict() for card in cards)
        all_failures.extend(_failure_cases(cube, probabilities, thresholds))

    stable_run = {
        "pilot_definition_sha256": sha256_value(
            {key: value for key, value in pilot.items() if key != "created_at_utc"}
        ),
        "model_name": model_name,
        "seed": seed,
        "max_train_pixels_per_tile": max_train_pixels_per_tile,
    }
    descriptor = asdict(model.descriptor)
    model_config = model.configuration() if hasattr(model, "configuration") else {}
    run_manifest = RunManifest(
        run_id=f"oxhyper-{sha256_value(stable_run)[:12]}",
        created_at_utc=datetime.now(UTC).isoformat(),
        evidence_grade=evidence_grade,
        data={
            "pilot_manifest": str(manifest_path),
            "pilot_manifest_sha256": sha256_value(pilot),
            "pilot_definition_sha256": stable_run["pilot_definition_sha256"],
            "dataset": pilot.get("dataset", {}),
            "records": [record.to_dict() for record in records],
        },
        model={**descriptor, "configuration": model_config},
        evaluation={
            "threshold_source": "validation_only",
            "thresholds": thresholds.tolist(),
            "seed": seed,
            "max_train_pixels_per_tile": max_train_pixels_per_tile,
        },
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": _installed_version("rasterio"),
            "scikit_learn": _installed_version("scikit-learn"),
            "platform": platform.platform(),
            "source_commit": os.environ.get("SPECTRASHIFT_SOURCE_COMMIT"),
        },
        limitations=tuple(pilot.get("limitations", [])),
    )
    output.mkdir(parents=True, exist_ok=True)
    hashes = {
        "run_manifest.json": write_json(output / "run_manifest.json", run_manifest.to_dict()),
        "metrics.json": write_json(output / "metrics.json", report),
        "thresholds.json": write_json(
            output / "thresholds.json",
            dict(zip(OXHYPER_CLASS_NAMES, thresholds.tolist(), strict=True)),
        ),
        "target_cards.json": write_json(output / "target_cards.json", all_cards),
        "failure_cases.json": write_json(output / "failure_cases.json", all_failures),
    }
    summary = {
        "run_id": run_manifest.run_id,
        "evidence_grade": run_manifest.evidence_grade,
        "model": model_name,
        "test_tiles": len(test_outputs),
        "macro_f1": report["macro_f1"],
        "macro_average_precision": report["macro_average_precision"],
        "mean_iou": report["mean_iou"],
        "targets": len(all_cards),
        "failure_cases": len(all_failures),
        "threshold_boundary_classes": [
            class_name
            for class_name, threshold in zip(
                OXHYPER_CLASS_NAMES, thresholds, strict=True
            )
            if np.isclose(threshold, threshold_candidates[0])
            or np.isclose(threshold, threshold_candidates[-1])
        ],
        "artifact_hashes": hashes,
        "warning": "Experimental pseudo-label benchmark; not field or deposit validation.",
    }
    write_json(output / "summary.json", summary)
    return summary
