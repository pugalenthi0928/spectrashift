from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spectrashift.evidence import synthetic_run_manifest, write_json
from spectrashift.metrics import (
    brier_score,
    expected_calibration_error,
    multilabel_report,
    selective_risk_curve,
)
from spectrashift.models.sam import SpectralAngleMapper
from spectrashift.synthetic import make_synthetic_cube
from spectrashift.targeting import extract_target_cards


def run_demo(output: Path, *, seed: int) -> dict[str, object]:
    cube, references = make_synthetic_cube(seed=seed)
    model = SpectralAngleMapper(references, scale=24.0)
    scores = model.predict(cube)
    if cube.labels is None:
        raise RuntimeError("synthetic fixture unexpectedly has no labels")

    valid_labels = cube.labels[cube.valid_mask]
    valid_scores = scores[cube.valid_mask]
    report = multilabel_report(valid_labels, valid_scores, threshold=0.5)
    report["expected_calibration_error"] = expected_calibration_error(valid_labels, valid_scores)
    report["brier_score"] = brier_score(valid_labels, valid_scores)
    report["selective_risk_curve"] = selective_risk_curve(valid_labels, valid_scores)
    for class_name, row in zip(cube.class_names, report["per_class"], strict=True):
        row["class_name"] = class_name

    cards = extract_target_cards(
        scores,
        cube.class_names,
        valid_mask=cube.valid_mask,
        spectral_agreement=scores,
        threshold=0.5,
        min_pixels=8,
        evidence_grade="synthetic-demonstration",
    )
    manifest = synthetic_run_manifest(
        seed=seed,
        scene_id=cube.scene_id,
        bands=cube.bands,
        class_names=cube.class_names,
        model_descriptor=asdict(model.descriptor),
    )

    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = write_json(output / "run_manifest.json", manifest.to_dict())
    metrics_hash = write_json(output / "metrics.json", report)
    cards_hash = write_json(output / "target_cards.json", [card.to_dict() for card in cards])
    np.savez_compressed(
        output / "synthetic_outputs.npz",
        wavelengths_nm=cube.wavelengths_nm,
        valid_mask=cube.valid_mask,
        labels=cube.labels,
        scores=scores,
    )
    summary = {
        "run_id": manifest.run_id,
        "evidence_grade": manifest.evidence_grade,
        "macro_f1": report["macro_f1"],
        "mean_iou": report["mean_iou"],
        "targets": len(cards),
        "artifact_hashes": {
            "run_manifest.json": manifest_hash,
            "metrics.json": metrics_hash,
            "target_cards.json": cards_hash,
        },
        "warning": "Synthetic software demonstration only; not geological evidence.",
    }
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrashift")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic synthetic integration demo")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=20260822)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.output, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

