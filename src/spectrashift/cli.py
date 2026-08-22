from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spectrashift.benchmark import run_oxhyper_benchmark
from spectrashift.data.oxhyper import (
    OXHYPER_MINI_REVISION,
    build_pilot_manifest,
    download_oxhyper_mini,
)
from spectrashift.evidence import synthetic_run_manifest, write_json
from spectrashift.metrics import (
    brier_score,
    expected_calibration_error,
    multilabel_report,
    selective_risk_curve,
)
from spectrashift.models.dofa import (
    DOFA_CHECKPOINT_SHA256,
    DOFA_SOURCE_REVISION,
    DofaConfig,
)
from spectrashift.models.sam import SpectralAngleMapper
from spectrashift.retrieval import query_evidence
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
        scene_id=cube.scene_id,
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

    download = subparsers.add_parser(
        "download-oxhyper-mini", help="download the public OxHyperMinerals development subset"
    )
    download.add_argument("--output", type=Path, default=Path("data/external/OxHyperMinerals_MINI"))
    download.add_argument("--revision", default=OXHYPER_MINI_REVISION)
    download.add_argument("--train-groups", type=int, default=4)
    download.add_argument("--validation-groups", type=int, default=2)
    download.add_argument("--test-groups", type=int, default=2)
    download.add_argument("--tiles-per-group", type=int, default=2)
    download.add_argument("--seed", type=int, default=20260822)

    index = subparsers.add_parser(
        "index-oxhyper", help="build a leakage-checked OxHyperMinerals pilot manifest"
    )
    index.add_argument("--dataset-root", type=Path, required=True)
    index.add_argument("--output", type=Path, default=Path("data/pilots/oxhyper-mini.json"))
    index.add_argument("--train-groups", type=int, default=4)
    index.add_argument("--validation-groups", type=int, default=2)
    index.add_argument("--test-groups", type=int, default=2)
    index.add_argument("--tiles-per-group", type=int, default=2)
    index.add_argument("--seed", type=int, default=20260822)
    index.add_argument("--hash-files", action="store_true")
    index.add_argument("--dataset-revision", default=OXHYPER_MINI_REVISION)

    benchmark = subparsers.add_parser(
        "benchmark-oxhyper", help="run a locked real-data OxHyperMinerals benchmark"
    )
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--dataset-root", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/oxhyper-mini"))
    benchmark.add_argument(
        "--model",
        choices=("prototype-sam", "pca-logistic", "dofa-frozen"),
        default="pca-logistic",
    )
    benchmark.add_argument("--max-train-pixels-per-tile", type=int, default=20000)
    benchmark.add_argument("--min-validation-positive-pixels", type=int, default=1)
    benchmark.add_argument("--seed", type=int, default=20260822)
    benchmark.add_argument("--dofa-source", type=Path)
    benchmark.add_argument("--dofa-checkpoint", type=Path)
    benchmark.add_argument("--dofa-source-revision", default=DOFA_SOURCE_REVISION)
    benchmark.add_argument("--dofa-checkpoint-sha256", default=DOFA_CHECKPOINT_SHA256)
    benchmark.add_argument("--dofa-input-size", type=int, default=112)
    benchmark.add_argument("--dofa-feature-layer", type=int, default=11)
    benchmark.add_argument("--dofa-device", default="cpu")
    benchmark.add_argument("--dofa-positive-patch-fraction", type=float, default=0.05)
    benchmark.add_argument("--dofa-minimum-valid-patch-fraction", type=float, default=0.8)

    query = subparsers.add_parser(
        "query-evidence",
        help="retrieve cited experiment evidence without changing model outputs",
    )
    query.add_argument("--root", type=Path, default=Path("results/oxhyper-mini"))
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.output, seed=args.seed), indent=2, sort_keys=True))
    elif args.command == "download-oxhyper-mini":
        path = download_oxhyper_mini(
            args.output,
            revision=args.revision,
            groups_per_split={
                "train": args.train_groups,
                "validation": args.validation_groups,
                "test": args.test_groups,
            },
            tiles_per_group=args.tiles_per_group,
            seed=args.seed,
        )
        print(json.dumps({"dataset_root": str(path)}, indent=2, sort_keys=True))
    elif args.command == "index-oxhyper":
        manifest = build_pilot_manifest(
            args.dataset_root,
            args.output,
            dataset_revision=args.dataset_revision,
            groups_per_split={
                "train": args.train_groups,
                "validation": args.validation_groups,
                "test": args.test_groups,
            },
            tiles_per_group=args.tiles_per_group,
            seed=args.seed,
            hash_files=args.hash_files,
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.output),
                    "records": len(manifest["records"]),
                    "excluded_leaking_groups": len(
                        manifest["selection"]["excluded_leaking_groups"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "benchmark-oxhyper":
        dofa_config = None
        if args.model == "dofa-frozen":
            if args.dofa_source is None or args.dofa_checkpoint is None:
                raise SystemExit("dofa-frozen requires --dofa-source and --dofa-checkpoint")
            dofa_config = DofaConfig(
                source_checkout=args.dofa_source,
                checkpoint=args.dofa_checkpoint,
                source_revision=args.dofa_source_revision,
                checkpoint_sha256=args.dofa_checkpoint_sha256,
                input_size=args.dofa_input_size,
                feature_layer=args.dofa_feature_layer,
                device=args.dofa_device,
                seed=args.seed,
            )
        summary = run_oxhyper_benchmark(
            args.manifest,
            args.dataset_root,
            args.output,
            model_name=args.model,
            max_train_pixels_per_tile=args.max_train_pixels_per_tile,
            min_validation_positive_pixels=args.min_validation_positive_pixels,
            seed=args.seed,
            dofa_config=dofa_config,
            dofa_positive_patch_fraction=args.dofa_positive_patch_fraction,
            dofa_minimum_valid_patch_fraction=args.dofa_minimum_valid_patch_fraction,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "query-evidence":
        print(
            json.dumps(
                query_evidence(args.root, args.question, top_k=args.top_k),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
