from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: str | Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    evidence_grade: str
    data: dict[str, Any]
    model: dict[str, Any]
    evaluation: dict[str, Any]
    environment: dict[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def synthetic_run_manifest(
    *,
    seed: int,
    scene_id: str,
    bands: int,
    class_names: tuple[str, ...],
    model_descriptor: dict[str, Any],
) -> RunManifest:
    stable_spec = {
        "seed": seed,
        "scene_id": scene_id,
        "bands": bands,
        "class_names": class_names,
        "model": model_descriptor,
    }
    return RunManifest(
        run_id=f"synthetic-{sha256_value(stable_spec)[:12]}",
        created_at_utc=datetime.now(UTC).isoformat(),
        evidence_grade="synthetic-demonstration",
        data={
            "scene_id": scene_id,
            "bands": bands,
            "class_names": class_names,
            "seed": seed,
            "label_provenance": "deterministic synthetic software fixture",
        },
        model=model_descriptor,
        evaluation={
            "threshold": 0.5,
            "calibration": "not fitted; similarity transform only",
        },
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        limitations=(
            "Synthetic spectra are not laboratory, airborne, or satellite measurements.",
            "Scores are not calibrated probabilities.",
            "Outputs must not be interpreted as geological evidence.",
        ),
    )

