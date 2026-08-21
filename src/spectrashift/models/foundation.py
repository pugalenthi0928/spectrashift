from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FoundationModelReadiness:
    family: str
    torch_available: bool
    checkpoint_exists: bool
    source_checkout_exists: bool
    ready: bool
    blockers: tuple[str, ...]


def inspect_foundation_model(
    family: str,
    *,
    checkpoint: str | Path | None,
    source_checkout: str | Path | None,
) -> FoundationModelReadiness:
    """Inspect local prerequisites without downloading code, data, or weights."""

    torch_available = importlib.util.find_spec("torch") is not None
    checkpoint_exists = checkpoint is not None and Path(checkpoint).is_file()
    source_exists = source_checkout is not None and Path(source_checkout).is_dir()
    blockers: list[str] = []
    if not torch_available:
        blockers.append("PyTorch is not installed")
    if not checkpoint_exists:
        blockers.append("a local checkpoint is required")
    if not source_exists:
        blockers.append("an audited local source checkout is required")
    return FoundationModelReadiness(
        family=family,
        torch_available=torch_available,
        checkpoint_exists=checkpoint_exists,
        source_checkout_exists=source_exists,
        ready=not blockers,
        blockers=tuple(blockers),
    )

