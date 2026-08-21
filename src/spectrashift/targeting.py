from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TargetCard:
    target_id: str
    scene_id: str
    class_name: str
    rank: int
    score: float
    pixel_count: int
    bounding_box_row_col: tuple[int, int, int, int]
    mean_probability: float
    mean_uncertainty: float
    mean_spectral_agreement: float | None
    spatial_coherence: float
    evidence_grade: str
    limitation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    visited = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
            result.append(component)
    return result


def extract_target_cards(
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    *,
    valid_mask: np.ndarray,
    uncertainty: np.ndarray | None = None,
    spectral_agreement: np.ndarray | None = None,
    threshold: float | np.ndarray = 0.5,
    min_pixels: int = 8,
    evidence_grade: str = "mechanism-tested",
    scene_id: str = "unknown-scene",
    max_targets: int | None = None,
) -> list[TargetCard]:
    """Convert thresholded class maps into explicit, ranked connected targets."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (height, width, classes)")
    if probabilities.shape[:2] != valid_mask.shape:
        raise ValueError("valid_mask must match probability spatial dimensions")
    if probabilities.shape[2] != len(class_names):
        raise ValueError("class_names must match probability class dimension")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    if max_targets is not None and max_targets < 1:
        raise ValueError("max_targets must be positive when provided")

    thresholds = np.asarray(threshold, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds, len(class_names))
    if thresholds.shape != (len(class_names),):
        raise ValueError("threshold must be scalar or contain one value per class")
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("thresholds must lie in [0, 1]")

    if uncertainty is None:
        uncertainty = 4.0 * probabilities * (1.0 - probabilities)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if uncertainty.shape != probabilities.shape:
        raise ValueError("uncertainty must match probabilities")
    if spectral_agreement is not None:
        spectral_agreement = np.asarray(spectral_agreement, dtype=np.float64)
        if spectral_agreement.shape != probabilities.shape:
            raise ValueError("spectral_agreement must match probabilities")

    raw_cards: list[dict[str, object]] = []
    scene_area = probabilities.shape[0] * probabilities.shape[1]
    for class_index, class_name in enumerate(class_names):
        class_mask = (probabilities[..., class_index] >= thresholds[class_index]) & valid_mask
        for component in _components(class_mask):
            if len(component) < min_pixels:
                continue
            rows = np.array([point[0] for point in component], dtype=int)
            cols = np.array([point[1] for point in component], dtype=int)
            class_probabilities = probabilities[rows, cols, class_index]
            class_uncertainty = uncertainty[rows, cols, class_index]
            area_score = min(1.0, len(component) / max(min_pixels * 8, scene_area * 0.02))
            box_height = int(rows.max() - rows.min() + 1)
            box_width = int(cols.max() - cols.min() + 1)
            coherence = len(component) / (box_height * box_width)
            if spectral_agreement is None:
                mean_spectral = None
                spectral_term = float(np.mean(class_probabilities))
            else:
                mean_spectral = float(np.mean(spectral_agreement[rows, cols, class_index]))
                spectral_term = mean_spectral
            mean_probability = float(np.mean(class_probabilities))
            mean_uncertainty = float(np.mean(class_uncertainty))
            score = float(
                np.clip(
                    0.45 * mean_probability
                    + 0.20 * spectral_term
                    + 0.20 * (1.0 - mean_uncertainty)
                    + 0.10 * coherence
                    + 0.05 * area_score,
                    0.0,
                    1.0,
                )
            )
            raw_cards.append(
                {
                    "class_name": class_name,
                    "score": score,
                    "pixel_count": len(component),
                    "bounding_box_row_col": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                    "mean_probability": mean_probability,
                    "mean_uncertainty": mean_uncertainty,
                    "mean_spectral_agreement": mean_spectral,
                    "spatial_coherence": float(coherence),
                }
            )

    raw_cards.sort(key=lambda card: (-float(card["score"]), str(card["class_name"])))
    if max_targets is not None:
        raw_cards = raw_cards[:max_targets]
    return [
        TargetCard(
            target_id=f"{scene_id}:T{rank:03d}",
            scene_id=scene_id,
            rank=rank,
            evidence_grade=evidence_grade,
            limitation=(
                "Screening evidence only; a surface spectral indicator is not proof of a "
                "subsurface economic deposit."
            ),
            **card,
        )
        for rank, card in enumerate(raw_cards, start=1)
    ]
