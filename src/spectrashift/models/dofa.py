from __future__ import annotations

import hashlib
import importlib
import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from spectrashift.contracts import HyperspectralCube
from spectrashift.models.base import ModelDescriptor

DOFA_REPOSITORY = "https://github.com/zhu-xlab/DOFA"
DOFA_SOURCE_REVISION = "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9"
DOFA_CHECKPOINT_REPOSITORY = "earthflow/DOFA"
DOFA_CHECKPOINT_REVISION = "7a5219e48d2f8848511b0fabea7920a8836bc480"
DOFA_CHECKPOINT_FILENAME = "DOFA_ViT_base_e100.pth"
DOFA_CHECKPOINT_SHA256 = "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resize_bilinear(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a small HWC grid without requiring a deep-learning runtime."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("values must have shape (height, width, channels)")
    if height < 1 or width < 1:
        raise ValueError("output dimensions must be positive")
    source_height, source_width, channels = values.shape
    if (source_height, source_width) == (height, width):
        return values.copy()

    source_x = np.arange(source_width, dtype=np.float64)
    target_x = np.linspace(0.0, max(source_width - 1, 0), width)
    horizontal = np.empty((source_height, width, channels), dtype=np.float32)
    for row in range(source_height):
        for channel in range(channels):
            horizontal[row, :, channel] = np.interp(target_x, source_x, values[row, :, channel])

    source_y = np.arange(source_height, dtype=np.float64)
    target_y = np.linspace(0.0, max(source_height - 1, 0), height)
    output = np.empty((height, width, channels), dtype=np.float32)
    for column in range(width):
        for channel in range(channels):
            output[:, column, channel] = np.interp(
                target_y, source_y, horizontal[:, column, channel]
            )
    return output


def pool_patch_labels(
    cube: HyperspectralCube,
    grid_shape: tuple[int, int],
    *,
    positive_fraction: float,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool pixel labels and validity into the foundation model's token grid."""

    if cube.labels is None:
        raise ValueError(f"{cube.scene_id} has no labels")
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_fraction must lie in (0, 1]")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1]")
    grid_height, grid_width = grid_shape
    if grid_height < 1 or grid_width < 1:
        raise ValueError("grid dimensions must be positive")

    pooled = np.zeros((grid_height, grid_width, cube.labels.shape[-1]), dtype=np.uint8)
    positive_share = np.zeros_like(pooled, dtype=np.float32)
    valid = np.zeros((grid_height, grid_width), dtype=bool)
    row_edges = np.linspace(0, cube.height, grid_height + 1, dtype=int)
    col_edges = np.linspace(0, cube.width, grid_width + 1, dtype=int)
    for row in range(grid_height):
        for column in range(grid_width):
            region = (
                slice(row_edges[row], row_edges[row + 1]),
                slice(col_edges[column], col_edges[column + 1]),
            )
            region_valid = cube.valid_mask[region]
            if region_valid.size == 0:
                continue
            valid[row, column] = float(region_valid.mean()) >= minimum_valid_fraction
            if not np.any(region_valid):
                continue
            shares = cube.labels[region][region_valid].mean(axis=0)
            positive_share[row, column] = shares
            pooled[row, column] = shares >= positive_fraction
    return pooled, valid, positive_share


class DenseFeatureEncoder(Protocol):
    descriptor: ModelDescriptor

    def fit_normalization(self, cubes: Iterable[HyperspectralCube]) -> None: ...

    def extract(self, cube: HyperspectralCube) -> np.ndarray: ...

    def configuration(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DofaConfig:
    source_checkout: Path
    checkpoint: Path
    source_revision: str = DOFA_SOURCE_REVISION
    checkpoint_sha256: str = DOFA_CHECKPOINT_SHA256
    input_size: int = 112
    feature_layer: int = 11
    device: str = "cpu"
    max_normalization_pixels_per_cube: int = 20000
    seed: int = 20260822


class DofaFeatureEncoder:
    """Pinned DOFA ViT-B token extractor for arbitrary hyperspectral channels."""

    def __init__(self, config: DofaConfig) -> None:
        if config.input_size % 16 != 0:
            raise ValueError("DOFA input_size must be divisible by its 16-pixel patch size")
        if not 0 <= config.feature_layer < 12:
            raise ValueError("feature_layer must select one of the 12 ViT-B blocks")
        self.config = config
        self.descriptor = ModelDescriptor(
            family="DOFA-ViT-B",
            revision=config.source_revision,
            adaptation="frozen-wavelength-aware-features+linear-probe",
            checkpoint=(
                f"{DOFA_CHECKPOINT_REPOSITORY}@{DOFA_CHECKPOINT_REVISION}/"
                f"{DOFA_CHECKPOINT_FILENAME}#{config.checkpoint_sha256}"
            ),
            evidence_grade="mechanism-tested",
        )
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._wavelengths_nm: np.ndarray | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._checkpoint_actual_sha256: str | None = None
        self._backbone_parameters: int | None = None
        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        source = self.config.source_checkout
        for required in (source / "dofa_v1.py", source / "wave_dynamic_layer.py"):
            if not required.is_file():
                raise ValueError(f"DOFA source file is missing: {required}")
        if not self.config.checkpoint.is_file():
            raise ValueError(f"DOFA checkpoint is missing: {self.config.checkpoint}")
        try:
            revision = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("DOFA source_checkout must be a Git checkout") from exc
        if revision != self.config.source_revision:
            raise ValueError(
                f"DOFA source revision {revision} does not match {self.config.source_revision}"
            )
        actual = sha256_file(self.config.checkpoint)
        if actual != self.config.checkpoint_sha256:
            raise ValueError(
                f"DOFA checkpoint SHA-256 {actual} does not match {self.config.checkpoint_sha256}"
            )
        self._checkpoint_actual_sha256 = actual

    def fit_normalization(self, cubes: Iterable[HyperspectralCube]) -> None:
        rng = np.random.default_rng(self.config.seed)
        samples: list[np.ndarray] = []
        wavelengths: np.ndarray | None = None
        for cube in cubes:
            if wavelengths is None:
                wavelengths = cube.wavelengths_nm.copy()
            elif not np.array_equal(wavelengths, cube.wavelengths_nm):
                raise ValueError("DOFA pilot cubes must share an identical wavelength grid")
            valid = cube.valid_spectra().astype(np.float32, copy=False)
            if valid.shape[0] > self.config.max_normalization_pixels_per_cube:
                indices = rng.choice(
                    valid.shape[0],
                    size=self.config.max_normalization_pixels_per_cube,
                    replace=False,
                )
                valid = valid[indices]
            samples.append(valid)
        if not samples or wavelengths is None:
            raise ValueError("at least one cube is required to fit DOFA normalization")
        stacked = np.concatenate(samples, axis=0)
        self._mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
        self._std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
        self._std = np.maximum(self._std, np.float32(1e-6))
        self._wavelengths_nm = wavelengths

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("PyTorch is required; install spectrashift[foundation]") from exc

        source = str(self.config.source_checkout.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        module = importlib.import_module("dofa_v1")
        module_path = Path(module.__file__).resolve() if module.__file__ else None
        if module_path is None or module_path.parent != Path(source):
            raise RuntimeError("imported DOFA module does not match the audited source checkout")
        model = module.vit_base_patch16(
            img_size=self.config.input_size,
            num_classes=0,
            global_pool=True,
        )
        try:
            payload = torch.load(
                self.config.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(self.config.checkpoint, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError("DOFA checkpoint does not contain a state dictionary")
        state = dict(state)
        if "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
            state["pos_embed"] = self._interpolate_position_embedding(
                state["pos_embed"], model.pos_embed.shape[1]
            )
        incompatible = model.load_state_dict(state, strict=False)
        required_prefixes = ("patch_embed.", "blocks.")
        missing_required = [
            key for key in incompatible.missing_keys if key.startswith(required_prefixes)
        ]
        if missing_required:
            raise ValueError(f"DOFA checkpoint is missing backbone keys: {missing_required[:5]}")
        model.eval().to(self.config.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._backbone_parameters = sum(parameter.numel() for parameter in model.parameters())
        self._model = model
        self._torch = torch

    def _interpolate_position_embedding(self, position: Any, target_tokens: int) -> Any:
        import torch
        from torch.nn import functional

        patch_tokens = position.shape[1] - 1
        source_side = math.isqrt(patch_tokens)
        target_side = math.isqrt(target_tokens - 1)
        if (
            source_side * source_side != patch_tokens
            or target_side * target_side != target_tokens - 1
        ):
            raise ValueError("DOFA position embeddings must form square token grids")
        class_token = position[:, :1]
        patch = position[:, 1:].reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
        patch = functional.interpolate(
            patch,
            size=(target_side, target_side),
            mode="bicubic",
            align_corners=False,
        )
        patch = patch.permute(0, 2, 3, 1).reshape(1, target_side * target_side, -1)
        return torch.cat((class_token, patch), dim=1)

    def extract(self, cube: HyperspectralCube) -> np.ndarray:
        if self._mean is None or self._std is None or self._wavelengths_nm is None:
            raise RuntimeError("fit_normalization must run before DOFA feature extraction")
        if not np.array_equal(cube.wavelengths_nm, self._wavelengths_nm):
            raise ValueError("cube wavelength grid differs from the fitted DOFA grid")
        self._load_model()
        torch = self._torch
        model = self._model
        if torch is None or model is None:  # pragma: no cover - defensive
            raise RuntimeError("DOFA runtime failed to initialize")

        normalized = (cube.reflectance.astype(np.float32) - self._mean) / self._std
        normalized[~cube.valid_mask] = 0.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.config.device, dtype=torch.float32)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.config.input_size, self.config.input_size),
            mode="bilinear",
            align_corners=False,
        )
        wavelengths_um = torch.as_tensor(
            cube.wavelengths_nm / 1000.0,
            device=self.config.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            tokens, _ = model.patch_embed(tensor, wavelengths_um)
            tokens = tokens + model.pos_embed[:, 1:]
            class_token = model.cls_token + model.pos_embed[:, :1]
            tokens = torch.cat((class_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
            for index, block in enumerate(model.blocks):
                tokens = block(tokens)
                if index == self.config.feature_layer:
                    break
            patch_tokens = model.fc_norm(tokens[:, 1:])
        side = math.isqrt(patch_tokens.shape[1])
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError("DOFA emitted a non-square token grid")
        return (
            patch_tokens[0]
            .reshape(side, side, patch_tokens.shape[-1])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def configuration(self) -> dict[str, Any]:
        serialized_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
            if key not in {"source_checkout", "checkpoint"}
        }
        return {
            **serialized_config,
            "repository": DOFA_REPOSITORY,
            "checkpoint_repository": DOFA_CHECKPOINT_REPOSITORY,
            "checkpoint_revision": DOFA_CHECKPOINT_REVISION,
            "checkpoint_actual_sha256": self._checkpoint_actual_sha256,
            "backbone_parameters": self._backbone_parameters,
            "normalization": "per-band mean/std fitted on sampled training pixels only",
            "wavelength_units_supplied_to_model": "micrometres",
        }


@dataclass(frozen=True)
class FrozenProbeConfig:
    positive_patch_fraction: float = 0.05
    minimum_valid_patch_fraction: float = 0.8
    regularization_c: float = 1.0
    max_iter: int = 500
    seed: int = 20260822


class DofaFrozenProbe:
    """Frozen DOFA token features with one supervised linear head per mineral."""

    def __init__(
        self,
        encoder: DenseFeatureEncoder,
        config: FrozenProbeConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or FrozenProbeConfig()
        self.descriptor = encoder.descriptor
        self._classifiers: list[Any] = []
        self._class_names: tuple[str, ...] = ()
        self._feature_dimension: int | None = None
        self._training_patches: int | None = None
        self._trainable_parameters: int | None = None

    def fit(self, cubes: Iterable[HyperspectralCube]) -> DofaFrozenProbe:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "scikit-learn is required; install spectrashift[foundation]"
            ) from exc
        cubes = list(cubes)
        if not cubes:
            raise ValueError("at least one training cube is required")
        self.encoder.fit_normalization(cubes)
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        class_names = cubes[0].class_names
        for cube in cubes:
            if cube.class_names != class_names:
                raise ValueError("training cubes must use identical class names")
            grid = np.asarray(self.encoder.extract(cube), dtype=np.float32)
            if grid.ndim != 3:
                raise ValueError("feature encoder must return (grid_height, grid_width, features)")
            pooled, valid, _ = pool_patch_labels(
                cube,
                grid.shape[:2],
                positive_fraction=self.config.positive_patch_fraction,
                minimum_valid_fraction=self.config.minimum_valid_patch_fraction,
            )
            features.append(grid[valid])
            labels.append(pooled[valid])
        matrix = np.concatenate(features)
        targets = np.concatenate(labels)
        if matrix.shape[0] < 4:
            raise ValueError("at least four valid training patches are required")
        self._classifiers = []
        for class_index in range(targets.shape[1]):
            target = targets[:, class_index]
            if np.unique(target).size != 2:
                raise ValueError(
                    f"class {class_names[class_index]} needs positive and negative training patches"
                )
            classifier = LogisticRegression(
                C=self.config.regularization_c,
                class_weight="balanced",
                max_iter=self.config.max_iter,
                random_state=self.config.seed + class_index,
                solver="lbfgs",
            )
            classifier.fit(matrix, target)
            self._classifiers.append(classifier)
        self._class_names = class_names
        self._feature_dimension = matrix.shape[1]
        self._training_patches = matrix.shape[0]
        self._trainable_parameters = (matrix.shape[1] + 1) * targets.shape[1]
        return self

    def predict(self, cube: HyperspectralCube) -> np.ndarray:
        if not self._classifiers or self._feature_dimension is None:
            raise RuntimeError("model must be fitted before prediction")
        grid = np.asarray(self.encoder.extract(cube), dtype=np.float32)
        if grid.ndim != 3 or grid.shape[-1] != self._feature_dimension:
            raise ValueError("feature encoder output does not match the fitted linear probe")
        flat = grid.reshape(-1, grid.shape[-1])
        probabilities = np.column_stack(
            [classifier.predict_proba(flat)[:, 1] for classifier in self._classifiers]
        ).astype(np.float32)
        probabilities = probabilities.reshape(*grid.shape[:2], len(self._classifiers))
        probabilities = _resize_bilinear(probabilities, cube.height, cube.width)
        probabilities[~cube.valid_mask] = 0.0
        return probabilities

    def configuration(self) -> dict[str, Any]:
        return {
            "probe": asdict(self.config),
            "feature_encoder": self.encoder.configuration(),
            "feature_dimension": self._feature_dimension,
            "training_patches": self._training_patches,
            "trainable_parameters": self._trainable_parameters,
            "prediction_map_policy": "bilinear upsample from frozen token-grid probabilities",
        }
