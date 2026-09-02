"""Small, experiment-oriented wrapper for the OpenIllumination OLAT data."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF


NUM_OLAT_LIGHTS = 142

# The OLAT cameras form an incomplete 4 x 6 grid.
OLAT_CAMERA_IDS = (
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D3",
    "D4",
    "D5",
    "D6",
)

OLAT_OBJECTS = {
    1: "obj_01_egg",
    2: "obj_02_sculpture",
    3: "obj_03_pumpkin",
    4: "obj_04_dolphin",
    5: "obj_05_hedgehog",
    6: "obj_06_chicken",
    7: "obj_07_pumpkin2",
    8: "obj_08_sculpture2",
    9: "obj_09_ball",
    10: "obj_10_pumpkin3",
    11: "obj_11_pine",
    12: "obj_12_pine2",
    13: "obj_13_mushroom",
    14: "obj_14_basketball",
    15: "obj_15_pumpkin4",
    16: "obj_16_friends_cup",
    17: "obj_17_pumpkin5",
    18: "obj_18_fabric_hat",
    19: "obj_19_cylinder",
    20: "obj_20_greenhead",
}

Camera = str | tuple[str, int]
MaskKind = Literal["object", "combined"]
ImageSize = int | tuple[int, int]


def camera_id(camera: Camera) -> str:
    """Normalize ``"C4"`` or ``("C", 4)`` and reject missing grid cells."""
    if isinstance(camera, tuple):
        if len(camera) != 2:
            raise ValueError("A camera coordinate must be a (row, column) pair.")
        row, column = camera
        camera = f"{str(row).upper()}{column}"
    else:
        camera = camera.upper()

    if camera not in OLAT_CAMERA_IDS:
        raise ValueError(
            f"Unknown OLAT camera {camera!r}. The available cameras are: "
            f"{', '.join(OLAT_CAMERA_IDS)}"
        )
    return camera


def _validate_light_index(index: int) -> int:
    index = int(index)
    if not 0 <= index < NUM_OLAT_LIGHTS:
        raise ValueError(f"Light index must be in [0, {NUM_OLAT_LIGHTS - 1}], got {index}.")
    return index


@dataclass(frozen=True, eq=False)
class OLATLightPattern:
    """Non-negative intensity weights for the 142 OLAT basis images."""

    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32)
        if weights.shape != (NUM_OLAT_LIGHTS,):
            raise ValueError(f"Expected {NUM_OLAT_LIGHTS} light weights, got shape {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Light weights must be finite and non-negative.")
        weights = weights.copy()
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def single(cls, index: int, intensity: float = 1.0) -> "OLATLightPattern":
        return cls.from_indices([index], intensity=intensity)

    @classmethod
    def from_indices(
        cls,
        indices: Iterable[int],
        intensity: float | Iterable[float] = 1.0,
    ) -> "OLATLightPattern":
        indices = [_validate_light_index(index) for index in indices]
        if np.isscalar(intensity):
            intensities = [float(intensity)] * len(indices)
        else:
            intensities = [float(value) for value in intensity]
            if len(intensities) != len(indices):
                raise ValueError("There must be one intensity per light index.")

        weights = np.zeros(NUM_OLAT_LIGHTS, dtype=np.float32)
        for index, value in zip(indices, intensities):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Light intensities must be finite and non-negative.")
            weights[index] += value
        return cls(weights)

    @classmethod
    def from_mapping(cls, weights_by_index: Mapping[int, float]) -> "OLATLightPattern":
        return cls.from_indices(weights_by_index.keys(), weights_by_index.values())

    @classmethod
    def random(
        cls,
        count: int,
        rng: np.random.Generator | None = None,
        intensity: float = 1.0,
    ) -> "OLATLightPattern":
        if not 1 <= count <= NUM_OLAT_LIGHTS:
            raise ValueError(f"count must be in [1, {NUM_OLAT_LIGHTS}].")
        rng = rng or np.random.default_rng()
        return cls.from_indices(rng.choice(NUM_OLAT_LIGHTS, size=count, replace=False), intensity)

    @property
    def active_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.weights))

    def without(self, indices: Iterable[int]) -> "OLATLightPattern":
        weights = self.weights.copy()
        weights[[_validate_light_index(index) for index in indices]] = 0
        return OLATLightPattern(weights)

    def dropout(
        self,
        probability: float,
        rng: np.random.Generator | None = None,
    ) -> "OLATLightPattern":
        if not 0 <= probability <= 1:
            raise ValueError("Dropout probability must be in [0, 1].")
        rng = rng or np.random.default_rng()
        weights = self.weights.copy()
        active = np.flatnonzero(weights)
        weights[active[rng.random(len(active)) < probability]] = 0
        return OLATLightPattern(weights)

    def shuffled(self, rng: np.random.Generator | None = None) -> "OLATLightPattern":
        """Move the current intensities to random light indices."""
        rng = rng or np.random.default_rng()
        weights = self.weights.copy()
        rng.shuffle(weights)
        return OLATLightPattern(weights)


@dataclass(frozen=True)
class OLATConfiguration:
    camera: str
    lights: OLATLightPattern

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", camera_id(self.camera))


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.maximum(np.asarray(image, dtype=np.float32), 0)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1 / 2.4) - 0.055)


class OpenIlluminationOLAT:
    """Access, combine, download, and sample configurations of one OLAT object."""

    repo_id = "OpenIllumination/OpenIllumination"

    def __init__(
        self,
        root: str | Path,
        object_id: int | str,
        *,
        download_missing: bool = False,
    ) -> None:
        self.root = Path(root)
        self.object_name = self._object_name(object_id)
        self.download_missing = download_missing

    @staticmethod
    def _object_name(object_id: int | str) -> str:
        if isinstance(object_id, int):
            try:
                return OLAT_OBJECTS[object_id]
            except KeyError as error:
                raise ValueError(f"OLAT object ID must be in [1, {len(OLAT_OBJECTS)}].") from error
        if object_id not in OLAT_OBJECTS.values():
            raise ValueError(
                f"Unknown OLAT object {object_id!r}; pass an integer ID or an exact OLAT object name."
            )
        return object_id

    @property
    def relative_root(self) -> Path:
        return Path("OLAT") / self.object_name

    @property
    def object_root(self) -> Path:
        return self.root / self.relative_root

    def image_path(self, camera: Camera, light_index: int) -> Path:
        return (
            self.object_root
            / "Lights"
            / f"{_validate_light_index(light_index):03d}"
            / "raw_undistorted"
            / f"{camera_id(camera)}.jpg"
        )

    def mask_path(self, camera: Camera, kind: MaskKind = "combined") -> Path:
        try:
            directory = {"object": "obj_masks", "combined": "com_masks"}[kind]
        except KeyError as error:
            raise ValueError("Mask kind must be 'object' or 'combined'.") from error
        return self.object_root / "output" / directory / f"{camera_id(camera)}.png"

    def metadata_path(self, split: Literal["train", "val", "test"]) -> Path:
        return self.object_root / "output" / f"transforms_{split}.json"

    @property
    def light_positions_path(self) -> Path:
        """Path to the dataset-wide light-stage coordinates supplied by the authors."""
        return self.root / "light_pos.npy"

    def _ensure(self, path: Path) -> Path:
        if path.exists():
            return path
        if not self.download_missing:
            raise FileNotFoundError(
                f"{path} is not available locally. Set download_missing=True or call download()."
            )

        from huggingface_hub import hf_hub_download

        relative_path = path.relative_to(self.root).as_posix()
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=relative_path,
                local_dir=self.root,
            )
        )

    def load_mask(self, camera: Camera, kind: MaskKind = "combined") -> np.ndarray:
        return np.asarray(Image.open(self._ensure(self.mask_path(camera, kind)))) > 0

    def load_metadata(self, split: Literal["train", "val", "test"]) -> dict[str, Any]:
        with self._ensure(self.metadata_path(split)).open() as file:
            return json.load(file)

    def load_light_positions(self, *, normalize: bool = False) -> np.ndarray:
        """Return the author-provided ``(142, 3)`` light positions.

        Rows use the same zero-based indexing as the OLAT image directories.  With
        ``normalize=True``, return unit directions while leaving the stored data
        untouched.
        """
        positions = np.asarray(
            np.load(self._ensure(self.light_positions_path), allow_pickle=False),
            dtype=np.float64,
        )
        if positions.shape != (NUM_OLAT_LIGHTS, 3):
            raise ValueError(
                f"Expected {NUM_OLAT_LIGHTS} three-dimensional light positions, "
                f"got shape {positions.shape}."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("Light positions must be finite.")
        if normalize:
            norms = np.linalg.norm(positions, axis=-1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Cannot normalize a light position at the origin.")
            positions = positions / norms
        else:
            positions = positions.copy()
        positions.setflags(write=False)
        return positions

    def camera_metadata(self, camera: Camera) -> dict[str, Any]:
        """Return the pose/intrinsics record and its train/val/test split."""
        camera = camera_id(camera)
        for split in ("train", "test", "val"):
            frame = self.load_metadata(split).get("frames", {}).get(camera)
            if frame is not None:
                return {**frame, "camera_id": camera, "split": split}
        raise KeyError(f"No camera metadata found for {camera}.")

    def load_olat(self, camera: Camera, light_index: int, *, linear: bool = True) -> np.ndarray:
        image = np.asarray(
            Image.open(self._ensure(self.image_path(camera, light_index))).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        return srgb_to_linear(image) if linear else image

    def composite_linear(
        self,
        configuration: OLATConfiguration,
        *,
        mask: MaskKind | None = None,
        subtract_background: bool = False,
    ) -> np.ndarray:
        active = configuration.lights.active_indices
        if not active:
            raise ValueError("Cannot composite a light pattern with no active lights.")

        foreground = None
        if mask is not None or subtract_background:
            foreground = self.load_mask(configuration.camera, mask or "object")

        composite = None
        for index in active:
            image = self.load_olat(configuration.camera, index, linear=True)
            if subtract_background:
                assert foreground is not None
                background = image[~foreground]
                if background.size:
                    image = np.maximum(image - np.median(background, axis=0), 0)
            weighted = image * configuration.lights.weights[index]
            composite = weighted if composite is None else composite + weighted

        assert composite is not None
        if mask is not None:
            assert foreground is not None
            composite = composite * foreground[..., None]
        return composite

    def render(
        self,
        configuration: OLATConfiguration,
        *,
        exposure_ev: float = 0.0,
        auto_exposure: bool = False,
        exposure_percentile: float = 99.0,
        mask: MaskKind | None = None,
        subtract_background: bool = False,
        tone_map: Literal["clip", "reinhard"] = "clip",
    ) -> np.ndarray:
        """Return display-ready sRGB in [0, 1] from an OLAT configuration."""
        image = self.composite_linear(
            configuration,
            mask=mask,
            subtract_background=subtract_background,
        )
        image = image * (2.0**exposure_ev)

        if auto_exposure:
            pixels = image
            if mask is not None:
                pixels = image[self.load_mask(configuration.camera, mask or "object")]
            white = np.percentile(pixels, exposure_percentile)
            if white > 0:
                image = image * (0.9 / white)

        if tone_map == "reinhard":
            image = image / (1.0 + image)
        elif tone_map != "clip":
            raise ValueError(f"Unknown tone mapper {tone_map!r}.")

        return np.clip(linear_to_srgb(image), 0, 1)

    def render_pair(
        self,
        pair: tuple[OLATConfiguration, OLATConfiguration],
        **render_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.render(pair[0], **render_kwargs), self.render(pair[1], **render_kwargs)

    @staticmethod
    def _resize_image(
        image: np.ndarray,
        size: ImageSize,
        *,
        center_crop: bool,
    ) -> np.ndarray:
        """Resize a float image without an intermediate integer or gamma conversion."""
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or any(value <= 0 for value in size):
            raise ValueError("size must be a positive integer or (height, width) pair.")
        size = tuple(int(value) for value in size)

        image_tensor = torch.from_numpy(image).movedim(-1, 0)
        if center_crop:
            side = min(image_tensor.shape[-2:])
            image_tensor = TF.center_crop(image_tensor, [side, side])
        return (
            TF.resize(image_tensor, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)
            .movedim(0, -1)
            .numpy()
        )

    @staticmethod
    def _batch_weights(
        weights: np.ndarray,
        light_indices: Iterable[int] | None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = weights[None]
        if weights.ndim != 2:
            raise ValueError(f"weights must be a one- or two-dimensional array, got shape {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Light weights must be finite and non-negative.")
        if weights.shape[0] == 0:
            raise ValueError("weights must contain at least one pattern.")
        if np.any(np.all(weights == 0, axis=1)):
            raise ValueError("Every light pattern must have at least one active light.")

        if light_indices is None:
            if weights.shape[1] != NUM_OLAT_LIGHTS:
                raise ValueError(
                    f"Full light weights must have {NUM_OLAT_LIGHTS} columns, got {weights.shape[1]}."
                )
            active = np.flatnonzero(np.any(weights != 0, axis=0))
            return weights[:, active], tuple(int(index) for index in active)

        indices = tuple(_validate_light_index(index) for index in light_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("light_indices must not contain duplicates.")
        if weights.shape[1] != len(indices):
            raise ValueError(
                f"Compact weights have {weights.shape[1]} columns but {len(indices)} light indices were provided."
            )
        return weights, indices

    def _render_basis(
        self,
        cameras: tuple[str, ...],
        light_indices: tuple[int, ...],
        *,
        size: ImageSize | None,
        center_crop: bool,
        mask: MaskKind | None,
        subtract_background: bool,
        background_sample_stride: int,
        max_workers: int,
    ) -> np.ndarray:
        basis_by_camera = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for camera in cameras:
                foreground = None
                if mask is not None or subtract_background:
                    foreground = self.load_mask(camera, mask or "object")

                def load_basis_image(index: int) -> np.ndarray:
                    image = self.load_olat(camera, index, linear=True)
                    if subtract_background:
                        assert foreground is not None
                        sampled_image = image[::background_sample_stride, ::background_sample_stride]
                        sampled_foreground = foreground[::background_sample_stride, ::background_sample_stride]
                        background = sampled_image[~sampled_foreground]
                        if background.size:
                            image = np.maximum(image - np.median(background, axis=0), 0)
                    if mask is not None:
                        assert foreground is not None
                        image = image * foreground[..., None]
                    if size is not None:
                        image = self._resize_image(image, size, center_crop=center_crop)
                    return image

                basis = list(executor.map(load_basis_image, light_indices))
                basis_by_camera.append(np.stack(basis))
        return np.stack(basis_by_camera)

    def iter_render_weight_batches(
        self,
        weights: np.ndarray,
        *,
        cameras: Iterable[Camera],
        light_indices: Iterable[int] | None = None,
        batch_size: int = 16,
        size: ImageSize | None = None,
        center_crop: bool = False,
        exposure_ev: float = 0.0,
        mask: MaskKind | None = None,
        subtract_background: bool = False,
        background_sample_stride: int = 1,
        max_workers: int = 1,
        tone_map: Literal["clip", "reinhard"] = "clip",
    ) -> Iterator[np.ndarray]:
        """Yield display-ready sRGB batches for many shared light mixtures.

        The output shape is ``(batch, camera, height, width, 3)``.  ``weights``
        may either have all 142 columns or only the columns named by
        ``light_indices``.  In the full form, OLAT images are loaded only for
        columns that are active in at least one row.  The selected image basis is
        loaded and optionally resized once, then reused for every yielded batch.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if background_sample_stride <= 0:
            raise ValueError("background_sample_stride must be positive.")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        cameras = tuple(camera_id(camera) for camera in cameras)
        if not cameras:
            raise ValueError("cameras must contain at least one camera.")
        weights, active_indices = self._batch_weights(weights, light_indices)
        basis = self._render_basis(
            cameras,
            active_indices,
            size=size,
            center_crop=center_crop,
            mask=mask,
            subtract_background=subtract_background,
            background_sample_stride=background_sample_stride,
            max_workers=max_workers,
        )

        for start in range(0, len(weights), batch_size):
            batch = np.einsum(
                "bk,ckhwq->bchwq",
                weights[start : start + batch_size],
                basis,
                optimize=True,
            )
            batch *= 2.0**exposure_ev
            if tone_map == "reinhard":
                batch /= 1.0 + batch
            elif tone_map != "clip":
                raise ValueError(f"Unknown tone mapper {tone_map!r}.")
            yield np.clip(linear_to_srgb(batch), 0, 1)

    def render_weight_batch(
        self,
        weights: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Render all weighted patterns in memory; see :meth:`iter_render_weight_batches`.

        Prefer the iterator for arrays whose rendered output would be large.
        """
        weights_array = np.asarray(weights)
        kwargs = {**kwargs, "batch_size": max(1, len(weights_array) if weights_array.ndim > 1 else 1)}
        return next(self.iter_render_weight_batches(weights_array, **kwargs))

    def configuration(
        self,
        camera: Camera,
        lights: OLATLightPattern | int | Iterable[int],
    ) -> OLATConfiguration:
        if isinstance(lights, int):
            lights = OLATLightPattern.single(lights)
        elif not isinstance(lights, OLATLightPattern):
            lights = OLATLightPattern.from_indices(lights)
        return OLATConfiguration(camera_id(camera), lights)

    def sample_pair(
        self,
        *,
        same: Literal["view", "light"],
        lights_per_pattern: int = 1,
        camera: Camera | None = None,
        other_camera: Camera | None = None,
        light_pattern: OLATLightPattern | None = None,
        other_light_pattern: OLATLightPattern | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[OLATConfiguration, OLATConfiguration]:
        """Sample two configurations sharing exactly the requested factor."""
        rng = rng or np.random.default_rng()
        if same == "light":
            if other_light_pattern is not None:
                raise ValueError("other_light_pattern cannot vary when same='light'.")
            pattern = light_pattern or OLATLightPattern.random(lights_per_pattern, rng)
            first_camera = camera_id(camera) if camera is not None else str(rng.choice(OLAT_CAMERA_IDS))
            if other_camera is None:
                choices = tuple(candidate for candidate in OLAT_CAMERA_IDS if candidate != first_camera)
                second_camera = str(rng.choice(choices))
            else:
                second_camera = camera_id(other_camera)
            if first_camera == second_camera:
                raise ValueError("same='light' requires two different cameras.")
            return OLATConfiguration(first_camera, pattern), OLATConfiguration(second_camera, pattern)
        if same == "view":
            if other_camera is not None:
                raise ValueError("other_camera cannot vary when same='view'.")
            shared_camera = camera_id(camera) if camera is not None else str(rng.choice(OLAT_CAMERA_IDS))
            first = light_pattern or OLATLightPattern.random(lights_per_pattern, rng)
            second = other_light_pattern or OLATLightPattern.random(lights_per_pattern, rng)
            while np.array_equal(first.weights, second.weights):
                if other_light_pattern is not None:
                    raise ValueError("same='view' requires two different light patterns.")
                second = OLATLightPattern.random(lights_per_pattern, rng)
            return OLATConfiguration(shared_camera, first), OLATConfiguration(shared_camera, second)
        raise ValueError("same must be either 'view' or 'light'.")

    def download(
        self,
        *,
        cameras: Iterable[Camera] = OLAT_CAMERA_IDS,
        light_indices: Iterable[int] = range(NUM_OLAT_LIGHTS),
        masks: Sequence[MaskKind] = ("object", "combined"),
        metadata: bool = True,
        light_positions: bool = True,
        max_workers: int = 8,
    ) -> list[Path]:
        """Download an exact JPEG subset, never the multi-gigabyte RAW captures.

        Downloads run concurrently using up to ``max_workers`` threads. Set
        ``max_workers=1`` to download serially.
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")

        cameras = tuple(camera_id(camera) for camera in cameras)
        light_indices = tuple(_validate_light_index(index) for index in light_indices)
        relative_paths = [
            self.image_path(camera, index).relative_to(self.root)
            for index in light_indices
            for camera in cameras
        ]
        relative_paths.extend(
            self.mask_path(camera, kind).relative_to(self.root)
            for kind in masks
            for camera in cameras
        )
        if metadata:
            relative_paths.extend(
                self.metadata_path(split).relative_to(self.root)
                for split in ("train", "val", "test")
            )
        if light_positions:
            relative_paths.append(self.light_positions_path.relative_to(self.root))

        def download_file(path: Path) -> Path | None:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        filename=path.as_posix(),
                        local_dir=self.root,
                    )
                )
            except RemoteEntryNotFoundError:
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return [path for path in executor.map(download_file, relative_paths) if path is not None]


__all__ = [
    "NUM_OLAT_LIGHTS",
    "OLAT_CAMERA_IDS",
    "OLAT_OBJECTS",
    "OLATConfiguration",
    "OLATLightPattern",
    "OpenIlluminationOLAT",
    "camera_id",
    "linear_to_srgb",
    "srgb_to_linear",
]
