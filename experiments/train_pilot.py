"""Stage-1 lighting-conditioned RayDer pilot on OpenIllumination OLAT.

This script deliberately trains only the light encoder and the zero-initialized
light-to-AdaRMSNorm projections. It builds a deterministic uint8 ``.npy`` render
pool once, then memory-maps that pool for validation and training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterator, Sequence

# Permit the documented ``python experiments/train_pilot.py`` invocation from a
# source checkout without requiring an editable install.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

import rayder.model as rayder_model
from rayder.model import Camera, RayDer_B, RayDer_L, RayDer_S, RayDer_XS
from rayder.open_illumination import OpenIlluminationOLAT
from train import make_scheduler


_BLOCK_MASK_CACHE: dict[tuple[str, int, int, int, str], object] = {}
_FLEX_ATTENTION_COMPILED = False
_COMPILED_FLEX_MARKER = "_rayder_dynamic_flex_attention"

# Keep the black crop background from becoming an unconstrained output channel,
# while preserving foreground-dominated optimization for these small objects.
BACKGROUND_LOSS_WEIGHT = 0.05

# Reserve part of every pattern's total energy for an all-sphere fill. Each LED
# receives only a tiny floor, while their sum exposes geometry that a single
# directional lobe would otherwise leave in darkness.
PILOT_LIGHT_FLOOR_FRACTION = 0.10

# Cache construction is I/O and preprocessing heavy. The resized basis is much
# smaller than the original OLAT frames, so trade additional host RAM for fewer
# render batches and parallel JPEG decode/background estimation.
PILOT_CACHE_CAMERA_BLOCK_SIZE = 12
PILOT_CACHE_RENDER_BATCH_SIZE = 64
PILOT_CACHE_RENDER_WORKERS = 8
PILOT_BACKGROUND_SAMPLE_STRIDE = 8

# This pilot intentionally targets the locally resident mini subset. Track D
# was not downloaded, and object 12 + 20 is kept completely untouched for final
# evaluation rather than being used for tuning or checkpoint selection.
PILOT_CAMERA_IDS = (
    "A1", "A2", "A3", "A4", "A5", "A6",
    "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6",
)
RESIDENT_OBJECT_IDS = (1, 2, 3, 4, 5, 6, 8, 12, 20)
FINAL_TEST_OBJECT_IDS = (12, 20)
TRAIN_OBJECT_IDS = tuple(object_id for object_id in RESIDENT_OBJECT_IDS if object_id not in FINAL_TEST_OBJECT_IDS)


@dataclass(frozen=True)
class Lobe:
    weights: np.ndarray
    condition: np.ndarray
    effective_support: float
    concentration: float


@dataclass(frozen=True)
class ExampleAssignment:
    source_indices: tuple[int, int, int]
    target_index: int
    same_view: bool


@dataclass
class PilotBatch:
    source: torch.Tensor
    target_source: torch.Tensor
    target_requested: torch.Tensor
    mask: torch.Tensor
    source_light: torch.Tensor
    requested_light: torch.Tensor
    camera_rotation: torch.Tensor
    camera_translation: torch.Tensor
    camera_focal: torch.Tensor


def sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction)


def project_lobe(
    directions: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    energy: float,
    prune_threshold: float,
    floor_fraction: float = 0.0,
) -> Lobe:
    """Project a spherical lobe plus an optional fixed-energy all-sphere fill."""
    if not 0 <= floor_fraction < 1:
        raise ValueError("floor_fraction must be in [0, 1).")
    angles = np.arccos(np.clip(directions @ mu, -1.0, 1.0))
    weights = np.exp(-(angles**2) / (2.0 * sigma**2))
    weights[weights < prune_threshold * weights.max()] = 0.0
    weights *= energy * (1.0 - floor_fraction) / weights.sum()
    weights += energy * floor_fraction / len(weights)
    probabilities = weights / weights.sum()
    effective_support = float(1.0 / np.square(probabilities).sum())
    concentration = float(np.linalg.norm(probabilities @ directions))
    condition = np.asarray([*mu, math.log(sigma), math.log(energy)], dtype=np.float32)
    return Lobe(weights.astype(np.float32), condition, effective_support, concentration)


def sample_lobes(
    directions: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    sigma_bins: Sequence[tuple[float, float]],
    energy_range: tuple[float, float],
    prune_threshold: float,
    support_range: tuple[float, float],
    min_concentration: float,
    floor_fraction: float = 0.0,
    max_attempts: int = 1000,
) -> list[Lobe]:
    """Stratify sigma and reject lobes that become diffuse on the discrete rig."""
    lobes: list[Lobe] = []
    attempts = 0
    while len(lobes) < count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Accepted only {len(lobes)}/{count} lobes after {max_attempts} attempts; "
                "relax --effective-support or --min-concentration."
            )
        attempts += 1
        low, high = sigma_bins[len(lobes) % len(sigma_bins)]
        sigma = float(np.exp(rng.uniform(math.log(low), math.log(high))))
        energy = float(np.exp(rng.uniform(math.log(energy_range[0]), math.log(energy_range[1]))))
        lobe = project_lobe(
            directions,
            sample_unit_vector(rng),
            sigma,
            energy,
            prune_threshold,
            floor_fraction=floor_fraction,
        )
        if support_range[0] <= lobe.effective_support <= support_range[1] and lobe.concentration >= min_concentration:
            lobes.append(lobe)
    rng.shuffle(lobes)
    return lobes


def camera_distance(first: str, second: str) -> float:
    return math.hypot(ord(first[0]) - ord(second[0]), int(first[1:]) - int(second[1:]))


def nearest_cameras(target: str, candidates: Sequence[str], count: int) -> tuple[str, ...]:
    return tuple(sorted(candidates, key=lambda camera: (camera_distance(target, camera), camera))[:count])


def heldout_camera_map(object_ids: Sequence[int], seed: int) -> dict[int, str]:
    """Choose different per-object columns without globally removing a camera."""
    offset = seed % len(PILOT_CAMERA_IDS)
    return {
        object_id: PILOT_CAMERA_IDS[(offset + index) % len(PILOT_CAMERA_IDS)]
        for index, object_id in enumerate(sorted(object_ids))
    }


def overlapping_blocks(cameras: Sequence[str], block_size: int) -> list[tuple[str, ...]]:
    if block_size < 4:
        raise ValueError("camera_block_size must be at least four (three sources plus one target).")
    if len(cameras) <= block_size:
        return [tuple(cameras)]
    stride = block_size - 3
    starts = list(range(0, len(cameras) - block_size + 1, stride))
    final_start = len(cameras) - block_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [tuple(cameras[start : start + block_size]) for start in starts]


def make_assignments(
    num_examples: int,
    num_cameras: int,
    rng: np.random.Generator,
    same_view_fraction: float,
    fixed_target: int | None = None,
) -> list[ExampleAssignment]:
    assignments = []
    for _ in range(num_examples):
        target = fixed_target if fixed_target is not None else int(rng.integers(num_cameras))
        same_view = fixed_target is None and rng.random() < same_view_fraction
        candidates = [index for index in range(num_cameras) if index != target]
        if same_view:
            sources = [target, *rng.choice(candidates, size=2, replace=False).tolist()]
            rng.shuffle(sources)
        else:
            sources = rng.choice(candidates, size=3, replace=False).tolist()
        assignments.append(ExampleAssignment(tuple(int(i) for i in sources), target, same_view))
    return assignments


def resized_mask(dataset: OpenIlluminationOLAT, camera: str, size: int) -> np.ndarray:
    mask = dataset.load_mask(camera, "object").astype(np.float32)[..., None]
    return dataset._resize_image(mask, size, center_crop=True) > 0.5


def camera_parameters(dataset: OpenIlluminationOLAT, camera: str, size: int):
    """Convert OpenIllumination's c2w/FoV metadata to RayDer's camera convention."""
    metadata = dataset.camera_metadata(camera)
    transform = np.asarray(metadata["transform_matrix"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera transform for {camera}, got {transform.shape}.")
    # Inputs are center-cropped to the original image width (calib_imgw) and
    # resized to a square, so the normalized focal is independent of calib_imgw.
    # Camera.get_rays maps the image edges to normalized coordinates +/-1, so
    # its focal parameter is twice the usual f_pixels / image_width value.
    focal = size / ((size - 1) * math.tan(float(metadata["camera_angle_x"]) / 2))
    return transform[:3, :3], transform[:3, 3], focal


@dataclass
class CachedObject:
    images: np.ndarray
    masks: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    focals: np.ndarray


def cache_manifest(args: argparse.Namespace) -> dict:
    floor_fraction = getattr(args, "light_floor_fraction", 0.0)
    background_sample_stride = getattr(args, "background_sample_stride", 1)
    manifest = {
        "version": 3 if floor_fraction or background_sample_stride != 1 else 2,
        "camera_focal_convention": "rayder_edge_pm1",
        "dtype": "uint8",
        "size": args.size,
        "cameras": list(PILOT_CAMERA_IDS),
        "objects": list(args.train_objects),
        "train_lobes": args.cache_train_lobes,
        "validation_lobes": args.cache_validation_lobes,
        "cache_seed": args.cache_seed,
        "val_seed": args.val_seed,
        "sigma_bins": [list(pair) for pair in args.sigma_bins],
        "energy_range": list(args.energy_range),
        "effective_support": list(args.effective_support),
        "min_concentration": args.min_concentration,
        "prune_threshold": args.prune_threshold,
        "source_direction": list(args.source_direction),
        "source_sigma": args.source_sigma,
        "source_energy": args.source_energy,
        "subtract_background": args.subtract_background,
    }
    if manifest["version"] >= 3:
        manifest["light_floor_fraction"] = floor_fraction
        manifest["background_sample_stride"] = background_sample_stride
    return manifest


def migrate_v1_cache_focals(args: argparse.Namespace, actual_manifest: dict, expected_manifest: dict) -> bool:
    """Upgrade the small auxiliary files without rebuilding rendered images."""
    legacy_manifest = dict(expected_manifest)
    legacy_manifest["version"] = 1
    legacy_manifest.pop("camera_focal_convention")
    if actual_manifest != legacy_manifest:
        return False

    for object_id in args.train_objects:
        path = args.cache_dir / f"object_{object_id:02d}_aux.npz"
        if not path.exists():
            return False
    for object_id in args.train_objects:
        path = args.cache_dir / f"object_{object_id:02d}_aux.npz"
        temporary = path.with_suffix(".npz.tmp")
        dataset = OpenIlluminationOLAT(args.data_root, object_id)
        corrected_focals = np.asarray(
            [camera_parameters(dataset, camera, args.size)[2] for camera in PILOT_CAMERA_IDS]
        )
        with np.load(path) as auxiliary, temporary.open("wb") as file:
            np.savez(
                file,
                masks=auxiliary["masks"],
                rotations=auxiliary["rotations"],
                translations=auxiliary["translations"],
                focals=corrected_focals,
            )
        os.replace(temporary, path)
    temporary_manifest = (args.cache_dir / "manifest.json").with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(expected_manifest, indent=2) + "\n")
    os.replace(temporary_manifest, args.cache_dir / "manifest.json")
    return True


def make_cache_lobes(directions: np.ndarray, args: argparse.Namespace) -> tuple[list[Lobe], list[Lobe]]:
    train_lobes = sample_lobes(
        directions,
        args.cache_train_lobes,
        np.random.default_rng(args.cache_seed),
        **lobe_kwargs(args),
    )
    validation_lobes = sample_lobes(
        directions,
        args.cache_validation_lobes,
        np.random.default_rng(args.val_seed),
        **lobe_kwargs(args),
    )
    return train_lobes, validation_lobes


def prepare_render_cache(
    args: argparse.Namespace,
    directions: np.ndarray,
    source_lobe: Lobe,
    logger: logging.Logger,
) -> tuple[list[Lobe], list[Lobe]]:
    """Build or validate the fixed uint8 render pool used by training."""
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "manifest.json"
    expected_manifest = cache_manifest(args)
    if manifest_path.exists() and not args.rebuild_cache:
        actual_manifest = json.loads(manifest_path.read_text())
        if actual_manifest != expected_manifest:
            if migrate_v1_cache_focals(args, actual_manifest, expected_manifest):
                logger.info("Migrated cached camera focals to the RayDer +/-1 image-plane convention.")
                actual_manifest = expected_manifest
            else:
                raise RuntimeError(
                    f"Cache settings do not match {manifest_path}. Use --rebuild-cache or a different --cache-dir."
                )
        complete = all(
            (args.cache_dir / f"object_{object_id:02d}_images.npy").exists()
            and (args.cache_dir / f"object_{object_id:02d}_aux.npz").exists()
            for object_id in args.train_objects
        )
        if complete:
            train_lobes, validation_lobes = make_cache_lobes(directions, args)
            logger.info("Using existing uint8 render cache: %s", args.cache_dir)
            return train_lobes, validation_lobes
        logger.info("Resuming incomplete uint8 render cache: %s", args.cache_dir)

    train_lobes, validation_lobes = make_cache_lobes(directions, args)
    all_lobes = [*train_lobes, *validation_lobes]
    weights = np.stack([source_lobe.weights, *(lobe.weights for lobe in all_lobes)])
    shape = (len(weights), len(PILOT_CAMERA_IDS), args.size, args.size, 3)
    gib_per_object = math.prod(shape) / 2**30
    logger.info(
        "Building uint8 cache with %d light rows x %d cameras (%.2f GiB/object, %.2f GiB total)",
        len(weights),
        len(PILOT_CAMERA_IDS),
        gib_per_object,
        gib_per_object * len(args.train_objects),
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(expected_manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    for object_number, object_id in enumerate(tqdm(args.train_objects, desc="Caching objects"), start=1):
        final_images = args.cache_dir / f"object_{object_id:02d}_images.npy"
        final_aux = args.cache_dir / f"object_{object_id:02d}_aux.npz"
        if not args.rebuild_cache and final_images.exists() and final_aux.exists():
            logger.info("Skipping completed cached object ID %d", object_id)
            continue
        logger.info("Caching object %d/%d (ID %d)", object_number, len(args.train_objects), object_id)
        dataset = OpenIlluminationOLAT(args.data_root, object_id)
        temporary_images = args.cache_dir / f"object_{object_id:02d}_images.npy.tmp"
        images = np.lib.format.open_memmap(temporary_images, mode="w+", dtype=np.uint8, shape=shape)
        progress = tqdm(total=len(weights) * len(PILOT_CAMERA_IDS), desc=f"Object {object_id} renders", leave=False)
        cache_camera_block_size = getattr(args, "cache_camera_block_size", args.camera_block_size)
        for camera_start in range(0, len(PILOT_CAMERA_IDS), cache_camera_block_size):
            camera_end = min(camera_start + cache_camera_block_size, len(PILOT_CAMERA_IDS))
            camera_block = PILOT_CAMERA_IDS[camera_start:camera_end]
            offset = 0
            for rendered in dataset.iter_render_weight_batches(
                weights,
                cameras=camera_block,
                batch_size=args.cache_render_batch_size,
                size=args.size,
                center_crop=True,
                mask="object",
                subtract_background=args.subtract_background,
                background_sample_stride=getattr(args, "background_sample_stride", 1),
                max_workers=getattr(args, "cache_render_workers", 1),
                tone_map="clip",
            ):
                quantized = np.rint(rendered * 255).clip(0, 255).astype(np.uint8)
                images[offset : offset + len(quantized), camera_start:camera_end] = quantized
                offset += len(quantized)
                progress.update(len(quantized) * len(camera_block))
        progress.close()
        images.flush()
        del images
        os.replace(temporary_images, final_images)

        masks = np.stack([resized_mask(dataset, camera, args.size) for camera in PILOT_CAMERA_IDS]).astype(np.uint8)
        camera_values = [camera_parameters(dataset, camera, args.size) for camera in PILOT_CAMERA_IDS]
        temporary_aux = args.cache_dir / f"object_{object_id:02d}_aux.npz.tmp"
        with temporary_aux.open("wb") as file:
            np.savez(
                file,
                masks=masks,
                rotations=np.stack([value[0] for value in camera_values]),
                translations=np.stack([value[1] for value in camera_values]),
                focals=np.asarray([value[2] for value in camera_values]),
            )
        os.replace(temporary_aux, final_aux)

    logger.info("Finished render cache: %s", args.cache_dir)
    return train_lobes, validation_lobes


def load_cached_object(args: argparse.Namespace, object_id: int) -> CachedObject:
    images = np.load(args.cache_dir / f"object_{object_id:02d}_images.npy", mmap_mode="r")
    with np.load(args.cache_dir / f"object_{object_id:02d}_aux.npz") as auxiliary:
        masks = auxiliary["masks"]
        rotations = auxiliary["rotations"]
        translations = auxiliary["translations"]
        focals = auxiliary["focals"]
    return CachedObject(
        images=images,
        masks=masks,
        rotations=rotations,
        translations=translations,
        focals=focals,
    )


def batches_from_cache(
    cache: CachedObject,
    cameras: Sequence[str],
    lobe_indices: Sequence[int],
    assignments: Sequence[ExampleAssignment],
    source_lobe: Lobe,
    all_lobes: Sequence[Lobe],
    args: argparse.Namespace,
) -> Iterator[PilotBatch]:
    """Gather logical batches from the memory-mapped uint8 render pool."""
    camera_indices = [PILOT_CAMERA_IDS.index(camera) for camera in cameras]
    source_render = cache.images[0, camera_indices]
    for start in range(0, len(assignments), args.batch_size):
        selected_assignments = assignments[start : start + args.batch_size]
        selected_indices = lobe_indices[start : start + args.batch_size]
        source = np.stack([source_render[list(item.source_indices)] for item in selected_assignments])
        target_source = np.stack([source_render[item.target_index] for item in selected_assignments])
        target_requested = np.stack(
            [
                cache.images[1 + lobe_index, camera_indices[item.target_index]]
                for lobe_index, item in zip(selected_indices, selected_assignments)
            ]
        )
        target_masks = np.stack([cache.masks[camera_indices[item.target_index]] for item in selected_assignments])
        camera_rows = [
            tuple(camera_indices[index] for index in (*item.source_indices, item.target_index))
            for item in selected_assignments
        ]
        selected_lobes = [all_lobes[index] for index in selected_indices]
        yield PilotBatch(
            source=torch.from_numpy(source.astype(np.float32)).div(127.5).sub(1),
            target_source=torch.from_numpy(target_source.astype(np.float32)).div(127.5).sub(1),
            target_requested=torch.from_numpy(target_requested.astype(np.float32)).div(127.5).sub(1),
            mask=torch.from_numpy(target_masks.astype(bool)),
            source_light=torch.from_numpy(np.stack([source_lobe.condition] * len(selected_assignments))),
            requested_light=torch.from_numpy(np.stack([lobe.condition for lobe in selected_lobes])),
            camera_rotation=torch.from_numpy(
                np.stack([[cache.rotations[index] for index in row] for row in camera_rows])
            ),
            camera_translation=torch.from_numpy(
                np.stack([[cache.translations[index] for index in row] for row in camera_rows])
            ),
            camera_focal=torch.from_numpy(
                np.asarray([[cache.focals[index] for index in row] for row in camera_rows])
            ),
        )


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() * mask).sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1)


def supported_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    foreground = masked_mse(prediction, target, mask)
    background = masked_mse(prediction, target, ~mask)
    return foreground + BACKGROUND_LOSS_WEIGHT * background


def foreground_lpips(
    perceptual_model: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average a spatial LPIPS map over the target foreground only.

    LPIPS uses NCHW tensors in [-1, 1]. Compositing both images onto black
    before evaluating the spatial map prevents arbitrary crop-background
    predictions from influencing features whose receptive fields cross the
    silhouette boundary.
    """
    prediction_nchw = prediction.permute(0, 3, 1, 2).float()
    target_nchw = target.permute(0, 3, 1, 2).float()
    mask_nchw = mask.permute(0, 3, 1, 2).bool()
    black = prediction_nchw.new_tensor(-1.0)
    prediction_nchw = torch.where(mask_nchw, prediction_nchw, black)
    target_nchw = torch.where(mask_nchw, target_nchw, black)
    spatial_distance = perceptual_model(prediction_nchw, target_nchw, normalize=False)
    if spatial_distance.shape[-2:] != mask_nchw.shape[-2:]:
        spatial_distance = F.interpolate(
            spatial_distance,
            size=mask_nchw.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    weights = mask_nchw.to(spatial_distance)
    per_example = (spatial_distance * weights).sum(dim=(1, 2, 3)) / weights.sum(
        dim=(1, 2, 3)
    ).clamp_min(1)
    return per_example.mean()


def perceptual_weight_at_step(weight: float, warmup_steps: int, step: int) -> float:
    if weight <= 0:
        return 0.0
    if warmup_steps <= 0:
        return weight
    return weight * min(1.0, max(0.0, step / warmup_steps))


def load_perceptual_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module | None:
    if args.perceptual_weight <= 0:
        return None
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "--perceptual-weight requires the optional 'lpips' package. Install it with `pip install lpips`."
        ) from error
    model = lpips.LPIPS(
        net=args.perceptual_net,
        spatial=True,
        eval_mode=True,
        # Keep the checkpoint's standard Dropout -> Conv module layout so the
        # learned calibration weights load into ``model.1``. ``eval_mode``
        # disables dropout stochastically during both training and validation.
        use_dropout=True,
    ).to(device)
    calibration_weights = [layer.model[-1].weight for layer in model.lins]
    if any(torch.any(weight < 0) for weight in calibration_weights):
        raise RuntimeError(
            "LPIPS calibration weights contain negative values. This usually means the pretrained "
            "checkpoint did not match the calibration-layer architecture."
        )
    model.eval().requires_grad_(False)
    return model


def _dataset_to_inferred_world_rotation(
    calibrated: Camera,
    inferred: Camera,
) -> torch.Tensor:
    """Fit the rotation taking dataset-world directions into RayDer's pose gauge."""
    # Camera rotations differ by a single world-frame rotation Q:
    # R_inferred ~= Q @ R_calibrated. Project the mean relative rotation onto SO(3).
    relative = inferred.R @ calibrated.R.transpose(-2, -1)
    u, _, vh = torch.linalg.svd(relative.sum(dim=1))
    orientation = u @ vh
    handedness = torch.linalg.det(orientation)
    correction = torch.diag_embed(
        torch.stack([torch.ones_like(handedness), torch.ones_like(handedness), handedness], dim=-1)
    )
    return u @ correction @ vh


def camera_conditioning_for_batch(
    model: torch.nn.Module,
    batch: PilotBatch,
    source: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Camera, torch.Tensor | None]:
    calibrated = Camera(
        R=batch.camera_rotation.to(device, non_blocking=True),
        t=batch.camera_translation.to(device, non_blocking=True),
        f=batch.camera_focal.to(device, non_blocking=True),
    )
    if not getattr(args, "infer_camera_poses", True):
        return calibrated, None

    # Match pretraining: estimate every source and target camera jointly from
    # their observed source-light images. The target image informs only its
    # camera condition; it is never passed to the reconstruction context.
    target_source = batch.target_source.to(device, non_blocking=True)[:, None]
    pose_images = torch.cat([source, target_source], dim=1)
    with torch.no_grad():
        inferred, _ = model._estimate_cameras(
            pose_images,
            temporal_ranks=torch.arange(pose_images.shape[1], device=device),
        )
    return inferred, _dataset_to_inferred_world_rotation(calibrated, inferred)


def cameras_for_batch(
    model: torch.nn.Module,
    batch: PilotBatch,
    source: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Camera:
    """Convenience wrapper used by evaluation code that only needs cameras."""
    return camera_conditioning_for_batch(model, batch, source, args, device)[0]


def predict_batch(model: torch.nn.Module, batch: PilotBatch, args: argparse.Namespace, device: torch.device):
    """Render the paired source/requested-light outputs for one cached batch."""
    source = batch.source.to(device, non_blocking=True)
    target_light = torch.stack([batch.source_light, batch.requested_light], dim=1).to(device)

    cameras, world_rotation = camera_conditioning_for_batch(model, batch, source, args, device)
    if world_rotation is not None:
        # Keep illumination global rather than camera-relative, but express it
        # in the same inferred world gauge as the Pluecker rays.
        target_light = target_light.clone()
        target_light[..., :3] = torch.einsum(
            "bij,bnj->bni", world_rotation.to(target_light), target_light[..., :3]
        )
    camera_target = Camera.cat([cameras[:, -1:], cameras[:, -1:]], dim=1)
    n_tokens = (args.size // model.total_spatial_downsample) ** 2
    # Each paired target is an independent rendering request and must see all
    # three context views. RayDer's training mask assumes aligned input/target
    # sequences; with 3 inputs and 2 targets it would expose only one context to
    # target 0 and two contexts to target 1.
    mask_key = ("all_sources", 3, 2, n_tokens, str(device))
    if mask_key not in _BLOCK_MASK_CACHE:
        _BLOCK_MASK_CACHE[mask_key] = model._inference_block_mask(3, 2, n_tokens, device)
    block_mask = _BLOCK_MASK_CACHE[mask_key]
    prediction = model._reconstruct(
        x_in=source,
        camera_in=cameras[:, :3],
        camera_target=camera_target,
        state_target=source.new_zeros(len(source), 2, model.d_dynamic_state),
        block_mask=block_mask,
        drop_state=True,
        target_light=target_light,
        # The two outputs differ only in requested light. Giving both the same
        # temporal coordinate keeps the difference loss free of position noise.
        temporal_positions=torch.tensor([0, 1, 2, 3, 3], device=device),
    )
    return prediction.unbind(dim=1)


def forward_loss(
    model: torch.nn.Module,
    batch: PilotBatch,
    args: argparse.Namespace,
    device: torch.device,
    *,
    perceptual_model: torch.nn.Module | None = None,
    perceptual_weight: float = 0.0,
    require_pair_symmetry: bool = False,
):
    target_source = batch.target_source.to(device, non_blocking=True)
    target_requested = batch.target_requested.to(device, non_blocking=True)
    mask = batch.mask.to(device, non_blocking=True)
    pred_source, pred_requested = predict_batch(model, batch, args, device)
    if require_pair_symmetry:
        # Under bfloat16 autocast, equivalent sparse-attention rows can differ by
        # one or two ULPs because the self token occupies a different physical
        # index in each row. The mask regression test checks exact connectivity;
        # this runtime check catches material output asymmetry.
        precision = torch.finfo(pred_source.dtype).eps
        tolerance = max(1e-5, 2 * precision)
        if not torch.allclose(pred_source, pred_requested, rtol=0.0, atol=tolerance):
            max_error = float((pred_source - pred_requested).abs().max())
            raise RuntimeError(
                "Zero-init paired targets are not symmetric; their camera, temporal position, "
                f"and attention visibility should be equivalent (dtype={pred_source.dtype}, "
                f"max difference {max_error:.6g}, tolerance {tolerance:.6g})."
            )
    loss_target = masked_mse(pred_requested, target_requested, mask)
    loss_source = masked_mse(pred_source, target_source, mask)
    loss_delta = masked_mse(pred_requested - pred_source, target_requested - target_source, mask)
    optimized_target = supported_mse(pred_requested, target_requested, mask)
    optimized_source = supported_mse(pred_source, target_source, mask)
    optimized_delta = supported_mse(pred_requested - pred_source, target_requested - target_source, mask)
    pixel_loss = optimized_target + args.gamma * optimized_source + args.beta * optimized_delta
    if perceptual_model is not None:
        # VGG LPIPS is numerically safer in fp32; the frozen network still
        # propagates gradients to the RayDer predictions.
        with torch.autocast(device_type=device.type, enabled=False):
            perceptual_target = foreground_lpips(
                perceptual_model, pred_requested, target_requested, mask
            )
            perceptual_source = foreground_lpips(
                perceptual_model, pred_source, target_source, mask
            )
        perceptual_loss = perceptual_target + args.gamma * perceptual_source
    else:
        perceptual_target = pixel_loss.new_zeros(())
        perceptual_source = pixel_loss.new_zeros(())
        perceptual_loss = pixel_loss.new_zeros(())
    objective = pixel_loss + perceptual_weight * perceptual_loss
    metrics = {
        # Keep ``loss`` as the original pixel/delta objective so validation and
        # checkpoint selection remain directly comparable across ablations.
        "loss": pixel_loss.detach(),
        "objective": objective.detach(),
        "perceptual_loss": perceptual_loss.detach(),
        "perceptual_target": perceptual_target.detach(),
        "perceptual_source": perceptual_source.detach(),
        "target_mse": loss_target.detach() / 4,
        "source_mse": loss_source.detach() / 4,
        "delta_mse": loss_delta.detach() / 4,
        "target_psnr": -10 * torch.log10((loss_target.detach() / 4).clamp_min(1e-12)),
    }
    return objective, metrics


def flex_attention_is_compiled() -> bool:
    """Inspect the callable actually used by ``rayder.model``, not a copied flag."""
    return bool(getattr(rayder_model.flex_attention, _COMPILED_FLEX_MARKER, False))


def configure_flex_attention(compile_flex_attention: bool) -> None:
    """Idempotently install compiled FlexAttention, including after module reloads."""
    global _FLEX_ATTENTION_COMPILED
    if compile_flex_attention and not flex_attention_is_compiled():
        # Evaluation exercises several batch sizes, sequence lengths, and head
        # counts. A shared dynamic=False wrapper exhausts Dynamo's per-function
        # recompile limit and silently falls back to eager dense attention.
        attention = getattr(
            rayder_model.flex_attention,
            "_torchdynamo_orig_callable",
            rayder_model.flex_attention,
        )
        # Dynamic shapes should keep this well below the limit, but retain a
        # safety margin rather than silently switching to the OOM-prone eager
        # implementation if a backend specializes an unexpected dimension.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        compiled_attention = torch.compile(attention, dynamic=True)
        setattr(compiled_attention, _COMPILED_FLEX_MARKER, True)
        rayder_model.flex_attention = compiled_attention
    _FLEX_ATTENTION_COMPILED = flex_attention_is_compiled()


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    # Calling FlexAttention eagerly falls back to dense score/mask tensors; at
    # 256px that can allocate several GiB even for batch size one.
    configure_flex_attention(args.compile_flex_attention)
    constructors = {"XS": RayDer_XS, "S": RayDer_S, "B": RayDer_B, "L": RayDer_L}
    model = constructors[args.model](d_light=args.light_dim, dynamic_state_dropout=0).to(device)
    checkpoint_path = args.pretrained
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download

        checkpoint_path = Path(hf_hub_download("CompVis/rayder", f"rayder_{args.model.lower()}.pt"))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    if "camera_tokens_2.weight" in state:
        state["nvs_tokens.weight"] = state.pop("camera_tokens_2.weight")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("light_encoder" not in key and "light_linear" not in key for key in missing):
        raise RuntimeError(f"Incompatible pretrained checkpoint; missing={missing}, unexpected={unexpected}")

    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "light_encoder" in name or "light_linear" in name:
            parameter.requires_grad_(True)
    return model


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, epoch: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    lighting_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if "light_encoder" in name or "light_linear" in name
    }
    torch.save(
        {
            "lighting": lighting_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def load_resume(path: Path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["lighting"], strict=False)
    if unexpected or any("light_encoder" in key or "light_linear" in key for key in missing):
        raise RuntimeError(f"Incompatible pilot checkpoint; missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["step"]), int(checkpoint.get("epoch", 0))


def all_reduce_gradients(parameters: Sequence[torch.nn.Parameter], world_size: int) -> None:
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


@torch.no_grad()
def validate(
    model,
    args,
    device,
    heldout_cameras,
    source_lobe,
    train_lobes,
    validation_lobes,
    rank,
    world_size,
    perceptual_model: torch.nn.Module | None = None,
    perceptual_weight: float = 0.0,
    require_pair_symmetry: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for block_index, object_id in enumerate(sorted(args.train_objects)):
        if block_index % world_size != rank:
            continue
        target = heldout_cameras[object_id]
        retained = [camera for camera in PILOT_CAMERA_IDS if camera != target]
        cameras = (*nearest_cameras(target, retained, 3), target)
        rng = np.random.default_rng(args.val_seed + object_id)
        lobe_indices = list(
            range(len(train_lobes), len(train_lobes) + args.val_examples_per_object)
        )
        assignments = make_assignments(len(lobe_indices), 4, rng, 0.0, fixed_target=3)
        cache = load_cached_object(args, object_id)
        for batch in batches_from_cache(
            cache,
            cameras,
            lobe_indices,
            assignments,
            source_lobe,
            [*train_lobes, *validation_lobes],
            args,
        ):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, metrics = forward_loss(
                    model,
                    batch,
                    args,
                    device,
                    perceptual_model=perceptual_model,
                    perceptual_weight=perceptual_weight,
                    require_pair_symmetry=require_pair_symmetry,
                )
            require_pair_symmetry = False
            batch_size = len(batch.source)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    metric_names = ("delta_mse", "loss", "source_mse", "target_mse", "target_psnr")
    if perceptual_model is not None:
        metric_names += ("perceptual_loss", "perceptual_source", "perceptual_target", "objective")
    packed = torch.tensor([count, *(sums.get(key, 0.0) for key in metric_names)], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    count = max(1.0, float(packed[0]))
    result = {key: float(value) / count for key, value in zip(metric_names, packed[1:].tolist())}
    model.train(was_training)
    return result


def lobe_kwargs(args: argparse.Namespace) -> dict:
    return {
        "sigma_bins": args.sigma_bins,
        "energy_range": tuple(args.energy_range),
        "prune_threshold": args.prune_threshold,
        "support_range": tuple(args.effective_support),
        "min_concentration": args.min_concentration,
        "floor_fraction": getattr(args, "light_floor_fraction", 0.0),
    }


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def train(args: argparse.Namespace) -> None:
    rank, world_size, device = setup_distributed()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = args.out_dir / args.run_name / timestamp
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger("rayder.pilot")

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    heldout_cameras = heldout_camera_map(args.train_objects, args.split_seed)
    if rank == 0:
        logger.info("Final held-out objects (never loaded): %s", args.test_objects)
        logger.info("Validation columns: %s", heldout_cameras)

    reference_dataset = OpenIlluminationOLAT(args.data_root, args.train_objects[0])
    directions = reference_dataset.load_light_positions(normalize=True)
    source_lobe = project_lobe(
        directions,
        np.asarray(args.source_direction, dtype=np.float64) / np.linalg.norm(args.source_direction),
        args.source_sigma,
        args.source_energy,
        args.prune_threshold,
        floor_fraction=getattr(args, "light_floor_fraction", 0.0),
    )
    if not (args.effective_support[0] <= source_lobe.effective_support <= args.effective_support[1]):
        logger.warning("Fixed source lobe has N_eff=%.2f, outside the requested target range.", source_lobe.effective_support)

    if rank == 0:
        train_lobes, validation_lobes = prepare_render_cache(args, directions, source_lobe, logger)
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        train_lobes, validation_lobes = make_cache_lobes(directions, args)
    if args.cache_only:
        if rank == 0:
            logger.info("Cache-only run complete.")
        return

    # Lighting modules must start identically because distributed workers average
    # gradients manually while reading different whole physical blocks.
    torch.manual_seed(args.seed)
    model = load_model(args, device)
    perceptual_model = load_perceptual_model(args, device)
    if rank == 0:
        logger.info("Compiled FlexAttention: %s", args.compile_flex_attention)
        logger.info(
            "Camera conditioning: %s",
            "jointly inferred source/target poses" if args.infer_camera_poses else "dataset-calibrated poses",
        )
        if perceptual_model is not None:
            logger.info(
                "Perceptual objective: LPIPS-%s, weight %.4g, ramp %d steps (foreground spatial mean)",
                args.perceptual_net,
                args.perceptual_weight,
                args.perceptual_warmup_steps,
            )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr, args.warmup_steps, args.max_steps, args.scheduler)
    step, start_epoch = 0, 0
    if args.resume is not None:
        step, start_epoch = load_resume(args.resume, model, optimizer, scheduler, device)
    if rank == 0:
        logger.info("Trainable parameters: %.3fM", sum(p.numel() for p in trainable) / 1e6)

    pbar = tqdm(total=args.max_steps, initial=step, disable=rank != 0, desc="Pilot training")
    best_val_loss = math.inf
    if args.val_frequency > 0 and step == 0:
        baseline = validate(
            model,
            args,
            device,
            heldout_cameras,
            source_lobe,
            train_lobes,
            validation_lobes,
            rank,
            world_size,
            perceptual_model=perceptual_model,
            perceptual_weight=args.perceptual_weight,
            require_pair_symmetry=True,
        )
        if rank == 0:
            (output / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2) + "\n")
            logger.info("Zero-init validation baseline: %s", ", ".join(f"{k}={v:.6g}" for k, v in baseline.items()))
            logger.info("Zero-init paired-target symmetry check passed.")
    epoch = start_epoch
    while step < args.max_steps:
        object_order = list(args.train_objects)
        np.random.default_rng(args.seed + epoch).shuffle(object_order)
        physical_index = 0
        for object_id in object_order:
            cache = load_cached_object(args, object_id)
            retained = [camera for camera in PILOT_CAMERA_IDS if camera != heldout_cameras[object_id]]
            blocks = overlapping_blocks(retained, args.camera_block_size)
            for local_block_index, cameras in enumerate(blocks):
                assigned_rank = physical_index % world_size
                physical_index += 1
                if assigned_rank != rank:
                    continue
                block_seed = args.seed + epoch * 1_000_003 + object_id * 1009 + local_block_index
                rng = np.random.default_rng(block_seed)
                lobe_indices = rng.integers(0, len(train_lobes), size=args.examples_per_block).tolist()
                assignments = make_assignments(len(lobe_indices), len(cameras), rng, args.same_view_fraction)
                for batch in batches_from_cache(
                    cache,
                    cameras,
                    lobe_indices,
                    assignments,
                    source_lobe,
                    train_lobes,
                    args,
                ):
                    optimizer.zero_grad(set_to_none=True)
                    current_perceptual_weight = perceptual_weight_at_step(
                        args.perceptual_weight, args.perceptual_warmup_steps, step
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss, metrics = forward_loss(
                            model,
                            batch,
                            args,
                            device,
                            perceptual_model=perceptual_model,
                            perceptual_weight=current_perceptual_weight,
                        )
                    loss.backward()
                    all_reduce_gradients(trainable, world_size)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    if rank == 0:
                        pbar.update(1)
                        postfix = {
                            "loss": f"{float(metrics['objective']):.4g}",
                            "pixel": f"{float(metrics['loss']):.4g}",
                            "psnr": f"{float(metrics['target_psnr']):.3f}",
                        }
                        if perceptual_model is not None:
                            postfix["lpips"] = f"{float(metrics['perceptual_loss']):.4g}"
                        pbar.set_postfix(**postfix)

                    if args.val_frequency > 0 and step % args.val_frequency == 0:
                        val = validate(
                            model,
                            args,
                            device,
                            heldout_cameras,
                            source_lobe,
                            train_lobes,
                            validation_lobes,
                            rank,
                            world_size,
                            perceptual_model=perceptual_model,
                            perceptual_weight=args.perceptual_weight,
                        )
                        if rank == 0:
                            logger.info("Validation step %d: %s", step, ", ".join(f"{k}={v:.6g}" for k, v in val.items()))
                            if val["loss"] < best_val_loss:
                                best_val_loss = val["loss"]
                                save_checkpoint(
                                    output / "checkpoints" / "best.pt", model, optimizer, scheduler, step, epoch, args
                                )
                    if rank == 0 and args.checkpoint_frequency > 0 and step % args.checkpoint_frequency == 0:
                        save_checkpoint(output / "checkpoints" / f"checkpoint_{step:07d}.pt", model, optimizer, scheduler, step, epoch, args)
                        save_checkpoint(output / "checkpoints" / "latest.pt", model, optimizer, scheduler, step, epoch, args)
                    if step >= args.max_steps:
                        break
                if step >= args.max_steps:
                    break
            if step >= args.max_steps:
                break
        epoch += 1

    if rank == 0:
        save_checkpoint(output / "checkpoints" / "latest.pt", model, optimizer, scheduler, step, epoch, args)
        logger.info("Finished at step %d; output: %s", step, output)
        pbar.close()


def parse_sigma_bins(value: str) -> list[tuple[float, float]]:
    try:
        bins = [tuple(float(number) for number in item.split(":")) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated low:high pairs, e.g. 0.12:0.2,0.2:0.3") from error
    if not bins or any(len(pair) != 2 or pair[0] <= 0 or pair[0] >= pair[1] for pair in bins):
        raise argparse.ArgumentTypeError("Every sigma bin must be a positive low:high pair.")
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default="openillumination-pilot-stage1")
    parser.add_argument("--model", choices=["XS", "S", "B", "L"], default="L")
    parser.add_argument("--light-dim", type=int, default=128)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--compile-flex-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-block-size", type=int, default=6)
    parser.add_argument("--examples-per-block", type=int, default=32)
    parser.add_argument("--batch-size", "--render-batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--same-view-fraction", type=float, default=0.25)
    parser.add_argument("--infer-camera-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtract-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-train-lobes", type=int, default=256)
    parser.add_argument("--cache-validation-lobes", type=int, default=32)
    parser.add_argument("--cache-camera-block-size", type=int, default=PILOT_CACHE_CAMERA_BLOCK_SIZE)
    parser.add_argument("--cache-render-batch-size", type=int, default=PILOT_CACHE_RENDER_BATCH_SIZE)
    parser.add_argument("--cache-render-workers", type=int, default=PILOT_CACHE_RENDER_WORKERS)
    parser.add_argument("--background-sample-stride", type=int, default=PILOT_BACKGROUND_SAMPLE_STRIDE)
    parser.add_argument("--cache-seed", type=int, default=31415)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")

    parser.add_argument("--sigma-bins", type=parse_sigma_bins, default=parse_sigma_bins("0.10:0.16,0.16:0.24,0.24:0.36"))
    parser.add_argument("--energy-range", type=float, nargs=2, default=(4.0, 8.0))
    parser.add_argument("--effective-support", type=float, nargs=2, default=(2.0, 20.0))
    parser.add_argument("--min-concentration", type=float, default=0.75)
    parser.add_argument("--prune-threshold", type=float, default=0.05)
    parser.add_argument("--source-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--source-sigma", type=float, default=0.24)
    parser.add_argument("--source-energy", type=float, default=6.0)
    parser.add_argument("--light-floor-fraction", type=float, default=PILOT_LIGHT_FLOOR_FRACTION)

    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=0.0,
        help="LPIPS auxiliary-loss weight; zero preserves the pixel-only recipe.",
    )
    parser.add_argument("--perceptual-net", choices=["alex", "vgg", "squeeze"], default="vgg")
    parser.add_argument("--perceptual-warmup-steps", type=int, default=500)
    parser.add_argument("--val-frequency", type=int, default=500)
    parser.add_argument("--val-examples-per-object", type=int, default=8)
    parser.add_argument("--checkpoint-frequency", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--val-seed", type=int, default=2026)
    args = parser.parse_args()
    args.train_objects = TRAIN_OBJECT_IDS
    args.test_objects = FINAL_TEST_OBJECT_IDS
    if args.cache_dir is None:
        fill_suffix = f"_fill{round(100 * args.light_floor_fraction):02d}" if args.light_floor_fraction else ""
        args.cache_dir = args.data_root / f"pilot_uint8_cache_{args.size}{fill_suffix}"
    if not 0 <= args.same_view_fraction <= 1:
        parser.error("--same-view-fraction must be in [0, 1].")
    if args.size <= 0 or args.size % 32:
        parser.error("--size must be positive and divisible by RayDer's spatial downsample factor (32).")
    if args.energy_range[0] <= 0 or args.energy_range[0] > args.energy_range[1]:
        parser.error("--energy-range must be a positive LOW HIGH pair.")
    if args.effective_support[0] <= 0 or args.effective_support[0] > args.effective_support[1]:
        parser.error("--effective-support must be a positive LOW HIGH pair.")
    if not 0 <= args.min_concentration <= 1 or not 0 <= args.prune_threshold < 1:
        parser.error("--min-concentration must be in [0, 1] and --prune-threshold in [0, 1).")
    if args.light_dim <= 0 or args.examples_per_block <= 0 or args.batch_size <= 0 or args.max_steps <= 0:
        parser.error("Light dimension, batch/example counts, and max steps must be positive.")
    if args.cache_train_lobes <= 0 or args.cache_validation_lobes < args.val_examples_per_object:
        parser.error("Cache lobe counts must be positive and cover --val-examples-per-object.")
    if args.cache_camera_block_size <= 0 or args.cache_render_batch_size <= 0:
        parser.error("Cache camera block size and render batch size must be positive.")
    if args.cache_render_workers <= 0 or args.background_sample_stride <= 0:
        parser.error("Cache render workers and background sample stride must be positive.")
    if (
        args.val_examples_per_object <= 0
        or args.gamma < 0
        or args.beta < 0
        or args.perceptual_weight < 0
        or args.perceptual_warmup_steps < 0
    ):
        parser.error("Validation examples must be positive and loss weights non-negative.")
    if np.linalg.norm(args.source_direction) == 0 or args.source_sigma <= 0 or args.source_energy <= 0:
        parser.error("The source direction must be nonzero and source sigma/energy must be positive.")
    if not 0 <= args.light_floor_fraction < 1:
        parser.error("--light-floor-fraction must be in [0, 1).")
    return args


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        train(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
