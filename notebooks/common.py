"""Shared utilities for the lighting-probe notebooks."""

import os
from pathlib import Path

import numpy as np
import torch
from einops import reduce as einops_reduce, repeat
from scipy.spatial import HalfspaceIntersection
from scipy.stats import rankdata, spearmanr
from transformers import AutoImageProcessor, AutoModel

from rayder.model import Camera, PLUECKER_DIM, RayDer_L, _prepare_temporal_ranks, make_axial_pos_3d


AXIS_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z")
DINO_ID = "devvcamp07/dinov3-mirror"


def select_axis_lights(dataset):
    directions = torch.tensor(dataset.load_light_positions(normalize=True), dtype=torch.float32)
    axes = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    indices = (directions @ axes.T).argmax(0)
    assert indices.unique().numel() == len(AXIS_LABELS)
    return directions, indices, directions[indices]


def sample_mixtures(num_mixtures, seed, concentrations=(0.25, 0.5, 1.0, 2.0)):
    concentrations = torch.tensor(concentrations)
    counts = torch.full((len(concentrations),), num_mixtures // len(concentrations))
    counts[: num_mixtures % len(concentrations)] += 1
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        weights = torch.cat([
            torch.distributions.Dirichlet(torch.full((6,), alpha)).sample((int(count),))
            for alpha, count in zip(concentrations, counts)
        ])
        order = torch.randperm(num_mixtures)
    return weights[order], torch.repeat_interleave(concentrations, counts)[order]


def load_rayder(device):
    from huggingface_hub import hf_hub_download

    checkpoint = os.environ.get("RAYDER_CHECKPOINT") or hf_hub_download("CompVis/rayder", "rayder_l.pt")
    state_dict = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if "camera_tokens_2.weight" in state_dict:
        state_dict["nvs_tokens.weight"] = state_dict.pop("camera_tokens_2.weight")
    model = RayDer_L().eval().requires_grad_(False)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def load_dino(device, model_id=DINO_ID):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().requires_grad_(False).to(device)
    mean = torch.tensor(processor.image_mean, device=device)[None, :, None, None]
    std = torch.tensor(processor.image_std, device=device)[None, :, None, None]
    return model, mean, std


@torch.no_grad()
@torch.compile(dynamic=False, fullgraph=False)
def rayder_features(model, x, temporal_ranks=None):
    """Return cameras, pose-head-normalized tokens, and nuisance projections."""
    B, T, H, W, _ = x.shape
    pos = repeat(
        make_axial_pos_3d(t=T, h=H, w=W, device=x.device),
        "(t h w) c -> b t h w c", b=B, t=T, h=H, w=W,
    ).clone()
    if temporal_ranks is None:
        temporal_ranks = torch.zeros(T, device=x.device, dtype=torch.long)
    ranks = _prepare_temporal_ranks(B, T, x.device, temporal_ranks)
    pos[..., 0] = ranks[:, :, None, None].to(pos)
    tokens = model.camera_tokens(torch.zeros(B, T, device=x.device, dtype=torch.long))
    registers_pos = einops_reduce(pos, "b t h w c -> b t c", "mean")
    view_type = model.view_type_embedding(torch.zeros((B, T), device=x.device, dtype=torch.long))
    _, tokens = model.backbone(
        x=torch.cat([x, x.new_zeros(B, T, H, W, PLUECKER_DIM)], -1),
        pos=pos,
        registers=tokens,
        registers_pos=registers_pos,
        cond_norm=view_type[:, :, None, None] + model.token_type_embedding.weight[0],
        registers_cond_norm=view_type + model.token_type_embedding.weight[1],
    )
    pose_features = model.camera_pose_head.norm(tokens)
    cameras = Camera.from_parameters(
        model.camera_pose_head.out_proj(pose_features).double(),
        model.intrinsics_head(tokens).double(),
    )
    return cameras, pose_features, model.dynamic_state_head(tokens)


def standardize(x):
    return (x - x.mean(0)) / x.std(0).clamp_min(1e-8)


def spearman(x, y):
    return float(spearmanr(x.numpy(), y.numpy()).statistic)


def partial_spearman(x, y, control):
    ranked = [torch.from_numpy(rankdata(value.numpy())).double() for value in (x, y, control)]
    design = torch.column_stack([torch.ones_like(ranked[2]), ranked[2]])
    residuals = [value - design @ torch.linalg.lstsq(design, value).solution for value in ranked[:2]]
    return torch.corrcoef(torch.stack(residuals))[0, 1].item()


def spherical_w1(weights, directions, chunk_size=4096):
    """All pairwise exact W1 distances on a fixed spherical support."""
    cost = torch.acos((directions @ directions.T).clamp(-1, 1))
    cost.fill_diagonal_(0)
    halfspaces = []
    for source in range(len(directions)):
        for target in range(len(directions)):
            if source != target:
                coefficients = torch.zeros(len(directions) - 1)
                if source:
                    coefficients[source - 1] += 1
                if target:
                    coefficients[target - 1] -= 1
                halfspaces.append(torch.cat([coefficients, -cost[source, target, None]]))
    vertices = HalfspaceIntersection(
        torch.stack(halfspaces).numpy(), np.zeros(len(directions) - 1)
    ).intersections
    vertices = torch.column_stack([torch.zeros(len(vertices)), torch.from_numpy(vertices).float()])
    left, right = torch.triu_indices(len(weights), len(weights), offset=1)
    distances = torch.cat([
        ((weights[left[start:stop]] - weights[right[start:stop]]) @ vertices.T).amax(1)
        for start in range(0, len(left), chunk_size)
        for stop in [min(start + chunk_size, len(left))]
    ])
    return distances, cost


def two_factor_energy(features):
    """Sum-to-zero view + lighting + interaction decomposition for (view, mixture, feature)."""
    values = features.double()
    mean = values.mean((0, 1))
    view = values.mean(1) - mean
    light = values.mean(0) - mean
    interaction = values - mean - view[:, None] - light[None]
    energies = torch.stack([
        view.square().sum(-1).mean(),
        light.square().sum(-1).mean(),
        interaction.square().sum(-1).mean(),
    ])
    total = (values - mean).square().sum(-1).mean()
    return energies, (total - energies.sum()).abs() / total


def variance_weighted_r2(actual, predicted):
    residual = (actual - predicted).square().sum()
    total = (actual - actual.mean(0)).square().sum()
    return 1 - residual / total


def dual_ridge_predict(train_kernel, test_train_kernel, targets, alpha=1e-6):
    """Affine kernel-ridge prediction, centered and scaled from training data only."""
    kernel, cross, targets = train_kernel.double(), test_train_kernel.double(), targets.double()
    column_mean, grand_mean = kernel.mean(0), kernel.mean()
    kernel = kernel - kernel.mean(1, keepdim=True) - column_mean[None] + grand_mean
    cross = cross - cross.mean(1, keepdim=True) - column_mean[None] + grand_mean
    scale = kernel.diagonal().mean().clamp_min(1e-12)
    kernel, cross = kernel / scale, cross / scale
    target_mean = targets.mean(0)
    coefficients = torch.linalg.solve(
        kernel + alpha * torch.eye(len(kernel), dtype=kernel.dtype), targets - target_mean,
    )
    return target_mean + cross @ coefficients


def pixel_basis_gram(basis_pixels):
    """Gram matrix of (view, light, pixel) OLAT bases, normalized per pixel channel."""
    bases = basis_pixels.double()
    if bases.ndim == 2:
        bases = bases[None]
    flat = bases.flatten(0, 1)
    return flat @ flat.T / flat.shape[1]


def view_weight_coefficients(weights, num_views):
    """Represent each (view, mixture) in the span of all view-specific OLAT bases."""
    num_lights = weights.shape[1]
    coefficients = weights.new_zeros(num_views, len(weights), num_views * num_lights)
    for view in range(num_views):
        coefficients[view, :, view * num_lights : (view + 1) * num_lights] = weights
    return coefficients


def annotated_heatmap(ax, values, labels, *, title, cmap="viridis", vmin=0, vmax=1):
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(title=title, xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    return image
