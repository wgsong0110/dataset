#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ 렌더링을 위한 공유 유틸리티 함수들

render_npz.py와 train_nerf.py에서 공통으로 사용하는 함수들을 모아둔 모듈.
- NPZ 로딩 (hierarchical GMM)
- Sampling wrappers
"""
import math, torch, numpy as np
from pathlib import Path
from .rendering_common import (
    quat_to_cov,
    softplus,
    softplus_inv,
    empty_img,
    compute_camera_intrinsics,
)

# ============================================================================
# NPZ 로딩
# ============================================================================

def load_npz_with_hierarchy(path, device):
    """NPZ 파일에서 hierarchical GMM 데이터 로드 (Format 4.0)

    Returns:
        means, covs, weights, rgb: 마지막 레벨 데이터
        level_means, level_covs, level_labels: 모든 레벨 데이터
    """
    data = np.load(str(path))
    keys = list(data.keys())
    if 'K' not in keys:
        raise ValueError(f"Format 4.0 NPZ만 지원합니다 ('K' 키 필요)")

    K = int(data['K'])
    depth = int(data['depth'])

    # 마지막 레벨 데이터
    last_level = depth - 1
    means = torch.tensor(data[f'level_{last_level}_means'], dtype=torch.float32, device=device)
    scales = torch.tensor(data[f'level_{last_level}_scales'], dtype=torch.float32, device=device)
    quats = torch.tensor(data[f'level_{last_level}_quats'], dtype=torch.float32, device=device)
    covs = quat_to_cov(quats, scales)
    weights = torch.ones(means.shape[0], dtype=torch.float32, device=device)
    rgb = torch.tensor(data[f'level_{last_level}_rgb'], dtype=torch.float32, device=device)

    # 모든 레벨 로드
    level_means, level_covs = [], []
    for lvl in range(depth):
        level_means.append(torch.tensor(data[f'level_{lvl}_means'], dtype=torch.float32, device=device))
        lvl_scales = torch.tensor(data[f'level_{lvl}_scales'], dtype=torch.float32, device=device)
        lvl_quats = torch.tensor(data[f'level_{lvl}_quats'], dtype=torch.float32, device=device)
        level_covs.append(quat_to_cov(lvl_quats, lvl_scales))

    # Tree 구조
    level_labels = []
    for lvl in range(depth - 1):
        n_children = level_means[lvl + 1].shape[0]
        if K > 0:
            child_to_parent = torch.arange(n_children, dtype=torch.int32, device=device) // K
        else:
            label_key = f'level_{lvl}_to_{lvl+1}_labels'
            child_to_parent = torch.tensor(data[label_key], dtype=torch.int32, device=device)
        level_labels.append(child_to_parent)

    # Leaf 정렬
    if len(level_labels) > 0:
        leaf_to_cluster = level_labels[-1]
    else:
        leaf_to_cluster = torch.zeros(means.shape[0], dtype=torch.int32, device=device)

    sort_idx = torch.argsort(leaf_to_cluster)
    means, covs, weights, rgb = means[sort_idx], covs[sort_idx], weights[sort_idx], rgb[sort_idx]
    level_means[-1], level_covs[-1] = means, covs

    leaf_to_cluster_sorted = leaf_to_cluster[sort_idx]
    if len(level_labels) > 0:
        level_labels[-1] = leaf_to_cluster_sorted

    return means, covs, weights, rgb, level_means, level_covs, level_labels

# Activation functions and utility functions imported from rendering_common
# (see imports at top of file)

# ============================================================================
# Ray Generation
# ============================================================================

def generate_rays(w: int, h: int, fx: float, fy: float, cx: float, cy: float,
                  c2w: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Generate ray directions for all pixels in the image

    Args:
        w, h: image dimensions
        fx, fy: focal lengths
        cx, cy: principal point
        c2w: (4, 4) camera-to-world transformation matrix
        device: torch device

    Returns:
        rays_d: (w*h, 3) normalized ray directions in world space
    """
    # Extract rotation matrix from c2w
    R_c2w = c2w[:3, :3]  # (3, 3)

    # Generate pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Compute normalized image coordinates
    x_norm = (x + 0.5 - cx) / fx
    y_norm = (y + 0.5 - cy) / fy

    # Ray directions in camera space
    dirs_cam = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1)  # (h, w, 3)

    # Normalize
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    # Transform to world space
    dirs_world = torch.matmul(dirs_cam, R_c2w.T)  # (h, w, 3)

    # Normalize again
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    # Flatten to (w*h, 3)
    rays_d = dirs_world.reshape(-1, 3).contiguous()

    return rays_d

# ============================================================================
# Sampling Wrappers
# ============================================================================

def sampling_with_intrinsics(
    method: str,
    level_means,
    level_covs,
    level_labels,
    weights: torch.Tensor,
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    cfg: dict,
    seed: int = None
):
    """Sampling function that accepts pre-computed camera intrinsics

    This is the new unified interface for sampling that accepts pre-computed
    intrinsics instead of a camera object. This enables:
    1. Computing intrinsics once and reusing them
    2. Using custom ray grids (e.g., for ray-batch training)

    Supported Methods:
        - "uniform": Uniform sampling at all Gaussian-ray intersections
        - "importance_uniform": Importance sampling + fixed samples per ray (RECOMMENDED)
        - "adaptive": Adaptive sample budget allocation per ray

    Args:
        method: "uniform", "importance_uniform", or "adaptive"
        level_means, level_covs, level_labels: hierarchical GMM data
        weights: gaussian weights
        c2w: (4, 4) camera-to-world transformation matrix
        fx, fy: focal lengths
        cx, cy: principal point
        w, h: image dimensions
        cfg: config dict with keys: dev, t_min, dt, margin,
             opacity_threshold, mahal_threshold, total_samples, min_samples_per_intersection,
             samples_per_ray (for importance_uniform)

    Returns:
        ridxs, ts, pnts: sampling results
    """
    import sampling  # CUDA extension

    c2w = c2w.to(cfg['dev']).contiguous()
    level_covs_flat = [c.reshape(-1, 9).contiguous() for c in level_covs]

    # Generate rays for all pixels (all methods now use ray-batch mode)
    rays_d = generate_rays(w, h, fx, fy, cx, cy, c2w, cfg['dev'])
    rays_o = c2w[:3, 3].expand(w * h, 3).contiguous()
    n_rays = w * h

    if method == "uniform":
        return sampling.uniform(
            level_means, level_covs_flat, level_labels, weights,
            rays_o, rays_d, n_rays,
            cfg['t_min'], cfg['dt'], cfg['margin'],
            cfg.get('opacity_threshold', 0.95), True, 0.0,
            cfg.get('mahal_threshold', 3.0)
        )
    elif method == "importance_uniform":
        # Use provided seed if given, otherwise generate random seed
        sampling_seed = seed if seed is not None else np.random.randint(0, 2**31)
        return sampling.importance_uniform(
            level_means, level_covs_flat, level_labels, weights,
            rays_o, rays_d, n_rays,
            cfg['t_min'], cfg['dt'], cfg['margin'],
            cfg.get('opacity_threshold', 0.95), True, 0.0,
            cfg.get('mahal_threshold', 3.0),
            cfg.get('samples_per_ray', 100),
            cfg.get('min_samples_per_intersection', 1),
            sampling_seed
        )
    elif method == "adaptive":
        # Use provided seed if given, otherwise generate random seed
        sampling_seed = seed if seed is not None else np.random.randint(0, 2**31)
        return sampling.adaptive(
            level_means, level_covs_flat, level_labels, weights,
            rays_o, rays_d, n_rays,
            cfg['t_min'], cfg['dt'], cfg['margin'],
            cfg.get('opacity_threshold', 0.95), True, 0.0,
            cfg.get('mahal_threshold', 3.0),
            cfg.get('total_budget', w*h*100),
            cfg.get('min_samples_per_ray', 10),
            cfg.get('max_samples_per_ray', 200),
            cfg.get('min_samples_per_intersection', 1),
            sampling_seed
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

def sampling_wrapper(method, level_means, level_covs, level_labels, weights, cam, cfg, seed=None):
    """Sampling wrapper for different methods (backward compatibility wrapper)

    This function maintains backward compatibility by accepting a camera object
    and internally using sampling_with_intrinsics.

    Args:
        method: "uniform", "importance_uniform", or "adaptive"
        level_means, level_covs, level_labels: hierarchical data
        weights: gaussian weights
        cam: camera object (must have c2w, FoVx, FoVy)
        cfg: config dict with keys: dev, w, h, t_min, dt, margin,
             opacity_threshold, mahal_threshold, total_samples, min_samples_per_intersection,
             samples_per_ray (for importance_uniform)
        seed: optional random seed for reproducible sampling (default: None for random)

    Returns:
        ridxs, ts, pnts (all methods now return 3 values)
    """
    # Compute intrinsics from camera
    fx, fy, cx, cy = compute_camera_intrinsics(cam, cfg['w'], cfg['h'])
    c2w = cam.c2w

    # Call the new sampling_with_intrinsics function
    return sampling_with_intrinsics(
        method, level_means, level_covs, level_labels, weights,
        c2w, fx, fy, cx, cy, cfg['w'], cfg['h'], cfg, seed
    )
