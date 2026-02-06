#!/usr/bin/env python3
"""
Render Mahalanobis Distance Intersection Masks

Ray-Gaussian intersection을 Mahalanobis distance threshold별로
binary mask로 시각화하고, GIF 애니메이션으로 저장하는 스크립트.
"""
import argparse
import os
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from script.lib.utils.camera_pool import CameraPool
from script.lib.utils.rendering_common import (
    compute_camera_intrinsics,
    generate_rays_full_image,
    quat_to_cov,
)

# ============================================================================
# 1. Load Leaf Gaussians
# ============================================================================

def load_leaf_gaussians(npz_path: str, device: torch.device):
    """Load leaf-level Gaussians from NPZ and compute covariance matrices.

    Args:
        npz_path: Path to the hierarchical NPZ file.
        device: Torch device.

    Returns:
        means: (G, 3) Gaussian centers.
        covs: (G, 3, 3) Covariance matrices.
        cov_invs: (G, 3, 3) Inverse covariance matrices.
    """
    data = np.load(npz_path, allow_pickle=True)
    depth = int(data["depth"])
    leaf_level = depth - 1

    means_np = data[f"level_{leaf_level}_means"]
    quats_np = data[f"level_{leaf_level}_quats"]
    scales_np = data[f"level_{leaf_level}_scales"]

    means = torch.tensor(means_np, dtype=torch.float32, device=device)
    quats = torch.tensor(quats_np, dtype=torch.float32, device=device)
    scales = torch.tensor(scales_np, dtype=torch.float32, device=device)

    # Covariance: Sigma = R @ diag(s^2) @ R^T
    covs = quat_to_cov(quats, scales)

    # Regularize and invert
    eps = 1e-6
    covs_reg = covs + eps * torch.eye(3, device=device).unsqueeze(0)
    cov_invs = torch.linalg.inv(covs_reg)

    print(f"Loaded {means.shape[0]} leaf Gaussians (level {leaf_level})")
    return means, covs, cov_invs


# ============================================================================
# 2. Ray-Gaussian Mahalanobis Distance
# ============================================================================

def ray_gaussian_mahalanobis(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    means: torch.Tensor,
    cov_invs: torch.Tensor,
) -> torch.Tensor:
    """Compute Mahalanobis distance between rays and Gaussians.

    For each ray (o, d) and Gaussian N(mu, Sigma):
      1. t* = (d^T Sigma_inv (mu - o)) / (d^T Sigma_inv d), clamped >= 0
      2. residual = o + t* d - mu
      3. mahal = sqrt(residual^T Sigma_inv residual)

    Args:
        rays_o: (R, 3) ray origins.
        rays_d: (R, 3) ray directions (normalized).
        means: (G, 3) Gaussian means.
        cov_invs: (G, 3, 3) inverse covariance matrices.

    Returns:
        (R, G) Mahalanobis distances.
    """
    R = rays_o.shape[0]
    G = means.shape[0]

    # diff = mu - o: (R, G, 3)
    diff = means.unsqueeze(0) - rays_o.unsqueeze(1)  # (R, G, 3)

    # Sigma_inv @ diff: (R, G, 3)
    # cov_invs: (G, 3, 3), diff: (R, G, 3)
    Si_diff = torch.einsum("gij,rgj->rgi", cov_invs, diff)  # (R, G, 3)

    # d^T Sigma_inv (mu - o): numerator for t*
    # rays_d: (R, 3) -> (R, 1, 3)
    d_exp = rays_d.unsqueeze(1)  # (R, 1, 3)
    numer = (d_exp * Si_diff).sum(dim=-1)  # (R, G)

    # d^T Sigma_inv d: denominator for t*
    Si_d = torch.einsum("gij,rj->rgi", cov_invs, rays_d)  # (R, G, 3)
    denom = (d_exp * Si_d).sum(dim=-1)  # (R, G)
    denom = denom.clamp(min=1e-10)

    # Optimal t, clamped to non-negative
    t_star = (numer / denom).clamp(min=0.0)  # (R, G)

    # Residual = o + t*d - mu
    # (R, 1, 3) + (R, G, 1) * (R, 1, 3) - (1, G, 3)
    residual = rays_o.unsqueeze(1) + t_star.unsqueeze(-1) * d_exp - means.unsqueeze(0)  # (R, G, 3)

    # Mahalanobis: sqrt(r^T Sigma_inv r)
    Si_res = torch.einsum("gij,rgj->rgi", cov_invs, residual)  # (R, G, 3)
    mahal_sq = (residual * Si_res).sum(dim=-1)  # (R, G)
    mahal_sq = mahal_sq.clamp(min=0.0)
    mahal = torch.sqrt(mahal_sq)

    return mahal


# ============================================================================
# 3. Chunked Min Mahalanobis
# ============================================================================

def compute_min_mahal_chunked(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    means: torch.Tensor,
    cov_invs: torch.Tensor,
    chunk_size: int = 4000,
) -> torch.Tensor:
    """Compute min Mahalanobis distance per ray, processing in chunks.

    Args:
        rays_o: (N, 3) ray origins.
        rays_d: (N, 3) ray directions.
        means: (G, 3) Gaussian means.
        cov_invs: (G, 3, 3) inverse covariance matrices.
        chunk_size: Number of rays per chunk.

    Returns:
        (N,) min Mahalanobis distance per ray.
    """
    N = rays_o.shape[0]
    min_mahal = torch.empty(N, device=rays_o.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_o = rays_o[start:end]
        chunk_d = rays_d[start:end]

        # (chunk, G)
        mahal = ray_gaussian_mahalanobis(chunk_o, chunk_d, means, cov_invs)
        min_mahal[start:end] = mahal.min(dim=-1).values

    return min_mahal


# ============================================================================
# 4. Render Mahalanobis Masks
# ============================================================================

def render_mahal_masks(
    cam,
    means: torch.Tensor,
    cov_invs: torch.Tensor,
    thresholds: list,
    w: int,
    h: int,
    device: torch.device,
    chunk_size: int = 4000,
) -> list:
    """Render binary masks for multiple Mahalanobis thresholds.

    Args:
        cam: Camera object with c2w, FoVx, FoVy.
        means: (G, 3) Gaussian means.
        cov_invs: (G, 3, 3) inverse covariance matrices.
        thresholds: List of threshold values.
        w, h: Output image dimensions.
        device: Torch device.
        chunk_size: Ray chunk size for memory management.

    Returns:
        List of (H, W) uint8 numpy arrays (0 or 255).
    """
    # Intrinsics
    fx, fy, cx, cy = compute_camera_intrinsics(cam, w, h)

    # Generate rays
    c2w = cam.c2w.to(device)
    rays_o, rays_d = generate_rays_full_image(c2w, fx, fy, cx, cy, w, h, device)

    # Flatten
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)

    # Compute min Mahalanobis per ray (single pass)
    min_mahal = compute_min_mahal_chunked(
        rays_o_flat, rays_d_flat, means, cov_invs, chunk_size
    )
    min_mahal_2d = min_mahal.reshape(h, w)

    # Threshold to binary masks
    masks = []
    for thr in thresholds:
        mask = (min_mahal_2d <= thr).cpu().numpy().astype(np.uint8) * 255
        masks.append(mask)

    return masks


# ============================================================================
# 5. Annotate Frame
# ============================================================================

def annotate_frame(mask: np.ndarray, text: str) -> np.ndarray:
    """Overlay threshold text on a mask image.

    Args:
        mask: (H, W) uint8 array.
        text: Text to overlay.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    # Convert grayscale mask to RGB
    img = Image.fromarray(mask).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a larger font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Draw text with background
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 6
    draw.rectangle([4, 4, 4 + tw + 2 * margin, 4 + th + 2 * margin], fill=(0, 0, 0))
    draw.text((4 + margin, 4 + margin), text, fill=(255, 255, 0), font=font)

    return np.array(img)


# ============================================================================
# 6. Save Masks as GIF
# ============================================================================

def save_masks_as_gif(
    masks: list,
    thresholds: list,
    out_dir: str,
    cam_idx: int,
):
    """Save individual PNGs and an animated GIF for one camera.

    Args:
        masks: List of (H, W) uint8 arrays.
        thresholds: Corresponding threshold values.
        out_dir: Base output directory.
        cam_idx: Camera index (for naming).
    """
    cam_name = f"cam_{cam_idx:03d}"
    cam_dir = os.path.join(out_dir, cam_name)
    os.makedirs(cam_dir, exist_ok=True)

    frames = []
    for mask, thr in zip(masks, thresholds):
        # Save individual PNG
        png_path = os.path.join(cam_dir, f"mahal_{thr}.png")
        Image.fromarray(mask).save(png_path)

        # Annotate for GIF
        frame = annotate_frame(mask, f"mahal <= {thr}")
        frames.append(frame)

    # Save GIF
    gif_path = os.path.join(out_dir, f"{cam_name}.gif")
    imageio.mimsave(gif_path, frames, duration=0.6, loop=0)
    print(f"  Saved {gif_path} ({len(frames)} frames)")


# ============================================================================
# 7. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render Mahalanobis distance intersection masks as GIF"
    )
    parser.add_argument(
        "--scene", type=str, default="blender/lego",
        help="Dataset/scene path (e.g. blender/lego)",
    )
    parser.add_argument(
        "--npz_name", type=str, default="512_h8x3.npz",
        help="NPZ filename under <scene>/reduced/",
    )
    parser.add_argument(
        "--width", type=int, default=400,
        help="Output image width",
    )
    parser.add_argument(
        "--height", type=int, default=400,
        help="Output image height",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=4000,
        help="Ray chunk size for memory management",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Parse scene into dataset/scene parts
    parts = args.scene.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"--scene must be 'dataset/scene', got '{args.scene}'")
    dataset, scene = parts

    # Base path: project root (one level up from script/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Load Gaussians ---
    npz_path = os.path.join(project_root, dataset, scene, "reduced", args.npz_name)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    means, covs, cov_invs = load_leaf_gaussians(npz_path, device)

    # --- Load Cameras ---
    cam_pool = CameraPool(
        rst=project_root,
        dataset=dataset,
        scene=scene,
        split="test",
        coordinate_system="blender",
        width=args.width,
        height=args.height,
    )
    print(f"CameraPool: {cam_pool}, {len(cam_pool)} cameras")

    # Select 4 cameras at evenly spaced indices (0, 50, 100, 150 from 200)
    cam_indices = [0, 50, 100, 150]
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    # Output directory
    scene_tag = f"{dataset}_{scene}"
    out_dir = os.path.join(project_root, "rst", "mahal_mask", scene_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # --- Render each camera ---
    for idx in cam_indices:
        print(f"\nRendering camera {idx} ...")
        cam = cam_pool[idx]
        masks = render_mahal_masks(
            cam, means, cov_invs, thresholds,
            args.width, args.height, device, args.chunk_size,
        )
        save_masks_as_gif(masks, thresholds, out_dir, idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
