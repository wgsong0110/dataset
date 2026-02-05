#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendering Common Functions

공통 렌더링 함수들 - train.py와 render_npz.py에서 공유
중복 코드 제거 및 일관성 있는 렌더링 파이프라인 제공

Created: 2025-12-21
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================================
# Activation Functions
# ============================================================================

def softplus(x: torch.Tensor, bias: float = -1.0) -> torch.Tensor:
    """Softplus activation with bias

    Args:
        x: Input tensor
        bias: Bias value (default: -1.0)

    Returns:
        F.softplus(x + bias)
    """
    return F.softplus(x + bias)

def softplus_inv(y: torch.Tensor, bias: float = -1.0, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of softplus function (numerically stable)

    Args:
        y: Input tensor (output of softplus)
        bias: Bias value (default: -1.0)
        eps: Small epsilon for numerical stability

    Returns:
        x such that softplus(x, bias) ≈ y
    """
    # softplus(x) = log(1 + exp(x))
    # y = log(1 + exp(x + bias))
    # exp(y) = 1 + exp(x + bias)
    # x + bias = log(exp(y) - 1)
    
    # Stable implementation:
    # For large y, log(exp(y) - 1) ≈ y
    y_clamped = y.clamp(min=eps)
    res = torch.where(y_clamped > 20.0, y_clamped, torch.log(torch.exp(y_clamped) - 1 + eps))
    return res - bias

# ============================================================================
# Geometry Functions
# ============================================================================

def quat_to_cov(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Convert quaternion rotation and scale to 3x3 covariance matrix

    Computes: Σ = R @ diag(s²) @ R^T
    where R is rotation matrix from quaternion

    Args:
        quats: (N, 4) quaternion (w, x, y, z)
        scales: (N, 3) scale per axis

    Returns:
        covs: (N, 3, 3) covariance matrices
    """
    eps = 1e-9
    q = quats / (quats.norm(dim=-1, keepdim=True) + eps)
    qw, qx, qy, qz = q.unbind(dim=-1)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)

    R = torch.stack([row0, row1, row2], dim=-2)
    diag = torch.diag_embed(scales * scales)

    return R @ diag @ R.transpose(-1, -2)

# ============================================================================
# Camera and Ray Generation
# ============================================================================

def compute_camera_intrinsics(cam, w: int, h: int, FoVx: Optional[float] = None, FoVy: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Compute camera intrinsic parameters from FoV

    Args:
        cam: Optional Camera object with FoVx, FoVy attributes
        w: Image width
        h: Image height
        FoVx: Optional field of view in x-direction
        FoVy: Optional field of view in y-direction

    Returns:
        fx, fy, cx, cy: Camera intrinsics
    """
    import math
    
    # If FoVy is provided but FoVx is not, compute FoVx from aspect ratio (render_npz style)
    if FoVy is not None and FoVx is None:
        fovx_val = 2 * math.atan((w / h) * math.tan(FoVy / 2))
        fovy_val = FoVy
    elif FoVx is not None and FoVy is not None:
        fovx_val = FoVx
        fovy_val = FoVy
    else:
        if cam is None:
            raise ValueError("Insufficient camera parameters provided.")
        fovx_val = cam.FoVx
        fovy_val = cam.FoVy

    fx = float(w / (2 * math.tan(fovx_val / 2)))
    fy = float(h / (2 * math.tan(fovy_val / 2)))
    cx, cy = float(w / 2), float(h / 2)
    return fx, fy, cx, cy

def generate_rays_full_image(
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    device: torch.device,
    convention: str = 'legacy_render_npz'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate all rays for full image

    Args:
        c2w: (4, 4) camera-to-world transformation matrix
        fx, fy: Focal lengths
        cx, cy: Principal point
        w, h: Image dimensions
        device: torch device
        convention: 'legacy_render_npz' (default), 'opencv', or 'opengl'

    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions (normalized)
    """
    # Create a copy of c2w to avoid modifying original
    c2w_work = c2w.clone()

    if convention.lower() == 'legacy_render_npz':
        # In this project, CameraPool already applies c2w[:3, 1:3] *= -1 during loading for JSON data.
        # Therefore, 'legacy_render_npz' here means we use the c2w as provided by CameraPool,
        # which behaves like OpenCV convention.
        actual_convention = 'opencv'
    else:
        actual_convention = convention.lower()

    # Generate pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )

    if actual_convention == 'opencv':
        # OpenCV: x-right, y-down, z-forward
        dirs = torch.stack([
            (j + 0.5 - cx) / fx,
            (i + 0.5 - cy) / fy,
            torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)
    else:
        # OpenGL: x-right, y-up, z-back (look-at is -z)
        dirs = torch.stack([
            (j + 0.5 - cx) / fx,
            -(i + 0.5 - cy) / fy,
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)

    # Normalize in camera space (to match original and train.py logic)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Transform to world space using (possibly flipped) c2w
    rays_d = torch.sum(dirs[..., None, :] * c2w_work[:3, :3], dim=-1)  # (H, W, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # Normalize
    rays_o = c2w_work[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d

def get_rays_at_pixel(
    camera,
    pixel_x: int,
    pixel_y: int,
    device: torch.device,
    convention: str = 'opencv'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate ray for a specific pixel

    Args:
        camera: Camera object (must have R, T, FoVx, FoVy, image_width, image_height attributes)
        pixel_x: x coordinate (column)
        pixel_y: y coordinate (row)
        device: torch device
        convention: 'opencv' (default) or 'opengl'

    Returns:
        rays_o: (3,) tensor, ray origin
        rays_d: (3,) tensor, ray direction (normalized)
    """
    # 1. Compute intrinsics
    w, h = camera.image_width, camera.image_height
    fx, fy, cx, cy = compute_camera_intrinsics(camera, w, h)

    # 2. Compute C2W matrix
    # GaussRF cameras store W2C (R, T). We need C2W.
    # W2C = [R | T] -> C2W = [R^T | -R^T * T]
    R = camera.R.to(device)
    T = camera.T.to(device)
    
    # C2W rotation is R transpose
    c2w_R = R.transpose(0, 1)
    # C2W translation is -R^T * T
    c2w_T = -torch.matmul(c2w_R, T)
    
    # 3. Generate Camera Space Direction
    if convention == 'opencv':
        # x-right, y-down, z-forward
        x_cam = (pixel_x + 0.5 - cx) / fx
        y_cam = (pixel_y + 0.5 - cy) / fy
        z_cam = 1.0
    else: 
        # opengl: x-right, y-up, z-back
        x_cam = (pixel_x + 0.5 - cx) / fx
        y_cam = -(pixel_y + 0.5 - cy) / fy
        z_cam = -1.0

    dir_cam = torch.tensor([x_cam, y_cam, z_cam], device=device, dtype=torch.float32)
    dir_cam = dir_cam / torch.norm(dir_cam) # Normalize

    # 4. Transform to World Space
    rays_d = torch.matmul(c2w_R, dir_cam)
    rays_d = rays_d / torch.norm(rays_d)
    
    rays_o = c2w_T

    return rays_o, rays_d

# ============================================================================
# Utility Functions
# ============================================================================

def empty_img(h: int, w: int, bg: torch.Tensor) -> torch.Tensor:
    """Create empty image filled with background color

    Args:
        h: Height
        w: Width
        bg: Background color tensor (3,)

    Returns:
        (h, w, 3) image filled with bg color
    """
    return bg.view(1, 1, 3).expand(h, w, 3).clone()


def create_nerf_sampler(near: float, far: float, n_samples: int):
    """Create a NeRF uniform sampler function

    Args:
        near: Near plane distance
        far: Far plane distance
        n_samples: Number of samples per ray

    Returns:
        Sampler function that takes (rays_o, rays_d, training) and returns (t_vals, points)
    """
    def sampler(rays_o: torch.Tensor, rays_d: torch.Tensor, training: bool):
        """Sample points uniformly along rays

        Args:
            rays_o: (n_rays, 3) ray origins
            rays_d: (n_rays, 3) ray directions
            training: whether in training mode

        Returns:
            t_vals: (n_rays, n_samples) sample distances
            points: (n_rays, n_samples, 3) sample points
        """
        n_rays = rays_o.shape[0]
        device = rays_o.device

        # Create uniform samples between near and far
        t_vals = torch.linspace(near, far, n_samples, device=device)
        t_vals = t_vals.expand(n_rays, n_samples)

        # Add random perturbation during training
        if training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # Compute sample points
        points = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]

        return t_vals, points

    return sampler