#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GaussRF NPZ Renderer (Legacy Compatible)

Renders images from GMM (NPZ) data using legacy camera conventions.
Supports various sampling methods and visualization modes.
"""

import os
import sys
import math
import torch
import numpy as np
import imageio
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add script directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # dataset directory
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'lib' / 'sampling'))
sys.path.insert(0, str(SCRIPT_DIR / 'lib' / 'rendering'))

from lib.utils.camera_pool import CameraPool
from lib.utils.npz_loader import load_npz_modules, find_largest_npz
from lib.utils.rendering_common import (
    generate_rays_full_image,
    compute_camera_intrinsics,
    empty_img
)
import sampling
import rendering

@torch.no_grad()
def render_frame(cam, gs_data, cfg):
    """Render a single frame matching render_npz.py exactly."""
    device = cfg['device']
    W, H = cam.image_width, cam.image_height
    
    # CameraPool in backup already applied c2w[:3, 1:3] *= -1.
    # We should use cam.c2w as is.
    c2w = cam.c2w.to(device)
    
    # Match backup/GaussRF__/lib/utils/rendering_common.py: compute_camera_intrinsics
    # Use cam.FoVx and cam.FoVy directly
    fx = float(W / (2 * math.tan(cam.FoVx / 2)))
    fy = float(H / (2 * math.tan(cam.FoVy / 2)))
    cx, cy = float(W / 2), float(H / 2)
    
    full_img = torch.zeros((H, W, 3), device=device)
    chunk_rows = cfg.get('chunk_rows', 50)
    
    for row_start in range(0, H, chunk_rows):
        row_end = min(row_start + chunk_rows, H)
        chunk_H = row_end - row_start
        
        # Match backup/GaussRF__/lib/utils/npz_rendering.py: generate_rays
        y, x = torch.meshgrid(
            torch.arange(row_start, row_end, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Add 0.5 offset to match backup/GaussRF__/lib/utils/npz_rendering.py
        x_norm = (x + 0.5 - cx) / fx
        y_norm = (y + 0.5 - cy) / fy
        
        # OpenCV convention: x-right, y-down, z-forward
        dirs_cam = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1) # (chunk_H, W, 3)
        dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)
        
        # Transform to world space
        # R_c2w = c2w[:3, :3]
        rays_d = torch.matmul(dirs_cam, c2w[:3, :3].T)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        
        rays_o_flat = rays_o.reshape(-1, 3).contiguous()
        rays_d_flat = rays_d.reshape(-1, 3).contiguous()
        n_rays_chunk = rays_o_flat.shape[0]
        
        # Sampling
        # Use t_min as dt/t_max placeholder to match the binding's position if necessary
        # However, importance_uniform_cuda in binding.cu expects:
        # t_min, dt, bbox_margin, ...
        # render_npz.py passes cfg['dt'] here. Default dt is 0.02.
        dt_val = cfg.get('dt', 0.02)
        
        ridxs, ts, pnts = sampling.importance_uniform(
            gs_data['level_means_list'],
            gs_data['level_covs_list'],
            gs_data['level_labels_list'],
            gs_data['leaf_weights'],
            rays_o_flat,
            rays_d_flat,
            n_rays_chunk,
            cfg['t_min'], dt_val,
            cfg['bbox_margin'],
            cfg['opacity_threshold'],
            cfg['use_actual_opacity'],
            cfg['weight_threshold'],
            cfg['mahal_threshold'],
            cfg['samples_per_ray'],
            cfg['min_samples_per_intersection'],
            cfg['seed']
        )

        if ridxs.numel() == 0:
            chunk_img = cfg['bg_color'].view(1, 1, 3).expand(chunk_H, W, 3)
        else:
            leaf_idx = -1
            means = gs_data['level_means_list'][leaf_idx]
            covs = gs_data['level_covs_list'][leaf_idx]
            weights = gs_data['leaf_weights']
            rgb = gs_data['level_rgb'][leaf_idx]
            
            from lib.density import compute_density_gmm
            samples_rgb, samples_sigma = compute_density_gmm(pnts, means, covs, rgb, weights, cfg['mahal_threshold'])
            
            sort_idx = torch.argsort(ridxs)
            ridxs_sorted = ridxs[sort_idx].int()
            ts_sorted = ts[sort_idx]
            sigma_sorted = samples_sigma[sort_idx]
            rgb_sorted = samples_rgb[sort_idx]

            _, out_rgb = rendering.volume_render(
                ridxs_sorted, ts_sorted, sigma_sorted, rgb_sorted, n_rays_chunk
            )
            chunk_img = out_rgb.reshape(chunk_H, W, 3)
            
        full_img[row_start:row_end] = chunk_img
        
    return full_img

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Scene: 'dataset/scene' (e.g. 'tank-temp/truck')")
    p.add_argument("--out_dir", default="rst/render")
    p.add_argument("--split", default="test", choices=["train", "test", "val"])
    p.add_argument("--npz_name", type=str, default=None, help="NPZ filename. If None, finds largest.")
    p.add_argument("--gpu", type=int, default=0)
    
    # Camera / Rendering params
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--t_min", type=float, default=0.1)
    p.add_argument("--t_max", type=float, default=10.0)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--bbox_margin", type=float, default=0.0)
    p.add_argument("--bg", default="1,1,1")
    p.add_argument("--convention", default="legacy_render_npz")
    
    # Sampling params
    p.add_argument("--method", default="importance_uniform", choices=["uniform", "importance_uniform", "adaptive"])
    p.add_argument("--samples_per_ray", type=int, default=100)
    p.add_argument("--frame_idx", type=int, default=None, help="Frame index to render. If None, renders all.")
    p.add_argument("--mahal_threshold", type=float, default=3.0)
    p.add_argument("--opacity_threshold", type=float, default=1.0)
    p.add_argument("--weight_threshold", type=float, default=0.0)
    
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    parts = args.scene.split('/')
    dataset = parts[0]
    scene_name = parts[1] if len(parts) > 1 else parts[0]
    
    scene_path = Path('/data/wgsong/dataset') / dataset / scene_name
    if args.npz_name:
        npz_path = scene_path / "reduced" / args.npz_name
    else:
        npz_path = find_largest_npz(scene_path)
        
    if not npz_path or not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found for scene {args.scene}")
        
    print(f"🚀 Loading NPZ with Hierarchy: {npz_path}")
    from lib.utils.npz_rendering import load_npz_with_hierarchy
    
    # This unified loader handles sorting, child-to-parent labels, and full covs
    means, covs, weights, rgb, level_means_list, level_covs_list, level_labels_list = \
        load_npz_with_hierarchy(npz_path, device)
        
    gs_data = {
        'level_means_list': level_means_list,
        'level_covs_list': [c.view(-1, 9).contiguous() for c in level_covs_list], # Ensure [N, 9] for kernels
        'level_labels_list': level_labels_list,
        'level_rgb': [rgb] * len(level_means_list), # Dummy list for interface compatibility
        'leaf_weights': weights
    }
    
    # Final leaf level RGB needs to be specific
    gs_data['level_rgb'][-1] = rgb
    
    print(f"📦 Loaded Hierarchical NPZ. Depth: {len(level_means_list)}, Leaf Gaussians: {len(weights)}")
    
    # 2. Setup Cameras
    cam_pool = CameraPool(rst="/data/wgsong/dataset", dataset=dataset, scene=scene_name, split=args.split, width=args.width, height=args.height)
    print(f"📸 Loaded {len(cam_pool)} cameras from {args.split} split")
    
    # 3. Render Loop
    frames = [args.frame_idx] if args.frame_idx is not None else range(len(cam_pool))
    out_path = PROJECT_ROOT / args.out_dir / f"{dataset}_{scene_name}_{args.method}"
    out_path.mkdir(parents=True, exist_ok=True)
    
    bg_color = torch.tensor([float(v) for v in args.bg.split(",")], device=device)
    
    cfg = {
        'device': device, 'bg_color': bg_color, 't_min': args.t_min, 't_max': args.t_max, 'dt': 0.02,
        'bbox_margin': args.bbox_margin, 'n_samples': args.n_samples,
        'sampling_method': args.method, 'samples_per_ray': args.samples_per_ray,
        'mahal_threshold': args.mahal_threshold, 'opacity_threshold': args.opacity_threshold,
        'weight_threshold': args.weight_threshold, 'use_actual_opacity': False,
        'min_samples_per_intersection': 1, 'seed': 42,
        'total_budget': (args.width or 800) * (args.height or 800) * 100,
        'min_samples_per_ray': 10, 'max_samples_per_ray': 200,
        'camera_convention': args.convention,
        'chunk_rows': 50 # Reduced to be safer
    }
    
    for i in frames:
        if i < 0 or i >= len(cam_pool):
            print(f"⚠️ Frame index {i} out of range [0, {len(cam_pool)-1}]")
            continue
            
        cam = cam_pool[i]
        frame = render_frame(cam, gs_data, cfg)
        
        # Save image
        img_data = (torch.nan_to_num(frame, nan=0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        save_name = out_path / f"{i:05d}.png"
        imageio.imwrite(save_name, img_data)
        print(f"🖼️ Saved frame {i} to: {save_name}")
        
        if len(frames) > 1 and (i + 1) % max(1, len(cam_pool) // 10) == 0:
            print(f"  Progress: {i+1}/{len(cam_pool)}")
            
    print(f"✅ Rendering completed. Results saved to: {out_path}")

if __name__ == "__main__":
    main()
