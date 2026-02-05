"""
NPZ Auto-discovery and Loading Module

This module provides utilities to:
1. Auto-discover NPZ files in scene directories
2. Load NPZ data for sampling, transform, and density modules
"""

import os
import re
from typing import Optional, Dict, List
import numpy as np
import torch
from pathlib import Path


def find_largest_npz(scene_dir: Path, cluster_num: Optional[int] = None) -> Optional[Path]:
    """
    Find the NPZ file with the largest cluster count in the scene directory or its 'reduced' subdirectory.

    Args:
        scene_dir: Path to the scene directory (e.g., Path("rsc/blender/lego"))
        cluster_num: Specific cluster number to look for. If None, finds the largest.

    Returns:
        Absolute path to the NPZ file with the largest/specified cluster count, or None if not found.
    """
    search_dirs = [scene_dir]
    if (scene_dir / "reduced").is_dir():
        search_dirs.append(scene_dir / "reduced")

    npz_files_found = []

    for s_dir in search_dirs:
        if cluster_num is not None:
            # Look for a specific cluster number
            expected_path = s_dir / f"clusters_{cluster_num}.npz"
            if expected_path.is_file():
                return expected_path
        else:
            # Find all clusters_*.npz files
            npz_files_found.extend(s_dir.glob("clusters_*.npz"))

    if cluster_num is not None:
        # If a specific cluster was requested but not found
        return None

    if not npz_files_found:
        return None

    max_cluster_num = -1
    largest_npz_path = None

    for npz_file in npz_files_found:
        try:
            # Extract cluster number from filename (e.g., clusters_5.npz -> 5)
            num_str = npz_file.stem.split('_')[-1]
            current_cluster_num = int(num_str)
            if current_cluster_num > max_cluster_num:
                max_cluster_num = current_cluster_num
                largest_npz_path = npz_file
        except (ValueError, IndexError):
            # Ignore files that don't match the expected pattern
            continue

    return largest_npz_path


def quat_to_cov_upper_tri(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Convert quaternions and scales to upper triangular covariance representation.

    Args:
        quats: [N, 4] quaternions (w, x, y, z)
        scales: [N, 3] scales (sx, sy, sz)

    Returns:
        [N, 6] upper triangular covariance (c00, c01, c02, c11, c12, c22)
    """
    N = quats.shape[0]
    covs = np.zeros((N, 6), dtype=np.float32)

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]

    # Rotation matrix from quaternion
    # R = [[1-2(yy+zz), 2(xy-wz), 2(xz+wy)],
    #      [2(xy+wz), 1-2(xx+zz), 2(yz-wx)],
    #      [2(xz-wy), 2(yz+wx), 1-2(xx+yy)]]

    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - w*z)
    r02 = 2 * (x*z + w*y)
    r10 = 2 * (x*y + w*z)
    r11 = 1 - 2 * (x*x + z*z)
    r12 = 2 * (y*z - w*x)
    r20 = 2 * (x*z - w*y)
    r21 = 2 * (y*z + w*x)
    r22 = 1 - 2 * (x*x + y*y)

    # Scale matrix S = diag(sx, sy, sz)
    # Covariance = R @ S @ S^T @ R^T = R @ diag(sx^2, sy^2, sz^2) @ R^T
    sx2, sy2, sz2 = sx*sx, sy*sy, sz*sz

    # Compute covariance matrix elements
    # C = R @ diag(sx2, sy2, sz2) @ R^T
    c00 = r00*r00*sx2 + r01*r01*sy2 + r02*r02*sz2
    c01 = r00*r10*sx2 + r01*r11*sy2 + r02*r12*sz2
    c02 = r00*r20*sx2 + r01*r21*sy2 + r02*r22*sz2
    c11 = r10*r10*sx2 + r11*r11*sy2 + r12*r12*sz2
    c12 = r10*r20*sx2 + r11*r21*sy2 + r12*r22*sz2
    c22 = r20*r20*sx2 + r21*r21*sy2 + r22*r22*sz2

    covs[:, 0] = c00
    covs[:, 1] = c01
    covs[:, 2] = c02
    covs[:, 3] = c11
    covs[:, 4] = c12
    covs[:, 5] = c22

    return covs


def load_npz_modules_cpu(
    npz_path: Path,
    use_transform: bool = False,
    use_density: bool = False,
    use_sampling: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load NPZ data as numpy arrays (CPU-only, no GPU operations).
    This version is safe for use with instant-ngp which has its own CUDA context.

    Args:
        npz_path: Path to NPZ file
        use_transform: Whether to load data for transform module
        use_density: Whether to load data for density module
        use_sampling: Whether to load data for sampling module

    Returns:
        Dictionary with keys 'transform', 'density', 'sampling' (if requested).
        Each contains numpy arrays needed for that module.

    Example:
        >>> data = load_npz_modules_cpu("scene.npz", use_density=True)
        >>> data['density']['means']  # numpy array on CPU
    """
    # Load NPZ file
    npz_data = np.load(npz_path)

    result = {}

    # Detect format: hierarchical (level_X_*) vs flat (means/covs/weights)
    is_hierarchical = 'level_0_means' in npz_data

    if is_hierarchical:
        # Find the finest level (highest level number with the most Gaussians)
        levels = []
        for key in npz_data.keys():
            if key.startswith('level_') and key.endswith('_means'):
                level_num = int(key.split('_')[1])
                levels.append(level_num)

        finest_level = max(levels)
        level_prefix = f'level_{finest_level}_'

        # Load from hierarchical format
        means = npz_data[level_prefix + 'means']
        scales = npz_data[level_prefix + 'scales']
        quats = npz_data[level_prefix + 'quats']
        weights = npz_data[level_prefix + 'weights']

        # Convert quaternions + scales to covariance
        covs = quat_to_cov_upper_tri(quats, scales)

        # Use weights as features
        features = weights.copy()

        # Check for primitive labels (leaf to cluster mapping)
        if 'primitive_labels' in npz_data:
            leaf_to_cluster = npz_data['primitive_labels'].astype(np.int32)
        else:
            leaf_to_cluster = np.arange(len(means), dtype=np.int32)
    else:
        # Load from flat format
        means = npz_data['means']
        weights = npz_data['weights']

        # Handle covariance
        if 'covs' in npz_data:
            covs = npz_data['covs']
        else:
            # Convert from quats + scales
            scales = npz_data['scales']
            quats = npz_data['quats']
            covs = quat_to_cov_upper_tri(quats, scales)

        # Handle features
        if 'features' in npz_data:
            features = npz_data['features']
        elif 'densities' in npz_data:
            features = npz_data['densities']
        else:
            features = weights.copy()

        # Handle leaf_to_cluster
        if 'leaf_to_cluster' in npz_data:
            leaf_to_cluster = npz_data['leaf_to_cluster']
        else:
            leaf_to_cluster = np.arange(len(means), dtype=np.int32)

    # For density module
    if use_density:
        result['density'] = {
            'means': means,
            'covs': covs,
            'features': features,
            'weights': weights,
            'leaf_to_cluster': leaf_to_cluster,
        }

    # For sampling module
    if use_sampling:
        # Check for hierarchical clustering data
        if 'cluster_start' in npz_data:
            cluster_start = npz_data['cluster_start']
        else:
            cluster_start = np.arange(len(means), dtype=np.int32)

        if 'cluster_count' in npz_data:
            cluster_count = npz_data['cluster_count']
        else:
            cluster_count = np.ones(len(means), dtype=np.int32)

        result['sampling'] = {
            'level_means': [means], # Wrap in list for std::vector<torch::Tensor>
            'level_covs': [covs],   # Wrap in list
            'level_labels': [leaf_to_cluster], # Use leaf_to_cluster as labels
            'leaf_weights': weights, # This is the missing part
        }

    # For transform module
    if use_transform:
        # Check for tree structure
        tree = npz_data['tree'] if 'tree' in npz_data else None

        result['transform'] = {
            'means': means,
            'covs': covs,
            'weights': weights,
            'tree': tree if tree is not None else np.array([]),
        }

    return result


def load_npz_modules(
    npz_path: Path,
    use_transform: bool = False,
    use_density: bool = False,
    use_sampling: bool = False,
    device: str = "cuda"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load NPZ data and prepare it for the specified modules.

    Args:
        npz_path: Path to NPZ file
        use_transform: Whether to load data for transform module
        use_density: Whether to load data for density module
        use_sampling: Whether to load data for sampling module
        device: Device to load tensors to ("cuda" or "cpu")

    Returns:
        Dictionary with keys 'transform', 'density', 'sampling' (if requested).
        Each contains tensors needed for that module.

    Example:
        >>> data = load_npz_modules("scene.npz", use_transform=True, use_density=True)
        >>> data['transform']['means']  # torch.Tensor on CUDA
    """
    # Load NPZ file
    npz_data = np.load(npz_path)

    result = {}

    # Common function to convert numpy to torch
    def to_torch(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    # Load hierarchy data (used by sampling and density)
    if use_sampling or use_density:
        # Load from npz_rendering module's format
        # If npz_rendering is not available, fall back to simple loading
        try:
            from lib.utils.npz_rendering import load_npz_with_hierarchy
            means, covs, weights, rgb, level_means, level_covs, level_labels = load_npz_with_hierarchy(
                npz_path, device=device
            )
            # Ensure level_labels is a list of tensors for consistency with std::vector<torch::Tensor>
            if not isinstance(level_labels, list):
                level_labels = [level_labels]

        except ImportError:
            print("Warning: lib.utils.npz_rendering not found. Falling back to basic NPZ loading.")
            # Basic loading if hierarchical module is not available
            means = to_torch(npz_data['means'])
            weights = to_torch(npz_data['weights'])
            
            if 'covs' in npz_data:
                covs = to_torch(npz_data['covs'])
            else:
                scales = to_torch(npz_data['scales'])
                quats = to_torch(npz_data['quats'])
                covs = to_torch(quat_to_cov_upper_tri(quats.cpu().numpy(), scales.cpu().numpy())) # Convert to numpy for quat_to_cov_upper_tri
            
            rgb = to_torch(npz_data.get('rgb', np.zeros_like(means))) # Fallback for rgb
            
            level_means = [means]
            level_covs = [covs]
            level_labels = [torch.arange(len(means), dtype=torch.int32, device=device)]


        # Build leaf_to_cluster, cluster_start, cluster_count from hierarchy
        N = means.shape[0]
        # For simple NPZ, assume leaf_to_cluster is identity
        leaf_to_cluster_val = torch.arange(N, dtype=torch.int32, device=device)
        
        # leaf_weights might be 'weights' from the NPZ directly
        leaf_weights_val = weights 


        # For sampling
        if use_sampling:
            result['sampling'] = {
                'level_means': level_means,                  # List of [N_l, 3] tensors
                'level_covs': level_covs,                    # List of [N_l, 3, 3] or [N_l, 6] tensors
                'level_labels': level_labels,                # List of [N_l] tensors (primitive labels or cluster indices)
                'leaf_weights': leaf_weights_val,            # [N] tensor
            }

        # For density
        if use_density:
            # Check if features exist in npz_data
            if 'features' in npz_data:
                features = to_torch(npz_data['features'])
            elif 'densities' in npz_data:
                features = to_torch(npz_data['densities'])
            else:
                # Use rgb as features
                features = rgb

            result['density'] = {
                'means': means,
                'covs': covs,
                'features': features,
                'weights': weights,
                'leaf_to_cluster': leaf_to_cluster_val,
            }

    # Load transform data (hierarchical GMM)
    if use_transform:
        # For transform, we need hierarchical structure
        from lib.transform import RosenblattTransform

        # Initialize transform to get the data in proper format
        transform_obj = RosenblattTransform(npz_path=str(npz_path), device=device) # Pass Path as str

        # Extract data from transform object
        # Note: RosenblattTransform should load and store the hierarchical data
        result['transform'] = {
            'means': transform_obj.means if hasattr(transform_obj, 'means') else None,
            'covs': transform_obj.covs if hasattr(transform_obj, 'covs') else None,
            'weights': transform_obj.weights if hasattr(transform_obj, 'weights') else None,
            'tree': transform_obj.tree if hasattr(transform_obj, 'tree') else None,
            'transform_obj': transform_obj,  # Store the object itself
        }

    return result


def validate_npz_data(data: Dict[str, Dict[str, torch.Tensor]]) -> bool:
    """
    Validate that the loaded NPZ data has the required fields.

    Args:
        data: Dictionary returned by load_npz_modules

    Returns:
        True if valid, False otherwise
    """
    required_fields = {
        'sampling': ['level_means', 'level_covs', 'level_labels', 'leaf_weights'],
        'transform': ['means', 'covs', 'weights'],
        'density': ['means', 'covs', 'features', 'weights', 'leaf_to_cluster'],
    }

    for module_name, fields in required_fields.items():
        if module_name in data:
            module_data = data[module_name]
            for field in fields:
                if field not in module_data:
                    print(f"❌ Missing field '{field}' in {module_name} data")
                    return False
                if module_data[field] is None:
                    print(f"❌ Field '{field}' is None in {module_name} data")
                    return False

    return True


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python npz_loader.py <scene_path>")
        sys.exit(1)

    scene_path = Path(sys.argv[1]) # Use Path object

    # Test auto-discovery
    print(f"Searching for NPZ in: {scene_path}")
    npz_path = find_largest_npz(scene_path)

    if npz_path:
        print(f"✅ Found NPZ: {npz_path}")

        # Test loading
        print("\nLoading NPZ data...")
        data = load_npz_modules(
            npz_path,
            use_transform=True,
            use_density=True,
            use_sampling=True,
            device="cuda"
        )

        print("\nLoaded modules:")
        for module_name, module_data in data.items():
            print(f"  {module_name}:")
            for key, value in module_data.items():
                if isinstance(value, (torch.Tensor, list)): # Handle list of tensors
                    if isinstance(value, list):
                        print(f"    {key}: list of {len(value)} tensors, first shape={value[0].shape}, dtype={value[0].dtype}, device={value[0].device}")
                    else:
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"    {key}: {type(value)}")

        # Validate
        if validate_npz_data(data):
            print("\n✅ NPZ data validation passed")
        else:
            print("\n❌ NPZ data validation failed")
    else:
        print(f"❌ No NPZ file found in {scene_path}") # Use scene_path directly
