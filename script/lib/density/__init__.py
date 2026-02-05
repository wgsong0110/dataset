"""GMM density computation CUDA extension."""

from __future__ import annotations

import torch
import importlib.util
from importlib import import_module

try:
    _ops = import_module("._density_cuda", __name__)
except ModuleNotFoundError:
    lite_spec = importlib.util.find_spec("density")
    if lite_spec is not None and lite_spec.origin.endswith(".so"):
        import types
        _ops = types.SimpleNamespace(_compute_density_indexed=None, _compute_density_clustered=None)
    else:
        raise

# Python wrapper: 통합 API (기본값: indexed 모드)
def compute_density_gmm(pnts, means, covs, features, weights, dist_threshold=4.0,
                        clustered=False, primitive_idx=None, leaf_to_cluster=None, cluster_start=None, cluster_count=None):
    """GMM density 계산.

    Args:
        clustered: False면 indexed 모드 (각 sample에 대해 best gaussian 자동 탐지),
                   True면 clustered 모드 (cluster 내 모든 gaussian 합산, primitive_idx 필요)
    """
    if clustered:
        if primitive_idx is None or leaf_to_cluster is None or cluster_start is None or cluster_count is None:
            raise ValueError("clustered=True requires primitive_idx, leaf_to_cluster, cluster_start, cluster_count")
        return _ops._compute_density_clustered(pnts, means, covs, features, weights,
                                                primitive_idx, leaf_to_cluster, cluster_start, cluster_count, dist_threshold)
    else:
        return _ops._compute_density_indexed(pnts, means, covs, features, weights, dist_threshold)

__all__ = ['compute_density_gmm']
