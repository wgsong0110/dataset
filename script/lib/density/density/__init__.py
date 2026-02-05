from ._density_cuda import _compute_density_indexed

def compute_density_gmm(pnts, means, covs, features, weights, dist_threshold=4.0):
    """
    User-facing wrapper for the CUDA-based GMM density computation.
    This function currently calls the 'indexed' version, which finds the
    best matching Gaussian for each point.
    """
    return _compute_density_indexed(
        pnts, means, covs, features, weights, dist_threshold
    )

__all__ = ['compute_density_gmm']
