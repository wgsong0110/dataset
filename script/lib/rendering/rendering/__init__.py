"""Volume rendering CUDA extension with autograd support."""
from __future__ import annotations
import torch
# import importlib.util # Removed
# import types # Removed
# from importlib import import_module # Removed

# Try to import the compiled CUDA module
try:
    from . import _rendering as _ops
except ImportError:
    # This should ideally not happen if setup.py is correct
    # Fallback to a dummy object or raise a more specific error
    print("Failed to import compiled CUDA rendering operations. Performance will be affected.")
    _ops = types.SimpleNamespace(
        volume_render_forward=None,
        volume_render_backward=None
    )


class VolumeRenderFunction(torch.autograd.Function):
    """
    Custom autograd function for volume rendering.

    Forward: RGB = volume_render(ridxs, ts, sigma, albedo)
    Backward: Computes gradients w.r.t. sigma and albedo
    """

    @staticmethod
    def forward(ctx, ridxs, ts, sigma, albedo, n_rays):
        """
        Forward pass of volume rendering.

        Args:
            ridxs: [N] ray indices (int32, sorted)
            ts: [N] depth values (float32)
            sigma: [N] density values (float32, requires_grad)
            albedo: [N, 3] color values (float32, requires_grad)
            n_rays: int, total number of rays

        Returns:
            out_ridxs: [n_rays] ray indices (0, 1, ..., n_rays-1)
            out_rgbs: [n_rays, 3] rendered RGB colors
        """
        # Call CUDA forward kernel
        out_rgbs, transmittance, alpha, weight = _ops.volume_render_forward(
            ridxs, ts, sigma, albedo, n_rays
        )

        # Save for backward
        ctx.save_for_backward(ridxs, ts, sigma, albedo, transmittance, alpha, weight)
        ctx.n_rays = n_rays

        # Return output (ridxs, rgbs)
        out_ridxs = torch.arange(n_rays, dtype=torch.int32, device=ridxs.device)
        return out_ridxs, out_rgbs

    @staticmethod
    def backward(ctx, grad_ridxs, grad_rgbs):
        """
        Backward pass of volume rendering.

        Args:
            grad_ridxs: gradient w.r.t. out_ridxs (ignored, not differentiable)
            grad_rgbs: [n_rays, 3] gradient w.r.t. out_rgbs

        Returns:
            grad_ridxs: None (not differentiable)
            grad_ts: None (not differentiable)
            grad_sigma: [N] gradient w.r.t. sigma
            grad_albedo: [N, 3] gradient w.r.t. albedo
            grad_n_rays: None (not differentiable)
        """
        ridxs, ts, sigma, albedo, transmittance, alpha, weight = ctx.saved_tensors
        n_rays = ctx.n_rays

        # Call CUDA backward kernel
        grad_sigma, grad_albedo = _ops.volume_render_backward(
            ridxs, ts, sigma, albedo,
            transmittance, alpha, weight,
            grad_rgbs, n_rays
        )

        # Return gradients (None for non-differentiable inputs)
        return None, None, grad_sigma, grad_albedo, None


def volume_render(ridxs, ts, sigma, albedo, n_rays):
    """
    Volume rendering with alpha compositing and autograd support.

    Performs volume rendering using the formula:
        C = Σ T_i * α_i * c_i
    where:
        T_i = exp(-Σ_{j<i} σ_j * δ_j)  (transmittance)
        α_i = 1 - exp(-σ_i * δ_i)       (alpha)
        δ_j = t_{j+1} - t_j              (depth interval)

    Args:
        ridxs: [N] torch.Tensor (int32) - ray indices (MUST BE SORTED)
        ts: [N] torch.Tensor (float32) - depth values
        sigma: [N] torch.Tensor (float32) - density values (differentiable)
        albedo: [N, 3] torch.Tensor (float32) - albedo/color values (differentiable)
        n_rays: int - total number of rays

    Returns:
        out_ridxs: [n_rays] torch.Tensor (int32) - ray indices [0, 1, ..., n_rays-1]
        out_rgbs: [n_rays, 3] torch.Tensor (float32) - rendered RGB colors (differentiable)

    Example:
        >>> ridxs = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device='cuda')
        >>> ts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], dtype=torch.float32, device='cuda')
        >>> sigma = torch.tensor([1.0, 2.0, 1.5, 1.0, 0.5], requires_grad=True, device='cuda')
        >>> albedo = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]],
        ...                        dtype=torch.float32, requires_grad=True, device='cuda')
        >>> ridxs_out, rgbs = volume_render(ridxs, ts, sigma, albedo, n_rays=2)
        >>> loss = rgbs.sum()
        >>> loss.backward()
        >>> print(sigma.grad)  # Gradients computed!
    """
    return VolumeRenderFunction.apply(ridxs, ts, sigma, albedo, n_rays)


__all__ = ['volume_render', 'VolumeRenderFunction']
