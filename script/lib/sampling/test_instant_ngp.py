#!/usr/bin/env python
"""Test script for instant-ngp sampling integration"""
import sys
import os
# Force using installed version, not local build
sys.path.insert(0, '/opt/conda/envs/gaussrf/lib/python3.9/site-packages')
# Set LD_LIBRARY_PATH for torch libraries
os.environ['LD_LIBRARY_PATH'] = '/opt/conda/envs/gaussrf/lib/python3.9/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import sampling

def test_module_import():
    """Test that the new API is available"""
    print("✓ Module imported successfully")
    assert hasattr(sampling, 'sample_instant_ngp'), "sample_instant_ngp function not found"
    print("✓ sample_instant_ngp function found")

def test_basic_sampling():
    """Test basic sampling with simple Gaussian"""
    device = torch.device('cuda:0')

    # Create simple test data
    n_gaussians = 10
    n_rays = 5

    # Create Gaussians along the ray path
    # Place them at regular intervals in front of the rays
    means = torch.zeros(n_gaussians, 3, device=device)
    for i in range(n_gaussians):
        means[i, 2] = 2.0 + i * 0.5  # Z position: 2.0, 2.5, 3.0, ...
        # Add some variation in X and Y
        means[i, 0] = (i % 3 - 1) * 0.2  # -0.2, 0, 0.2, ...
        means[i, 1] = ((i // 3) % 3 - 1) * 0.2

    # Create covariance matrices (isotropic for simplicity)
    covs = torch.zeros(n_gaussians, 9, device=device)
    for i in range(n_gaussians):
        # Diagonal covariance: sigma^2 * I
        sigma = 0.3  # Larger sigma to ensure intersection
        covs[i, 0] = sigma * sigma  # C[0,0]
        covs[i, 4] = sigma * sigma  # C[1,1]
        covs[i, 8] = sigma * sigma  # C[2,2]

    weights = torch.ones(n_gaussians, device=device)

    # Create rays: simple forward-facing rays
    rays_o = torch.zeros(n_rays, 3, device=device)
    rays_d = torch.zeros(n_rays, 3, device=device)
    rays_d[:, 2] = 1.0  # Point forward in +Z direction

    # Add some variation to ray origins
    rays_o[:, 0] = torch.linspace(-0.3, 0.3, n_rays, device=device)
    rays_o[:, 1] = torch.linspace(-0.3, 0.3, n_rays, device=device)

    print(f"\n📊 Test configuration:")
    print(f"   - Gaussians: {n_gaussians}")
    print(f"   - Rays: {n_rays}")
    print(f"   - Samples per ray: 100")

    try:
        # Call the instant-ngp sampling function
        ridxs, ts, points = sampling.sample_instant_ngp(
            means=means,
            covs=covs,
            weights=weights,
            rays_o=rays_o,
            rays_d=rays_d,
            n_rays=n_rays,
            t_min=0.0,
            mahal_threshold=3.0,
            opacity_threshold=0.99,
            samples_per_ray=100,
            min_samples_per_intersection=1,
            seed=42
        )

        print(f"\n✓ Sampling succeeded!")
        print(f"   - Output samples: {ridxs.shape[0]}")
        print(f"   - Ray indices shape: {ridxs.shape}")
        print(f"   - t values shape: {ts.shape}")
        print(f"   - Points shape: {points.shape}")

        # Basic validation
        assert ridxs.device == device, "Output not on correct device"
        assert ts.device == device, "Output not on correct device"
        assert points.device == device, "Output not on correct device"

        assert ridxs.shape[0] == ts.shape[0], "Mismatched output sizes"
        assert points.shape == (ts.shape[0], 3), "Incorrect points shape"

        assert torch.all(ridxs >= 0) and torch.all(ridxs < n_rays), "Invalid ray indices"
        assert torch.all(ts >= 0.0), "Negative t values"

        print("\n✅ All validation checks passed!")

        # Print some statistics
        if ridxs.shape[0] > 0:
            print(f"\n📈 Statistics:")
            print(f"   - t range: [{ts.min().item():.4f}, {ts.max().item():.4f}]")
            print(f"   - Samples per ray (avg): {ridxs.shape[0] / n_rays:.1f}")
        else:
            print(f"\n⚠️ Warning: No samples generated (no intersections found)")

        return True

    except Exception as e:
        print(f"\n❌ Sampling failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Testing instant-ngp GMM Sampling Integration")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    # Test module import
    test_module_import()

    # Test basic sampling
    success = test_basic_sampling()

    print("\n" + "=" * 70)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)

if __name__ == '__main__':
    main()
