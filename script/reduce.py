import os, sys, torch, numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

# src.clustering 모듈 import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(PROJECT_ROOT) / 'src'))

import clustering as clustering_lib

C0 = 0.28209479177387814  # SH DC 정규화 상수
COV_EPS = 1e-7  # 공분산 정규화 상수

# SH → RGB 변환
def SH2RGB(sh):
    return sh * C0 + 0.5

# PLY 파일을 torch tensor로 로드
def load_ply_as_tensors(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32)
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1), dtype=torch.float32)
    scales = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1), dtype=torch.float32)
    scales = torch.exp(scales)
    opacities = torch.tensor(v["opacity"], dtype=torch.float32).unsqueeze(1)
    f_cols = [f for f in v.dtype.names if f.startswith("f_")]
    f_cols.sort(key=lambda s: (0 if "f_dc" in s else 1, int(s.split("_")[-1])))
    features = torch.tensor(np.stack([v[c] for c in f_cols], axis=1), dtype=torch.float32)
    return xyz, quats, scales, opacities, features

# Scale, Quat → Covariance 변환 (배치)
def scale_quat_to_cov(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Scale과 Quaternion으로부터 공분산 행렬 계산: Σ = R @ diag(s^2) @ R^T"""
    q = quats / (quats.norm(dim=-1, keepdim=True) + 1e-9)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy),
        2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx),
        2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    S_sq = torch.diag_embed(scales ** 2)
    return R @ S_sq @ R.transpose(-1, -2)

# 클러스터별 Covariance 역계산 (가중 공분산 + primitive 공분산)
def compute_cluster_covariance(means: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, min_variance: float = 1e-6):
    """
    각 클러스터에 속한 점들의 분포로부터 공분산 행렬 계산 (Moment Matching)
    Σ_cluster = Σ_i w_i * [(μ_i - μ_c)(μ_i - μ_c)^T + Σ_i] / Σ_i w_i

    Args:
        means: (N, 3) 원본 gaussian means
        labels: (N,) 클러스터 할당
        centers: (K, 3) 클러스터 중심
        weights: (N,) 가중치
        scales: (N, 3) 원본 gaussian scales
        quats: (N, 4) 원본 gaussian quaternions
        min_variance: 최소 분산 (수치 안정성)

    Returns:
        covs: (K, 3, 3) 각 클러스터의 공분산 행렬 (양정치 보장)
    """
    K = centers.shape[0]
    device = means.device
    covs = torch.zeros(K, 3, 3, device=device)
    labels_long = labels.long()

    # 모든 primitive의 공분산 미리 계산
    prim_covs = scale_quat_to_cov(scales, quats)  # (N, 3, 3)

    for k in range(K):
        mask = (labels_long == k)
        n_points = mask.sum().item()

        if n_points == 0:
            # 빈 클러스터: identity * min_variance
            covs[k] = torch.eye(3, device=device) * min_variance
            continue

        # 가중 공분산 계산
        sub_means = means[mask]  # (n, 3)
        sub_weights = weights[mask]  # (n,)
        sub_prim_covs = prim_covs[mask]  # (n, 3, 3)
        center = centers[k]  # (3,)
        w_sum = sub_weights.sum()

        if n_points == 1:
            # 단일 점: primitive 자체의 공분산 사용
            covs[k] = sub_prim_covs[0]
            continue

        # 중심으로부터의 편차
        diff = sub_means - center.unsqueeze(0)  # (n, 3)

        # 위치 분산: Σ_i w_i * (x_i - μ)(x_i - μ)^T / Σ_i w_i
        weighted_diff = diff * sub_weights.unsqueeze(1).sqrt()  # (n, 3)
        pos_cov = (weighted_diff.T @ weighted_diff) / w_sum  # (3, 3)

        # Primitive 공분산의 가중 평균: Σ_i w_i * Σ_i / Σ_i w_i
        weighted_prim_covs = sub_prim_covs * sub_weights.view(-1, 1, 1)  # (n, 3, 3)
        avg_prim_cov = weighted_prim_covs.sum(dim=0) / w_sum  # (3, 3)

        # Moment Matching: 위치 분산 + primitive 공분산
        cov = pos_cov + avg_prim_cov

        # 대칭화
        cov = (cov + cov.T) / 2

        # 양정치 보장 (eigenvalue clamping)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp(min=min_variance)
        cov = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

        covs[k] = cov

    return covs

# NPZ 저장 (Format 4.0)
def save_tensors_to_npz(out_path: Path, hierarchy, K: int, depth: int, primitive_labels: torch.Tensor):
    """Hierarchy 데이터를 NPZ로 저장 (Format 4.0)"""
    save_dict = {
        'format_version': '4.0',
        'K': K,
        'depth': depth,
        'primitive_labels': primitive_labels.cpu().numpy(),
    }

    for level_idx, (lvl_mu, lvl_scales, lvl_quats, lvl_w, lvl_rgb) in enumerate(hierarchy):
        save_dict[f'level_{level_idx}_means'] = lvl_mu.cpu().numpy()
        save_dict[f'level_{level_idx}_scales'] = lvl_scales.cpu().numpy()
        save_dict[f'level_{level_idx}_quats'] = lvl_quats.cpu().numpy()
        save_dict[f'level_{level_idx}_weights'] = lvl_w.cpu().numpy()
        save_dict[f'level_{level_idx}_rgb'] = lvl_rgb.cpu().numpy()

    np.savez(str(out_path), **save_dict)
    print(f"✅ Saved {depth} levels (K={K}, format=4.0) to {out_path}")

# 메인 함수
def reduce_scene(scene: str, branching_factor: int = 8, depth: int = 1, max_iters: int = 20, min_variance: float = 1e-6, tileB: int = 64, seed: int = 0, distance_metric: str = 'euclidean'):
    # Scene 파싱
    parts = scene.split('/')
    if len(parts) == 2:
        dataset, scene_name = parts
    elif len(parts) == 1:
        scene_name = parts[0]
        dataset = scene_name
    else:
        raise ValueError(f"Scene must be 'scene' or 'dataset/scene', got: {scene}")

    # 경로 설정
    in_path = Path('/data/wgsong/dataset') / dataset / scene_name / 'gs.ply'
    out_dir = Path('/data/wgsong/dataset') / dataset / scene_name / 'reduced'
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 PLY를 찾을 수 없습니다: {in_path}")

    # PLY 로드
    print(f"📂 Loading {in_path}")
    means, quats, scales, opacities, features = load_ply_as_tensors(in_path)
    means = means.cuda().contiguous()
    quats = quats.cuda().contiguous()
    scales = scales.cuda().contiguous()
    opacities = opacities.cuda().contiguous()
    features = features.cuda().contiguous()
    total_points = means.shape[0]

    # 가중치 계산 (opacity 기반)
    opacities_activated = torch.sigmoid(opacities).view(-1)
    weights = opacities_activated.contiguous()

    # Hierarchical K-means
    import time
    final_k = branching_factor ** depth

    if distance_metric == 'euclidean':
        print(f"⚙️ {total_points} → {final_k} 클러스터로 축소 중 (Euclidean K-means, means-based)...")
        print(f"🔄 Running BFS hierarchical K-means (K={branching_factor}, depth={depth}, max_iters={max_iters})...")
        start_time = time.time()

        # Euclidean K-means 호출 (means만 사용)
        cluster_result = clustering_lib.clustering_euclidean_bfs(
            means,
            weights=weights,
            K=branching_factor,
            depth=depth,
            max_iters=max_iters,
            tol=1e-4,
        )

        elapsed = time.time() - start_time
        print(f"✅ Clustering completed in {elapsed:.2f}s ({depth} levels)")

        # 결과 추출
        primitive_labels = cluster_result['primitive_labels']  # (N,) int32
        level_mu = cluster_result['level_mu']  # list of (K^lvl, 3)
        level_cov = None  # Euclidean 모드에서는 수동 계산 필요
        level_w = None

    elif distance_metric == 'w2':
        # Primitive covariance 계산
        print(f"🔧 Computing primitive covariances...")
        covs = scale_quat_to_cov(scales, quats).contiguous()

        print(f"⚙️ {total_points} → {final_k} 클러스터로 축소 중 (W2 K-means, CUDA accelerated)...")
        print(f"🔄 Running BFS hierarchical K-means (K={branching_factor}, depth={depth}, max_iters={max_iters})...")
        start_time = time.time()

        # CUDA 커널 호출 (clustering_bfs)
        cluster_result = clustering_lib.clustering_bfs(
            means, covs,
            weights=weights,
            K=branching_factor,
            depth=depth,
            max_iters=max_iters,
            tol=1e-4,
            tileB=tileB,
            seed=seed,
            reseed_empty=True
        )

        elapsed = time.time() - start_time
        print(f"✅ Clustering completed in {elapsed:.2f}s ({depth} levels)")

        # 결과 추출
        primitive_labels = cluster_result['primitive_labels']  # (N,) int32
        level_mu = cluster_result['level_mu']  # list of (K^lvl, 3)
        level_cov = cluster_result['level_cov']  # list of (K^lvl, 3, 3)
        level_w = cluster_result['level_w']  # list of (K^lvl,)

    else:
        raise ValueError(f"Invalid distance_metric: {distance_metric}. Must be 'euclidean' or 'w2'.")

    # RGB 준비
    features_dc = features[:, :3]
    features_rgb = SH2RGB(features_dc)
    weighted_rgb = features_rgb * weights.unsqueeze(1)
    two_pi_pow = (2.0 * 3.141592653589793) ** (3.0 / 2.0)

    # 각 레벨 처리
    print(f"🔧 Processing {depth} levels...")
    hierarchy_processed = []

    for level_idx in range(depth):
        lvl_centers = level_mu[level_idx]  # (K^lvl, 3)
        lvl_k = lvl_centers.shape[0]

        # Primitive → Level 매핑 계산 (implicit tree 구조)
        if level_idx == depth - 1:
            lvl_labels_long = primitive_labels.long()
        else:
            divisor = branching_factor ** (depth - 1 - level_idx)
            lvl_labels_long = (primitive_labels // divisor).long()

        # Covariance 및 Scale/Quaternion 계산
        if distance_metric == 'euclidean':
            # Euclidean 모드: 수동으로 covariance 계산
            lvl_cov = compute_cluster_covariance(means, lvl_labels_long, lvl_centers, weights, scales, quats, min_variance)
            # CUDA 커널로 변환
            lvl_scales, lvl_quats = clustering_lib.cov_to_scale_quat(lvl_cov)
        else:  # w2
            # W2 모드: 이미 계산된 covariance 사용
            lvl_cov = level_cov[level_idx]  # (K^lvl, 3, 3)
            lvl_scales, lvl_quats = clustering_lib.cov_to_scale_quat(lvl_cov)

        # RGB 계산 (가중 평균)
        lvl_rgb = torch.zeros(lvl_k, 3, device=features.device, dtype=features.dtype)
        lvl_rgb_weight = torch.zeros(lvl_k, device=features.device, dtype=torch.float32)
        lvl_index = lvl_labels_long.unsqueeze(1).expand_as(weighted_rgb)
        lvl_rgb.scatter_add_(0, lvl_index, weighted_rgb)
        lvl_rgb_weight.scatter_add_(0, lvl_labels_long, weights)
        lvl_nonzero = lvl_rgb_weight > 0
        lvl_rgb[lvl_nonzero] /= lvl_rgb_weight[lvl_nonzero].unsqueeze(1)

        # Weight 계산 (opacity sum * sqrt(det(cov)))
        lvl_opacity_sum = torch.zeros(lvl_k, device=opacities_activated.device, dtype=torch.float32)
        lvl_opacity_sum.scatter_add_(0, lvl_labels_long, opacities_activated)
        lvl_det = torch.det(lvl_cov).clamp(min=1e-10)
        lvl_sqrt_det = torch.sqrt(lvl_det)
        lvl_w = lvl_opacity_sum * two_pi_pow * lvl_sqrt_det

        hierarchy_processed.append((lvl_centers, lvl_scales, lvl_quats, lvl_w, lvl_rgb))
        print(f"   Level {level_idx}: {lvl_k} clusters, cov min={lvl_cov.min().item():.6f}, max={lvl_cov.max().item():.6f}")

    # 파일명 결정
    actual_k = hierarchy_processed[-1][0].shape[0]
    metric_suffix = "euc" if distance_metric == 'euclidean' else "w2"
    if depth > 1:
        filename_base = f"{actual_k}_h{branching_factor}x{depth}_{metric_suffix}"
    else:
        filename_base = f"{actual_k}_{metric_suffix}"

    # NPZ 저장
    out_path_npz = out_dir / f"{filename_base}.npz"
    save_tensors_to_npz(out_path_npz, hierarchy_processed, branching_factor, depth, primitive_labels.int())
    print(f"📁 Output: {out_path_npz}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical K-means로 Gaussian Splatting PLY 축소 (Euclidean/W2 거리 지원)")
    parser.add_argument("--scene", type=str, required=True, help="Scene: 'dataset/scene' (예: 'tank-temp/truck')")
    parser.add_argument("--branching_factor", type=int, default=8, help="각 레벨에서 분할할 클러스터 수")
    parser.add_argument("--depth", type=int, default=1, help="분할 깊이 (최종 클러스터 수 = branching_factor^depth)")
    parser.add_argument("--max_iters", type=int, default=20, help="K-means 최대 반복 횟수")
    parser.add_argument("--min_variance", type=float, default=1e-6, help="최소 분산 (공분산 정규화)")
    parser.add_argument("--distance_metric", type=str, default='euclidean', choices=['euclidean', 'w2'],
                        help="거리 메트릭: 'euclidean' (means 기반, 기본값) 또는 'w2' (Wasserstein-2, CUDA)")
    parser.add_argument("--tileB", type=int, default=64, help="CUDA tile size for shared memory (W2 모드만)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for FPS initialization (W2 모드만)")
    args = parser.parse_args()

    reduce_scene(
        args.scene,
        branching_factor=args.branching_factor,
        depth=args.depth,
        max_iters=args.max_iters,
        min_variance=args.min_variance,
        tileB=args.tileB,
        seed=args.seed,
        distance_metric=args.distance_metric
    )
