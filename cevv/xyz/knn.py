"""
K-nearest neighbor distance computation for 3D point clouds.
"""

from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree


def compute_nn_distances(
    points: torch.Tensor,
    k: int = 1,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute k-th nearest neighbor distances for points.

    Uses scipy's cKDTree for efficient O(N log N) computation.
    Optionally samples a subset of points to reduce computation for very large point clouds.

    Args:
        points: Point coordinates, shape (N, 3).
        k: Which nearest neighbor distance to compute (1 = nearest, 2 = second nearest, etc.).
        sample_size: If provided, randomly sample this many points for distance estimation.
            If None or >= N, use all points.
        seed: Random seed for reproducible sampling.

    Returns:
        Tensor of k-th nearest neighbor distances, shape (M,) where M = min(sample_size, N).

    Raises:
        ValueError: If there are not enough points to compute k-th neighbor (n_points <= k).
    """
    device = points.device
    points_np = points.detach().cpu().numpy().astype(np.float64)
    n_points = points_np.shape[0]

    if n_points <= k:
        raise ValueError(
            f"Not enough points to compute {k}-th nearest neighbor. "
            f"Got {n_points} points, need at least {k + 1}."
        )

    # Optionally subsample for large point clouds
    if sample_size is not None and sample_size < n_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, size=sample_size, replace=False)
        query_points = points_np[indices]
    else:
        query_points = points_np

    tree = cKDTree(points_np)
    # Query k+1 neighbors because the first neighbor is the point itself
    distances, _ = tree.query(query_points, k=k + 1)

    # Extract the k-th neighbor distance (index k, since index 0 is self with distance 0)
    nn_distances = distances[:, k]

    return torch.from_numpy(nn_distances).to(device=device, dtype=points.dtype)


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing compute_nn_distances")
    print("=" * 60)

    # Test 1: Simple grid points
    print("\n[Test 1] Simple 2x2x2 grid with spacing 1.0")
    grid_points = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.float32)
    nn_dist = compute_nn_distances(grid_points, k=1)
    print(f"  Points shape: {grid_points.shape}")
    print(f"  NN distances: {nn_dist}")
    print(f"  Expected: all 1.0 (grid spacing)")
    assert torch.allclose(nn_dist, torch.ones(8)), "Test 1 failed!"
    print("  PASSED")

    # Test 2: Different k values
    print("\n[Test 2] k=2 on grid (second nearest neighbor)")
    nn_dist_k2 = compute_nn_distances(grid_points, k=2)
    print(f"  NN distances (k=2): {nn_dist_k2}")
    print(f"  Expected: all 1.0 (each point has multiple neighbors at distance 1)")
    assert torch.allclose(nn_dist_k2, torch.ones(8)), "Test 2 failed!"
    print("  PASSED")

    # Test 3: k=4 should give sqrt(2) for corner points (face diagonal)
    print("\n[Test 3] k=4 on grid (fourth nearest neighbor)")
    nn_dist_k4 = compute_nn_distances(grid_points, k=4)
    print(f"  NN distances (k=4): {nn_dist_k4}")
    print(f"  Expected: all sqrt(2) ~ 1.414 (face diagonal neighbors)")
    # Each corner has 3 edge neighbors at dist 1, then 3 face-diagonal neighbors at sqrt(2)
    assert torch.allclose(nn_dist_k4, torch.full((8,), 2**0.5), atol=1e-5), "Test 3 failed!"
    print("  PASSED")

    # Test 4: Sampling
    print("\n[Test 4] Sampling from larger point cloud")
    large_points = torch.rand(1000, 3)
    nn_dist_full = compute_nn_distances(large_points, k=1)
    nn_dist_sampled = compute_nn_distances(large_points, k=1, sample_size=100, seed=42)
    print(f"  Full cloud: {len(nn_dist_full)} distances")
    print(f"  Sampled: {len(nn_dist_sampled)} distances")
    print(f"  Full mean: {nn_dist_full.mean():.4f}, Sampled mean: {nn_dist_sampled.mean():.4f}")
    assert len(nn_dist_sampled) == 100, "Test 4 failed!"
    print("  PASSED")

    # Test 5: Error on insufficient points
    print("\n[Test 5] Error when n_points <= k")
    try:
        compute_nn_distances(torch.rand(2, 3), k=2)
        print("  FAILED - should have raised ValueError")
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASSED")

    # Test 6: Reproducibility with seed
    print("\n[Test 6] Reproducibility with seed")
    dist1 = compute_nn_distances(large_points, k=1, sample_size=50, seed=123)
    dist2 = compute_nn_distances(large_points, k=1, sample_size=50, seed=123)
    assert torch.allclose(dist1, dist2), "Test 6 failed!"
    print("  Same seed gives same results: PASSED")

    # Test 7: Performance test
    print("\n[Test 7] Performance test")
    for n in [10_000, 100_000, 500_000]:
        points = torch.rand(n, 3)

        start = time.perf_counter()
        _ = compute_nn_distances(points, k=1, sample_size=10000, seed=42)
        elapsed = time.perf_counter() - start
        print(f"  {n:>7,} points (sampled 10k): {elapsed:.3f}s")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
