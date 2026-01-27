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
