"""
Dense region scale estimation from nearest neighbor distances.
"""

import torch


def compute_dense_scale(
    nn_distances: torch.Tensor,
    quantile: float = 0.05,
) -> float:
    """
    Compute the dense region scale from nearest neighbor distances.

    Takes a low quantile of the NN distances to estimate the characteristic
    scale in the densest regions of the point cloud.

    Args:
        nn_distances: Nearest neighbor distances, shape (M,).
        quantile: The quantile to use, in range (0, 1). E.g., 0.01 or 0.05 for dense regions.

    Returns:
        The dense scale estimate (d_dense).

    Raises:
        ValueError: If all distances are zero (all points are duplicates).
    """
    # Filter out zero distances (can occur with duplicate points)
    valid_distances = nn_distances[nn_distances > 0]
    if valid_distances.numel() == 0:
        raise ValueError(
            "All nearest neighbor distances are zero. "
            "This typically means all points are duplicates."
        )

    # torch.quantile requires float
    dense_scale = torch.quantile(valid_distances.float(), quantile).item()

    return dense_scale
