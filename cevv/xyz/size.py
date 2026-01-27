"""
Quantization step size calculation.
"""

from typing import Optional


def compute_step_size(
    dense_scale: float,
    alpha: float = 0.2,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
) -> float:
    """
    Compute quantization step size from dense scale.

    The step size is computed as:
        step_size = alpha * dense_scale

    Args:
        dense_scale: The dense region scale estimate (d_dense), typically from
            compute_dense_scale().
        alpha: Scaling factor for step size (typical range: 0.1 to 0.25).
            Smaller values preserve more detail but increase data size.
        min_step: Optional minimum step size to prevent extremely fine quantization.
        max_step: Optional maximum step size to prevent overly coarse quantization.

    Returns:
        The quantization step size (delta).

    Example:
        >>> from cevv.xyz import compute_nn_distances, compute_dense_scale, compute_step_size
        >>> nn_distances = compute_nn_distances(points, k=1, sample_size=10000)
        >>> dense_scale = compute_dense_scale(nn_distances, quantile=0.05)
        >>> step_size = compute_step_size(dense_scale, alpha=0.2)
    """
    step_size = alpha * dense_scale

    if min_step is not None:
        step_size = max(step_size, min_step)
    if max_step is not None:
        step_size = min(step_size, max_step)

    return step_size
