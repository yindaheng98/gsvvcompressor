"""
XYZ coordinate quantization and dequantization.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .knn import compute_nn_distances
from .dense import compute_dense_scale
from .size import compute_step_size


@dataclass
class XYZQuantConfig:
    """
    Configuration for XYZ coordinate quantization.

    Attributes:
        step_size: The quantization step size (delta).
        origin: The origin offset for quantization (median of coordinates).
    """
    step_size: float
    origin: torch.Tensor  # shape (3,)


def compute_quant_config(
    points: torch.Tensor,
    k: int = 1,
    sample_size: Optional[int] = 10000,
    seed: Optional[int] = 42,
    quantile: float = 0.05,
    alpha: float = 0.2,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
) -> XYZQuantConfig:
    """
    Compute quantization configuration from points.

    Internally computes step_size via:
        1. compute_nn_distances() - k-th nearest neighbor distances
        2. compute_dense_scale() - low quantile of NN distances
        3. compute_step_size() - alpha * dense_scale

    Args:
        points: Point coordinates, shape (N, 3).
        k: Which nearest neighbor to use (1 = nearest).
        sample_size: Number of points to sample for NN estimation.
            Set to None to use all points. Default 10000 balances speed and accuracy.
        seed: Random seed for reproducible sampling.
        quantile: Quantile of NN distances to use (e.g., 0.01 or 0.05).
            Lower values target denser regions.
        alpha: Scaling factor for step size (typical range: 0.1 to 0.25).
            Smaller values preserve more detail but increase data size.
        min_step: Optional minimum step size to prevent extremely fine quantization.
        max_step: Optional maximum step size to prevent overly coarse quantization.

    Returns:
        XYZQuantConfig with step size and origin (median of coordinates).
    """
    nn_distances = compute_nn_distances(points, k=k, sample_size=sample_size, seed=seed)
    dense_scale = compute_dense_scale(nn_distances, quantile=quantile)
    step_size = compute_step_size(dense_scale, alpha=alpha, min_step=min_step, max_step=max_step)
    origin = points.median(dim=0).values
    return XYZQuantConfig(step_size=step_size, origin=origin)


def quantize_xyz(
    points: torch.Tensor,
    config: XYZQuantConfig,
) -> torch.Tensor:
    """
    Quantize XYZ coordinates using the given configuration.

    Applies the formula:
        quantized = round((points - origin) / step_size)

    Args:
        points: Point coordinates, shape (N, 3).
        config: Quantization configuration (step size and origin).

    Returns:
        Quantized coordinates as integers, shape (N, 3), dtype torch.int32.
    """
    normalized = (points - config.origin) / config.step_size
    quantized = torch.round(normalized).to(torch.int32)
    return quantized


def dequantize_xyz(
    quantized: torch.Tensor,
    config: XYZQuantConfig,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize XYZ coordinates back to floating point.

    Applies the formula:
        points = quantized * step_size + origin

    Args:
        quantized: Quantized coordinates, shape (N, 3), dtype torch.int32.
        config: Quantization configuration (step size and origin).
        dtype: Output dtype for the dequantized coordinates.

    Returns:
        Dequantized coordinates, shape (N, 3).
    """
    return quantized.to(dtype) * config.step_size + config.origin


def estimate_quantization_error(
    points: torch.Tensor,
    config: XYZQuantConfig,
) -> dict:
    """
    Estimate the quantization error statistics.

    Useful for debugging and validating quantization parameters.

    Args:
        points: Original point coordinates, shape (N, 3).
        config: Quantization configuration.

    Returns:
        Dictionary with error statistics:
            - 'max_error': Maximum absolute error across all coordinates.
            - 'mean_error': Mean absolute error.
            - 'rmse': Root mean squared error.
            - 'relative_max_error': Max error relative to step size.
    """
    quantized = quantize_xyz(points, config)
    reconstructed = dequantize_xyz(quantized, config, dtype=points.dtype)

    error = (points - reconstructed).abs()

    return {
        'max_error': error.max().item(),
        'mean_error': error.mean().item(),
        'rmse': torch.sqrt((error ** 2).mean()).item(),
        'relative_max_error': error.max().item() / config.step_size,
    }
