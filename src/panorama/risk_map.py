"""Risk-map generation for seam-aware stitching.

The objective is to mark regions where seam crossing is dangerous:
- depth discontinuity
- semantic/object boundaries
- strong edges
- high residual alignment error
"""

import cv2
import numpy as np


def normalize_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize an array to [0, 1] for stable weighted fusion."""
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)


def depth_discontinuity_from_depth(depth_map: np.ndarray) -> np.ndarray:
    """Compute depth discontinuity as gradient magnitude.

    Intuition: large depth jumps often produce visible stitching artifacts.
    """
    gx = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return normalize_01(np.sqrt(gx * gx + gy * gy))


def edge_strength(gray: np.ndarray) -> np.ndarray:
    """Edge strength map from Canny output."""
    edges = cv2.Canny(gray, 80, 160)
    return normalize_01(edges)


def build_risk_map(
    depth_disc: np.ndarray,
    semantic_boundary: np.ndarray,
    edge_map: np.ndarray,
    residual_map: np.ndarray,
    w_depth: float = 0.30,
    w_sem: float = 0.25,
    w_edge: float = 0.20,
    w_res: float = 0.25,
) -> np.ndarray:
    """Fuse multiple cues into a seam-risk map.

    This is intentionally simple and transparent for student use.
    Later, you can replace it with a trainable fusion network.
    """
    risk = (
        w_depth * normalize_01(depth_disc)
        + w_sem * normalize_01(semantic_boundary)
        + w_edge * normalize_01(edge_map)
        + w_res * normalize_01(residual_map)
    )
    return normalize_01(risk)
