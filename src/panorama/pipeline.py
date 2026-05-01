"""Integrated 5-phase panorama pipeline."""

import cv2
import numpy as np

from .alignment import BaselineHomographyAligner
from .blend import alpha_blend
from .dynamic_mask import estimate_dynamic_mask
from .refine import seam_local_refine
from .risk_map import build_risk_map, depth_discontinuity_from_depth, edge_strength
from .seam import min_vertical_seam, seam_band_mask
from .types import PairSample


def run_full_pipeline(sample: PairSample) -> dict:
    """Run all stages from phase1 to phase5 on a single pair."""
    aligned = BaselineHomographyAligner().run(sample)

    gray = cv2.cvtColor(sample.image_a, cv2.COLOR_BGR2GRAY)
    depth_proxy = gray.astype(np.float32)
    depth_disc = depth_discontinuity_from_depth(depth_proxy)
    edge = edge_strength(gray)

    dyn_mask = estimate_dynamic_mask(sample.image_a, sample.image_b).astype(np.float32)
    semantic_boundary = np.clip(edge + 0.5 * dyn_mask, 0.0, 1.0)

    risk = build_risk_map(depth_disc, semantic_boundary, edge, aligned.residual_flow_mag)
    seam_cost = risk * aligned.overlap_mask.astype(np.float32)
    seam = min_vertical_seam(seam_cost)

    band = seam_band_mask(seam, width=64, img_w=sample.image_a.shape[1])
    refined = seam_local_refine(sample.image_a, aligned.warped_b_to_a, band)
    composed = alpha_blend(sample.image_a, refined, seam, feather=32)

    return {
        "aligned": aligned.warped_b_to_a,
        "risk_map": risk,
        "dynamic_mask": dyn_mask,
        "seam_band": band,
        "refined": refined,
        "composed": composed,
    }
