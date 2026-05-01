"""Run a readable, educational panorama pipeline on one image pair."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from panorama.alignment import BaselineHomographyAligner
from panorama.refine import seam_local_refine
from panorama.risk_map import build_risk_map, depth_discontinuity_from_depth, edge_strength
from panorama.seam import min_vertical_seam, seam_band_mask
from panorama.types import PairSample


def load_gray_depth_proxy(img: np.ndarray) -> np.ndarray:
    """Temporary depth proxy for local testing.

    Real project step: replace with MiDaS/DPT depth prediction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_a", required=True)
    parser.add_argument("--img_b", required=True)
    parser.add_argument("--out_dir", default="outputs/demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = cv2.imread(args.img_a)
    b = cv2.imread(args.img_b)
    if a is None or b is None:
        raise FileNotFoundError("Failed to read input image(s).")

    sample = PairSample(pair_id="demo", image_a=a, image_b=b)

    aligner = BaselineHomographyAligner()
    aligned = aligner.run(sample)

    depth_proxy = load_gray_depth_proxy(a)
    depth_disc = depth_discontinuity_from_depth(depth_proxy)
    edge = edge_strength(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))

    semantic_boundary = edge
    risk = build_risk_map(depth_disc, semantic_boundary, edge, aligned.residual_flow_mag)

    seam_cost = risk * aligned.overlap_mask.astype(np.float32)
    seam = min_vertical_seam(seam_cost)
    band = seam_band_mask(seam, width=64, img_w=a.shape[1])

    refined = seam_local_refine(a, aligned.warped_b_to_a, band)

    cv2.imwrite(str(out_dir / "warped_b.png"), aligned.warped_b_to_a)
    cv2.imwrite(str(out_dir / "risk_map.png"), (risk * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "seam_band.png"), band * 255)
    cv2.imwrite(str(out_dir / "refined.png"), refined)


if __name__ == "__main__":
    main()
