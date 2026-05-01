"""Batch evaluation runner for experiment comparison (Phase 5)."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from panorama.alignment import BaselineHomographyAligner
from panorama.dynamic_mask import estimate_dynamic_mask
from panorama.metrics import overlap_l1, seam_band_l1
from panorama.refine import seam_local_refine
from panorama.risk_map import build_risk_map, depth_discontinuity_from_depth, edge_strength
from panorama.seam import min_vertical_seam, seam_band_mask
from panorama.types import PairSample


def run_pair(path_a: Path, path_b: Path) -> dict:
    a = cv2.imread(str(path_a))
    b = cv2.imread(str(path_b))
    sample = PairSample(pair_id=path_a.stem, image_a=a, image_b=b)
    aligned = BaselineHomographyAligner().run(sample)

    depth_proxy = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    depth_disc = depth_discontinuity_from_depth(depth_proxy)
    edge = edge_strength(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    dyn = estimate_dynamic_mask(a, b).astype(np.float32)

    risk = build_risk_map(depth_disc, edge + 0.5 * dyn, edge, aligned.residual_flow_mag)
    seam = min_vertical_seam(risk * aligned.overlap_mask.astype(np.float32))
    band = seam_band_mask(seam, 64, a.shape[1])
    refined = seam_local_refine(a, aligned.warped_b_to_a, band)

    return {
        "overlap_l1": overlap_l1(a, aligned.warped_b_to_a, aligned.overlap_mask),
        "seam_band_l1_before": seam_band_l1(a, aligned.warped_b_to_a, band),
        "seam_band_l1_after": seam_band_l1(a, refined, band),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_txt", required=True)
    args = p.parse_args()

    lines = [x.strip() for x in Path(args.pairs_txt).read_text().splitlines() if x.strip()]
    rows = []
    for line in lines:
        pa, pb = line.split(',')
        rows.append(run_pair(Path(pa), Path(pb)))

    keys = rows[0].keys()
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        print(f"{k}: mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    main()
