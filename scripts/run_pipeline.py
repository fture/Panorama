"""Run a readable, educational panorama pipeline on one image pair."""

import argparse
from pathlib import Path

import cv2

from panorama.pipeline import run_full_pipeline
from panorama.types import PairSample


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

    outputs = run_full_pipeline(sample)

    cv2.imwrite(str(out_dir / "aligned.png"), outputs["aligned"])
    cv2.imwrite(str(out_dir / "risk_map.png"), (outputs["risk_map"] * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "dynamic_mask.png"), (outputs["dynamic_mask"] * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "seam_band.png"), outputs["seam_band"] * 255)
    cv2.imwrite(str(out_dir / "refined.png"), outputs["refined"])
    cv2.imwrite(str(out_dir / "composed.png"), outputs["composed"])


if __name__ == "__main__":
    main()
