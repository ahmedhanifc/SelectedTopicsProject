#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis_tools.config import get_dataset_config
from analysis_tools.error_analysis import analyze_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV reports and artifacts for SASVi predictions.")
    parser.add_argument("--frames_root", type=Path, required=True, help="Root directory containing input video frames.")
    parser.add_argument("--pred_root", type=Path, required=True, help="Root directory containing predicted masks.")
    parser.add_argument("--gt_root", type=Path, required=True, help="Root directory containing ground-truth masks.")
    parser.add_argument("--output_root", type=Path, required=True, help="Directory to write analysis.csv, summary.json, and artifacts.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["CADIS", "CHOLECSEG8K", "CATARACT1K"], help="Dataset type used for palette decoding.")
    parser.add_argument("--confidence_root", type=Path, default=None, help="Optional root directory containing per-frame confidence maps exported during inference.")
    parser.add_argument("--pred_mask_suffix", type=str, default="_rgb_mask.png", help="Filename suffix for predicted masks.")
    parser.add_argument("--gt_mask_suffix", type=str, default="_rgb_mask.png", help="Filename suffix for ground-truth masks.")
    parser.add_argument("--image_suffix", type=str, default=".jpg", help="Filename suffix for input frames.")
    parser.add_argument("--ignore_index", type=int, default=None, help="Optional override for dataset ignore index.")
    parser.add_argument("--background_index", type=int, default=None, help="Optional override for dataset background index.")
    args = parser.parse_args()

    dataset_config = get_dataset_config(
        dataset_type=args.dataset_type,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
    )

    summary = analyze_predictions(
        frame_root=args.frames_root,
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        dataset_config=dataset_config,
        confidence_root=args.confidence_root,
        pred_mask_suffix=args.pred_mask_suffix,
        gt_mask_suffix=args.gt_mask_suffix,
        image_suffix=args.image_suffix,
    )
    print(f"Analyzed {summary['frames_analyzed']} frame(s) into {args.output_root}")


if __name__ == "__main__":
    main()
