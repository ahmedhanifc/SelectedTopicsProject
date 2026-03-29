from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SAM2_ROOT = REPO_ROOT / "src" / "sam2"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM2_ROOT))

from analysis_tools.config import get_dataset_config
from analysis_tools.inference_export import INFERENCE_METADATA_COLUMNS, write_markdown_report, write_rows_to_csv
from sam2.build_sam import build_sam2_video_predictor
from src.sam2.eval_sasvi import discover_video_dirs, get_primary_visual_palette, sasvi_inference

from parallelization_and_streaming.parallel_disagreement_sasvi import (
    build_dataset_config,
    build_overseer_model,
    filter_video_entries,
    write_run_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timing wrapper for the original disagreement-gated SASVI pipeline."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--sam2_cfg", type=str, required=True, help="SAM2 config file.")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint.")
    parser.add_argument("--overseer_checkpoint", type=str, required=True, help="Path to Overseer checkpoint.")
    parser.add_argument(
        "--overseer_type",
        type=str,
        required=True,
        choices=["MaskRCNN", "DETR", "Mask2Former"],
        help="Overseer model type.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["CADIS", "CHOLECSEG8K", "CATARACT1K"],
        help="Dataset type.",
    )
    parser.add_argument("--base_video_dir", type=str, required=True, help="Directory containing video frame folders.")
    parser.add_argument(
        "--gt_root_dir",
        type=str,
        default=None,
        help="Optional root directory for ground-truth masks. If omitted, GT masks are searched next to the input frames.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default= REPO_ROOT / "disagreement_frame_outputs",
        help="Root directory for original disagreement pipeline outputs.",
    )
    parser.add_argument(
        "--analysis_output_root",
        type=Path,
        default= REPO_ROOT / "analysis_output_disagreement",
        help="Root directory for original disagreement pipeline analysis outputs.",
    )
    parser.add_argument("--video_name", type=str, default=None, help="Optional single video folder name.")
    parser.add_argument("--video_names", type=str, nargs="+", default=None, help="Optional list of video names.")
    parser.add_argument("--frame_name", type=str, default=None, help="Optional single frame name.")
    parser.add_argument("--start_frame", type=int, default=None, help="Optional first frame index.")
    parser.add_argument("--end_frame", type=int, default=None, help="Optional last frame index.")
    parser.add_argument("--max_videos", type=int, default=None, help="Optional cap on number of videos.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Threshold for SAM2 mask logits.")
    parser.add_argument("--apply_postprocessing", action="store_true", help="Enable SAM2 postprocessing.")
    parser.add_argument("--save_binary_mask", action="store_true", help="Also save binary masks.")
    parser.add_argument("--save_overseer_masks", action="store_true", help="Also export Overseer masks.")
    parser.add_argument(
        "--enable_disagreement_gate",
        action="store_true",
        help="Enable disagreement-gated corrective re-prompting.",
    )
    parser.add_argument(
        "--disagreement_iou_threshold",
        type=float,
        default=0.5,
        help="Bad-frame IoU threshold.",
    )
    parser.add_argument(
        "--disagreement_bad_frames",
        type=int,
        default=2,
        help="Consecutive bad frames needed to trigger disagreement re-prompting.",
    )
    parser.add_argument(
        "--enable_boundary_distance_gate",
        action="store_true",
        help="Also use boundary distance as a disagreement trigger.",
    )
    parser.add_argument(
        "--boundary_distance_threshold",
        type=float,
        default=20.0,
        help="Boundary distance trigger threshold.",
    )
    parser.add_argument(
        "--save_disagreement_visuals",
        action="store_true",
        help="Save disagreement debug overlays.",
    )
    parser.add_argument(
        "--max_disagreement_visuals",
        type=int,
        default=10,
        help="Maximum disagreement visuals to save per video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_name is not None and (args.start_frame is not None or args.end_frame is not None):
        raise ValueError("--frame_name cannot be combined with --start_frame/--end_frame")
    if args.video_name is not None and args.video_names is not None:
        raise ValueError("Use either --video_name or --video_names, not both")
    if args.disagreement_bad_frames < 1:
        raise ValueError("--disagreement_bad_frames must be >= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    video_entries = discover_video_dirs(args.base_video_dir)
    if not video_entries:
        raise RuntimeError(f"No video folders with frames found under {args.base_video_dir}")
    video_entries = filter_video_entries(video_entries, args.video_name, args.video_names, args.max_videos)
    if not video_entries:
        raise RuntimeError("No videos left to process after filtering.")

    cfg = build_dataset_config(args.dataset_type, args.overseer_type)
    gt_dataset_config = get_dataset_config(args.dataset_type)
    overseer_model = build_overseer_model(args, cfg)

    hydra_overrides_extra = ["++model.non_overlap_masks=true"]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    run_timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = args.output_root / run_timestamp
    run_analysis_dir = args.analysis_output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_analysis_dir.mkdir(parents=True, exist_ok=True)
    (run_analysis_dir / "inference").mkdir(parents=True, exist_ok=True)

    output_mask_dir = run_output_dir / "output_masks"
    overseer_mask_dir = run_output_dir / "overseer_masks" if args.save_overseer_masks else None

    write_run_parameters(
        run_analysis_dir / "run_parameters.csv",
        [
            ("run_name", run_timestamp),
            ("base_video_dir", args.base_video_dir),
            ("gt_root_dir", args.gt_root_dir),
            ("output_root", str(run_output_dir)),
            ("analysis_output_dir", str(run_analysis_dir)),
            ("sam2_cfg", args.sam2_cfg),
            ("sam2_checkpoint", args.sam2_checkpoint),
            ("overseer_checkpoint", args.overseer_checkpoint),
            ("overseer_type", args.overseer_type),
            ("dataset_type", args.dataset_type),
            ("device", args.device),
            ("score_thresh", args.score_thresh),
            ("save_binary_mask", args.save_binary_mask),
            ("save_overseer_masks", args.save_overseer_masks),
            ("enable_disagreement_gate", args.enable_disagreement_gate),
            ("disagreement_iou_threshold", args.disagreement_iou_threshold),
            ("disagreement_bad_frames", args.disagreement_bad_frames),
            ("enable_boundary_distance_gate", args.enable_boundary_distance_gate),
            ("boundary_distance_threshold", args.boundary_distance_threshold),
            ("save_disagreement_visuals", args.save_disagreement_visuals),
            ("max_disagreement_visuals", args.max_disagreement_visuals),
            ("video_name", args.video_name),
            ("video_names", " ".join(args.video_names) if args.video_names else ""),
            ("frame_name", args.frame_name),
            ("start_frame", args.start_frame),
            ("end_frame", args.end_frame),
            ("max_videos", args.max_videos),
        ],
    )

    overall_start = time.perf_counter()
    all_trace_rows: list[dict] = []
    video_summaries: list[dict] = []

    print(f"Running original disagreement SASVI on {len(video_entries)} videos")
    for index, (video_name, video_dir) in enumerate(video_entries, start=1):
        print(f"\n[{index}/{len(video_entries)}] Processing {video_name}")
        video_start = time.perf_counter()
        video_rows, video_summary = sasvi_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            output_mask_dir=str(output_mask_dir),
            video_name=video_name,
            video_dir=video_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_name=args.frame_name,
            overseer_type=args.overseer_type,
            overseer_mask_dir=str(overseer_mask_dir) if overseer_mask_dir is not None else None,
            overseer_model=overseer_model,
            nnunet_model=None,
            num_classes=int(cfg["num_classes"]),
            ignore_indices=list(cfg["ignore_indices"]),
            shift_by_1=bool(cfg["shift_by_1"]),
            palette=list(get_primary_visual_palette(int(cfg["num_classes"]))),
            dataset_type=args.dataset_type,
            gt_dataset_config=gt_dataset_config,
            gt_root_dir=args.gt_root_dir,
            score_thresh=args.score_thresh,
            save_binary_mask=args.save_binary_mask,
            analysis_output_dir=str(run_analysis_dir),
            enable_disagreement_gate=args.enable_disagreement_gate,
            disagreement_iou_threshold=args.disagreement_iou_threshold,
            disagreement_bad_frames=args.disagreement_bad_frames,
            enable_boundary_distance_gate=args.enable_boundary_distance_gate,
            boundary_distance_threshold=args.boundary_distance_threshold,
            save_disagreement_visuals=args.save_disagreement_visuals,
            max_disagreement_visuals=args.max_disagreement_visuals,
        )
        video_elapsed = time.perf_counter() - video_start
        frames_processed = int(video_summary.get("frames_processed", 0))
        video_summary["sasvi_seconds"] = video_elapsed
        video_summary["overseer_cache_seconds"] = None
        video_summary["throughput_fps"] = frames_processed / video_elapsed if video_elapsed > 0 else 0.0
        all_trace_rows.extend(video_rows)
        video_summaries.append(video_summary)
        print(
            f"{video_name}: {frames_processed} frames, "
            f"original SASVI {video_elapsed:.3f}s, "
            f"{video_summary['throughput_fps']:.3f} FPS"
        )

    if all_trace_rows:
        write_rows_to_csv(
            run_analysis_dir / "inference" / "inference_metadata.csv",
            all_trace_rows,
            INFERENCE_METADATA_COLUMNS,
        )
        for video_name, _ in video_entries:
            video_rows = [row for row in all_trace_rows if row.get("video_name") == video_name]
            if video_rows:
                write_rows_to_csv(
                    run_analysis_dir / "inference" / f"{video_name}_metadata.csv",
                    [
                        {
                            key: row.get(key)
                            for key in [
                                "video_name",
                                "frame_name",
                                "frame_idx",
                                "num_objects",
                                "confidence_path",
                                "confidence_mean",
                                "confidence_std",
                                "confidence_min",
                                "confidence_max",
                            ]
                        }
                        for row in video_rows
                    ],
                    [
                        "video_name",
                        "frame_name",
                        "frame_idx",
                        "num_objects",
                        "confidence_path",
                        "confidence_mean",
                        "confidence_std",
                        "confidence_min",
                        "confidence_max",
                    ],
                )
        write_markdown_report(
            run_analysis_dir / "inference" / "disagreement_gate_report.md",
            config={
                "device": args.device,
                "dataset_type": args.dataset_type,
                "base_video_dir": args.base_video_dir,
                "gt_root_dir": args.gt_root_dir,
                "video_name": args.video_name,
                "video_names": args.video_names,
                "frame_name": args.frame_name,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "enable_disagreement_gate": args.enable_disagreement_gate,
                "disagreement_iou_threshold": args.disagreement_iou_threshold,
                "disagreement_bad_frames": args.disagreement_bad_frames,
                "enable_boundary_distance_gate": args.enable_boundary_distance_gate,
                "boundary_distance_threshold": args.boundary_distance_threshold,
                "save_disagreement_visuals": args.save_disagreement_visuals,
                "max_disagreement_visuals": args.max_disagreement_visuals,
            },
            video_summaries=video_summaries,
            trace_rows=all_trace_rows,
        )

    overall_elapsed = time.perf_counter() - overall_start
    total_frames = sum(int(item.get("frames_processed", 0)) for item in video_summaries)
    overall_fps = total_frames / overall_elapsed if overall_elapsed > 0 else 0.0

    timing_summary = {
        "base_video_dir": args.base_video_dir,
        "output_root": str(run_output_dir),
        "analysis_output_dir": str(run_analysis_dir),
        "dataset_type": args.dataset_type,
        "overseer_type": args.overseer_type,
        "device": args.device,
        "run_name": run_timestamp,
        "video_count": len(video_summaries),
        "total_frames": total_frames,
        "elapsed_seconds": overall_elapsed,
        "throughput_fps": overall_fps,
        "save_seconds": None,
        "avg_save_ms": None,
        "videos": [
            {
                "video_name": item["video_name"],
                "frames_processed": item["frames_processed"],
                "segments_started": item["segments_started"],
                "class_change_reprompts": item["class_change_reprompts"],
                "disagreement_reprompts": item["disagreement_reprompts"],
                "mean_iou": item["mean_iou"],
                "min_iou": item["min_iou"],
                "overseer_cache_seconds": item.get("overseer_cache_seconds"),
                "sasvi_seconds": item["sasvi_seconds"],
                "throughput_fps": item["throughput_fps"],
            }
            for item in video_summaries
        ],
    }
    timing_path = run_output_dir / "timing_summary.json"
    timing_path.write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")

    print("\nPipeline complete")
    print(f"Output masks: {output_mask_dir}")
    print(f"Analysis output: {run_analysis_dir}")
    print(f"Timing summary: {timing_path}")
    print(f"Total frames processed: {total_frames}")
    print(f"Elapsed time: {overall_elapsed:.3f} seconds")
    print(f"End-to-end throughput: {overall_fps:.3f} FPS")


if __name__ == "__main__":
    main()
