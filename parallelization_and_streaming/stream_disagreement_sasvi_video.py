from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_tools.config import get_dataset_config
from analysis_tools.inference_export import (
    INFERENCE_METADATA_COLUMNS,
    write_markdown_report,
    write_rows_to_csv,
)
from parallel_disagreement_sasvi import (
    SENTINEL,
    SaveMetrics,
    build_dataset_config,
    build_overseer_model,
    build_predictor,
    parallel_disagreement_inference,
    save_worker,
    write_run_parameters,
)
from stream_sasvi_video import overlay_mask
from src.sam2.eval_sasvi import get_per_obj_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run disagreement-gated SASVI directly on an input video by extracting frames internally and rendering an output mp4."
    )
    parser.add_argument("--input_video", type=Path, required=True, help="Path to input video.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "stream_disagreement_outputs",
        help="Root directory where a timestamped run folder will be created.",
    )
    parser.add_argument(
        "--analysis_output_root",
        type=Path,
        default=REPO_ROOT / "analysis_output_stream_disagreement",
        help="Root directory for analysis artifacts.",
    )
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
        help="Dataset/domain configuration.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Threshold for SAM2 mask logits.")
    parser.add_argument("--render_alpha", type=float, default=0.35, help="Overlay alpha for rendered output.")
    parser.add_argument("--save_binary_mask", action="store_true", help="Also save NPZ binary masks.")
    parser.add_argument("--save_overseer_masks", action="store_true", help="Also export Overseer masks.")
    parser.add_argument("--save_queue_size", type=int, default=32, help="Async save queue size.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Extract every Nth frame from the input video.",
    )
    parser.add_argument(
        "--output_image_ext",
        type=str,
        default=".jpg",
        choices=[".jpg", ".png"],
        help="Image extension for extracted frames.",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="Forwarded to the SAM2 predictor build.",
    )
    parser.add_argument(
        "--enable_disagreement_gate",
        action="store_true",
        help="Enable disagreement-gated corrective re-prompting.",
    )
    parser.add_argument(
        "--disagreement_iou_threshold",
        type=float,
        default=0.5,
        help="Trigger a bad disagreement frame when SAM2 vs Overseer IoU falls below this value.",
    )
    parser.add_argument(
        "--disagreement_bad_frames",
        type=int,
        default=2,
        help="Minimum consecutive bad disagreement frames before corrective re-prompting.",
    )
    parser.add_argument(
        "--enable_boundary_distance_gate",
        action="store_true",
        help="Also allow disagreement triggering using foreground boundary distance.",
    )
    parser.add_argument(
        "--boundary_distance_threshold",
        type=float,
        default=20.0,
        help="Boundary-distance trigger threshold in pixels.",
    )
    parser.add_argument(
        "--save_disagreement_visuals",
        action="store_true",
        help="Save disagreement overlay images.",
    )
    parser.add_argument(
        "--max_disagreement_visuals",
        type=int,
        default=10,
        help="Maximum disagreement visuals to save.",
    )
    parser.add_argument(
        "--overseer_batch_size",
        type=int,
        default=8,
        help="Batch size for the upfront Overseer cache.",
    )
    parser.add_argument(
        "--gt_root_dir",
        type=str,
        help="Optional root directory for GT masks. If omitted, GT metrics are skipped.",
    )
    parser.add_argument(
        "--keep_extracted_frames",
        action="store_true",
        help="Keep the extracted frame directory after the run.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars for Overseer caching.",
    )
    return parser.parse_args()


def extract_video_frames(
    input_video: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int | None,
    image_ext: str,
) -> tuple[int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_frame_idx = 0
    saved_frame_idx = 0

    try:
        while True:
            if max_frames is not None and saved_frame_idx >= max_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_frame_idx % max(frame_stride, 1) != 0:
                source_frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_name = f"{saved_frame_idx:05d}{image_ext}"
            Image.fromarray(frame_rgb).save(output_dir / frame_name)
            saved_frame_idx += 1
            source_frame_idx += 1
    finally:
        capture.release()

    return saved_frame_idx, source_fps


def render_overlay_video(
    *,
    frames_dir: Path,
    output_masks_dir: Path,
    output_video: Path,
    video_name: str,
    render_alpha: float,
    output_fps: float,
    ignore_indices: list[int],
    shift_by_1: bool,
    palette: list[int],
) -> int:
    raw_mask_dir = output_masks_dir / "raw" / video_name
    frame_paths = {
        path.stem: path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if not frame_paths:
        raise RuntimeError(f"No extracted frames found in {frames_dir}")

    mask_paths = sorted(raw_mask_dir.glob("*_rgb_mask.png"))
    if not mask_paths:
        raise RuntimeError(f"No rendered masks found in {raw_mask_dir}")

    writer = None
    rendered = 0
    try:
        for mask_path in mask_paths:
            frame_name = mask_path.name[: -len("_rgb_mask.png")]
            frame_path = frame_paths.get(frame_name)
            if frame_path is None:
                continue
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
            height, width = frame_rgb.shape[:2]
            per_obj_mask = get_per_obj_mask(
                mask_path=str(raw_mask_dir),
                frame_name=frame_name,
                use_binary_mask=False,
                width=width,
                height=height,
                ignore_indices=ignore_indices,
                shift_by_1=shift_by_1,
                palette=palette,
            )
            overlay = overlay_mask(
                frame_rgb=frame_rgb,
                per_obj_mask=per_obj_mask,
                height=height,
                width=width,
                shift_by_1=shift_by_1,
                palette=palette,
                alpha=render_alpha,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            rendered += 1
    finally:
        if writer is not None:
            writer.release()

    return rendered


def maybe_cleanup_frames(frames_root: Path, keep_frames: bool) -> None:
    if keep_frames:
        return
    for path in sorted(frames_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    if args.disagreement_bad_frames < 1:
        raise ValueError("--disagreement_bad_frames must be >= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")

    cfg = build_dataset_config(args.dataset_type, args.overseer_type)
    gt_dataset_config = get_dataset_config(args.dataset_type)
    predictor = build_predictor(args)
    overseer_model = build_overseer_model(args, cfg)
    nnunet_model = None

    run_timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = args.output_root / run_timestamp
    run_analysis_dir = args.analysis_output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_analysis_dir.mkdir(parents=True, exist_ok=True)
    (run_analysis_dir / "inference").mkdir(parents=True, exist_ok=True)

    video_name = args.input_video.stem
    frames_root = run_output_dir / "extracted_frames"
    frames_dir = frames_root / video_name
    output_mask_dir = run_output_dir / "output_masks"
    output_video = run_output_dir / "output_video.mp4"
    overseer_mask_dir = run_output_dir / "overseer_masks" if args.save_overseer_masks else None
    if overseer_mask_dir is not None:
        overseer_mask_dir.mkdir(parents=True, exist_ok=True)

    extraction_start = time.perf_counter()
    extracted_frames, source_fps = extract_video_frames(
        input_video=args.input_video,
        output_dir=frames_dir,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        image_ext=args.output_image_ext,
    )
    extraction_seconds = time.perf_counter() - extraction_start
    if extracted_frames == 0:
        raise RuntimeError(f"No frames were extracted from {args.input_video}")
    output_fps = max(source_fps / max(args.frame_stride, 1), 1.0)

    write_run_parameters(
        run_analysis_dir / "run_parameters.csv",
        [
            ("run_name", run_timestamp),
            ("input_video", str(args.input_video)),
            ("extracted_frames_dir", str(frames_dir)),
            ("output_root", str(run_output_dir)),
            ("analysis_output_dir", str(run_analysis_dir)),
            ("sam2_cfg", args.sam2_cfg),
            ("sam2_checkpoint", args.sam2_checkpoint),
            ("overseer_checkpoint", args.overseer_checkpoint),
            ("overseer_type", args.overseer_type),
            ("dataset_type", args.dataset_type),
            ("device", args.device),
            ("score_thresh", args.score_thresh),
            ("frame_stride", args.frame_stride),
            ("max_frames", args.max_frames),
            ("save_binary_mask", args.save_binary_mask),
            ("save_overseer_masks", args.save_overseer_masks),
            ("enable_disagreement_gate", args.enable_disagreement_gate),
            ("disagreement_iou_threshold", args.disagreement_iou_threshold),
            ("disagreement_bad_frames", args.disagreement_bad_frames),
            ("enable_boundary_distance_gate", args.enable_boundary_distance_gate),
            ("boundary_distance_threshold", args.boundary_distance_threshold),
            ("save_disagreement_visuals", args.save_disagreement_visuals),
            ("max_disagreement_visuals", args.max_disagreement_visuals),
            ("overseer_batch_size", args.overseer_batch_size),
            ("render_alpha", args.render_alpha),
            ("output_fps", output_fps),
            ("gt_root_dir", args.gt_root_dir or ""),
        ],
    )

    save_queue: queue.Queue = queue.Queue(maxsize=max(args.save_queue_size, 1))
    save_stop_event = threading.Event()
    save_exception_queue: queue.Queue = queue.Queue()
    save_metrics = SaveMetrics()
    save_thread = threading.Thread(
        target=save_worker,
        args=(save_queue, save_stop_event, save_metrics, save_exception_queue),
        daemon=True,
    )
    save_thread.start()

    overall_start = time.perf_counter()
    trace_rows: list[dict] = []
    video_summary: dict[str, object] = {}

    try:
        trace_rows, video_summary = parallel_disagreement_inference(
            predictor=predictor,
            base_video_dir=str(frames_root),
            output_mask_dir=str(output_mask_dir),
            video_name=video_name,
            video_dir=str(frames_dir),
            overseer_type=args.overseer_type,
            overseer_mask_dir=str(overseer_mask_dir) if overseer_mask_dir is not None else None,
            overseer_model=overseer_model,
            nnunet_model=nnunet_model,
            num_classes=int(cfg["num_classes"]),
            ignore_indices=list(cfg["ignore_indices"]),
            shift_by_1=bool(cfg["shift_by_1"]),
            palette=list(cfg["palette"]),
            dataset_type=args.dataset_type,
            gt_dataset_config=gt_dataset_config,
            analysis_output_dir=str(run_analysis_dir),
            save_queue=save_queue,
            save_stop_event=save_stop_event,
            start_frame=None,
            end_frame=None,
            frame_name=None,
            gt_root_dir=args.gt_root_dir,
            score_thresh=args.score_thresh,
            save_binary_mask=args.save_binary_mask,
            enable_disagreement_gate=args.enable_disagreement_gate,
            disagreement_iou_threshold=args.disagreement_iou_threshold,
            disagreement_bad_frames=args.disagreement_bad_frames,
            enable_boundary_distance_gate=args.enable_boundary_distance_gate,
            boundary_distance_threshold=args.boundary_distance_threshold,
            save_disagreement_visuals=args.save_disagreement_visuals,
            max_disagreement_visuals=args.max_disagreement_visuals,
            overseer_batch_size=args.overseer_batch_size,
            disable_tqdm=args.disable_tqdm,
        )

        save_queue.put(SENTINEL)
        save_thread.join()
        if not save_exception_queue.empty():
            raise save_exception_queue.get()

        render_start = time.perf_counter()
        rendered_frames = render_overlay_video(
            frames_dir=frames_dir,
            output_masks_dir=output_mask_dir,
            output_video=output_video,
            video_name=video_name,
            render_alpha=args.render_alpha,
            output_fps=output_fps,
            ignore_indices=list(cfg["ignore_indices"]),
            shift_by_1=bool(cfg["shift_by_1"]),
            palette=list(cfg["palette"]),
        )
        render_seconds = time.perf_counter() - render_start
    except Exception:
        save_stop_event.set()
        try:
            save_queue.put_nowait(SENTINEL)
        except queue.Full:
            pass
        save_thread.join(timeout=2.0)
        raise

    total_elapsed = time.perf_counter() - overall_start
    save_summary = save_metrics.summary()

    metadata_path = run_analysis_dir / "inference" / "inference_metadata.csv"
    write_rows_to_csv(metadata_path, trace_rows, INFERENCE_METADATA_COLUMNS)
    write_markdown_report(
        run_analysis_dir / "inference" / "disagreement_gate_report.md",
        config={
            "input_video": str(args.input_video),
            "dataset_type": args.dataset_type,
            "device": args.device,
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "enable_disagreement_gate": args.enable_disagreement_gate,
            "disagreement_iou_threshold": args.disagreement_iou_threshold,
            "disagreement_bad_frames": args.disagreement_bad_frames,
            "enable_boundary_distance_gate": args.enable_boundary_distance_gate,
            "boundary_distance_threshold": args.boundary_distance_threshold,
            "save_disagreement_visuals": args.save_disagreement_visuals,
            "max_disagreement_visuals": args.max_disagreement_visuals,
            "overseer_batch_size": args.overseer_batch_size,
        },
        video_summaries=[video_summary],
        trace_rows=trace_rows,
    )

    timing_summary = {
        "run_name": run_timestamp,
        "input_video": str(args.input_video),
        "dataset_type": args.dataset_type,
        "overseer_type": args.overseer_type,
        "device": args.device,
        "frame_stride": args.frame_stride,
        "extracted_frames": extracted_frames,
        "rendered_frames": rendered_frames,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "extraction_seconds": extraction_seconds,
        "sasvi_total_seconds": total_elapsed,
        "render_seconds": render_seconds,
        "save_seconds": save_summary["save_seconds"],
        "save_tasks": save_summary["save_tasks"],
        "throughput_fps": rendered_frames / total_elapsed if total_elapsed > 0 else 0.0,
        "video_summary": video_summary,
    }
    with (run_output_dir / "timing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(timing_summary, handle, indent=2)

    maybe_cleanup_frames(frames_root, args.keep_extracted_frames)

    print("\nPipeline complete")
    print(f"Input video: {args.input_video}")
    print(f"Rendered video: {output_video}")
    print(f"Mask directory: {output_mask_dir / 'raw' / video_name}")
    print(f"Analysis directory: {run_analysis_dir}")
    print(f"Extracted frames: {extracted_frames}")
    print(f"Rendered frames: {rendered_frames}")
    print(f"Elapsed time: {total_elapsed:.3f} seconds")
    print(f"End-to-end throughput: {timing_summary['throughput_fps']:.3f} FPS")


if __name__ == "__main__":
    main()
