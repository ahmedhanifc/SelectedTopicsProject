from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "sam2"))

from sam2.build_sam import build_sam2_video_predictor
from src.sam2.eval_sasvi import save_masks_to_dir
from parallelization_and_streaming.stream_sasvi_video import (
    SENTINEL,
    DatasetConfig,
    Metrics,
    OverseerRuntime,
    dataset_config,
    overlay_mask,
    put_with_retry,
    worker_wrapper,
)


@dataclass
class FramePacket:
    frame_idx: int
    frame_name: str
    frame_rgb: np.ndarray


@dataclass
class OverseerPacket:
    frame_idx: int
    frame_name: str
    frame_rgb: np.ndarray
    per_obj_mask: dict[int, np.ndarray]


@dataclass
class OutputPacket:
    frame_idx: int
    frame_name: str
    frame_rgb: np.ndarray
    per_obj_mask: dict[int, np.ndarray]


def list_video_dirs(base_video_dir: Path) -> list[Path]:
    return sorted([path for path in base_video_dir.iterdir() if path.is_dir()])


def list_frame_names(video_dir: Path, dataset_type: str) -> list[str]:
    frame_names = [
        path.stem
        for path in video_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if dataset_type.upper() == "CADIS":
        frame_names.sort(key=lambda name: int(name[5:]))
    else:
        frame_names.sort(key=lambda name: int(name))
    return frame_names


def stage_a_load_frames(
    frame_paths: list[Path],
    frame_names: list[str],
    output_queue: queue.Queue,
    metrics: Metrics,
    max_frames: int | None,
    stop_event: threading.Event,
) -> None:
    limit = len(frame_paths) if max_frames is None else min(len(frame_paths), max_frames)
    try:
        for frame_idx in range(limit):
            if stop_event.is_set():
                break

            start = time.perf_counter()
            frame_bgr = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read frame '{frame_paths[frame_idx]}'")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            metrics.add_stage("capture", time.perf_counter() - start)

            while not stop_event.is_set():
                try:
                    output_queue.put(
                        FramePacket(
                            frame_idx=frame_idx,
                            frame_name=frame_names[frame_idx],
                            frame_rgb=frame_rgb,
                        ),
                        timeout=0.2,
                    )
                    break
                except queue.Full:
                    continue
    finally:
        put_with_retry(output_queue, SENTINEL, stop_event)


def stage_b_overseer(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    runtime: OverseerRuntime,
    gpu_lock: threading.Lock,
    metrics: Metrics,
    stop_event: threading.Event,
) -> None:
    while True:
        if stop_event.is_set():
            put_with_retry(output_queue, SENTINEL, stop_event)
            return
        try:
            packet = input_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if packet is SENTINEL:
            put_with_retry(output_queue, SENTINEL, stop_event)
            return

        start = time.perf_counter()
        with gpu_lock:
            per_obj_mask = runtime.predict(
                packet.frame_rgb,
                reshape_size=(packet.frame_rgb.shape[1], packet.frame_rgb.shape[0]),
            )
        metrics.add_stage("overseer", time.perf_counter() - start)

        while not stop_event.is_set():
            try:
                output_queue.put(
                    OverseerPacket(
                        frame_idx=packet.frame_idx,
                        frame_name=packet.frame_name,
                        frame_rgb=packet.frame_rgb,
                        per_obj_mask=per_obj_mask,
                    ),
                    timeout=0.2,
                )
                break
            except queue.Full:
                continue


def stage_c_sam2(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    video_dir: Path,
    predictor,
    num_classes: int,
    score_thresh: float,
    gpu_lock: threading.Lock,
    metrics: Metrics,
    stop_event: threading.Event,
    offload_state_to_cpu: bool,
    sam2_window: int,
) -> None:
    with gpu_lock:
        inference_state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=True,
        )
        for obj_id in range(num_classes):
            predictor._obj_id_to_idx(inference_state, obj_id)

    processed_in_window = 0

    while True:
        if stop_event.is_set():
            put_with_retry(output_queue, SENTINEL, stop_event)
            return
        try:
            packet = input_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if packet is SENTINEL:
            put_with_retry(output_queue, SENTINEL, stop_event)
            return

        start = time.perf_counter()
        with gpu_lock:
            if sam2_window > 0 and processed_in_window >= sam2_window:
                predictor.reset_state(inference_state)
                for obj_id in range(num_classes):
                    predictor._obj_id_to_idx(inference_state, obj_id)
                processed_in_window = 0

            if packet.per_obj_mask:
                for obj_id, object_mask in packet.per_obj_mask.items():
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=packet.frame_idx,
                        obj_id=int(obj_id),
                        mask=object_mask,
                    )
            else:
                dummy_mask = np.zeros(packet.frame_rgb.shape[:2], dtype=bool)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=packet.frame_idx,
                    obj_id=0,
                    mask=dummy_mask,
                )

            propagated = list(
                predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=packet.frame_idx,
                    max_frame_num_to_track=0,
                    reverse=False,
                )
            )

        current = next((item for item in propagated if item[0] == packet.frame_idx), propagated[-1])
        _, out_obj_ids, out_mask_logits = current
        per_obj_output_mask = {
            int(out_obj_id): (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        processed_in_window += 1
        metrics.add_stage("sam2", time.perf_counter() - start)

        while not stop_event.is_set():
            try:
                output_queue.put(
                    OutputPacket(
                        frame_idx=packet.frame_idx,
                        frame_name=packet.frame_name,
                        frame_rgb=packet.frame_rgb,
                        per_obj_mask=per_obj_output_mask,
                    ),
                    timeout=0.2,
                )
                break
            except queue.Full:
                continue


def stage_d_save_masks(
    input_queue: queue.Queue,
    output_mask_dir: Path,
    overlay_dir: Path | None,
    video_name: str,
    save_binary_mask: bool,
    cfg: DatasetConfig,
    render_alpha: float,
    metrics: Metrics,
    total_frames: int | None,
    progress_interval: int,
    disable_tqdm: bool,
    stop_event: threading.Event,
) -> None:
    if overlay_dir is not None:
        (overlay_dir / video_name).mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(
        total=total_frames,
        desc=video_name,
        unit="frame",
        leave=True,
        disable=disable_tqdm,
    )

    while True:
        if stop_event.is_set():
            break
        try:
            packet = input_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if packet is SENTINEL:
            break

        start = time.perf_counter()
        height, width = packet.frame_rgb.shape[:2]

        save_masks_to_dir(
            output_mask_dir=str(output_mask_dir),
            video_name=video_name,
            frame_name=packet.frame_name,
            per_obj_output_mask=packet.per_obj_mask,
            height=height,
            width=width,
            output_palette=cfg.palette,
            save_binary_mask=save_binary_mask,
            num_classes=cfg.num_classes,
            shift_by_1=cfg.shift_by_1,
        )

        if overlay_dir is not None:
            overlay = overlay_mask(
                packet.frame_rgb,
                packet.per_obj_mask,
                height,
                width,
                cfg.shift_by_1,
                cfg.palette,
                render_alpha,
            )
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_dir / video_name / f"{packet.frame_name}.jpg"), overlay_bgr)

        metrics.add_stage("render", time.perf_counter() - start)
        metrics.mark_output()
        progress_bar.update(1)

        if progress_interval > 0 and (packet.frame_idx + 1) % progress_interval == 0:
            progress = metrics.snapshot()
            progress_bar.set_postfix(
                elapsed_s=f"{progress['elapsed_seconds']:.1f}",
                fps=f"{progress['throughput_fps']:.2f}",
            )

    progress_bar.close()
    metrics.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel SASVI dataset-frame pipeline with timing and per-frame progress."
    )
    parser.add_argument("--base_video_dir", type=Path, required=True, help="Root directory containing frame folders.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("parallel_frame_outputs"),
        help="Root directory for masks, optional overlays, and summary JSON.",
    )
    parser.add_argument("--sam2_cfg", type=str, required=True, help="SAM2 config file.")
    parser.add_argument("--sam2_checkpoint", type=Path, required=True, help="Path to SAM2 checkpoint.")
    parser.add_argument("--overseer_checkpoint", type=Path, required=True, help="Path to overseer checkpoint.")
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
    parser.add_argument("--queue_size", type=int, default=2, help="Bounded queue size between stages.")
    parser.add_argument("--render_alpha", type=float, default=0.35, help="Overlay alpha for optional overlay frames.")
    parser.add_argument("--save_binary_mask", action="store_true", help="Also save NPZ binary masks.")
    parser.add_argument("--save_overlays", action="store_true", help="Also save overlay JPGs for each frame.")
    parser.add_argument("--max_videos", type=int, default=None, help="Optional cap on the number of clips.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on frames per clip.")
    parser.add_argument(
        "--offload_state_to_cpu",
        action="store_true",
        help="Offload SAM2 state to CPU to reduce VRAM usage at the cost of speed.",
    )
    parser.add_argument(
        "--sam2_window",
        type=int,
        default=64,
        help="Reset SAM2 tracking state every N processed frames to avoid long-run slowdown.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=25,
        help="Refresh FPS in the progress bar every N saved frames.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars for cleaner logs or timing-sensitive runs.",
    )
    parser.add_argument(
        "--metrics_output_path",
        type=Path,
        default=None,
        help="Optional path for the timing summary JSON. Defaults to <output_root>/timing_summary.json.",
    )
    return parser.parse_args()


def run_video_pipeline(
    video_dir: Path,
    args: argparse.Namespace,
    cfg: DatasetConfig,
    overseer_runtime: OverseerRuntime,
    predictor,
) -> dict[str, float | int | str]:
    frame_names = list_frame_names(video_dir, args.dataset_type)
    if not frame_names:
        raise RuntimeError(f"No JPG frames found in '{video_dir}'")

    frame_paths = [video_dir / f"{frame_name}.jpg" for frame_name in frame_names]
    if args.max_frames is not None and args.max_frames > 0:
        frame_names = frame_names[: args.max_frames]
        frame_paths = frame_paths[: args.max_frames]

    metrics = Metrics()
    gpu_lock = threading.Lock()
    stop_event = threading.Event()
    exception_queue: queue.Queue = queue.Queue()

    queue_ab: queue.Queue = queue.Queue(maxsize=args.queue_size)
    queue_bc: queue.Queue = queue.Queue(maxsize=args.queue_size)
    queue_cd: queue.Queue = queue.Queue(maxsize=args.queue_size)

    output_mask_dir = args.output_root / "output_masks"
    overlay_dir = args.output_root / "overlay_frames" if args.save_overlays else None

    threads = [
        threading.Thread(
            target=worker_wrapper,
            args=(
                stage_a_load_frames,
                exception_queue,
                frame_paths,
                frame_names,
                queue_ab,
                metrics,
                args.max_frames,
                stop_event,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=worker_wrapper,
            args=(stage_b_overseer, exception_queue, queue_ab, queue_bc, overseer_runtime, gpu_lock, metrics, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=worker_wrapper,
            args=(
                stage_c_sam2,
                exception_queue,
                queue_bc,
                queue_cd,
                video_dir,
                predictor,
                cfg.num_classes,
                args.score_thresh,
                gpu_lock,
                metrics,
                stop_event,
                args.offload_state_to_cpu,
                args.sam2_window,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=worker_wrapper,
            args=(
                stage_d_save_masks,
                exception_queue,
                queue_cd,
                output_mask_dir,
                overlay_dir,
                video_dir.name,
                args.save_binary_mask,
                cfg,
                args.render_alpha,
                metrics,
                len(frame_names),
                args.progress_interval,
                args.disable_tqdm,
                stop_event,
            ),
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            while thread.is_alive():
                thread.join(timeout=0.2)
                if not exception_queue.empty():
                    raise exception_queue.get()
    except KeyboardInterrupt:
        stop_event.set()
        for queue_obj in (queue_ab, queue_bc, queue_cd):
            try:
                queue_obj.put_nowait(SENTINEL)
            except queue.Full:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
        raise

    if not exception_queue.empty():
        raise exception_queue.get()

    summary = metrics.summary()
    return {
        "video_name": video_dir.name,
        "frames": int(summary["frames"]),
        "elapsed_seconds": summary["elapsed_seconds"],
        "throughput_fps": summary["throughput_fps"],
        "avg_capture_ms": summary["avg_capture_ms"],
        "avg_overseer_ms": summary["avg_overseer_ms"],
        "avg_sam2_ms": summary["avg_sam2_ms"],
        "avg_render_ms": summary["avg_render_ms"],
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not args.base_video_dir.exists():
        raise FileNotFoundError(f"Base video directory not found: {args.base_video_dir}")

    video_dirs = list_video_dirs(args.base_video_dir)
    if args.max_videos is not None and args.max_videos > 0:
        video_dirs = video_dirs[: args.max_videos]
    if not video_dirs:
        raise RuntimeError(f"No frame directories found in '{args.base_video_dir}'")

    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg = dataset_config(args.dataset_type)
    overseer_runtime = OverseerRuntime(
        overseer_type=args.overseer_type,
        dataset_type=args.dataset_type,
        checkpoint=args.overseer_checkpoint,
        device=args.device,
    )
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=str(args.sam2_checkpoint),
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )

    overall_start = time.perf_counter()
    summaries: list[dict[str, float | int | str]] = []

    print(f"Running parallel SASVI on {len(video_dirs)} frame directories from {args.base_video_dir}")
    for index, video_dir in enumerate(video_dirs, start=1):
        print(f"\n[{index}/{len(video_dirs)}] Processing {video_dir.name}")
        summary = run_video_pipeline(video_dir, args, cfg, overseer_runtime, predictor)
        summaries.append(summary)
        print(
            f"{video_dir.name}: {summary['frames']} frames in {summary['elapsed_seconds']:.3f}s "
            f"({summary['throughput_fps']:.3f} FPS)"
        )

    overall_elapsed = time.perf_counter() - overall_start
    total_frames = sum(int(item["frames"]) for item in summaries)
    overall_fps = total_frames / overall_elapsed if overall_elapsed > 0 else 0.0

    summary_payload = {
        "base_video_dir": str(args.base_video_dir),
        "output_root": str(args.output_root),
        "dataset_type": args.dataset_type,
        "overseer_type": args.overseer_type,
        "device": args.device,
        "video_count": len(summaries),
        "total_frames": total_frames,
        "elapsed_seconds": overall_elapsed,
        "throughput_fps": overall_fps,
        "videos": summaries,
    }
    summary_path = args.metrics_output_path or (args.output_root / "timing_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nPipeline complete")
    print(f"Output masks: {args.output_root / 'output_masks'}")
    if args.save_overlays:
        print(f"Overlay frames: {args.output_root / 'overlay_frames'}")
    print(f"Timing summary: {summary_path}")
    print(f"Total frames processed: {total_frames}")
    print(f"Elapsed time: {overall_elapsed:.3f} seconds")
    print(f"End-to-end throughput: {overall_fps:.3f} FPS")


if __name__ == "__main__":
    main()
