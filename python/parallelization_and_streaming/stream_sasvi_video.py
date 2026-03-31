from __future__ import annotations

import argparse
import datetime as dt
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import (
    DetrForSegmentation,
    DetrImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "sam2"))

from sam2.build_sam import build_sam2_video_predictor
from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import (
    get_cataract1k_colormap,
)
from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import (
    get_cholecseg8k_colormap,
)
from src.data import insert_component_masks, remap_labels
from src.model import get_model_instance_segmentation
from src.sam2.eval_sasvi import put_per_obj_mask, save_masks_to_dir
from src.utils import process_detr_outputs, process_mask2former_outputs


SENTINEL = object()


@dataclass
class FramePacket:
    frame_idx: int
    source_frame_idx: int
    frame_rgb: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class OverseerPacket:
    frame_idx: int
    source_frame_idx: int
    frame_rgb: np.ndarray
    per_obj_mask: dict[int, np.ndarray]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputPacket:
    frame_idx: int
    source_frame_idx: int
    frame_rgb: np.ndarray
    per_obj_mask: dict[int, np.ndarray]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    num_classes: int
    ignore_indices: list[int]
    shift_by_1: bool
    palette: list[int]
    maskrcnn_hidden_ft: int
    maskrcnn_backbone: str


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stage_seconds = {
            "capture": 0.0,
            "overseer": 0.0,
            "sam2": 0.0,
            "render": 0.0,
        }
        self.frames_out = 0
        self.start_wall = time.perf_counter()
        self.end_wall: float | None = None

    def add_stage(self, stage: str, seconds: float) -> None:
        with self.lock:
            self.stage_seconds[stage] += seconds

    def mark_output(self) -> None:
        with self.lock:
            self.frames_out += 1

    def finish(self) -> None:
        with self.lock:
            self.end_wall = time.perf_counter()

    def summary(self) -> dict[str, float]:
        with self.lock:
            elapsed = (self.end_wall or time.perf_counter()) - self.start_wall
            frames = self.frames_out
            return {
                "frames": float(frames),
                "elapsed_seconds": elapsed,
                "throughput_fps": frames / elapsed if elapsed > 0 else 0.0,
                "avg_capture_ms": 1000 * self.stage_seconds["capture"] / frames if frames else 0.0,
                "avg_overseer_ms": 1000 * self.stage_seconds["overseer"] / frames if frames else 0.0,
                "avg_sam2_ms": 1000 * self.stage_seconds["sam2"] / frames if frames else 0.0,
                "avg_render_ms": 1000 * self.stage_seconds["render"] / frames if frames else 0.0,
            }

    def snapshot(self) -> dict[str, float]:
        with self.lock:
            elapsed = (self.end_wall or time.perf_counter()) - self.start_wall
            frames = self.frames_out
            return {
                "frames": float(frames),
                "elapsed_seconds": elapsed,
                "throughput_fps": frames / elapsed if elapsed > 0 else 0.0,
            }


def put_with_retry(output_queue: queue.Queue, item: object, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def flatten_palette(palette: Any) -> list[int]:
    palette_array = np.asarray(palette, dtype=np.uint8).reshape(-1)
    return palette_array.tolist()


def dataset_config(dataset_type: str) -> DatasetConfig:
    dataset_type = dataset_type.upper()
    if dataset_type == "CADIS":
        return DatasetConfig(
            num_classes=18,
            ignore_indices=[255],
            shift_by_1=True,
            palette=flatten_palette(get_cadis_colormap()),
            maskrcnn_hidden_ft=32,
            maskrcnn_backbone="ResNet18",
        )
    if dataset_type == "CHOLECSEG8K":
        return DatasetConfig(
            num_classes=13,
            ignore_indices=[],
            shift_by_1=False,
            palette=flatten_palette(get_cholecseg8k_colormap()),
            maskrcnn_hidden_ft=64,
            maskrcnn_backbone="ResNet50",
        )
    if dataset_type == "CATARACT1K":
        return DatasetConfig(
            num_classes=14,
            ignore_indices=[],
            shift_by_1=False,
            palette=flatten_palette(get_cataract1k_colormap()),
            maskrcnn_hidden_ft=32,
            maskrcnn_backbone="ResNet18",
        )
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'")


class OverseerRuntime:
    def __init__(
        self,
        overseer_type: str,
        dataset_type: str,
        checkpoint: Path,
        device: str,
    ) -> None:
        self.overseer_type = overseer_type
        self.dataset_type = dataset_type.upper()
        self.device = device
        self.cfg = dataset_config(self.dataset_type)

        if self.overseer_type == "MaskRCNN":
            model = get_model_instance_segmentation(
                num_classes=self.cfg.num_classes,
                trainable_backbone_layers=0,
                hidden_ft=self.cfg.maskrcnn_hidden_ft,
                custom_in_ft_box=None,
                custom_in_ft_mask=None,
                backbone=self.cfg.maskrcnn_backbone,
                img_size=(299, 299),
            )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model = model.to(device)
            model.eval()
            self.model = model
            self.transform = T.Compose(
                [T.ToTensor(), T.Resize((299, 299)), T.Normalize(0.0, 1.0)]
            )
        elif self.overseer_type == "DETR":
            num_train_classes = self.cfg.num_classes - len(self.cfg.ignore_indices) + 1
            self.model = DetrForSegmentation.from_pretrained(
                "facebook/detr-resnet-50-panoptic",
                num_labels=num_train_classes,
                ignore_mismatched_sizes=True,
                num_queries=100,
            ).to(device)
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.model.eval()
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
            self.transform = A.Compose([A.Resize(200, 200), A.Normalize(0.0, 1.0)])
            self.num_train_classes = num_train_classes
        elif self.overseer_type == "Mask2Former":
            num_train_classes = self.cfg.num_classes - len(self.cfg.ignore_indices) + 1
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-base-coco-instance",
                num_labels=num_train_classes,
                ignore_mismatched_sizes=True,
                num_queries=20,
            ).to(device)
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.model.eval()
            self.processor = Mask2FormerImageProcessor.from_pretrained(
                "facebook/mask2former-swin-base-coco-instance",
                reduce_labels=True,
            )
            self.transform = A.Compose([A.Resize(299, 299), A.Normalize(0.0, 1.0)])
            self.num_train_classes = num_train_classes
        else:
            raise ValueError(f"Unsupported overseer_type '{self.overseer_type}'")

    @torch.inference_mode()
    def predict(self, frame_rgb: np.ndarray, reshape_size: tuple[int, int]) -> dict[int, np.ndarray]:
        if self.overseer_type == "MaskRCNN":
            image = self.transform(frame_rgb).to(self.device)
            predictions = self.model([image])
            pred_mask = predictions[0]["masks"]
            pred_labels = predictions[0]["labels"]
            if self.cfg.shift_by_1:
                pred_labels = pred_labels - 1
            remapped_pred_labels = remap_labels(
                pred_labels, self.cfg.num_classes, self.cfg.ignore_indices
            )
            binary_pred_mask = (pred_mask > 0.5).int()
            padded_binary_pred_mask = insert_component_masks(
                binary_pred_mask,
                remapped_pred_labels,
                self.cfg.num_classes,
                ignore_index=self.cfg.ignore_indices[0] if self.cfg.ignore_indices else None,
            )
            mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()
            if reshape_size is not None:
                width, height = reshape_size
                mask = np.array(
                    [
                        cv2.resize(item.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for item in mask
                    ]
                )
            object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
            if self.cfg.shift_by_1:
                background_class = self.cfg.num_classes - 1
                object_ids = object_ids[object_ids != background_class]
            return {int(object_id): mask[object_id] for object_id in object_ids}

        transformed = self.transform(image=frame_rgb)["image"]

        if self.overseer_type == "DETR":
            inputs = self.processor(
                images=[transformed],
                do_rescale=False,
                return_tensors="pt",
            ).to(self.device)
            predictions = self.model(**inputs)
            processed = process_detr_outputs(
                predictions,
                image_size=(reshape_size[1], reshape_size[0]),
                num_labels=self.num_train_classes,
                threshold=0.0,
            )[0]
        else:
            tensor = (
                torch.from_numpy(np.ascontiguousarray(transformed.transpose(2, 0, 1)))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            predictions = self.model(tensor)
            processed = process_mask2former_outputs(
                predictions,
                image_size=(reshape_size[1], reshape_size[0]),
                num_labels=self.num_train_classes,
                threshold=0.0,
            )[0]

        pred_mask = processed["masks"]
        pred_labels = processed["labels"]

        if self.dataset_type == "CHOLECSEG8K":
            pred_labels = pred_labels - 1
        if self.cfg.shift_by_1:
            pred_labels = pred_labels - 1

        remapped_pred_labels = remap_labels(
            pred_labels, self.cfg.num_classes, self.cfg.ignore_indices
        )
        padded_binary_pred_mask = insert_component_masks(
            pred_mask,
            remapped_pred_labels,
            self.cfg.num_classes,
            ignore_index=self.cfg.ignore_indices[0] if self.cfg.ignore_indices else None,
        )
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]

        if self.dataset_type == "CHOLECSEG8K":
            object_ids = object_ids[object_ids != 0]
        if self.dataset_type == "CADIS":
            object_ids = object_ids[object_ids != 17]

        return {int(object_id): mask[object_id] for object_id in object_ids}


def overlay_mask(
    frame_rgb: np.ndarray,
    per_obj_mask: dict[int, np.ndarray],
    height: int,
    width: int,
    shift_by_1: bool,
    palette: list[int],
    alpha: float,
) -> np.ndarray:
    label_mask = put_per_obj_mask(per_obj_mask, height, width, shift_by_1)
    mask_img = Image.fromarray(label_mask)
    mask_img.putpalette(palette)
    colour_mask = np.array(mask_img.convert("RGB"))
    overlay = cv2.addWeighted(frame_rgb, 1.0 - alpha, colour_mask, alpha, 0.0)
    return overlay


def stage_a_capture(
    input_video: Path,
    output_queue: queue.Queue,
    metrics: Metrics,
    max_frames: int | None,
    stop_event: threading.Event,
    frame_stride: int,
) -> None:
    capture = cv2.VideoCapture(str(input_video))
    source_frame_idx = 0
    output_frame_idx = 0
    try:
        while True:
            if stop_event.is_set():
                break
            if max_frames is not None and output_frame_idx >= max_frames:
                break
            start = time.perf_counter()
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_frame_idx % frame_stride != 0:
                source_frame_idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            elapsed = time.perf_counter() - start
            metrics.add_stage("capture", elapsed)
            while not stop_event.is_set():
                try:
                    output_queue.put(
                        FramePacket(
                            frame_idx=output_frame_idx,
                            source_frame_idx=source_frame_idx,
                            frame_rgb=frame_rgb,
                        ),
                        timeout=0.2,
                    )
                    break
                except queue.Full:
                    continue
            source_frame_idx += 1
            output_frame_idx += 1
    finally:
        capture.release()
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
        elapsed = time.perf_counter() - start
        metrics.add_stage("overseer", elapsed)

        timings = dict(packet.timings)
        timings["overseer"] = elapsed
        while not stop_event.is_set():
            try:
                output_queue.put(
                    OverseerPacket(
                        frame_idx=packet.frame_idx,
                        source_frame_idx=packet.source_frame_idx,
                        frame_rgb=packet.frame_rgb,
                        per_obj_mask=per_obj_mask,
                        timings=timings,
                    ),
                    timeout=0.2,
                )
                break
            except queue.Full:
                continue


def stage_c_sam2(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    input_video: Path,
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
            video_path=str(input_video),
            offload_video_to_cpu=True,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=False,
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
                        frame_idx=packet.source_frame_idx,
                        obj_id=int(obj_id),
                        mask=object_mask,
                    )
            else:
                dummy_mask = np.zeros(packet.frame_rgb.shape[:2], dtype=bool)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=packet.source_frame_idx,
                    obj_id=0,
                    mask=dummy_mask,
                )

            propagated = list(
                predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=packet.source_frame_idx,
                    max_frame_num_to_track=0,
                    reverse=False,
                )
            )

        current = next(
            (item for item in propagated if item[0] == packet.source_frame_idx),
            propagated[-1],
        )
        _, out_obj_ids, out_mask_logits = current
        per_obj_output_mask = {
            int(out_obj_id): (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        processed_in_window += 1

        elapsed = time.perf_counter() - start
        metrics.add_stage("sam2", elapsed)

        timings = dict(packet.timings)
        timings["sam2"] = elapsed
        while not stop_event.is_set():
            try:
                output_queue.put(
                    OutputPacket(
                        frame_idx=packet.frame_idx,
                        source_frame_idx=packet.source_frame_idx,
                        frame_rgb=packet.frame_rgb,
                        per_obj_mask=per_obj_output_mask,
                        timings=timings,
                    ),
                    timeout=0.2,
                )
                break
            except queue.Full:
                continue


def stage_d_render(
    input_queue: queue.Queue,
    output_video: Path,
    output_masks_dir: Path,
    video_name: str,
    save_binary_mask: bool,
    cfg: DatasetConfig,
    render_alpha: float,
    metrics: Metrics,
    total_frames: int | None,
    progress_interval: int,
    stop_event: threading.Event,
    output_fps: float,
) -> None:
    writer = None
    progress_bar = tqdm(total=total_frames, desc="stream_sasvi", unit="frame")
    output_masks_dir.mkdir(parents=True, exist_ok=True)

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
        frame_name = f"{packet.frame_idx:05d}"

        save_masks_to_dir(
            output_mask_dir=str(output_masks_dir),
            video_name=video_name,
            frame_name=frame_name,
            per_obj_output_mask=packet.per_obj_mask,
            height=height,
            width=width,
            output_palette=cfg.palette,
            save_binary_mask=save_binary_mask,
            num_classes=cfg.num_classes,
            shift_by_1=cfg.shift_by_1,
        )

        overlay = overlay_mask(
            packet.frame_rgb,
            packet.per_obj_mask,
            height,
            width,
            cfg.shift_by_1,
            cfg.palette,
            render_alpha,
        )

        if writer is None:
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )

        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        elapsed = time.perf_counter() - start
        metrics.add_stage("render", elapsed)
        metrics.mark_output()
        progress_bar.update(1)
        if progress_interval > 0 and (packet.frame_idx + 1) % progress_interval == 0:
            progress = metrics.snapshot()
            progress_bar.set_postfix(
                elapsed_s=f"{progress['elapsed_seconds']:.1f}",
                fps=f"{progress['throughput_fps']:.2f}",
            )

    if writer is not None:
        writer.release()
    progress_bar.close()
    metrics.finish()


def worker_wrapper(fn, exception_queue: queue.Queue, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        exception_queue.put(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal queued SASVi video pipeline with rendered video and mask export."
    )
    parser.add_argument("--input_video", type=Path, required=True, help="Path to input video (.mp4 recommended).")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("stream_outputs"),
        help="Root directory where a timestamped run folder will be created.",
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
    parser.add_argument("--render_alpha", type=float, default=0.35, help="Overlay alpha for rendered output.")
    parser.add_argument("--save_binary_mask", action="store_true", help="Also save NPZ binary masks.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame. Use 2 or 3 for much faster test runs.",
    )
    parser.add_argument(
        "--offload_state_to_cpu",
        action="store_true",
        help="Offload SAM2 state to CPU to reduce VRAM usage at the cost of speed.",
    )
    parser.add_argument(
        "--sam2_window",
        type=int,
        default=64,
        help="Reset SAM2 tracking state every N processed frames to avoid slowdown on long videos.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=25,
        help="Print progress every N rendered frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    run_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_root / run_timestamp
    args.output_video = run_dir / "output_video.mp4"
    args.output_masks_dir = run_dir / "output_masks"
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_masks_dir.mkdir(parents=True, exist_ok=True)

    video_meta = cv2.VideoCapture(str(args.input_video))
    source_fps = video_meta.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video_meta.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_meta.release()
    output_fps = max(source_fps / max(args.frame_stride, 1), 1.0)
    if total_frames > 0:
        total_frames = (total_frames + max(args.frame_stride, 1) - 1) // max(args.frame_stride, 1)
    if args.max_frames is not None and args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames) if total_frames > 0 else args.max_frames

    cfg = dataset_config(args.dataset_type)
    metrics = Metrics()
    gpu_lock = threading.Lock()
    stop_event = threading.Event()
    exception_queue: queue.Queue = queue.Queue()

    queue_ab: queue.Queue = queue.Queue(maxsize=args.queue_size)
    queue_bc: queue.Queue = queue.Queue(maxsize=args.queue_size)
    queue_cd: queue.Queue = queue.Queue(maxsize=args.queue_size)

    overseer_runtime = OverseerRuntime(
        overseer_type=args.overseer_type,
        dataset_type=args.dataset_type,
        checkpoint=args.overseer_checkpoint,
        device=args.device,
    )

    hydra_overrides_extra = ["++model.non_overlap_masks=true"]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=str(args.sam2_checkpoint),
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    video_name = args.input_video.stem
    threads = [
        threading.Thread(
            target=worker_wrapper,
            args=(
                stage_a_capture,
                exception_queue,
                args.input_video,
                queue_ab,
                metrics,
                args.max_frames,
                stop_event,
                max(args.frame_stride, 1),
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
                args.input_video,
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
                stage_d_render,
                exception_queue,
                queue_cd,
                args.output_video,
                args.output_masks_dir,
                video_name,
                args.save_binary_mask,
                cfg,
                args.render_alpha,
                metrics,
                total_frames if total_frames > 0 else None,
                args.progress_interval,
                stop_event,
                output_fps,
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
        print("\nInterrupted by user, shutting down...")
        for q in (queue_ab, queue_bc, queue_cd):
            try:
                q.put_nowait(SENTINEL)
            except queue.Full:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
        return

    if not exception_queue.empty():
        raise exception_queue.get()

    summary = metrics.summary()
    print("\nPipeline complete")
    print(f"Rendered video: {args.output_video}")
    print(f"Mask directory: {args.output_masks_dir / video_name}")
    print(f"Frames processed: {int(summary['frames'])}")
    print(f"Elapsed time: {summary['elapsed_seconds']:.3f} seconds")
    print(f"End-to-end throughput: {summary['throughput_fps']:.3f} FPS")
    print(f"Average Stage A capture/preprocess: {summary['avg_capture_ms']:.2f} ms/frame")
    print(f"Average Stage B overseer: {summary['avg_overseer_ms']:.2f} ms/frame")
    print(f"Average Stage C SAM2: {summary['avg_sam2_ms']:.2f} ms/frame")
    print(f"Average Stage D render/output: {summary['avg_render_ms']:.2f} ms/frame")


if __name__ == "__main__":
    main()
