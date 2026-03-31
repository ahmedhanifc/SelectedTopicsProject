from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
from transformers import DetrForSegmentation, Mask2FormerForUniversalSegmentation

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SAM2_ROOT = REPO_ROOT / "src" / "sam2"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAM2_ROOT))

from analysis_tools.inference_export import (
    INFERENCE_METADATA_COLUMNS,
    compute_confidence_map,
    save_confidence_map,
    summarise_confidence_map,
    write_markdown_report,
    write_rows_to_csv,
)
from analysis_tools.config import get_dataset_config
from analysis_tools.error_analysis import compute_frame_metrics, load_dataset_mask
from sam2.build_sam import build_sam2_video_predictor
from src.data import insert_component_masks, remap_labels
from src.model import get_model_instance_segmentation
from src.sam2.eval_sasvi import (
    DETR,
    Mask2Former,
    MaskRCNN,
    choose_duplicate_label,
    compute_boundary_distance,
    compute_disagreement_metrics,
    discover_video_dirs,
    get_per_obj_mask,
    get_primary_visual_palette,
    get_points_from_mask,
    get_unique_label,
    infer_frame_sort_key,
    list_video_frames,
    put_per_obj_mask,
    per_obj_mask_to_label_map,
    resolve_gt_mask_path,
    save_disagreement_visual,
    save_masks_to_dir,
)
from src.utils import process_detr_outputs, process_mask2former_outputs

try:
    from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
    from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import (
        get_cataract1k_colormap,
    )
    from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import (
        get_cholecseg8k_colormap,
    )
except ModuleNotFoundError:
    from src.sam2.eval_sasvi import (
        get_cadis_colormap,
        get_cataract1k_colormap,
        get_cholecseg8k_colormap,
    )

SENTINEL = object()


@dataclass
class SaveTask:
    output_mask_dir: str
    video_name: str
    frame_name: str
    per_obj_output_mask: dict[int, np.ndarray]
    height: int
    width: int
    output_palette: list[int]
    save_binary_mask: bool
    num_classes: int
    shift_by_1: bool
    confidence_path: str | None = None
    confidence_map: np.ndarray | None = None
    overseer_mask_dir: str | None = None
    per_obj_overseer_mask: dict[int, np.ndarray] | None = None
    disagreement_visual_path: str | None = None
    frame_path: str | None = None
    sam_foreground: np.ndarray | None = None
    overseer_foreground: np.ndarray | None = None


class SaveMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seconds = 0.0
        self.tasks = 0

    def add(self, seconds: float) -> None:
        with self.lock:
            self.seconds += seconds
            self.tasks += 1

    def summary(self) -> dict[str, float]:
        with self.lock:
            return {
                "save_seconds": self.seconds,
                "save_tasks": float(self.tasks),
                "avg_save_ms": (1000.0 * self.seconds / self.tasks) if self.tasks else 0.0,
            }


def save_worker(
    task_queue: queue.Queue,
    stop_event: threading.Event,
    save_metrics: SaveMetrics,
    exception_queue: queue.Queue,
) -> None:
    try:
        while True:
            task = task_queue.get()
            if task is SENTINEL:
                return
            if stop_event.is_set():
                return

            start = time.perf_counter()
            save_masks_to_dir(
                output_mask_dir=task.output_mask_dir,
                video_name=task.video_name,
                frame_name=task.frame_name,
                per_obj_output_mask=task.per_obj_output_mask,
                height=task.height,
                width=task.width,
                output_palette=task.output_palette,
                save_binary_mask=task.save_binary_mask,
                num_classes=task.num_classes,
                shift_by_1=task.shift_by_1,
            )

            if task.overseer_mask_dir is not None and task.per_obj_overseer_mask is not None:
                save_masks_to_dir(
                    output_mask_dir=task.overseer_mask_dir,
                    video_name=task.video_name,
                    frame_name=task.frame_name,
                    per_obj_output_mask=task.per_obj_overseer_mask,
                    height=task.height,
                    width=task.width,
                    output_palette=task.output_palette,
                    save_binary_mask=task.save_binary_mask,
                    num_classes=task.num_classes,
                    shift_by_1=task.shift_by_1,
                )

            if task.confidence_path is not None and task.confidence_map is not None:
                save_confidence_map(task.confidence_path, task.confidence_map)

            if (
                task.disagreement_visual_path is not None
                and task.frame_path is not None
                and task.sam_foreground is not None
                and task.overseer_foreground is not None
            ):
                save_disagreement_visual(
                    path=task.disagreement_visual_path,
                    frame_path=task.frame_path,
                    sam_foreground=task.sam_foreground,
                    overseer_foreground=task.overseer_foreground,
                )

            save_metrics.add(time.perf_counter() - start)
    except Exception as exc:  # pragma: no cover
        exception_queue.put(exc)
        stop_event.set()


def flatten_palette(palette: np.ndarray | list[int]) -> list[int]:
    return np.asarray(palette, dtype=np.uint8).reshape(-1).tolist()


def build_dataset_config(dataset_type: str, overseer_type: str) -> dict[str, object]:
    dataset_type = dataset_type.upper()
    if dataset_type == "CADIS":
        return {
            "num_classes": 18,
            "ignore_indices": [255],
            "shift_by_1": True,
            "palette": flatten_palette(get_primary_visual_palette(18)),
            "maskrcnn_hidden_ft": 32,
            "maskrcnn_backbone": "ResNet18",
        }
    if dataset_type == "CHOLECSEG8K":
        return {
            "num_classes": 13,
            "ignore_indices": [0] if overseer_type == "Mask2Former" else [],
            "shift_by_1": False,
            "palette": flatten_palette(get_primary_visual_palette(13)),
            "maskrcnn_hidden_ft": 64,
            "maskrcnn_backbone": "ResNet50",
        }
    if dataset_type == "CATARACT1K":
        return {
            "num_classes": 14,
            "ignore_indices": [0] if overseer_type in {"DETR", "Mask2Former"} else [],
            "shift_by_1": False,
            "palette": flatten_palette(get_primary_visual_palette(14)),
            "maskrcnn_hidden_ft": 32,
            "maskrcnn_backbone": "ResNet18",
        }
    raise NotImplementedError(f"Unsupported dataset_type '{dataset_type}'")


def build_overseer_model(args: argparse.Namespace, cfg: dict[str, object]):
    num_classes = int(cfg["num_classes"])
    ignore_indices = list(cfg["ignore_indices"])
    shift_by_1 = bool(cfg["shift_by_1"])

    if args.overseer_type == "MaskRCNN":
        model = get_model_instance_segmentation(
            num_classes=num_classes,
            trainable_backbone_layers=0,
            hidden_ft=int(cfg["maskrcnn_hidden_ft"]),
            custom_in_ft_box=None,
            custom_in_ft_mask=None,
            backbone=str(cfg["maskrcnn_backbone"]),
            img_size=(299, 299),
        )
        model.load_state_dict(torch.load(args.overseer_checkpoint, weights_only=True, map_location="cpu"))
        model = model.to(args.device)
        model.eval()
        return MaskRCNN(model, shift_by_1, ignore_indices, num_classes, device=args.device)

    if args.overseer_type == "DETR":
        num_train_classes = num_classes - len(ignore_indices) + 1
        model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=100,
        ).to(args.device)
        model.load_state_dict(torch.load(args.overseer_checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        return DETR(
            model,
            shift_by_1,
            ignore_indices,
            num_classes,
            num_train_classes,
            args.dataset_type,
            device=args.device,
        )

    if args.overseer_type == "Mask2Former":
        num_train_classes = num_classes - len(ignore_indices) + 1
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-instance",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=20,
        ).to(args.device)
        model.load_state_dict(torch.load(args.overseer_checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        return Mask2Former(
            model,
            shift_by_1,
            ignore_indices,
            num_classes,
            num_train_classes,
            args.dataset_type,
            device=args.device,
        )

    raise NotImplementedError(f"Unsupported overseer_type '{args.overseer_type}'")


def build_predictor(args: argparse.Namespace):
    hydra_overrides_extra = ["++model.non_overlap_masks=true"]
    return build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )


def filter_video_entries(
    video_entries: list[tuple[str, str]],
    video_name: str | None,
    video_names: list[str] | None,
    max_videos: int | None,
) -> list[tuple[str, str]]:
    requested_video_names = None
    if video_name is not None:
        requested_video_names = {video_name}
    elif video_names is not None:
        requested_video_names = set(video_names)

    if requested_video_names is not None:
        video_entries = [entry for entry in video_entries if entry[0] in requested_video_names]
        found_names = {name for name, _ in video_entries}
        missing_names = sorted(requested_video_names - found_names)
        if missing_names:
            raise RuntimeError(f"Requested video(s) not found: {missing_names}")

    if max_videos is not None and max_videos > 0:
        video_entries = video_entries[:max_videos]

    return video_entries


def chunked(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def build_frame_subset_dir_local(
    video_name: str,
    frame_infos: list[dict[str, str]],
    output_root: str | None = None,
) -> tuple[str, str]:
    cache_root = REPO_ROOT / "parallelization_and_streaming" / "_subset_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    subset_root = cache_root / f"{video_name}_{int(time.time() * 1000)}"
    os.makedirs(subset_root, exist_ok=True)
    for frame_info in frame_infos:
        dst_path = subset_root / f"{frame_info['stem']}{frame_info['ext']}"
        shutil.copy2(frame_info["path"], dst_path)
    subset_root_str = str(subset_root)
    return subset_root_str, subset_root_str


def postprocess_maskrcnn_prediction(
    overseer_model: MaskRCNN,
    prediction: dict,
    reshape_size: tuple[int, int] | None,
) -> dict[int, np.ndarray]:
    pred_mask = prediction["masks"]
    pred_labels = prediction["labels"]
    if overseer_model.shift_by_1:
        pred_labels = pred_labels - 1
    remapped_pred_labels = remap_labels(
        pred_labels,
        overseer_model.num_classes,
        overseer_model.ignore_indices,
    )
    binary_pred_mask = (pred_mask > 0.5).int()
    padded_binary_pred_mask = insert_component_masks(
        binary_pred_mask,
        remapped_pred_labels,
        overseer_model.num_classes,
        ignore_index=overseer_model.ignore_indices[0] if overseer_model.ignore_indices else None,
    )
    mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()
    if reshape_size is not None:
        mask = np.array(
            [
                resize(
                    item,
                    reshape_size,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(bool)
                for item in mask
            ]
        )
    object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
    if overseer_model.shift_by_1:
        background_class = overseer_model.num_classes - 1
        object_ids = object_ids[object_ids != background_class]
    return {int(object_id): mask[object_id] for object_id in object_ids}


def postprocess_transformer_prediction(
    overseer_model: Mask2Former | DETR,
    prediction: dict,
    reshape_size: tuple[int, int] | None,
) -> dict[int, np.ndarray]:
    pred_mask = prediction["masks"]
    pred_labels = prediction["labels"]

    ignore_indices = list(overseer_model.ignore_indices)
    if overseer_model.dataset_type == "CHOLECSEG8K":
        pred_labels = pred_labels - 1
    if overseer_model.dataset_type == "CATARACT1K":
        ignore_indices = []
    if overseer_model.shift_by_1:
        pred_labels = pred_labels - 1

    remapped_pred_labels = remap_labels(pred_labels, overseer_model.num_classes, ignore_indices)
    padded_binary_pred_mask = insert_component_masks(
        pred_mask,
        remapped_pred_labels,
        overseer_model.num_classes,
        ignore_index=ignore_indices[0] if ignore_indices else None,
    )
    mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

    if reshape_size is not None:
        mask = np.array(
            [
                resize(
                    item,
                    reshape_size,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(bool)
                for item in mask
            ]
        )

    object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
    per_obj_mask = {int(object_id): mask[object_id] for object_id in object_ids}

    if overseer_model.dataset_type == "CHOLECSEG8K":
        per_obj_mask = {object_id: obj_mask for object_id, obj_mask in per_obj_mask.items() if object_id != 0}
    if overseer_model.dataset_type == "CADIS":
        per_obj_mask = {object_id: obj_mask for object_id, obj_mask in per_obj_mask.items() if object_id != 17}
    return per_obj_mask


@torch.inference_mode()
def batched_overseer_predict(
    overseer_model,
    image_paths: list[str],
    reshape_size: tuple[int, int],
) -> list[dict[int, np.ndarray]]:
    if isinstance(overseer_model, MaskRCNN):
        images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        tensors = [overseer_model.transform(image).to(overseer_model.device) for image in images]
        predictions = overseer_model.model(tensors)
        return [
            postprocess_maskrcnn_prediction(overseer_model, prediction, reshape_size)
            for prediction in predictions
        ]

    if isinstance(overseer_model, Mask2Former):
        images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        transformed = [overseer_model.transform(image=image)["image"] for image in images]
        tensors = [A.pytorch.functional.img_to_tensor(image).to(overseer_model.device) for image in transformed]
        batch_tensor = torch.stack(tensors)
        predictions = overseer_model.model(batch_tensor)
        processed = process_mask2former_outputs(
            predictions,
            image_size=reshape_size,
            num_labels=overseer_model.num_train_classes,
            threshold=0.0,
        )
        return [
            postprocess_transformer_prediction(overseer_model, prediction, reshape_size)
            for prediction in processed
        ]

    if isinstance(overseer_model, DETR):
        images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        transformed = [overseer_model.transform(image=image)["image"] for image in images]
        inputs = overseer_model.processor(
            images=transformed,
            do_rescale=False,
            return_tensors="pt",
        ).to(overseer_model.device)
        predictions = overseer_model.model(**inputs)
        processed = process_detr_outputs(
            predictions,
            image_size=reshape_size,
            num_labels=overseer_model.num_train_classes,
            threshold=0.0,
        )
        return [
            postprocess_transformer_prediction(overseer_model, prediction, reshape_size)
            for prediction in processed
        ]

    raise TypeError(f"Unsupported overseer model type: {type(overseer_model)}")


def build_overseer_cache(
    frame_infos: list[dict[str, str]],
    overseer_model,
    reshape_size: tuple[int, int],
    batch_size: int,
    disable_tqdm: bool,
    video_name: str,
) -> tuple[dict[int, dict[int, np.ndarray]], float]:
    start = time.perf_counter()
    cache: dict[int, dict[int, np.ndarray]] = {}
    iterator = list(chunked(frame_infos, max(batch_size, 1)))
    progress = tqdm(
        iterator,
        desc=f"overseer:{video_name}",
        unit="batch",
        leave=False,
        disable=disable_tqdm,
    )
    for batch in progress:
        batch_paths = [item["path"] for item in batch]
        predictions = batched_overseer_predict(overseer_model, batch_paths, reshape_size)
        for frame_info, prediction in zip(batch, predictions):
            cache[int(frame_info["local_idx"])] = prediction
    return cache, time.perf_counter() - start


def enqueue_task(task_queue: queue.Queue, stop_event: threading.Event, task: SaveTask) -> None:
    while not stop_event.is_set():
        try:
            task_queue.put(task, timeout=0.2)
            return
        except queue.Full:
            continue


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def parallel_disagreement_inference(
    *,
    predictor,
    base_video_dir: str,
    output_mask_dir: str,
    video_name: str,
    overseer_type: str,
    overseer_mask_dir: str | None,
    overseer_model,
    nnunet_model,
    num_classes: int,
    ignore_indices: list[int],
    shift_by_1: bool,
    palette: list[int],
    dataset_type: str,
    gt_dataset_config,
    analysis_output_dir: str,
    save_queue: queue.Queue,
    save_stop_event: threading.Event,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_name: str | None = None,
    video_dir: str | None = None,
    gt_root_dir: str | None = None,
    score_thresh: float = 0.0,
    save_binary_mask: bool = False,
    enable_disagreement_gate: bool = False,
    disagreement_iou_threshold: float = 0.5,
    disagreement_bad_frames: int = 2,
    enable_boundary_distance_gate: bool = False,
    boundary_distance_threshold: float = 20.0,
    save_disagreement_visuals: bool = False,
    max_disagreement_visuals: int = 10,
    overseer_batch_size: int = 8,
    disable_tqdm: bool = False,
) -> tuple[list[dict], dict[str, float | int | str]]:
    video_dir = video_dir or os.path.join(base_video_dir, video_name)
    frame_infos = list_video_frames(video_dir)
    if not frame_infos:
        raise RuntimeError(f"No input frames found in {video_dir}")

    if frame_name is not None:
        frame_infos = [info for info in frame_infos if info["stem"] == frame_name]
        if not frame_infos:
            raise RuntimeError(f"Frame '{frame_name}' not found in {video_dir}")

    if start_frame is not None:
        frame_infos = [info for info in frame_infos if infer_frame_sort_key(info["stem"]) >= start_frame]
    if end_frame is not None:
        frame_infos = [info for info in frame_infos if infer_frame_sort_key(info["stem"]) <= end_frame]
    if not frame_infos:
        raise RuntimeError(f"No frames left to process for {video_name} after filtering.")

    for local_idx, frame_info in enumerate(frame_infos):
        frame_info["local_idx"] = local_idx

    frame_names = [info["stem"] for info in frame_infos]
    frame_path_by_name = {info["stem"]: info["path"] for info in frame_infos}

    predictor_video_dir = video_dir
    temp_subset_root = None
    subset_mode = frame_name is not None or start_frame is not None or end_frame is not None
    if subset_mode:
        temp_subset_root, predictor_video_dir = build_frame_subset_dir_local(
            video_name=video_name,
            frame_infos=frame_infos,
            output_root=analysis_output_dir,
        )

    inference_state = predictor.init_state(
        video_path=predictor_video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    confidence_dir = os.path.join(analysis_output_dir, "inference", "confidence_maps", video_name)
    os.makedirs(confidence_dir, exist_ok=True)

    disagreement_visual_dir = None
    if save_disagreement_visuals:
        disagreement_visual_dir = os.path.join(
            analysis_output_dir,
            "inference",
            "disagreement_visuals",
            video_name,
        )
        os.makedirs(disagreement_visual_dir, exist_ok=True)

    raw_output_mask_dir = os.path.join(output_mask_dir, "raw")
    smoothed_output_mask_dir = os.path.join(output_mask_dir, "smoothed")
    os.makedirs(os.path.join(raw_output_mask_dir, video_name), exist_ok=True)
    os.makedirs(os.path.join(smoothed_output_mask_dir, video_name), exist_ok=True)

    if overseer_mask_dir is not None:
        os.makedirs(os.path.join(overseer_mask_dir, "raw", video_name), exist_ok=True)
        os.makedirs(os.path.join(overseer_mask_dir, "smoothed", video_name), exist_ok=True)

    overseer_cache, overseer_cache_seconds = build_overseer_cache(
        frame_infos=frame_infos,
        overseer_model=overseer_model,
        reshape_size=(height, width),
        batch_size=overseer_batch_size,
        disable_tqdm=disable_tqdm,
        video_name=video_name,
    )

    trace_rows: list[dict] = []
    inference_rows: dict[str, dict] = {}
    video_summary = {
        "video_name": video_name,
        "frames_processed": 0,
        "segments_started": 0,
        "class_change_reprompts": 0,
        "disagreement_reprompts": 0,
        "mean_iou": 1.0,
        "min_iou": 1.0,
        "gt_frames_evaluated": 0,
        "gt_macro_iou_mean": None,
        "gt_macro_dice_mean": None,
        "gt_pixel_accuracy_mean": None,
        "overseer_cache_seconds": overseer_cache_seconds,
        "sasvi_seconds": 0.0,
        "throughput_fps": 0.0,
    }

    per_frame_ious = []
    gt_macro_ious = []
    gt_macro_dices = []
    gt_pixel_accuracies = []
    prompt_label_list = []
    negative_duplicate_list = []
    old_label = []
    current_label = []
    break_endless_loop = False
    disagreement_visual_count = 0
    idx = 0
    sasvi_start = time.perf_counter()

    try:
        while idx < len(frame_names):
            segment_id = video_summary["segments_started"] + 1
            video_summary["segments_started"] += 1
            segment_start_idx = idx
            disagreement_counter = 0

            print(f"[segment] video={video_name} segment={segment_id} start_frame={frame_names[idx]} idx={idx}")
            predictor.reset_state(inference_state=inference_state)
            buffer_length = 25

            per_obj_input_mask = overseer_cache[idx]

            if idx > 0:
                if dataset_type == "CADIS":
                    per_obj_previous_mask = get_per_obj_mask(
                        mask_path=os.path.join(raw_output_mask_dir, video_name),
                        frame_name=frame_names[idx],
                        use_binary_mask=False,
                        width=width,
                        height=height,
                        ignore_indices=ignore_indices,
                        shift_by_1=shift_by_1,
                        palette=palette,
                    )

                    ignore_class_input_mask = ~np.array(list(per_obj_input_mask.values())).any(axis=0)
                    ignore_class_previous_mask = ~np.array(list(per_obj_previous_mask.values())).any(axis=0)
                    additional_points = np.argwhere(ignore_class_input_mask & ~ignore_class_previous_mask)
                    merged_previous_mask = put_per_obj_mask(per_obj_previous_mask, height, width, shift_by_1)
                    for pos in additional_points:
                        val = merged_previous_mask[tuple(pos)]
                        if val in list(set(old_label) & set(current_label)):
                            per_obj_input_mask[val][tuple(pos)] = True

                for obj_id in prompt_label_list:
                    selected_point = get_points_from_mask(per_obj_input_mask, label_ids=[obj_id])
                    if selected_point[obj_id]:
                        points = np.array(selected_point[obj_id], dtype=np.float32)
                        labels = np.ones((len(selected_point[obj_id]),), dtype=np.int32)
                        predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )

                    false_positions = np.argwhere(per_obj_input_mask[obj_id])
                    for obj_input_mask in per_obj_input_mask:
                        if obj_input_mask != obj_id:
                            for false_pos in false_positions:
                                per_obj_input_mask[obj_input_mask][tuple(false_pos)] = False

                yet_another_unique_label = old_label
                break_endless_loop = True
                negative_duplicate_list = [x for xs in negative_duplicate_list for x in xs]
                old_label = list(
                    (((set(old_label) & set(current_label)) | set(prompt_label_list)) - set(negative_duplicate_list))
                )
            else:
                yet_another_unique_label = []
                negative_duplicate_list = []

            for object_id, object_mask in per_obj_input_mask.items():
                if idx == 0 or (idx > 0 and object_id in yet_another_unique_label and object_id not in negative_duplicate_list):
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=idx,
                        obj_id=object_id,
                        mask=object_mask,
                    )

            if len(inference_state["point_inputs_per_obj"]) == 0 and len(inference_state["mask_inputs_per_obj"]) == 0:
                print("Empty overseer mask, using dummy background point prompt. Happens when camera move out of scene.")
                dummy_id = num_classes if shift_by_1 else 0
                dummy_mask = np.zeros((height, width), dtype=bool)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx,
                    obj_id=dummy_id,
                    mask=dummy_mask,
                )

            trigger_next_idx = None
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                idx = out_frame_idx
                per_obj_output_mask = {
                    int(out_obj_id): (out_mask_logits[i] > score_thresh).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                current_overseer_mask = overseer_cache[out_frame_idx]
                disagreement_metrics = compute_disagreement_metrics(
                    per_obj_sam_mask=per_obj_output_mask,
                    per_obj_overseer_mask=current_overseer_mask,
                    height=height,
                    width=width,
                )

                boundary_distance = None
                if enable_boundary_distance_gate:
                    boundary_distance = compute_boundary_distance(
                        disagreement_metrics["sam_foreground"],
                        disagreement_metrics["overseer_foreground"],
                    )

                confidence_map = None
                confidence_path = None
                confidence_stats = {
                    "confidence_path": None,
                    "confidence_mean": None,
                    "confidence_std": None,
                    "confidence_min": None,
                    "confidence_max": None,
                }
                if len(out_obj_ids) > 0:
                    confidence_map = compute_confidence_map(
                        out_mask_logits,
                        object_ids=out_obj_ids,
                        score_thresh=score_thresh,
                    )
                    confidence_path = os.path.join(
                        confidence_dir,
                        f"{frame_names[out_frame_idx]}_confidence.png",
                    )
                    confidence_stats = {
                        "confidence_path": confidence_path,
                        **summarise_confidence_map(confidence_map),
                    }

                inference_rows[frame_names[out_frame_idx]] = {
                    "video_name": video_name,
                    "frame_name": frame_names[out_frame_idx],
                    "frame_idx": out_frame_idx,
                    "segment_id": segment_id,
                    "num_objects": len(out_obj_ids),
                    **confidence_stats,
                }

                gt_mask_path = resolve_gt_mask_path(
                    frame_name=frame_names[out_frame_idx],
                    video_dir=video_dir,
                    video_name=video_name,
                    base_video_dir=base_video_dir,
                    gt_root_dir=gt_root_dir,
                )
                gt_metrics = {
                    "gt_mask_path": gt_mask_path,
                    "gt_pixel_accuracy": None,
                    "gt_macro_iou": None,
                    "gt_macro_dice": None,
                    "gt_error_rate": None,
                }
                if gt_mask_path is not None:
                    pred_label_map = per_obj_mask_to_label_map(
                        per_obj_mask=per_obj_output_mask,
                        height=height,
                        width=width,
                        shift_by_1=shift_by_1,
                    )
                    gt_label_map = load_dataset_mask(gt_mask_path, gt_dataset_config)
                    gt_frame_metrics = compute_frame_metrics(
                        pred_mask=pred_label_map,
                        gt_mask=gt_label_map,
                        dataset_config=gt_dataset_config,
                    )
                    if gt_frame_metrics["macro_iou"] is not None:
                        gt_metrics = {
                            "gt_mask_path": gt_mask_path,
                            "gt_pixel_accuracy": gt_frame_metrics["pixel_accuracy"],
                            "gt_macro_iou": gt_frame_metrics["macro_iou"],
                            "gt_macro_dice": gt_frame_metrics["macro_dice"],
                            "gt_error_rate": gt_frame_metrics["error_rate"],
                        }
                        gt_macro_ious.append(gt_frame_metrics["macro_iou"])
                        gt_macro_dices.append(gt_frame_metrics["macro_dice"])
                        gt_pixel_accuracies.append(gt_frame_metrics["pixel_accuracy"])
                        video_summary["gt_frames_evaluated"] += 1

                bad_disagreement_frame = False
                if enable_disagreement_gate and out_frame_idx > segment_start_idx:
                    bad_disagreement_frame = disagreement_metrics["iou"] < disagreement_iou_threshold
                    if enable_boundary_distance_gate and boundary_distance is not None:
                        bad_disagreement_frame = bad_disagreement_frame or boundary_distance > boundary_distance_threshold
                    disagreement_counter = disagreement_counter + 1 if bad_disagreement_frame else 0
                else:
                    disagreement_counter = 0

                video_summary["frames_processed"] += 1
                per_frame_ious.append(disagreement_metrics["iou"])

                future_n_frame = min(10, len(frame_names) - idx)
                per_obj_input_mask_n = []
                duplicate_list = []
                negative_duplicate_list = []
                partial_duplicate_list = []
                prompt_label_list = []
                class_change_trigger = False
                disagreement_trigger = False
                reprompt_executed = False
                reprompt_reason = ""

                for n in range(future_n_frame):
                    per_obj_input_mask_n.append(overseer_cache[idx + n])

                unique_lable_n = get_unique_label(per_obj_input_mask_n)
                if idx == 0:
                    old_label = unique_lable_n[0]
                else:
                    current_label = unique_lable_n[0]
                    buffer_length -= 1
                    if buffer_length == 0:
                        trigger_next_idx = min(idx + 1, len(frame_names))
                        break

                    if old_label != current_label:
                        new_obj_label = list(set(current_label) - set(old_label))
                        unique_lable_n = unique_lable_n[1:4]
                        for unique_l in unique_lable_n:
                            new_obj_label = list(set(new_obj_label) & set(unique_l))

                        if new_obj_label:
                            for new_obj in new_obj_label:
                                mask1 = per_obj_input_mask_n[0][new_obj]
                                for obj in current_label:
                                    mask2 = per_obj_input_mask_n[0][obj]
                                    true_positions_1 = mask1 == 1
                                    true_positions_2 = mask2 == 1
                                    matching_true_positions = np.logical_and(true_positions_1, true_positions_2)
                                    similarity_1 = np.sum(matching_true_positions) / np.sum(true_positions_1)
                                    similarity_2 = np.sum(matching_true_positions) / np.sum(true_positions_2)

                                    if similarity_1 >= 0.70 and similarity_2 >= 0.70 and obj != new_obj:
                                        if not duplicate_list:
                                            duplicate_list.append([obj, new_obj])
                                        else:
                                            common_element_flag = False
                                            for sublist in duplicate_list:
                                                if list(set(sublist) & set([obj, new_obj])):
                                                    common_element_flag = True
                                                    new_element = list(set([obj, new_obj]) - set(sublist))
                                                    if new_element:
                                                        sublist.append(new_element[0])
                                            if not common_element_flag:
                                                duplicate_list.append([obj, new_obj])
                                    elif similarity_1 >= 0.70 and similarity_2 < 0.70:
                                        partial_duplicate_list.append(new_obj)

                            if duplicate_list:
                                negative_duplicate_list = duplicate_list
                                for sublist in duplicate_list:
                                    selections, use_nnunet = choose_duplicate_label(
                                        per_obj_mask_n=per_obj_input_mask_n,
                                        duplicate_label=sublist,
                                    )
                                    prompt_label = max(selections, key=selections.get)
                                    if use_nnunet and nnunet_model is not None:
                                        per_obj_input_mask_nnunet = nnunet_model.get_prediction(
                                            frame_path_by_name[frame_names[idx]],
                                            reshape_size=(width, height),
                                        )
                                        area_to_check = per_obj_input_mask_n[0][prompt_label]
                                        for key in selections:
                                            if key in per_obj_input_mask_nnunet:
                                                selections[key] += np.sum(
                                                    np.logical_and(area_to_check, per_obj_input_mask_nnunet[key])
                                                )
                                        prompt_label = max(selections, key=selections.get)

                                    if prompt_label not in old_label:
                                        prompt_label_list.append(prompt_label)
                                    negative_duplicate_list = [
                                        [x for x in slist if x != prompt_label]
                                        for slist in negative_duplicate_list
                                    ]

                            if new_obj_label:
                                single_label = list(
                                    (set(new_obj_label) - set([x for xs in duplicate_list for x in xs]))
                                    - set(partial_duplicate_list)
                                )
                                if single_label:
                                    for ixn in single_label:
                                        if ixn not in old_label:
                                            prompt_label_list.append(ixn)

                            if prompt_label_list:
                                if break_endless_loop:
                                    break_endless_loop = False
                                    idx += 1
                                    continue
                                class_change_trigger = True
                                reprompt_executed = True
                                reprompt_reason = "class-change"
                                video_summary["class_change_reprompts"] += 1
                                trigger_next_idx = idx
                                print(
                                    f"[reprompt] video={video_name} frame={frame_names[idx]} idx={idx} "
                                    f"reason=class-change labels={prompt_label_list}"
                                )
                            break_endless_loop = False

                    old_label = list(set(old_label) & set(current_label)) + prompt_label_list

                if (
                    not class_change_trigger
                    and enable_disagreement_gate
                    and out_frame_idx > segment_start_idx
                    and disagreement_counter >= disagreement_bad_frames
                ):
                    disagreement_trigger = True
                    reprompt_executed = True
                    reprompt_reason = "disagreement"
                    trigger_next_idx = idx
                    prompt_label_list = []
                    negative_duplicate_list = []
                    current_label = sorted(set(current_overseer_mask.keys()))
                    old_label = current_label.copy()
                    video_summary["disagreement_reprompts"] += 1
                    print(
                        f"[reprompt] video={video_name} frame={frame_names[idx]} idx={idx} "
                        f"reason=disagreement iou={disagreement_metrics['iou']:.4f} "
                        f"counter={disagreement_counter}"
                    )

                trace_row = {
                    **inference_rows[frame_names[out_frame_idx]],
                    "sam2_vs_overseer_iou": disagreement_metrics["iou"],
                    "sam2_vs_overseer_fg_iou": disagreement_metrics["foreground_iou"],
                    "sam2_vs_overseer_boundary_distance": boundary_distance,
                    "disagreement_bad_frame": bad_disagreement_frame,
                    "disagreement_counter": disagreement_counter,
                    "class_change_trigger": class_change_trigger,
                    "disagreement_trigger": disagreement_trigger,
                    "reprompt_executed": reprompt_executed,
                    "reprompt_reason": reprompt_reason,
                    **gt_metrics,
                }
                trace_rows.append(trace_row)
                print(
                    f"[trace] video={video_name} frame={frame_names[out_frame_idx]} idx={out_frame_idx} "
                    f"segment={segment_id} iou={disagreement_metrics['iou']:.4f} "
                    f"fg_iou={disagreement_metrics['foreground_iou']:.4f} "
                    f"boundary={'n/a' if boundary_distance is None or not np.isfinite(boundary_distance) else f'{boundary_distance:.4f}'} "
                    f"class_change={class_change_trigger} disagreement={disagreement_trigger} "
                    f"bad={bad_disagreement_frame} counter={disagreement_counter} "
                    f"reprompt={reprompt_executed} reason={reprompt_reason or 'none'}"
                )

                disagreement_visual_path = None
                if (
                    disagreement_visual_dir is not None
                    and disagreement_visual_count < max_disagreement_visuals
                    and (bad_disagreement_frame or reprompt_executed)
                ):
                    disagreement_visual_path = os.path.join(
                        disagreement_visual_dir,
                        f"{frame_names[out_frame_idx]}_segment{segment_id}_disagreement.png",
                    )
                    disagreement_visual_count += 1

                enqueue_task(
                    save_queue,
                    save_stop_event,
                    SaveTask(
                        output_mask_dir=output_mask_dir,
                        video_name=video_name,
                        frame_name=frame_names[out_frame_idx],
                        per_obj_output_mask=per_obj_output_mask,
                        height=height,
                        width=width,
                        output_palette=palette,
                        save_binary_mask=save_binary_mask,
                        num_classes=num_classes,
                        shift_by_1=shift_by_1,
                        confidence_path=confidence_path,
                        confidence_map=confidence_map,
                        overseer_mask_dir=overseer_mask_dir,
                        per_obj_overseer_mask=current_overseer_mask if overseer_mask_dir is not None else None,
                        disagreement_visual_path=disagreement_visual_path,
                        frame_path=frame_path_by_name[frame_names[out_frame_idx]] if disagreement_visual_path else None,
                        sam_foreground=disagreement_metrics["sam_foreground"] if disagreement_visual_path else None,
                        overseer_foreground=disagreement_metrics["overseer_foreground"] if disagreement_visual_path else None,
                    ),
                )

                if reprompt_executed:
                    break
                idx += 1

            if trigger_next_idx is None:
                if idx >= len(frame_names) - 1:
                    idx = len(frame_names)
                else:
                    idx += 1
            else:
                idx = trigger_next_idx
    finally:
        if temp_subset_root is not None:
            try:
                for frame_info in frame_infos:
                    os.unlink(os.path.join(predictor_video_dir, os.path.basename(frame_info["path"])))
                os.rmdir(predictor_video_dir)
                os.rmdir(temp_subset_root)
            except OSError:
                pass

    sasvi_seconds = time.perf_counter() - sasvi_start
    video_summary["sasvi_seconds"] = sasvi_seconds
    if per_frame_ious:
        video_summary["mean_iou"] = float(np.mean(per_frame_ious))
        video_summary["min_iou"] = float(np.min(per_frame_ious))
    if gt_macro_ious:
        video_summary["gt_macro_iou_mean"] = float(np.mean(gt_macro_ious))
        video_summary["gt_macro_dice_mean"] = float(np.mean(gt_macro_dices))
        video_summary["gt_pixel_accuracy_mean"] = float(np.mean(gt_pixel_accuracies))

    total_elapsed = overseer_cache_seconds + sasvi_seconds
    video_summary["throughput_fps"] = (
        video_summary["frames_processed"] / total_elapsed if total_elapsed > 0 else 0.0
    )
    return trace_rows, video_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel disagreement-gated SASVI with batched Overseer caching and async output export."
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
        default= REPO_ROOT / "parallel_disagreement_frame_outputs",
        help="Root directory for parallel disagreement frame outputs.",
    )
    parser.add_argument(
        "--analysis_output_root",
        type=Path,
        default= REPO_ROOT / "analysis_output_disagreement_parallel",
        help="Root directory for parallel disagreement analysis outputs.",
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
    parser.add_argument(
        "--overseer_batch_size",
        type=int,
        default=8,
        help="Batch size for precomputing Overseer masks. Higher values usually improve throughput.",
    )
    parser.add_argument(
        "--save_queue_size",
        type=int,
        default=32,
        help="Queue size for asynchronous save tasks.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def write_run_parameters(path: Path, rows: list[tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key, value in rows:
            writer.writerow([key, value])


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
    predictor = build_predictor(args)
    overseer_model = build_overseer_model(args, cfg)
    nnunet_model = None

    run_timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = args.output_root / run_timestamp
    run_analysis_dir = args.analysis_output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_analysis_dir.mkdir(parents=True, exist_ok=True)
    (run_analysis_dir / "inference").mkdir(parents=True, exist_ok=True)

    output_mask_dir = run_output_dir / "output_masks"
    overseer_mask_dir = run_output_dir / "overseer_masks" if args.save_overseer_masks else None
    if overseer_mask_dir is not None:
        overseer_mask_dir.mkdir(parents=True, exist_ok=True)

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
            ("overseer_batch_size", args.overseer_batch_size),
            ("save_queue_size", args.save_queue_size),
            ("video_name", args.video_name),
            ("video_names", " ".join(args.video_names) if args.video_names else ""),
            ("frame_name", args.frame_name),
            ("start_frame", args.start_frame),
            ("end_frame", args.end_frame),
            ("max_videos", args.max_videos),
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
    all_trace_rows: list[dict] = []
    video_summaries: list[dict] = []

    try:
        print(f"Running parallel disagreement SASVI on {len(video_entries)} videos")
        for index, (video_name, video_dir) in enumerate(video_entries, start=1):
            print(f"\n[{index}/{len(video_entries)}] Processing {video_name}")
            video_rows, video_summary = parallel_disagreement_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                output_mask_dir=str(output_mask_dir),
                video_name=video_name,
                video_dir=video_dir,
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
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                frame_name=args.frame_name,
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
            all_trace_rows.extend(video_rows)
            video_summaries.append(video_summary)
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
            print(
                f"{video_name}: {video_summary['frames_processed']} frames, "
                f"overseer cache {video_summary['overseer_cache_seconds']:.3f}s, "
                f"SASVI {video_summary['sasvi_seconds']:.3f}s, "
                f"{video_summary['throughput_fps']:.3f} FPS"
            )

        save_queue.put(SENTINEL)
        save_thread.join()
        if not save_exception_queue.empty():
            raise save_exception_queue.get()
    except Exception:
        save_stop_event.set()
        try:
            save_queue.put_nowait(SENTINEL)
        except queue.Full:
            pass
        save_thread.join(timeout=1.0)
        raise

    if all_trace_rows:
        write_rows_to_csv(
            run_analysis_dir / "inference" / "inference_metadata.csv",
            all_trace_rows,
            INFERENCE_METADATA_COLUMNS,
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
                "overseer_batch_size": args.overseer_batch_size,
            },
            video_summaries=video_summaries,
            trace_rows=all_trace_rows,
        )

    overall_elapsed = time.perf_counter() - overall_start
    total_frames = sum(int(item["frames_processed"]) for item in video_summaries)
    overall_fps = total_frames / overall_elapsed if overall_elapsed > 0 else 0.0
    save_summary = save_metrics.summary()

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
        "save_seconds": save_summary["save_seconds"],
        "avg_save_ms": save_summary["avg_save_ms"],
        "videos": [
            {
                "video_name": item["video_name"],
                "frames_processed": item["frames_processed"],
                "segments_started": item["segments_started"],
                "class_change_reprompts": item["class_change_reprompts"],
                "disagreement_reprompts": item["disagreement_reprompts"],
                "mean_iou": item["mean_iou"],
                "min_iou": item["min_iou"],
                "overseer_cache_seconds": item["overseer_cache_seconds"],
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
