from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import DatasetConfig
from .inference_export import write_rows_to_csv


REPORT_COLUMNS = [
    "video_name",
    "frame_name",
    "image_path",
    "pred_mask_path",
    "gt_mask_path",
    "confidence_path",
    "artifact_dir",
    "pixel_accuracy",
    "macro_iou",
    "macro_dice",
    "error_pixels",
    "error_rate",
    "false_positive_pixels",
    "false_negative_pixels",
    "class_confusion_pixels",
    "temporal_iou_prev",
    "num_pred_classes",
    "num_gt_classes",
    "confidence_mean",
    "confidence_std",
    "confidence_min",
    "confidence_max",
    "pred_classes",
    "gt_classes",
    "per_class_iou",
    "per_class_dice",
]


def _load_png_mask(path: Path, palette: np.ndarray | None) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.array(image)

    if mask.ndim == 2:
        return mask.astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] >= 3:
        if palette is None:
            raise ValueError(f"Palette is required to decode RGB mask: {path}")
        rgb = mask[:, :, :3].astype(np.uint8)
        label_mask = np.full(rgb.shape[:2], fill_value=255, dtype=np.uint8)
        for label, color in enumerate(palette):
            matches = np.all(rgb == color, axis=-1)
            label_mask[matches] = label
        return label_mask

    raise ValueError(f"Unsupported mask format in {path}")


def _load_confidence_map(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as image:
        confidence = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return confidence


def _resize_to_match(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if mask.shape[:2] == (height, width):
        return mask
    image = Image.fromarray(mask)
    resized = image.resize((width, height), resample=Image.NEAREST)
    return np.array(resized, dtype=mask.dtype)


def _collect_label_metrics(pred_mask: np.ndarray,
                           gt_mask: np.ndarray,
                           labels: list[int],
                           valid_mask: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    iou_scores: dict[str, float] = {}
    dice_scores: dict[str, float] = {}
    for label in labels:
        pred_binary = np.logical_and(pred_mask == label, valid_mask)
        gt_binary = np.logical_and(gt_mask == label, valid_mask)
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        denom = pred_binary.sum() + gt_binary.sum()
        iou_scores[str(label)] = float(intersection / union) if union > 0 else 1.0
        dice_scores[str(label)] = float((2 * intersection) / denom) if denom > 0 else 1.0
    return iou_scores, dice_scores


def _error_breakdown(pred_mask: np.ndarray,
                     gt_mask: np.ndarray,
                     valid_mask: np.ndarray,
                     background_index: int | None) -> tuple[int, int, int]:
    if background_index is None:
        error_pixels = np.logical_and(pred_mask != gt_mask, valid_mask)
        return int(error_pixels.sum()), 0, 0

    false_positive = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask != background_index, gt_mask == background_index))
    false_negative = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask == background_index, gt_mask != background_index))
    class_confusion = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask != background_index, gt_mask != background_index))
    return int(false_positive.sum()), int(false_negative.sum()), int(class_confusion.sum())


def _build_error_map(pred_mask: np.ndarray,
                     gt_mask: np.ndarray,
                     valid_mask: np.ndarray,
                     background_index: int | None) -> np.ndarray:
    height, width = pred_mask.shape
    error_map = np.zeros((height, width, 3), dtype=np.uint8)
    if background_index is None:
        error_pixels = np.logical_and(pred_mask != gt_mask, valid_mask)
        error_map[error_pixels] = np.array([255, 64, 64], dtype=np.uint8)
        return error_map

    false_positive = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask != background_index, gt_mask == background_index))
    false_negative = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask == background_index, gt_mask != background_index))
    class_confusion = np.logical_and.reduce((valid_mask, pred_mask != gt_mask, pred_mask != background_index, gt_mask != background_index))
    error_map[false_positive] = np.array([255, 64, 64], dtype=np.uint8)
    error_map[false_negative] = np.array([64, 128, 255], dtype=np.uint8)
    error_map[class_confusion] = np.array([255, 196, 64], dtype=np.uint8)
    return error_map


def _colorize_mask(mask: np.ndarray, palette: np.ndarray, ignore_index: int | None) -> np.ndarray:
    colorized = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        colorized[mask == label] = color
    if ignore_index is not None:
        colorized[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return colorized


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    blended = ((1.0 - alpha) * base.astype(np.float32) + alpha * overlay.astype(np.float32)).clip(0, 255)
    return blended.astype(np.uint8)


def _make_overlay(image: np.ndarray,
                  pred_rgb: np.ndarray,
                  gt_rgb: np.ndarray,
                  error_map: np.ndarray,
                  confidence_map: np.ndarray | None) -> np.ndarray:
    image = image[:, :, :3].astype(np.uint8)
    pred_overlay = _blend(image, pred_rgb, alpha=0.45)
    gt_overlay = _blend(image, gt_rgb, alpha=0.45)
    error_overlay = _blend(image, error_map, alpha=0.60)

    if confidence_map is None:
        confidence_rgb = np.zeros_like(image)
    else:
        confidence_uint8 = (np.clip(confidence_map, 0.0, 1.0) * 255).astype(np.uint8)
        confidence_rgb = np.stack([confidence_uint8] * 3, axis=-1)

    top = np.concatenate([image, pred_overlay], axis=1)
    bottom = np.concatenate([gt_overlay, error_overlay], axis=1)
    if confidence_map is not None:
        confidence_panel = np.concatenate([confidence_rgb, confidence_rgb], axis=1)
        overlay = np.concatenate([top, bottom, confidence_panel], axis=0)
    else:
        overlay = np.concatenate([top, bottom], axis=0)
    return overlay


def _frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.replace("_rgb_mask", "")
    try:
        return int(stem), stem
    except ValueError:
        return 0, stem


def analyze_predictions(frame_root: Path,
                        pred_root: Path,
                        gt_root: Path,
                        output_root: Path,
                        dataset_config: DatasetConfig,
                        confidence_root: Path | None = None,
                        pred_mask_suffix: str = "_rgb_mask.png",
                        gt_mask_suffix: str = "_rgb_mask.png",
                        image_suffix: str = ".jpg") -> dict[str, Any]:
    report_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset_type": dataset_config.dataset_type,
        "videos": {},
        "frames_analyzed": 0,
    }

    video_dirs = sorted(path for path in pred_root.iterdir() if path.is_dir())
    for video_dir in video_dirs:
        video_name = video_dir.name
        pred_files = sorted(video_dir.glob(f"*{pred_mask_suffix}"), key=_frame_sort_key)
        if not pred_files:
            continue

        prev_pred_mask: np.ndarray | None = None
        summary["videos"][video_name] = {"frames": 0}

        for pred_file in pred_files:
            frame_name = pred_file.name[:-len(pred_mask_suffix)]
            frame_path = frame_root / video_name / f"{frame_name}{image_suffix}"
            gt_path = gt_root / video_name / f"{frame_name}{gt_mask_suffix}"
            if not frame_path.exists():
                continue
            if not gt_path.exists():
                continue

            pred_mask = _load_png_mask(pred_file, dataset_config.palette)
            gt_mask = _load_png_mask(gt_path, dataset_config.palette)
            pred_mask = _resize_to_match(pred_mask, gt_mask.shape[:2])

            confidence_path = None
            confidence_map = None
            if confidence_root is not None:
                confidence_path = confidence_root / video_name / f"{frame_name}_confidence.png"
                confidence_map = _load_confidence_map(confidence_path)
                if confidence_map is not None:
                    confidence_map = _resize_to_match((confidence_map * 255).astype(np.uint8), gt_mask.shape[:2]).astype(np.float32) / 255.0

            with Image.open(frame_path) as image_handle:
                image = np.array(image_handle.convert("RGB"))
            if image.shape[:2] != gt_mask.shape[:2]:
                image = np.array(Image.fromarray(image).resize((gt_mask.shape[1], gt_mask.shape[0]), resample=Image.BILINEAR))

            ignore_index = dataset_config.ignore_index
            valid_mask = np.ones_like(gt_mask, dtype=bool)
            if ignore_index is not None:
                valid_mask &= gt_mask != ignore_index

            valid_pixels = int(valid_mask.sum())
            if valid_pixels == 0:
                continue

            labels = sorted(set(np.unique(pred_mask[valid_mask]).tolist()) | set(np.unique(gt_mask[valid_mask]).tolist()))
            iou_scores, dice_scores = _collect_label_metrics(pred_mask, gt_mask, labels, valid_mask)
            macro_iou = float(np.mean(list(iou_scores.values()))) if iou_scores else 0.0
            macro_dice = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
            pixel_accuracy = float((pred_mask[valid_mask] == gt_mask[valid_mask]).mean())
            error_pixels = int(np.logical_and(pred_mask != gt_mask, valid_mask).sum())
            error_rate = float(error_pixels / valid_pixels)
            false_positive_pixels, false_negative_pixels, class_confusion_pixels = _error_breakdown(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                valid_mask=valid_mask,
                background_index=dataset_config.background_index,
            )

            temporal_iou_prev = ""
            if prev_pred_mask is not None:
                temporal_intersection = np.logical_and(prev_pred_mask == pred_mask, valid_mask).sum()
                temporal_union = valid_pixels
                temporal_iou_prev = float(temporal_intersection / temporal_union) if temporal_union > 0 else ""
            prev_pred_mask = pred_mask.copy()

            pred_classes = sorted(np.unique(pred_mask[valid_mask]).tolist())
            gt_classes = sorted(np.unique(gt_mask[valid_mask]).tolist())
            pred_rgb = _colorize_mask(pred_mask, dataset_config.palette, dataset_config.ignore_index)
            gt_rgb = _colorize_mask(gt_mask, dataset_config.palette, dataset_config.ignore_index)
            error_map = _build_error_map(pred_mask, gt_mask, valid_mask, dataset_config.background_index)
            overlay = _make_overlay(image, pred_rgb, gt_rgb, error_map, confidence_map)

            artifact_dir = output_root / "artifacts" / video_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pred_artifact_path = artifact_dir / f"{frame_name}_pred_rgb.png"
            gt_artifact_path = artifact_dir / f"{frame_name}_gt_rgb.png"
            error_artifact_path = artifact_dir / f"{frame_name}_error_map.png"
            overlay_artifact_path = artifact_dir / f"{frame_name}_overlay.png"
            image_artifact_path = artifact_dir / f"{frame_name}_image.png"

            Image.fromarray(image).save(image_artifact_path)
            Image.fromarray(pred_rgb).save(pred_artifact_path)
            Image.fromarray(gt_rgb).save(gt_artifact_path)
            Image.fromarray(error_map).save(error_artifact_path)
            Image.fromarray(overlay).save(overlay_artifact_path)

            confidence_stats = {
                "confidence_mean": "",
                "confidence_std": "",
                "confidence_min": "",
                "confidence_max": "",
            }
            confidence_artifact_path = ""
            if confidence_map is not None:
                confidence_image = (np.clip(confidence_map, 0.0, 1.0) * 255).astype(np.uint8)
                confidence_artifact = artifact_dir / f"{frame_name}_confidence.png"
                Image.fromarray(confidence_image, mode="L").save(confidence_artifact)
                confidence_artifact_path = str(confidence_artifact)
                confidence_stats = {
                    "confidence_mean": float(confidence_map.mean()),
                    "confidence_std": float(confidence_map.std()),
                    "confidence_min": float(confidence_map.min()),
                    "confidence_max": float(confidence_map.max()),
                }

            report_rows.append({
                "video_name": video_name,
                "frame_name": frame_name,
                "image_path": str(frame_path),
                "pred_mask_path": str(pred_file),
                "gt_mask_path": str(gt_path),
                "confidence_path": confidence_artifact_path,
                "artifact_dir": str(artifact_dir),
                "pixel_accuracy": pixel_accuracy,
                "macro_iou": macro_iou,
                "macro_dice": macro_dice,
                "error_pixels": error_pixels,
                "error_rate": error_rate,
                "false_positive_pixels": false_positive_pixels,
                "false_negative_pixels": false_negative_pixels,
                "class_confusion_pixels": class_confusion_pixels,
                "temporal_iou_prev": temporal_iou_prev,
                "num_pred_classes": len(pred_classes),
                "num_gt_classes": len(gt_classes),
                "pred_classes": json.dumps(pred_classes),
                "gt_classes": json.dumps(gt_classes),
                "per_class_iou": json.dumps(iou_scores),
                "per_class_dice": json.dumps(dice_scores),
                **confidence_stats,
            })
            summary["videos"][video_name]["frames"] += 1
            summary["frames_analyzed"] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    write_rows_to_csv(output_root / "analysis.csv", report_rows, REPORT_COLUMNS)

    if report_rows:
        summary["macro_iou_mean"] = float(np.mean([row["macro_iou"] for row in report_rows]))
        summary["macro_dice_mean"] = float(np.mean([row["macro_dice"] for row in report_rows]))
        summary["pixel_accuracy_mean"] = float(np.mean([row["pixel_accuracy"] for row in report_rows]))
        summary["error_rate_mean"] = float(np.mean([row["error_rate"] for row in report_rows]))
    else:
        summary["macro_iou_mean"] = 0.0
        summary["macro_dice_mean"] = 0.0
        summary["pixel_accuracy_mean"] = 0.0
        summary["error_rate_mean"] = 0.0

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
