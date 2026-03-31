from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import DatasetConfig
from .inference_export import write_rows_to_csv
from .visual_config import (
    BACKGROUND_BLUR_RADIUS,
    BACKGROUND_DIM_FACTOR,
    ERROR_OVERLAY_ALPHA,
    GROUND_TRUTH_OVERLAY_ALPHA,
    HIGH_CONFIDENCE_SPACING,
    LOW_CONFIDENCE_SPACING,
    MEDIUM_CONFIDENCE_SPACING,
    PATTERN_LINE_WIDTH_DIVISOR,
    PATTERN_LINE_WIDTH_MIN,
    POLKA_DOT_RADIUS,
    PREDICTION_OVERLAY_ALPHA,
    TRIANGLE_SIDE,
    TRIANGLE_TILE_SIZE,
)


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
    "foreground_confidence_mean",
    "foreground_confidence_std",
    "foreground_confidence_min",
    "foreground_confidence_max",
    "foreground_pixels",
    "low_confidence_pixels",
    "medium_confidence_pixels",
    "high_confidence_pixels",
    "low_confidence_ratio",
    "medium_confidence_ratio",
    "high_confidence_ratio",
    "pred_classes",
    "gt_classes",
    "per_class_iou",
    "per_class_dice",
]

PER_VIDEO_SUMMARY_COLUMNS = [
    "video_name",
    "frames_evaluated",
    "macro_iou",
    "macro_dice",
    "pixel_accuracy",
    "error_rate",
    "false_positive_pixels",
    "false_negative_pixels",
    "class_confusion_pixels",
    "confidence_frames",
    "foreground_confidence_mean",
    "foreground_confidence_std",
    "foreground_confidence_min",
    "foreground_confidence_max",
    "low_confidence_ratio",
    "medium_confidence_ratio",
    "high_confidence_ratio",
]

PER_CLASS_SUMMARY_COLUMNS = [
    "class_id",
    "frames_present",
    "mean_iou",
    "mean_dice",
]

WORST_FRAMES_COLUMNS = [
    "video_name",
    "frame_name",
    "macro_iou",
    "macro_dice",
    "pixel_accuracy",
    "error_rate",
    "false_positive_pixels",
    "false_negative_pixels",
    "class_confusion_pixels",
    "foreground_confidence_mean",
    "low_confidence_ratio",
    "medium_confidence_ratio",
    "high_confidence_ratio",
    "artifact_dir",
]


def load_png_mask(path: Path, palette: np.ndarray | None) -> np.ndarray:
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


def load_dataset_mask(path: Path | str, dataset_config: DatasetConfig) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as image:
        mask = np.array(image)

    if mask.ndim == 2:
        if dataset_config.rgb_label_map:
            scalar_map = {
                rgb[0]: label
                for rgb, label in dataset_config.rgb_label_map.items()
                if rgb[0] == rgb[1] == rgb[2]
            }
            unique_values = set(np.unique(mask).tolist())
            if unique_values and unique_values.issubset(set(scalar_map.keys())):
                mapped = np.full(mask.shape, fill_value=255, dtype=np.uint8)
                for value, label in scalar_map.items():
                    mapped[mask == value] = label
                return mapped
        return mask.astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] >= 3 and dataset_config.rgb_label_map:
        rgb = mask[:, :, :3].astype(np.uint8)
        unique_colors = {
            tuple(color.tolist())
            for color in np.unique(rgb.reshape(-1, rgb.shape[-1]), axis=0)
        }
        known_colors = set(dataset_config.rgb_label_map.keys())
        if unique_colors and unique_colors.issubset(known_colors):
            label_mask = np.full(rgb.shape[:2], fill_value=255, dtype=np.uint8)
            for color, label in dataset_config.rgb_label_map.items():
                label_mask[np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)] = label
            return label_mask

    return load_png_mask(path, dataset_config.palette)


def compute_frame_metrics(pred_mask: np.ndarray,
                          gt_mask: np.ndarray,
                          dataset_config: DatasetConfig) -> dict[str, Any]:
    pred_mask = _resize_to_match(pred_mask.astype(np.uint8), gt_mask.shape[:2])
    ignore_index = dataset_config.ignore_index
    valid_mask = np.ones_like(gt_mask, dtype=bool)
    if ignore_index is not None:
        valid_mask &= gt_mask != ignore_index

    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        return {
            "pixel_accuracy": None,
            "macro_iou": None,
            "macro_dice": None,
            "error_pixels": None,
            "error_rate": None,
            "false_positive_pixels": None,
            "false_negative_pixels": None,
            "class_confusion_pixels": None,
            "num_pred_classes": 0,
            "num_gt_classes": 0,
            "pred_classes": [],
            "gt_classes": [],
            "per_class_iou": {},
            "per_class_dice": {},
        }

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

    return {
        "pixel_accuracy": pixel_accuracy,
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "error_pixels": error_pixels,
        "error_rate": error_rate,
        "false_positive_pixels": false_positive_pixels,
        "false_negative_pixels": false_negative_pixels,
        "class_confusion_pixels": class_confusion_pixels,
        "num_pred_classes": len(np.unique(pred_mask[valid_mask]).tolist()),
        "num_gt_classes": len(np.unique(gt_mask[valid_mask]).tolist()),
        "pred_classes": sorted(np.unique(pred_mask[valid_mask]).tolist()),
        "gt_classes": sorted(np.unique(gt_mask[valid_mask]).tolist()),
        "per_class_iou": iou_scores,
        "per_class_dice": dice_scores,
    }


def _load_confidence_map(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as image:
        confidence = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return confidence


def _summarize_foreground_confidence(
    confidence_map: np.ndarray | None,
    pred_mask: np.ndarray,
    valid_mask: np.ndarray,
    background_index: int | None,
    ignore_index: int | None,
    low_threshold: float,
    medium_threshold: float,
) -> dict[str, float | int | str]:
    empty_stats: dict[str, float | int | str] = {
        "foreground_confidence_mean": "",
        "foreground_confidence_std": "",
        "foreground_confidence_min": "",
        "foreground_confidence_max": "",
        "foreground_pixels": 0,
        "low_confidence_pixels": 0,
        "medium_confidence_pixels": 0,
        "high_confidence_pixels": 0,
        "low_confidence_ratio": "",
        "medium_confidence_ratio": "",
        "high_confidence_ratio": "",
    }
    if confidence_map is None:
        return empty_stats

    foreground_mask = valid_mask.copy()
    if background_index is not None:
        foreground_mask &= pred_mask != background_index
    if ignore_index is not None:
        foreground_mask &= pred_mask != ignore_index

    foreground_pixels = int(foreground_mask.sum())
    if foreground_pixels == 0:
        return empty_stats

    foreground_confidence = np.clip(confidence_map[foreground_mask], 0.0, 1.0)
    low_confidence_pixels = int((foreground_confidence <= low_threshold).sum())
    medium_confidence_pixels = int(
        np.logical_and(foreground_confidence > low_threshold, foreground_confidence <= medium_threshold).sum()
    )
    high_confidence_pixels = int((foreground_confidence > medium_threshold).sum())

    return {
        "foreground_confidence_mean": float(foreground_confidence.mean()),
        "foreground_confidence_std": float(foreground_confidence.std()),
        "foreground_confidence_min": float(foreground_confidence.min()),
        "foreground_confidence_max": float(foreground_confidence.max()),
        "foreground_pixels": foreground_pixels,
        "low_confidence_pixels": low_confidence_pixels,
        "medium_confidence_pixels": medium_confidence_pixels,
        "high_confidence_pixels": high_confidence_pixels,
        "low_confidence_ratio": float(low_confidence_pixels / foreground_pixels),
        "medium_confidence_ratio": float(medium_confidence_pixels / foreground_pixels),
        "high_confidence_ratio": float(high_confidence_pixels / foreground_pixels),
    }


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


def _make_pattern_mask(shape_hw: tuple[int, int], spacing: int, pattern_name: str) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.indices((height, width))
    if pattern_name == "stripes":
        line_width = max(PATTERN_LINE_WIDTH_MIN, spacing // PATTERN_LINE_WIDTH_DIVISOR)
        return ((xx + yy) % spacing) < line_width

    if pattern_name == "polka_dots":
        center = spacing // 2
        radius = min(POLKA_DOT_RADIUS, max(1, spacing // 3))
        dot_x = (xx % spacing) - center
        dot_y = (yy % spacing) - center
        return (dot_x * dot_x + dot_y * dot_y) <= (radius * radius)

    if pattern_name == "triangles":
        tile = max(TRIANGLE_TILE_SIZE, spacing * 2)
        tile_x = xx // tile
        tile_y = yy // tile
        local_x = xx % tile
        local_y = yy % tile

        # Deterministic per-tile jitter so triangles feel scattered instead of gridded.
        seed = (tile_x * 73 + tile_y * 151 + 17) % 997
        offset_x = seed % max(1, tile - TRIANGLE_SIDE - 2)
        offset_y = (seed * 7) % max(1, tile - TRIANGLE_SIDE - 2)

        triangle_height = TRIANGLE_SIDE
        base_x = 1 + offset_x
        base_y = 1 + offset_y
        rel_x = local_x - base_x
        rel_y = local_y - base_y

        inside_height = np.logical_and(rel_y >= 0, rel_y <= triangle_height)
        half_width = ((triangle_height - rel_y) / triangle_height) * (triangle_height / 2.0)
        center_x = triangle_height / 2.0
        inside_width = np.abs(rel_x - center_x) <= half_width
        return np.logical_and(inside_height, inside_width)

    raise ValueError(f"Unsupported pattern: {pattern_name}")


def _apply_pattern_overlay(image: np.ndarray,
                           region_mask: np.ndarray,
                           color: np.ndarray,
                           spacing: int,
                           pattern_name: str) -> np.ndarray:
    if not region_mask.any():
        return image
    pattern_mask = _make_pattern_mask(image.shape[:2], spacing=spacing, pattern_name=pattern_name)
    mask = np.logical_and(region_mask, pattern_mask)
    output = image.copy()
    output[mask] = color
    return output


def _apply_confidence_band_patterns(
    image: np.ndarray,
    confidence_map: np.ndarray | None,
    region_mask: np.ndarray,
    low_threshold: float,
    medium_threshold: float,
    pattern_color: np.ndarray,
) -> np.ndarray:
    if confidence_map is None or not region_mask.any():
        return image

    low_confidence_mask, medium_confidence_mask, high_confidence_mask = _confidence_band_masks(
        confidence_map=confidence_map,
        region_mask=region_mask,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
    )

    patterned = _apply_pattern_overlay(
        image,
        region_mask=high_confidence_mask,
        color=pattern_color,
        spacing=HIGH_CONFIDENCE_SPACING,
        pattern_name="stripes",
    )
    patterned = _apply_pattern_overlay(
        patterned,
        region_mask=medium_confidence_mask,
        color=pattern_color,
        spacing=MEDIUM_CONFIDENCE_SPACING,
        pattern_name="polka_dots",
    )
    patterned = _apply_pattern_overlay(
        patterned,
        region_mask=low_confidence_mask,
        color=pattern_color,
        spacing=LOW_CONFIDENCE_SPACING,
        pattern_name="triangles",
    )
    return patterned


def _soften_image(image: np.ndarray,
                  blur_radius: float = BACKGROUND_BLUR_RADIUS,
                  dim_factor: float = BACKGROUND_DIM_FACTOR) -> np.ndarray:
    softened = Image.fromarray(image.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    softened_np = np.array(softened, dtype=np.float32)
    softened_np *= dim_factor
    return np.clip(softened_np, 0, 255).astype(np.uint8)


def _stack_confidence_rgb(confidence_map: np.ndarray | None, image_shape: tuple[int, int, int]) -> np.ndarray:
    if confidence_map is None:
        return np.zeros(image_shape, dtype=np.uint8)
    confidence_uint8 = (np.clip(confidence_map, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([confidence_uint8] * 3, axis=-1)


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _add_panel_titles(canvas: np.ndarray, labels: list[tuple[str, tuple[int, int]]]) -> np.ndarray:
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for text, (x, y) in labels:
        _draw_label(draw, x, y, text, font)
    return np.array(image)


def _make_legend_strip(width: int,
                       has_confidence: bool,
                       low_threshold: float,
                       medium_threshold: float) -> np.ndarray:
    font = ImageFont.load_default()
    legend_height = 100 if has_confidence else 56
    legend = Image.new("RGB", (width, legend_height), color=(24, 24, 24))
    draw = ImageDraw.Draw(legend)
    draw.text((12, 8), "Legend", fill=(255, 255, 255), font=font)

    error_items = [
        ("False positive", (255, 64, 64)),
        ("False negative", (64, 128, 255)),
        ("Class confusion", (255, 196, 64)),
        ("Correct / no error", (0, 0, 0)),
    ]
    x = 12
    y = 28
    for label, color in error_items:
        draw.rectangle((x, y, x + 14, y + 14), fill=color, outline=(255, 255, 255))
        draw.text((x + 22, y), label, fill=(235, 235, 235), font=font)
        x += 150

    if has_confidence:
        y = 52
        draw.text((12, y), "Confidence patterns on correct / no-error pixels:", fill=(255, 255, 255), font=font)
        x = 250
        draw.rectangle((x, y, x + 34, y + 14), fill=(0, 0, 0), outline=(255, 255, 255))
        for offset in range(-14, 35, 4):
            draw.line((x + offset, y, x + offset + 14, y + 14), fill=(255, 255, 255), width=1)
        draw.text((x + 42, y), f"low <= {low_threshold:.2f}", fill=(235, 235, 235), font=font)

        x += 150
        draw.rectangle((x, y, x + 34, y + 14), fill=(0, 0, 0), outline=(255, 255, 255))
        for offset in range(-14, 35, 7):
            draw.line((x + offset, y, x + offset + 14, y + 14), fill=(255, 255, 255), width=1)
        draw.text((x + 42, y), f"{low_threshold:.2f} < medium <= {medium_threshold:.2f}", fill=(235, 235, 235), font=font)

        x += 210
        draw.rectangle((x, y, x + 34, y + 14), fill=(0, 0, 0), outline=(255, 255, 255))
        for offset in range(-14, 35, 10):
            draw.line((x + offset, y, x + offset + 14, y + 14), fill=(180, 180, 180), width=1)
        draw.text((x + 42, y), f"high > {medium_threshold:.2f}", fill=(235, 235, 235), font=font)

    return np.array(legend)


def _confidence_band_masks(confidence_map: np.ndarray,
                           region_mask: np.ndarray,
                           low_threshold: float,
                           medium_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_confidence_mask = np.logical_and(region_mask, confidence_map <= low_threshold)
    medium_confidence_mask = np.logical_and(
        region_mask,
        np.logical_and(confidence_map > low_threshold, confidence_map <= medium_threshold),
    )
    high_confidence_mask = np.logical_and(region_mask, confidence_map > medium_threshold)
    return low_confidence_mask, medium_confidence_mask, high_confidence_mask


def _annotate_error_map(base_canvas: np.ndarray,
                        soften_base_canvas: bool,
                        error_map: np.ndarray,
                        confidence_map: np.ndarray | None,
                        valid_mask: np.ndarray,
                        low_threshold: float,
                        medium_threshold: float) -> np.ndarray:
    base_canvas = base_canvas[:, :, :3].astype(np.uint8)
    background = _soften_image(base_canvas) if soften_base_canvas else base_canvas.copy()
    annotated_map = background.copy()
    error_pixels = error_map.any(axis=2)
    if error_pixels.any():
        annotated_map[error_pixels] = _blend(background[error_pixels], error_map[error_pixels], alpha=ERROR_OVERLAY_ALPHA)
    annotated_map = _apply_confidence_band_patterns(
        annotated_map,
        confidence_map=confidence_map,
        region_mask=np.logical_and(valid_mask, ~error_map.any(axis=2)),
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
        pattern_color=np.array([255, 255, 255], dtype=np.uint8),
    )
    return annotated_map


def _build_confidence_stripe_mask(shape_hw: tuple[int, int],
                                  confidence_map: np.ndarray,
                                  min_period: int = 4,
                                  max_period: int = 18,
                                  stripe_width: int = 2) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.indices((height, width))
    confidence = np.clip(confidence_map, 0.0, 1.0)
    periods = np.rint(max_period - confidence * (max_period - min_period)).astype(np.int32)
    periods = np.clip(periods, min_period, max_period)
    stripe_mask = ((xx + yy) % periods) < stripe_width
    return stripe_mask


def _apply_confidence_pattern(pred_mask: np.ndarray,
                              pred_rgb: np.ndarray,
                              confidence_map: np.ndarray | None,
                              low_confidence_threshold: float,
                              medium_confidence_threshold: float,
                              background_index: int | None,
                              ignore_index: int | None) -> np.ndarray:
    if confidence_map is None:
        return pred_rgb.copy()

    patterned = pred_rgb.copy()
    foreground_mask = np.ones_like(pred_mask, dtype=bool)
    if background_index is not None:
        foreground_mask &= pred_mask != background_index
    if ignore_index is not None:
        foreground_mask &= pred_mask != ignore_index

    return _apply_confidence_band_patterns(
        patterned,
        confidence_map=confidence_map,
        region_mask=foreground_mask,
        low_threshold=low_confidence_threshold,
        medium_threshold=medium_confidence_threshold,
        pattern_color=np.array([255, 255, 255], dtype=np.uint8),
    )


def _make_overlay(image: np.ndarray,
                  error_canvas: np.ndarray,
                  base_label: str,
                  soften_error_canvas: bool,
                  pred_rgb: np.ndarray,
                  gt_rgb: np.ndarray,
                  error_map: np.ndarray,
                  confidence_map: np.ndarray | None,
                  valid_mask: np.ndarray,
                  low_confidence_threshold: float,
                  medium_confidence_threshold: float) -> np.ndarray:
    image = image[:, :, :3].astype(np.uint8)
    error_canvas = error_canvas[:, :, :3].astype(np.uint8)
    pred_overlay = _blend(image, pred_rgb, alpha=PREDICTION_OVERLAY_ALPHA)
    gt_overlay = _blend(image, gt_rgb, alpha=GROUND_TRUTH_OVERLAY_ALPHA)
    error_background = _soften_image(error_canvas) if soften_error_canvas else error_canvas.copy()
    error_overlay = error_background.copy()
    error_pixels = error_map.any(axis=2)
    if error_pixels.any():
        error_overlay[error_pixels] = _blend(error_background[error_pixels], error_map[error_pixels], alpha=ERROR_OVERLAY_ALPHA)

    confidence_rgb = _stack_confidence_rgb(confidence_map, image.shape)

    if confidence_map is not None:
        pred_overlay = _apply_confidence_band_patterns(
            pred_overlay,
            confidence_map=confidence_map,
            region_mask=valid_mask.copy(),
            low_threshold=low_confidence_threshold,
            medium_threshold=medium_confidence_threshold,
            pattern_color=np.array([255, 255, 255], dtype=np.uint8),
        )
        error_overlay = _apply_confidence_band_patterns(
            error_overlay,
            confidence_map=confidence_map,
            region_mask=np.logical_and(valid_mask, ~error_map.any(axis=2)),
            low_threshold=low_confidence_threshold,
            medium_threshold=medium_confidence_threshold,
            pattern_color=np.array([255, 255, 255], dtype=np.uint8),
        )

    top = np.concatenate([image, pred_overlay], axis=1)
    bottom = np.concatenate([gt_overlay, error_overlay], axis=1)
    if confidence_map is not None:
        confidence_panel = np.concatenate([confidence_rgb, confidence_rgb], axis=1)
        overlay = np.concatenate([top, bottom, confidence_panel], axis=0)
    else:
        overlay = np.concatenate([top, bottom], axis=0)
    overlay = _add_panel_titles(
        overlay,
        labels=[
            (base_label, (10, 8)),
            ("Prediction overlay", (image.shape[1] + 10, 8)),
            ("Ground truth overlay", (10, image.shape[0] + 8)),
            ("Error overlay", (image.shape[1] + 10, image.shape[0] + 8)),
            *((("Confidence map", (10, image.shape[0] * 2 + 8)),) if confidence_map is not None else ()),
        ],
    )
    return overlay


def _frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.replace("_rgb_mask", "")
    try:
        return int(stem), stem
    except ValueError:
        return 0, stem


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _dir_has_images(path: Path) -> bool:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    try:
        for entry in os.scandir(path):
            if entry.is_file() and Path(entry.name).suffix.lower() in image_suffixes:
                return True
    except OSError:
        return False
    return False


def _build_logical_video_name_map(root_dir: Path, candidate_names: set[str]) -> dict[str, str]:
    mapping = {name: name for name in candidate_names}
    try:
        top_level_entries = sorted(root_dir.iterdir(), key=lambda path: path.name)
    except OSError:
        return mapping

    for entry in top_level_entries:
        if not entry.is_dir():
            continue
        if _dir_has_images(entry):
            if entry.name in mapping:
                mapping[entry.name] = entry.name
            continue

        try:
            child_entries = sorted(entry.iterdir(), key=lambda path: path.name)
        except OSError:
            continue

        for child in child_entries:
            if child.is_dir() and child.name in mapping and _dir_has_images(child):
                mapping[child.name] = entry.name

    return mapping


def _aggregate_video_rows(report_rows: list[dict[str, Any]], logical_name_map: dict[str, str]) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in report_rows:
        logical_name = logical_name_map.get(row["video_name"], row["video_name"])
        grouped_rows.setdefault(logical_name, []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for logical_name in sorted(grouped_rows):
        rows = grouped_rows[logical_name]
        confidence_rows = [row for row in rows if row.get("foreground_confidence_mean") not in ("", None)]
        aggregated_rows.append({
            "video_name": logical_name,
            "frames_evaluated": len(rows),
            "macro_iou": _mean_or_zero([float(row["macro_iou"]) for row in rows]),
            "macro_dice": _mean_or_zero([float(row["macro_dice"]) for row in rows]),
            "pixel_accuracy": _mean_or_zero([float(row["pixel_accuracy"]) for row in rows]),
            "error_rate": _mean_or_zero([float(row["error_rate"]) for row in rows]),
            "false_positive_pixels": int(sum(int(row["false_positive_pixels"]) for row in rows)),
            "false_negative_pixels": int(sum(int(row["false_negative_pixels"]) for row in rows)),
            "class_confusion_pixels": int(sum(int(row["class_confusion_pixels"]) for row in rows)),
            "confidence_frames": len(confidence_rows),
            "foreground_confidence_mean": _mean_or_zero([float(row["foreground_confidence_mean"]) for row in confidence_rows]) if confidence_rows else None,
            "foreground_confidence_std": _mean_or_zero([float(row["foreground_confidence_std"]) for row in confidence_rows]) if confidence_rows else None,
            "foreground_confidence_min": min(float(row["foreground_confidence_min"]) for row in confidence_rows) if confidence_rows else None,
            "foreground_confidence_max": max(float(row["foreground_confidence_max"]) for row in confidence_rows) if confidence_rows else None,
            "low_confidence_ratio": _mean_or_zero([float(row["low_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
            "medium_confidence_ratio": _mean_or_zero([float(row["medium_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
            "high_confidence_ratio": _mean_or_zero([float(row["high_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
        })
    return aggregated_rows


def _aggregate_dataset_row(report_rows: list[dict[str, Any]], per_video_rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence_rows = [row for row in report_rows if row.get("foreground_confidence_mean") not in ("", None)]
    return {
        "videos_evaluated": len(per_video_rows),
        "frames_evaluated": len(report_rows),
        "macro_iou": _mean_or_zero([float(row["macro_iou"]) for row in report_rows]),
        "macro_dice": _mean_or_zero([float(row["macro_dice"]) for row in report_rows]),
        "pixel_accuracy": _mean_or_zero([float(row["pixel_accuracy"]) for row in report_rows]),
        "error_rate": _mean_or_zero([float(row["error_rate"]) for row in report_rows]),
        "false_positive_pixels": int(sum(int(row["false_positive_pixels"]) for row in report_rows)),
        "false_negative_pixels": int(sum(int(row["false_negative_pixels"]) for row in report_rows)),
        "class_confusion_pixels": int(sum(int(row["class_confusion_pixels"]) for row in report_rows)),
        "confidence_frames": len(confidence_rows),
        "foreground_confidence_mean": _mean_or_zero([float(row["foreground_confidence_mean"]) for row in confidence_rows]) if confidence_rows else None,
        "foreground_confidence_std": _mean_or_zero([float(row["foreground_confidence_std"]) for row in confidence_rows]) if confidence_rows else None,
        "foreground_confidence_min": min(float(row["foreground_confidence_min"]) for row in confidence_rows) if confidence_rows else None,
        "foreground_confidence_max": max(float(row["foreground_confidence_max"]) for row in confidence_rows) if confidence_rows else None,
        "low_confidence_ratio": _mean_or_zero([float(row["low_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
        "medium_confidence_ratio": _mean_or_zero([float(row["medium_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
        "high_confidence_ratio": _mean_or_zero([float(row["high_confidence_ratio"]) for row in confidence_rows]) if confidence_rows else None,
    }


def _aggregate_class_rows(report_rows: list[dict[str, Any]], dataset_config: DatasetConfig) -> list[dict[str, Any]]:
    per_class_values: dict[int, dict[str, list[float] | int]] = {
        class_id: {"iou_values": [], "dice_values": [], "frames_present": 0}
        for class_id in range(dataset_config.num_classes)
    }
    if dataset_config.ignore_index is not None:
        per_class_values.setdefault(int(dataset_config.ignore_index), {"iou_values": [], "dice_values": [], "frames_present": 0})

    for row in report_rows:
        per_class_iou = json.loads(row["per_class_iou"])
        per_class_dice = json.loads(row["per_class_dice"])
        class_ids = set(per_class_iou.keys()) | set(per_class_dice.keys())
        for class_id_str in class_ids:
            class_id = int(class_id_str)
            bucket = per_class_values.setdefault(class_id, {"iou_values": [], "dice_values": [], "frames_present": 0})
            if class_id_str in per_class_iou:
                bucket["iou_values"].append(float(per_class_iou[class_id_str]))  # type: ignore[index]
            if class_id_str in per_class_dice:
                bucket["dice_values"].append(float(per_class_dice[class_id_str]))  # type: ignore[index]
            bucket["frames_present"] = int(bucket["frames_present"]) + 1

    class_rows: list[dict[str, Any]] = []
    for class_id in sorted(per_class_values):
        bucket = per_class_values[class_id]
        if int(bucket["frames_present"]) == 0:
            continue
        class_rows.append({
            "class_id": class_id,
            "frames_present": int(bucket["frames_present"]),
            "mean_iou": _mean_or_zero(list(bucket["iou_values"])),  # type: ignore[arg-type]
            "mean_dice": _mean_or_zero(list(bucket["dice_values"])),  # type: ignore[arg-type]
        })
    return class_rows


def _build_worst_frame_rows(report_rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        report_rows,
        key=lambda row: (float(row["macro_iou"]), -float(row["error_rate"]), row["video_name"], row["frame_name"]),
    )
    return [
        {
            "video_name": row["video_name"],
            "frame_name": row["frame_name"],
            "macro_iou": row["macro_iou"],
            "macro_dice": row["macro_dice"],
            "pixel_accuracy": row["pixel_accuracy"],
            "error_rate": row["error_rate"],
            "false_positive_pixels": row["false_positive_pixels"],
            "false_negative_pixels": row["false_negative_pixels"],
            "class_confusion_pixels": row["class_confusion_pixels"],
            "foreground_confidence_mean": row.get("foreground_confidence_mean"),
            "low_confidence_ratio": row.get("low_confidence_ratio"),
            "medium_confidence_ratio": row.get("medium_confidence_ratio"),
            "high_confidence_ratio": row.get("high_confidence_ratio"),
            "artifact_dir": row["artifact_dir"],
        }
        for row in sorted_rows[:limit]
    ]


def _build_lowest_confidence_frame_rows(report_rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    confidence_rows = [row for row in report_rows if row.get("foreground_confidence_mean") not in ("", None)]
    sorted_rows = sorted(
        confidence_rows,
        key=lambda row: (
            float(row["foreground_confidence_mean"]),
            -float(row.get("low_confidence_ratio", 0.0)),
            row["video_name"],
            row["frame_name"],
        ),
    )
    return [
        {
            "video_name": row["video_name"],
            "frame_name": row["frame_name"],
            "foreground_confidence_mean": row.get("foreground_confidence_mean"),
            "foreground_confidence_std": row.get("foreground_confidence_std"),
            "foreground_confidence_min": row.get("foreground_confidence_min"),
            "foreground_confidence_max": row.get("foreground_confidence_max"),
            "low_confidence_ratio": row.get("low_confidence_ratio"),
            "medium_confidence_ratio": row.get("medium_confidence_ratio"),
            "high_confidence_ratio": row.get("high_confidence_ratio"),
            "artifact_dir": row["artifact_dir"],
        }
        for row in sorted_rows[:limit]
    ]


def _write_markdown_report(
    path: Path,
    *,
    config: dict[str, Any],
    per_video_rows: list[dict[str, Any]],
    dataset_row: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
    worst_video_rows: list[dict[str, Any]],
    worst_frame_rows: list[dict[str, Any]],
    lowest_confidence_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Error Analysis Report",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Per-Video Summary",
        "",
        "| Video | Frames Evaluated | Macro IoU | Macro Dice | Pixel Accuracy | Error Rate | False Positive Pixels | False Negative Pixels | Class Confusion Pixels | Confidence Frames | Foreground Confidence Mean | Low Confidence Ratio | Medium Confidence Ratio | High Confidence Ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in per_video_rows:
        lines.append(
            f"| {row['video_name']} | {row['frames_evaluated']} | {_format_metric(row['macro_iou'])} | "
            f"{_format_metric(row['macro_dice'])} | {_format_metric(row['pixel_accuracy'])} | "
            f"{_format_metric(row['error_rate'])} | {row['false_positive_pixels']} | "
            f"{row['false_negative_pixels']} | {row['class_confusion_pixels']} | {row['confidence_frames']} | "
            f"{_format_metric(row['foreground_confidence_mean'])} | {_format_metric(row['low_confidence_ratio'])} | "
            f"{_format_metric(row['medium_confidence_ratio'])} | {_format_metric(row['high_confidence_ratio'])} |"
        )

    lines.extend([
        "",
        "## Dataset Summary",
        "",
        "| Videos Evaluated | Frames Evaluated | Macro IoU | Macro Dice | Pixel Accuracy | Error Rate | False Positive Pixels | False Negative Pixels | Class Confusion Pixels | Confidence Frames | Foreground Confidence Mean | Foreground Confidence Std | Foreground Confidence Min | Foreground Confidence Max | Low Confidence Ratio | Medium Confidence Ratio | High Confidence Ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {dataset_row['videos_evaluated']} | {dataset_row['frames_evaluated']} | "
            f"{_format_metric(dataset_row['macro_iou'])} | {_format_metric(dataset_row['macro_dice'])} | "
            f"{_format_metric(dataset_row['pixel_accuracy'])} | {_format_metric(dataset_row['error_rate'])} | "
            f"{dataset_row['false_positive_pixels']} | {dataset_row['false_negative_pixels']} | "
            f"{dataset_row['class_confusion_pixels']} | {dataset_row['confidence_frames']} | "
            f"{_format_metric(dataset_row['foreground_confidence_mean'])} | {_format_metric(dataset_row['foreground_confidence_std'])} | "
            f"{_format_metric(dataset_row['foreground_confidence_min'])} | {_format_metric(dataset_row['foreground_confidence_max'])} | "
            f"{_format_metric(dataset_row['low_confidence_ratio'])} | {_format_metric(dataset_row['medium_confidence_ratio'])} | "
            f"{_format_metric(dataset_row['high_confidence_ratio'])} |"
        ),
        "",
        "## Confidence Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Frames With Confidence Maps | {dataset_row['confidence_frames']} |",
        f"| Foreground Confidence Mean | {_format_metric(dataset_row['foreground_confidence_mean'])} |",
        f"| Foreground Confidence Std | {_format_metric(dataset_row['foreground_confidence_std'])} |",
        f"| Foreground Confidence Min | {_format_metric(dataset_row['foreground_confidence_min'])} |",
        f"| Foreground Confidence Max | {_format_metric(dataset_row['foreground_confidence_max'])} |",
        f"| Low Confidence Ratio | {_format_metric(dataset_row['low_confidence_ratio'])} |",
        f"| Medium Confidence Ratio | {_format_metric(dataset_row['medium_confidence_ratio'])} |",
        f"| High Confidence Ratio | {_format_metric(dataset_row['high_confidence_ratio'])} |",
        "",
        "## Per-Class Summary",
        "",
        "| Class ID | Frames Present | Mean IoU | Mean Dice |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in per_class_rows:
        lines.append(
            f"| {row['class_id']} | {row['frames_present']} | "
            f"{_format_metric(row['mean_iou'])} | {_format_metric(row['mean_dice'])} |"
        )

    total_error_pixels = (
        int(dataset_row["false_positive_pixels"])
        + int(dataset_row["false_negative_pixels"])
        + int(dataset_row["class_confusion_pixels"])
    )
    lines.extend([
        "",
        "## Error Breakdown",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| False Positive Pixels | {dataset_row['false_positive_pixels']} |",
        f"| False Negative Pixels | {dataset_row['false_negative_pixels']} |",
        f"| Class Confusion Pixels | {dataset_row['class_confusion_pixels']} |",
        f"| Total Error Pixels | {total_error_pixels} |",
        "",
        "## Worst Videos / Worst Frames",
        "",
        "### Worst Videos",
        "",
        "| Video | Frames Evaluated | Macro IoU | Macro Dice | Pixel Accuracy | Error Rate | False Positive Pixels | False Negative Pixels | Class Confusion Pixels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in worst_video_rows:
        lines.append(
            f"| {row['video_name']} | {row['frames_evaluated']} | {_format_metric(row['macro_iou'])} | "
            f"{_format_metric(row['macro_dice'])} | {_format_metric(row['pixel_accuracy'])} | "
            f"{_format_metric(row['error_rate'])} | {row['false_positive_pixels']} | "
            f"{row['false_negative_pixels']} | {row['class_confusion_pixels']} |"
        )

    lines.extend([
        "",
        "### Worst Frames",
        "",
        "| Video | Frame | Macro IoU | Macro Dice | Pixel Accuracy | Error Rate | False Positive Pixels | False Negative Pixels | Class Confusion Pixels | Foreground Confidence Mean | Low Confidence Ratio | Medium Confidence Ratio | High Confidence Ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in worst_frame_rows:
        lines.append(
            f"| {row['video_name']} | {row['frame_name']} | {_format_metric(row['macro_iou'])} | "
            f"{_format_metric(row['macro_dice'])} | {_format_metric(row['pixel_accuracy'])} | "
            f"{_format_metric(row['error_rate'])} | {row['false_positive_pixels']} | "
            f"{row['false_negative_pixels']} | {row['class_confusion_pixels']} | "
            f"{_format_metric(row['foreground_confidence_mean'])} | {_format_metric(row['low_confidence_ratio'])} | "
            f"{_format_metric(row['medium_confidence_ratio'])} | {_format_metric(row['high_confidence_ratio'])} |"
        )

    lines.extend([
        "",
        "### Lowest Confidence Frames",
        "",
        "| Video | Frame | Foreground Confidence Mean | Foreground Confidence Std | Foreground Confidence Min | Foreground Confidence Max | Low Confidence Ratio | Medium Confidence Ratio | High Confidence Ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in lowest_confidence_rows:
        lines.append(
            f"| {row['video_name']} | {row['frame_name']} | "
            f"{_format_metric(row['foreground_confidence_mean'])} | {_format_metric(row['foreground_confidence_std'])} | "
            f"{_format_metric(row['foreground_confidence_min'])} | {_format_metric(row['foreground_confidence_max'])} | "
            f"{_format_metric(row['low_confidence_ratio'])} | {_format_metric(row['medium_confidence_ratio'])} | "
            f"{_format_metric(row['high_confidence_ratio'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_predictions(frame_root: Path,
                        pred_root: Path,
                        gt_root: Path,
                        output_root: Path,
                        dataset_config: DatasetConfig,
                        confidence_root: Path | None = None,
                        pred_mask_suffix: str = "_rgb_mask.png",
                        gt_mask_suffix: str = "_rgb_mask.png",
                        image_suffix: str = ".jpg",
                        low_confidence_threshold: float = 0.35,
                        medium_confidence_threshold: float = 0.60) -> dict[str, Any]:
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

            pred_mask = load_png_mask(pred_file, dataset_config.palette)
            gt_mask = load_dataset_mask(gt_path, dataset_config)
            metrics = compute_frame_metrics(pred_mask=pred_mask, gt_mask=gt_mask, dataset_config=dataset_config)
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

            if metrics["pixel_accuracy"] is None:
                continue
            ignore_index = dataset_config.ignore_index
            valid_mask = np.ones_like(gt_mask, dtype=bool)
            if ignore_index is not None:
                valid_mask &= gt_mask != ignore_index

            valid_pixels = int(valid_mask.sum())
            if valid_pixels == 0:
                continue

            temporal_iou_prev = ""
            if prev_pred_mask is not None:
                temporal_intersection = np.logical_and(prev_pred_mask == pred_mask, valid_mask).sum()
                temporal_union = valid_pixels
                temporal_iou_prev = float(temporal_intersection / temporal_union) if temporal_union > 0 else ""
            prev_pred_mask = pred_mask.copy()

            pred_classes = metrics["pred_classes"]
            gt_classes = metrics["gt_classes"]
            pred_rgb = _colorize_mask(pred_mask, dataset_config.palette, dataset_config.ignore_index)
            pred_pattern_rgb = _apply_confidence_pattern(
                pred_mask=pred_mask,
                pred_rgb=pred_rgb,
                confidence_map=confidence_map,
                low_confidence_threshold=low_confidence_threshold,
                medium_confidence_threshold=medium_confidence_threshold,
                background_index=dataset_config.background_index,
                ignore_index=dataset_config.ignore_index,
            )
            gt_rgb = _colorize_mask(gt_mask, dataset_config.palette, dataset_config.ignore_index)
            mask_backdrop = gt_rgb.copy()
            error_map = _build_error_map(pred_mask, gt_mask, valid_mask, dataset_config.background_index)
            overlay = _make_overlay(
                image=image,
                error_canvas=image,
                base_label="Input image",
                soften_error_canvas=True,
                pred_rgb=pred_rgb,
                gt_rgb=gt_rgb,
                error_map=error_map,
                confidence_map=confidence_map,
                valid_mask=valid_mask,
                low_confidence_threshold=low_confidence_threshold,
                medium_confidence_threshold=medium_confidence_threshold,
            )
            overlay_mask = _make_overlay(
                image=image,
                error_canvas=mask_backdrop,
                base_label="Input image",
                soften_error_canvas=False,
                pred_rgb=pred_rgb,
                gt_rgb=gt_rgb,
                error_map=error_map,
                confidence_map=confidence_map,
                valid_mask=valid_mask,
                low_confidence_threshold=low_confidence_threshold,
                medium_confidence_threshold=medium_confidence_threshold,
            )
            error_map_annotated = _annotate_error_map(
                base_canvas=image,
                soften_base_canvas=True,
                error_map=error_map,
                confidence_map=confidence_map,
                valid_mask=valid_mask,
                low_threshold=low_confidence_threshold,
                medium_threshold=medium_confidence_threshold,
            )
            error_map_mask_annotated = _annotate_error_map(
                base_canvas=mask_backdrop,
                soften_base_canvas=False,
                error_map=error_map,
                confidence_map=confidence_map,
                valid_mask=valid_mask,
                low_threshold=low_confidence_threshold,
                medium_threshold=medium_confidence_threshold,
            )

            artifact_dir = output_root / "artifacts" / video_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pred_artifact_path = artifact_dir / f"{frame_name}_pred_rgb.png"
            pred_pattern_artifact_path = artifact_dir / f"{frame_name}_pred_confidence_pattern.png"
            gt_artifact_path = artifact_dir / f"{frame_name}_gt_rgb.png"
            error_artifact_path = artifact_dir / f"{frame_name}_error_map.png"
            error_mask_artifact_path = artifact_dir / f"{frame_name}_error_map_mask.png"
            overlay_artifact_path = artifact_dir / f"{frame_name}_overlay.png"
            overlay_mask_artifact_path = artifact_dir / f"{frame_name}_overlay_mask.png"
            image_artifact_path = artifact_dir / f"{frame_name}_image.png"

            Image.fromarray(image).save(image_artifact_path)
            Image.fromarray(pred_rgb).save(pred_artifact_path)
            Image.fromarray(pred_pattern_rgb).save(pred_pattern_artifact_path)
            Image.fromarray(gt_rgb).save(gt_artifact_path)
            Image.fromarray(error_map_annotated).save(error_artifact_path)
            Image.fromarray(error_map_mask_annotated).save(error_mask_artifact_path)
            Image.fromarray(overlay).save(overlay_artifact_path)
            Image.fromarray(overlay_mask).save(overlay_mask_artifact_path)

            confidence_stats = {
                "confidence_mean": "",
                "confidence_std": "",
                "confidence_min": "",
                "confidence_max": "",
            }
            foreground_confidence_stats: dict[str, float | int | str] = {
                "foreground_confidence_mean": "",
                "foreground_confidence_std": "",
                "foreground_confidence_min": "",
                "foreground_confidence_max": "",
                "foreground_pixels": 0,
                "low_confidence_pixels": 0,
                "medium_confidence_pixels": 0,
                "high_confidence_pixels": 0,
                "low_confidence_ratio": "",
                "medium_confidence_ratio": "",
                "high_confidence_ratio": "",
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
                foreground_confidence_stats = _summarize_foreground_confidence(
                    confidence_map=confidence_map,
                    pred_mask=pred_mask,
                    valid_mask=valid_mask,
                    background_index=dataset_config.background_index,
                    ignore_index=dataset_config.ignore_index,
                    low_threshold=low_confidence_threshold,
                    medium_threshold=medium_confidence_threshold,
                )

            report_rows.append({
                "video_name": video_name,
                "frame_name": frame_name,
                "image_path": str(frame_path),
                "pred_mask_path": str(pred_file),
                "gt_mask_path": str(gt_path),
                "confidence_path": confidence_artifact_path,
                "artifact_dir": str(artifact_dir),
                "pixel_accuracy": metrics["pixel_accuracy"],
                "macro_iou": metrics["macro_iou"],
                "macro_dice": metrics["macro_dice"],
                "error_pixels": metrics["error_pixels"],
                "error_rate": metrics["error_rate"],
                "false_positive_pixels": metrics["false_positive_pixels"],
                "false_negative_pixels": metrics["false_negative_pixels"],
                "class_confusion_pixels": metrics["class_confusion_pixels"],
                "temporal_iou_prev": temporal_iou_prev,
                "num_pred_classes": len(pred_classes),
                "num_gt_classes": len(gt_classes),
                "pred_classes": json.dumps(pred_classes),
                "gt_classes": json.dumps(gt_classes),
                "per_class_iou": json.dumps(metrics["per_class_iou"]),
                "per_class_dice": json.dumps(metrics["per_class_dice"]),
                **confidence_stats,
                **foreground_confidence_stats,
            })
            summary["videos"][video_name]["frames"] += 1
            summary["frames_analyzed"] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    write_rows_to_csv(output_root / "analysis.csv", report_rows, REPORT_COLUMNS)

    logical_name_map = _build_logical_video_name_map(
        frame_root,
        {str(row["video_name"]) for row in report_rows},
    )
    per_video_rows = _aggregate_video_rows(report_rows, logical_name_map)
    dataset_row = _aggregate_dataset_row(report_rows, per_video_rows)
    per_class_rows = _aggregate_class_rows(report_rows, dataset_config)
    worst_video_rows = sorted(per_video_rows, key=lambda row: (float(row["macro_iou"]), -float(row["error_rate"]), row["video_name"]))[:10]
    worst_frame_rows = _build_worst_frame_rows(report_rows)
    lowest_confidence_rows = _build_lowest_confidence_frame_rows(report_rows)

    write_rows_to_csv(output_root / "per_video_summary.csv", per_video_rows, PER_VIDEO_SUMMARY_COLUMNS)
    write_rows_to_csv(output_root / "per_class_summary.csv", per_class_rows, PER_CLASS_SUMMARY_COLUMNS)
    write_rows_to_csv(output_root / "worst_frames.csv", worst_frame_rows, WORST_FRAMES_COLUMNS)
    write_rows_to_csv(
        output_root / "lowest_confidence_frames.csv",
        lowest_confidence_rows,
        [
            "video_name",
            "frame_name",
            "foreground_confidence_mean",
            "foreground_confidence_std",
            "foreground_confidence_min",
            "foreground_confidence_max",
            "low_confidence_ratio",
            "medium_confidence_ratio",
            "high_confidence_ratio",
            "artifact_dir",
        ],
    )

    _write_markdown_report(
        output_root / "report.md",
        config={
            "dataset_type": dataset_config.dataset_type,
            "frames_root": str(frame_root),
            "pred_root": str(pred_root),
            "gt_root": str(gt_root),
            "output_root": str(output_root),
            "confidence_root": str(confidence_root) if confidence_root is not None else None,
            "pred_mask_suffix": pred_mask_suffix,
            "gt_mask_suffix": gt_mask_suffix,
            "image_suffix": image_suffix,
            "ignore_index": dataset_config.ignore_index,
            "background_index": dataset_config.background_index,
            "confidence_low_threshold": low_confidence_threshold,
            "confidence_medium_threshold": medium_confidence_threshold,
        },
        per_video_rows=per_video_rows,
        dataset_row=dataset_row,
        per_class_rows=per_class_rows,
        worst_video_rows=worst_video_rows,
        worst_frame_rows=worst_frame_rows,
        lowest_confidence_rows=lowest_confidence_rows,
    )

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
    summary["videos_evaluated"] = dataset_row["videos_evaluated"]
    summary["false_positive_pixels"] = dataset_row["false_positive_pixels"]
    summary["false_negative_pixels"] = dataset_row["false_negative_pixels"]
    summary["class_confusion_pixels"] = dataset_row["class_confusion_pixels"]
    summary["confidence_frames"] = dataset_row["confidence_frames"]
    summary["foreground_confidence_mean"] = dataset_row["foreground_confidence_mean"]
    summary["foreground_confidence_std"] = dataset_row["foreground_confidence_std"]
    summary["foreground_confidence_min"] = dataset_row["foreground_confidence_min"]
    summary["foreground_confidence_max"] = dataset_row["foreground_confidence_max"]
    summary["low_confidence_ratio"] = dataset_row["low_confidence_ratio"]
    summary["medium_confidence_ratio"] = dataset_row["medium_confidence_ratio"]
    summary["high_confidence_ratio"] = dataset_row["high_confidence_ratio"]

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
