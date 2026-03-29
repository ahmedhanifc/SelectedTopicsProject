from __future__ import annotations

import json
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
    PREDICTION_OVERLAY_ALPHA,
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


def _make_pattern_mask(shape_hw: tuple[int, int], spacing: int) -> np.ndarray:
    height, width = shape_hw
    yy, xx = np.indices((height, width))
    line_width = max(PATTERN_LINE_WIDTH_MIN, spacing // PATTERN_LINE_WIDTH_DIVISOR)
    return ((xx + yy) % spacing) < line_width


def _apply_pattern_overlay(image: np.ndarray,
                           region_mask: np.ndarray,
                           color: np.ndarray,
                           spacing: int) -> np.ndarray:
    if not region_mask.any():
        return image
    pattern_mask = _make_pattern_mask(image.shape[:2], spacing=spacing)
    mask = np.logical_and(region_mask, pattern_mask)
    output = image.copy()
    output[mask] = color
    return output


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
                        has_confidence: bool,
                        low_threshold: float,
                        medium_threshold: float) -> np.ndarray:
    base_canvas = base_canvas[:, :, :3].astype(np.uint8)
    background = _soften_image(base_canvas) if soften_base_canvas else base_canvas.copy()
    annotated_map = background.copy()
    error_pixels = error_map.any(axis=2)
    if error_pixels.any():
        annotated_map[error_pixels] = _blend(background[error_pixels], error_map[error_pixels], alpha=ERROR_OVERLAY_ALPHA)
    if confidence_map is not None:
        correct_region_mask = np.logical_and(valid_mask, ~error_map.any(axis=2))
        low_confidence_mask, medium_confidence_mask, high_confidence_mask = _confidence_band_masks(
            confidence_map=confidence_map,
            region_mask=correct_region_mask,
            low_threshold=low_threshold,
            medium_threshold=medium_threshold,
        )
        annotated_map = _apply_pattern_overlay(
            annotated_map,
            region_mask=high_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=HIGH_CONFIDENCE_SPACING,
        )
        annotated_map = _apply_pattern_overlay(
            annotated_map,
            region_mask=medium_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=MEDIUM_CONFIDENCE_SPACING,
        )
        annotated_map = _apply_pattern_overlay(
            annotated_map,
            region_mask=low_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=LOW_CONFIDENCE_SPACING,
        )
    legend = _make_legend_strip(
        width=annotated_map.shape[1],
        has_confidence=has_confidence,
        low_threshold=low_threshold,
        medium_threshold=medium_threshold,
    )
    return np.concatenate([annotated_map, legend], axis=0)


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
        confidence_region_mask = valid_mask.copy()
        low_confidence_mask, medium_confidence_mask, high_confidence_mask = _confidence_band_masks(
            confidence_map=confidence_map,
            region_mask=confidence_region_mask,
            low_threshold=low_confidence_threshold,
            medium_threshold=medium_confidence_threshold,
        )
        pred_overlay = _apply_pattern_overlay(
            pred_overlay,
            region_mask=high_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=HIGH_CONFIDENCE_SPACING,
        )
        pred_overlay = _apply_pattern_overlay(
            pred_overlay,
            region_mask=medium_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=MEDIUM_CONFIDENCE_SPACING,
        )
        pred_overlay = _apply_pattern_overlay(
            pred_overlay,
            region_mask=low_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=LOW_CONFIDENCE_SPACING,
        )
        correct_region_mask = np.logical_and(valid_mask, ~error_map.any(axis=2))
        low_correct_confidence_mask, medium_correct_confidence_mask, high_correct_confidence_mask = _confidence_band_masks(
            confidence_map=confidence_map,
            region_mask=correct_region_mask,
            low_threshold=low_confidence_threshold,
            medium_threshold=medium_confidence_threshold,
        )
        error_overlay = _apply_pattern_overlay(
            error_overlay,
            region_mask=high_correct_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=HIGH_CONFIDENCE_SPACING,
        )
        error_overlay = _apply_pattern_overlay(
            error_overlay,
            region_mask=medium_correct_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=MEDIUM_CONFIDENCE_SPACING,
        )
        error_overlay = _apply_pattern_overlay(
            error_overlay,
            region_mask=low_correct_confidence_mask,
            color=np.array([255, 255, 255], dtype=np.uint8),
            spacing=LOW_CONFIDENCE_SPACING,
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
    legend = _make_legend_strip(
        width=overlay.shape[1],
        has_confidence=confidence_map is not None,
        low_threshold=low_confidence_threshold,
        medium_threshold=medium_confidence_threshold,
    )
    return np.concatenate([overlay, legend], axis=0)


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
                has_confidence=confidence_map is not None,
                low_threshold=low_confidence_threshold,
                medium_threshold=medium_confidence_threshold,
            )
            error_map_mask_annotated = _annotate_error_map(
                base_canvas=mask_backdrop,
                soften_base_canvas=False,
                error_map=error_map,
                confidence_map=confidence_map,
                valid_mask=valid_mask,
                has_confidence=confidence_map is not None,
                low_threshold=low_confidence_threshold,
                medium_threshold=medium_confidence_threshold,
            )

            artifact_dir = output_root / "artifacts" / video_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pred_artifact_path = artifact_dir / f"{frame_name}_pred_rgb.png"
            gt_artifact_path = artifact_dir / f"{frame_name}_gt_rgb.png"
            error_artifact_path = artifact_dir / f"{frame_name}_error_map.png"
            error_mask_artifact_path = artifact_dir / f"{frame_name}_error_map_mask.png"
            overlay_artifact_path = artifact_dir / f"{frame_name}_overlay.png"
            overlay_mask_artifact_path = artifact_dir / f"{frame_name}_overlay_mask.png"
            image_artifact_path = artifact_dir / f"{frame_name}_image.png"

            Image.fromarray(image).save(image_artifact_path)
            Image.fromarray(pred_rgb).save(pred_artifact_path)
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
