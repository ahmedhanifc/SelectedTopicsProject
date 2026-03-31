from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch


INFERENCE_METADATA_COLUMNS = [
    "video_name",
    "frame_name",
    "frame_idx",
    "segment_id",
    "num_objects",
    "confidence_path",
    "confidence_mean",
    "confidence_std",
    "confidence_min",
    "confidence_max",
    "sam2_vs_overseer_iou",
    "sam2_vs_overseer_fg_iou",
    "sam2_vs_overseer_boundary_distance",
    "disagreement_bad_frame",
    "disagreement_counter",
    "class_change_trigger",
    "disagreement_trigger",
    "reprompt_executed",
    "reprompt_reason",
    "gt_mask_path",
    "gt_pixel_accuracy",
    "gt_macro_iou",
    "gt_macro_dice",
    "gt_error_rate",
]


def compute_confidence_map(mask_logits: torch.Tensor,
                           object_ids: list[int] | torch.Tensor,
                           score_thresh: float) -> np.ndarray:
    logits = mask_logits.detach().float().cpu()
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3:
        raise ValueError(f"Expected mask logits in [N,H,W] or [H,W] format, got {tuple(logits.shape)}")

    probs = torch.sigmoid(logits)
    if isinstance(object_ids, torch.Tensor):
        object_ids = object_ids.detach().cpu().tolist()
    object_ids = [int(object_id) for object_id in object_ids]

    if probs.shape[0] != len(object_ids):
        raise ValueError(
            "Number of logits channels must match number of object ids, "
            f"got {probs.shape[0]} channels and {len(object_ids)} ids"
        )

    probs_np = probs.numpy().astype(np.float32)
    confidence = 1.0 - probs_np.max(axis=0)
    thresholded_masks = probs_np > float(score_thresh)
    object_index = {object_id: index for index, object_id in enumerate(object_ids)}

    # Mirror put_per_obj_mask(): higher label ids overwrite lower ones on overlap.
    for object_id in sorted(object_ids, reverse=True):
        mask = thresholded_masks[object_index[object_id]]
        confidence[mask] = probs_np[object_index[object_id]][mask]

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def summarise_confidence_map(confidence_map: np.ndarray) -> dict[str, float]:
    return {
        "confidence_mean": float(confidence_map.mean()),
        "confidence_std": float(confidence_map.std()),
        "confidence_min": float(confidence_map.min()),
        "confidence_max": float(confidence_map.max()),
    }


def save_confidence_map(path: str | Path, confidence_map: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(confidence_map, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(output_path)


def write_rows_to_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _format_metric(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _dir_has_frames(path: str | Path) -> bool:
    frame_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    try:
        for entry in os.scandir(path):
            if entry.is_file() and Path(entry.name).suffix.lower() in frame_suffixes:
                return True
    except OSError:
        return False
    return False


def _build_logical_video_name_map(base_video_dir: str | None, candidate_names: set[str]) -> dict[str, str]:
    mapping = {name: name for name in candidate_names}
    if not base_video_dir:
        return mapping

    try:
        top_level_entries = sorted(os.scandir(base_video_dir), key=lambda entry: entry.name)
    except OSError:
        return mapping

    for entry in top_level_entries:
        if not entry.is_dir():
            continue
        if _dir_has_frames(entry.path):
            if entry.name in mapping:
                mapping[entry.name] = entry.name
            continue

        try:
            child_entries = sorted(os.scandir(entry.path), key=lambda child: child.name)
        except OSError:
            continue

        for child in child_entries:
            if child.is_dir() and child.name in mapping and _dir_has_frames(child.path):
                mapping[child.name] = entry.name

    return mapping


def _summarise_group(
    logical_video_name: str,
    summary_items: list[dict],
    trace_items: list[dict],
) -> dict[str, float | int | str | None]:
    gt_rows = [row for row in trace_items if row.get("gt_macro_iou") is not None]
    finite_boundary_values = [
        float(row["sam2_vs_overseer_boundary_distance"])
        for row in trace_items
        if row.get("sam2_vs_overseer_boundary_distance") is not None
        and np.isfinite(row["sam2_vs_overseer_boundary_distance"])
    ]
    return {
        "video_name": logical_video_name,
        "frames_processed": int(sum(int(item.get("frames_processed", 0)) for item in summary_items)),
        "segments_started": int(sum(int(item.get("segments_started", 0)) for item in summary_items)),
        "class_change_reprompts": int(sum(int(item.get("class_change_reprompts", 0)) for item in summary_items)),
        "disagreement_reprompts": int(sum(int(item.get("disagreement_reprompts", 0)) for item in summary_items)),
        "mean_iou": _mean_or_none([
            float(row["sam2_vs_overseer_iou"])
            for row in trace_items
            if row.get("sam2_vs_overseer_iou") is not None
        ]),
        "min_iou": (
            float(min(float(row["sam2_vs_overseer_iou"]) for row in trace_items if row.get("sam2_vs_overseer_iou") is not None))
            if any(row.get("sam2_vs_overseer_iou") is not None for row in trace_items)
            else None
        ),
        "gt_macro_iou_mean": _mean_or_none([float(row["gt_macro_iou"]) for row in gt_rows]),
        "gt_macro_dice_mean": _mean_or_none([float(row["gt_macro_dice"]) for row in gt_rows]),
        "gt_pixel_accuracy_mean": _mean_or_none([float(row["gt_pixel_accuracy"]) for row in gt_rows]),
        "boundary_distance_mean": _mean_or_none(finite_boundary_values),
    }


def write_markdown_report(
    path: str | Path,
    *,
    config: dict,
    video_summaries: list[dict],
    trace_rows: list[dict],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_enabled = bool(config.get("enable_boundary_distance_gate"))

    candidate_names = {
        str(item["video_name"])
        for item in video_summaries
        if item.get("video_name") is not None
    }
    candidate_names.update(
        str(row["video_name"])
        for row in trace_rows
        if row.get("video_name") is not None
    )
    logical_name_map = _build_logical_video_name_map(config.get("base_video_dir"), candidate_names)

    grouped_summaries: dict[str, list[dict]] = {}
    grouped_trace_rows: dict[str, list[dict]] = {}
    for summary in video_summaries:
        logical_name = logical_name_map.get(summary["video_name"], summary["video_name"])
        grouped_summaries.setdefault(logical_name, []).append(summary)
    for row in trace_rows:
        logical_name = logical_name_map.get(row.get("video_name"), row.get("video_name"))
        grouped_trace_rows.setdefault(logical_name, []).append(row)

    aggregated_video_summaries = []
    for logical_name in sorted(set(grouped_summaries) | set(grouped_trace_rows)):
        aggregated_video_summaries.append(
            _summarise_group(
                logical_video_name=logical_name,
                summary_items=grouped_summaries.get(logical_name, []),
                trace_items=grouped_trace_rows.get(logical_name, []),
            )
        )

    dataset_summary = _summarise_group(
        logical_video_name="dataset",
        summary_items=video_summaries,
        trace_items=trace_rows,
    )

    lines = [
        "# Disagreement-Gated Re-prompting Report",
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
        (
            "| Video | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | "
            "Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc | Mean Boundary Dist |"
            if boundary_enabled else
            "| Video | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | "
            "Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            if boundary_enabled else
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ])
    for summary in aggregated_video_summaries:
        row = (
            f"| {summary['video_name']} | {summary['frames_processed']} | {summary['segments_started']} | "
            f"{summary['class_change_reprompts']} | {summary['disagreement_reprompts']} | "
            f"{_format_metric(summary.get('mean_iou'))} | {_format_metric(summary.get('min_iou'))} | "
            f"{_format_metric(summary.get('gt_macro_iou_mean'))} | "
            f"{_format_metric(summary.get('gt_macro_dice_mean'))} | "
            f"{_format_metric(summary.get('gt_pixel_accuracy_mean'))} |"
        )
        if boundary_enabled:
            row = row[:-1] + f" {_format_metric(summary.get('boundary_distance_mean'))} |"
        lines.append(row)

    lines.extend([
        "",
        "## Dataset Summary",
        "",
        (
            "| Dataset | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | "
            "Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc | Mean Boundary Dist |"
            if boundary_enabled else
            "| Dataset | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | "
            "Mean IoU | Min IoU | GT Macro IoU | GT Macro Dice | GT Pixel Acc |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            if boundary_enabled else
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ])
    dataset_row = (
        f"| all | {dataset_summary['frames_processed']} | {dataset_summary['segments_started']} | "
        f"{dataset_summary['class_change_reprompts']} | {dataset_summary['disagreement_reprompts']} | "
        f"{_format_metric(dataset_summary.get('mean_iou'))} | {_format_metric(dataset_summary.get('min_iou'))} | "
        f"{_format_metric(dataset_summary.get('gt_macro_iou_mean'))} | "
        f"{_format_metric(dataset_summary.get('gt_macro_dice_mean'))} | "
        f"{_format_metric(dataset_summary.get('gt_pixel_accuracy_mean'))} |"
    )
    if boundary_enabled:
        dataset_row = dataset_row[:-1] + f" {_format_metric(dataset_summary.get('boundary_distance_mean'))} |"
    lines.append(dataset_row)

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    total_class_change = sum(item["class_change_reprompts"] for item in aggregated_video_summaries)
    total_disagreement = sum(item["disagreement_reprompts"] for item in aggregated_video_summaries)
    if total_disagreement > 0:
        lines.append(
            f"Disagreement gating triggered `{total_disagreement}` corrective re-prompt(s), "
            f"in addition to `{total_class_change}` class-change re-prompt(s)."
        )
    else:
        lines.append(
            f"No disagreement-based re-prompts fired. Class-change re-prompts fired `{total_class_change}` time(s)."
        )

    gt_rows = [row for row in trace_rows if row.get("gt_macro_iou") is not None]
    if gt_rows:
        lines.append(
            f" Ground-truth comparison was available for `{len(gt_rows)}` frame(s): "
            f"mean GT macro IoU `{np.mean([row['gt_macro_iou'] for row in gt_rows]):.4f}`, "
            f"mean GT macro Dice `{np.mean([row['gt_macro_dice'] for row in gt_rows]):.4f}`, "
            f"mean GT pixel accuracy `{np.mean([row['gt_pixel_accuracy'] for row in gt_rows]):.4f}`."
        )

    sampled_rows = trace_rows[: min(len(trace_rows), 60)]
    lines.extend([
        "",
        "## Sampled IoU Trace",
        "",
        (
            "| Video | Frame | Idx | Segment | IoU | FG IoU | GT IoU | GT Dice | GT Acc | Boundary Dist | "
            "Bad | Counter | Class-change | Disagreement | Re-prompt | Reason |"
            if boundary_enabled else
            "| Video | Frame | Idx | Segment | IoU | FG IoU | GT IoU | GT Dice | GT Acc | "
            "Bad | Counter | Class-change | Disagreement | Re-prompt | Reason |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |"
            if boundary_enabled else
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |"
        ),
    ])
    for row in sampled_rows:
        logical_name = logical_name_map.get(row.get("video_name"), row.get("video_name", ""))
        boundary_distance = row.get("sam2_vs_overseer_boundary_distance")
        gt_iou = row.get("gt_macro_iou")
        gt_dice = row.get("gt_macro_dice")
        gt_acc = row.get("gt_pixel_accuracy")
        trace_line = (
            f"| {logical_name} | {row.get('frame_name', '')} | {row.get('frame_idx', '')} | "
            f"{row.get('segment_id', '')} | {row.get('sam2_vs_overseer_iou', float('nan')):.4f} | "
            f"{row.get('sam2_vs_overseer_fg_iou', float('nan')):.4f} | "
            f"{_format_metric(gt_iou)} | "
            f"{_format_metric(gt_dice)} | "
            f"{_format_metric(gt_acc)} | "
            f"{row.get('disagreement_bad_frame', False)} | {row.get('disagreement_counter', 0)} | "
            f"{row.get('class_change_trigger', False)} | {row.get('disagreement_trigger', False)} | "
            f"{row.get('reprompt_executed', False)} | {row.get('reprompt_reason', '')} |"
        )
        if boundary_enabled:
            trace_line = trace_line.replace(
                f"| {row.get('disagreement_bad_frame', False)} |",
                f"| {_format_metric(boundary_distance)} | {row.get('disagreement_bad_frame', False)} |",
                1,
            )
        lines.append(trace_line)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
