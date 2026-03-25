from __future__ import annotations

import csv
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
]


def compute_confidence_map(mask_logits: torch.Tensor) -> np.ndarray:
    logits = mask_logits.detach().float().cpu()
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3:
        raise ValueError(f"Expected mask logits in [N,H,W] or [H,W] format, got {tuple(logits.shape)}")

    probs = torch.sigmoid(logits)
    confidence = probs.max(dim=0).values
    return confidence.numpy().astype(np.float32)


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


def write_markdown_report(
    path: str | Path,
    *,
    config: dict,
    video_summaries: list[dict],
    trace_rows: list[dict],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        "## Video Summary",
        "",
        "| Video | Frames | Segments | Class-change re-prompts | Disagreement re-prompts | Mean IoU | Min IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in video_summaries:
        lines.append(
            f"| {summary['video_name']} | {summary['frames_processed']} | {summary['segments_started']} | "
            f"{summary['class_change_reprompts']} | {summary['disagreement_reprompts']} | "
            f"{summary['mean_iou']:.4f} | {summary['min_iou']:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    total_class_change = sum(item["class_change_reprompts"] for item in video_summaries)
    total_disagreement = sum(item["disagreement_reprompts"] for item in video_summaries)
    if total_disagreement > 0:
        lines.append(
            f"Disagreement gating triggered `{total_disagreement}` corrective re-prompt(s), "
            f"in addition to `{total_class_change}` class-change re-prompt(s)."
        )
    else:
        lines.append(
            f"No disagreement-based re-prompts fired. Class-change re-prompts fired `{total_class_change}` time(s)."
        )

    sampled_rows = trace_rows[: min(len(trace_rows), 60)]
    lines.extend([
        "",
        "## Sampled IoU Trace",
        "",
        "| Video | Frame | Idx | Segment | IoU | FG IoU | Boundary Dist | Bad | Counter | Class-change | Disagreement | Re-prompt | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
    ])
    for row in sampled_rows:
        boundary_distance = row.get("sam2_vs_overseer_boundary_distance")
        if boundary_distance is None or not np.isfinite(boundary_distance):
            boundary_value = "n/a"
        else:
            boundary_value = f"{boundary_distance:.4f}"
        lines.append(
            f"| {row.get('video_name', '')} | {row.get('frame_name', '')} | {row.get('frame_idx', '')} | "
            f"{row.get('segment_id', '')} | {row.get('sam2_vs_overseer_iou', float('nan')):.4f} | "
            f"{row.get('sam2_vs_overseer_fg_iou', float('nan')):.4f} | {boundary_value} | "
            f"{row.get('disagreement_bad_frame', False)} | {row.get('disagreement_counter', 0)} | "
            f"{row.get('class_change_trigger', False)} | {row.get('disagreement_trigger', False)} | "
            f"{row.get('reprompt_executed', False)} | {row.get('reprompt_reason', '')} |"
        )

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
