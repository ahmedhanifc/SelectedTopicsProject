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
    "num_objects",
    "confidence_path",
    "confidence_mean",
    "confidence_std",
    "confidence_min",
    "confidence_max",
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
