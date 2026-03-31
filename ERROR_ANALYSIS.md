# Error Analysis

This document explains the error-analysis module, what it produces, and how to run it.

## Overview

The error-analysis workflow is a second pass on top of SASVi predictions. It does two things:

1. Reads predicted RGB masks, ground-truth masks, and optional confidence maps.
2. Produces a report bundle with frame-level metrics, video-level summaries, and confidence-pattern artifacts.

The module is implemented in:

- `analysis_tools/run_error_analysis.py`
- `analysis_tools/error_analysis.py`
- `analysis_tools/inference_export.py`

## What It Measures

For each frame, the module computes:

- pixel accuracy
- macro IoU
- macro Dice
- error rate
- false-positive pixels
- false-negative pixels
- class-confusion pixels

If confidence maps are available, it also computes confidence statistics over the predicted foreground only:

- foreground confidence mean
- foreground confidence std
- foreground confidence min / max
- low-confidence ratio
- medium-confidence ratio
- high-confidence ratio

This is the confidence of the class that appears in the generated mask at each predicted foreground pixel. The confidence export now uses the same thresholding rule as the actual mask generation, so the confidence report matches the saved prediction more faithfully.

## Confidence Patterns

When confidence maps are present, the prediction mask artifact uses one pattern family per confidence band:

- high confidence: stripes
- medium confidence: polka dots
- low confidence: triangles

Default thresholds:

- low: `<= 0.35`
- medium: `0.35 < c <= 0.60`
- high: `> 0.60`

## Main Outputs

The offline report writes:

- `analysis.csv`
- `per_video_summary.csv`
- `per_class_summary.csv`
- `worst_frames.csv`
- `lowest_confidence_frames.csv`
- `summary.json`
- `report.md`
- per-frame artifacts under `artifacts/<video_name>/`

The most useful artifact files are:

- `<frame>_pred_confidence_pattern.png`
- `<frame>_error_map_mask.png`
- `<frame>_overlay_mask.png`
- `<frame>_confidence.png`

## Running It

Recommended workflow:

1. Run inference with `main.py run` and make sure `--analysis-output-dir` is produced.
2. Run `main.py error-analysis` on the generated masks.
3. Inspect the report under `reports/`.

Example for `video01_00160` with the disagreement pipeline:

```bash
python main.py run \
  --mode disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --video-name video01_00160 \
  --disagreement-threshold 0.90 \
  --bad-frames 2 \
  --run-name video01_00160_confidence_check
```

Then run offline error analysis:

```bash
python main.py error-analysis \
  --dataset CHOLECSEG8K \
  --frames-root frame_root \
  --pred-root output_masks_video01_00160_confidence_check/raw \
  --gt-root gt_root \
  --confidence-root analysis_output_video01_00160_confidence_check/inference/confidence_maps \
  --run-name video01_00160_confidence_check_raw
```

By default, that writes the report bundle here:

```text
reports/error_analysis_video01_00160_confidence_check_raw/
```

## What To Check

Use `report.md` first. It now includes:

- per-video summary table
- dataset summary table
- confidence summary table
- worst frames table
- lowest confidence frames table

Use this to answer:

- Are most predicted foreground pixels low confidence?
- Is confidence improving or degrading across the video?
- Are the lowest-confidence frames also the worst-IoU frames?
- Are confidence patterns visually matching obvious hard regions?

Then inspect:

- `artifacts/video01_00160/*_pred_confidence_pattern.png`

to verify that the pattern overlays look sensible on the mask itself.
