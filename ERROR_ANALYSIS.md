# Error Analysis

This document describes the offline error-analysis module used to inspect SASVi predictions, confidence behavior, and per-frame failure modes.

## Purpose

The error-analysis pipeline is designed to answer two questions:

1. How well do the predicted masks match ground truth?
2. How confident was the model in the class that was finally assigned at each predicted pixel?

The module operates as a post-processing and reporting layer on top of saved SASVi outputs. It does not change inference. It reads exported masks and confidence maps, computes frame- and video-level metrics, and writes a report bundle under `reports/`.

## Implementation

The workflow is implemented in:

- [analysis_tools/run_error_analysis.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools/run_error_analysis.py)
- [analysis_tools/error_analysis.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools/error_analysis.py)
- [analysis_tools/inference_export.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools/inference_export.py)

Inference-time confidence export is triggered from:

- [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py)

## Confidence Definition

Confidence is computed in a label-aligned way.

For each pixel:

1. SASVi produces object logits.
2. The normal inference pipeline resolves the final saved label map using the same overlap and overwrite rules used for mask export.
3. The confidence map stores the probability of the class that actually won at that pixel in the final saved mask.

This is important because the confidence map is meant to explain the saved prediction, not a different intermediate representation.

In practical terms, the confidence reported by this module answers:

`How confident was the model in the class label that appears in the final output mask at this pixel?`

The offline report summarizes confidence over predicted foreground pixels only.

## Metrics

For each analyzed frame, the module computes:

- pixel accuracy
- macro IoU
- macro Dice
- error rate
- false-positive pixels
- false-negative pixels
- class-confusion pixels

When confidence maps are available, it also computes:

- foreground confidence mean
- foreground confidence std
- foreground confidence min
- foreground confidence max
- foreground pixel count
- low-confidence ratio
- medium-confidence ratio
- high-confidence ratio

## Confidence Bands And Patterns

The default confidence thresholds are:

- low: `c < 0.50`
- medium: `0.50 <= c < 0.80`
- high: `c >= 0.80`

The current pattern mapping is:

- high confidence: polka dots
- medium confidence: triangles
- low confidence: stripes

Pattern styling is controlled from:

- [visual_config.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools/visual_config.py)

The current implementation uses:

- scattered high-confidence polka dots
- scattered medium-confidence triangles
- low-confidence stripes with widened spacing and thinner lines
- variant-aware confidence lookup, so raw-mask analysis uses raw-aligned confidence maps and smoothed-mask analysis uses smoothed-aligned confidence maps when available

## Outputs

Running offline error analysis writes a report bundle containing:

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

## Report Structure

The markdown report contains:

- configuration summary
- per-video summary table
- dataset summary table
- confidence summary table
- per-class summary table
- error breakdown table
- worst-frame table
- lowest-confidence-frame table

This makes it possible to compare geometric segmentation quality and confidence behavior in the same place.

## Recommended Workflow

The recommended workflow has two steps.

### 1. Run inference

Example for `video01_00160` with the disagreement-enabled pipeline:

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

This produces:

- masks under `output_masks_video01_00160_confidence_check/`
- confidence maps under `analysis_output_video01_00160_confidence_check/inference/confidence_maps/`

### 2. Run offline error analysis

```bash
python main.py error-analysis \
  --dataset CHOLECSEG8K \
  --frames-root frame_root \
  --pred-root output_masks_video01_00160_confidence_check/raw \
  --gt-root gt_root \
  --confidence-root analysis_output_video01_00160_confidence_check/inference/confidence_maps \
  --run-name video01_00160_confidence_check_raw
```

By default, this writes the report bundle to:

```text
reports/error_analysis_video01_00160_confidence_check_raw/
```

## Single-Frame Check

For quick visual debugging, it is often useful to run a single frame first.

Example:

```bash
python main.py run \
  --mode disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --video-name video01_00160 \
  --frame-name 0001 \
  --disagreement-threshold 0.90 \
  --bad-frames 2 \
  --run-name video01_00160_oneframe_colorcheck
```

Then:

```bash
python main.py error-analysis \
  --dataset CHOLECSEG8K \
  --frames-root frame_root \
  --pred-root output_masks_video01_00160_oneframe_colorcheck/raw \
  --gt-root gt_root \
  --confidence-root analysis_output_video01_00160_oneframe_colorcheck/inference/confidence_maps \
  --run-name video01_00160_oneframe_colorcheck_raw
```

This is useful for checking:

- class palette changes
- confidence pattern styling
- whether the confidence summary matches the visual artifact

## How To Interpret Results

Use the report together with the patterned masks.

Questions to ask:

- Are the lowest-confidence frames also the lowest-IoU frames?
- Are high-confidence regions visually stable and clinically plausible?
- Are low-confidence stripes concentrated at boundaries, small structures, or class transitions?
- Does the confidence summary agree with what the patterned mask suggests?

If the patterned masks and the report disagree strongly, the confidence export should be treated as suspect and re-validated against the final saved label map.

## Notes

- The report summarizes confidence on predicted foreground only.
- Ground-truth quality metrics and confidence metrics should be interpreted together, not in isolation.
- The offline analysis is intended for inspection and reporting, not for modifying the underlying model output.
