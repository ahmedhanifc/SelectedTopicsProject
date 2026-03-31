# Disagreement-Gated Re-prompting

This document describes the disagreement-gated re-prompting module implemented on top of the baseline SASVi inference pipeline.

## Purpose

The disagreement module is designed to improve reliability during prompt-and-propagate segmentation by identifying cases where SAM2 and the Overseer diverge over time.

Baseline SASVi already supports re-prompting through class-change logic. The disagreement extension adds a second corrective trigger based on spatial disagreement between:

- the current SAM2 propagated mask
- the current Overseer prediction for the same frame

The objective is to catch temporal drift even when class labels do not obviously change.

## Implementation

The main implementation lives in:

- [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py)

Relevant functions and sections include:

- disagreement metrics: [compute_disagreement_metrics](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L429)
- optional boundary distance: [compute_boundary_distance](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L405)
- disagreement visualization export: [save_disagreement_visual](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L448)
- gate logic inside inference: [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L1213)
- CLI flags: [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L1566)

Reporting support is provided through:

- [inference_export.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools/inference_export.py)

## Core Idea

For each propagated frame, the pipeline compares the current SAM2 output against the Overseer output for the same frame.

The disagreement module can then:

1. detect whether disagreement is sufficiently large
2. count how many consecutive frames remain in that bad state
3. trigger corrective re-prompting when the configured threshold is reached

This logic is intentionally conservative and sits on top of the baseline SASVi control flow rather than replacing it.

## Metrics Used

The current implementation computes:

- class-aware disagreement IoU
- merged foreground IoU
- optional foreground boundary distance

The primary disagreement signal is SAM2-vs-Overseer IoU. The foreground IoU and boundary distance are logged to help diagnose whether a trigger reflects a semantic mismatch, a geometric drift, or both.

## Trigger Logic

At a high level, the disagreement gate works as follows:

1. propagate the current segment with SAM2
2. retrieve the Overseer prediction for the same frame
3. compute disagreement metrics
4. mark the frame as a bad disagreement frame if the configured criteria are met
5. increment a consecutive bad-frame counter
6. trigger corrective re-prompting when the counter reaches the configured threshold

The gate only evaluates disagreement after the segment-start frame. This avoids triggering immediately on the prompt frame itself.

## Boundary Trigger

The repository also supports an optional boundary-distance trigger.

When enabled, large foreground contour deviation can also mark a frame as bad, even if IoU alone is not enough to trigger.

This is useful for cases where region shape drift is clinically relevant but coarse overlap remains deceptively acceptable.

## Outputs

When `--analysis_output_dir` is provided, disagreement-aware inference writes:

- `inference_metadata.csv`
- `disagreement_gate_report.md`
- optional disagreement visualizations under `inference/disagreement_visuals/<video_name>/`

These outputs are useful for:

- auditing when the disagreement gate fired
- checking the reason for re-prompting
- analyzing IoU trends across frames
- comparing disagreement-enabled runs against baseline runs

## Main Configuration Flags

The key CLI options are:

- `--enable_disagreement_gate`
- `--disagreement_iou_threshold`
- `--disagreement_bad_frames`
- `--enable_boundary_distance_gate`
- `--boundary_distance_threshold`
- `--save_disagreement_visuals`
- `--max_disagreement_visuals`

These are defined in:

- [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py#L1566)

## Example Commands

### Baseline run

```bash
cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint ./sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint ../../checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir ../../dataset \
  --video_name video01_00160 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir ../../output_masks_baseline \
  --analysis_output_dir ../../analysis_output_baseline
```

### Disagreement-enabled run

```bash
cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint ./sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint ../../checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir ../../dataset \
  --video_name video01_00160 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir ../../output_masks_disagreement \
  --analysis_output_dir ../../analysis_output_disagreement \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2 \
  --save_disagreement_visuals \
  --max_disagreement_visuals 6
```

### Disagreement plus boundary-distance trigger

```bash
cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint ./sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint ../../checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir ../../dataset \
  --video_name video01_00160 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir ../../output_masks_disagreement_bd \
  --analysis_output_dir ../../analysis_output_disagreement_bd \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2 \
  --enable_boundary_distance_gate \
  --boundary_distance_threshold 20.0
```

## How To Verify It

### Terminal trace

Look for per-frame trace lines that include:

- disagreement IoU
- foreground IoU
- boundary distance
- bad-frame status
- disagreement counter
- whether a re-prompt was executed
- re-prompt reason

### CSV metadata

Inspect:

- `class_change_trigger`
- `disagreement_trigger`
- `reprompt_executed`
- `reprompt_reason`
- `sam2_vs_overseer_iou`
- `sam2_vs_overseer_fg_iou`
- `sam2_vs_overseer_boundary_distance`
- `disagreement_counter`

### Markdown report

Inspect:

- `analysis_output_*/inference/disagreement_gate_report.md`

This report summarizes configuration, re-prompt counts, disagreement behavior, and sampled trace rows.

## Practical Notes

- Single-frame runs are not meaningful for evaluating disagreement-gated temporal logic, because the gate relies on consecutive bad frames.
- The disagreement module is most informative on short multi-frame clips where spatial drift can accumulate.
- Boundary distance should be interpreted as an optional geometric fallback, not a replacement for the main disagreement signal.

## Scope

This module should be understood as an extension to the baseline SASVi pipeline. It does not redesign the full prompt-and-propagate framework. Instead, it adds a corrective control mechanism intended to improve robustness when propagation begins to drift away from the Overseer signal.
