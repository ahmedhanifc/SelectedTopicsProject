# Disagreement-Gated Re-prompting

## What changed

This change extends the existing SASVi inference pipeline so re-prompting can now happen for two reasons:

1. The original class-change trigger still works as before.
2. A new disagreement trigger can force corrective re-prompting when SAM2 and the Overseer disagree spatially for multiple consecutive frames.

The new logic lives in `src/sam2/eval_sasvi.py` inside the main `sasvi_inference(...)` propagation loop, where both of these masks are available for the same frame:

- current SAM2 propagated mask
- current Overseer prediction for that frame

## Files changed

- `src/sam2/eval_sasvi.py`
  - added disagreement metrics
  - added disagreement gate state and trigger logic
  - added per-frame trace logging
  - added CLI flags for enabling and tuning the gate
  - added optional disagreement debug visual export
  - extended the existing dataset-path handling so raw nested `dataset/videoXX/videoXX_XXXXX` folders still work

- `analysis_tools/inference_export.py`
  - extended CSV export columns with disagreement and re-prompt trace fields
  - added markdown report writer for post-run summaries

- `src/sam2/sam2/utils/misc.py`
  - already updated earlier to support frame folders containing `.png` images, which is required for your dataset layout

## Implemented behavior

For each processed frame during propagation:

1. SASVi gets the current SAM2 output mask.
2. SASVi gets the current Overseer mask for the same frame.
3. It computes:
   - mean class-wise IoU over the union of present labels
   - merged foreground IoU
   - optional foreground boundary distance
4. If disagreement gating is enabled and the IoU stays below threshold for `N` consecutive frames, SASVi breaks the current propagation segment and starts a corrective re-prompt from that frame.

The disagreement gate is conservative by default:

- disabled by default
- IoU threshold default: `0.5`
- consecutive bad frames default: `2`

## Trigger precedence

If both triggers could happen on the same frame:

- class-change trigger takes precedence
- disagreement trigger is used as an additional fallback when class labels do not change but the segmentation drifts spatially

## Trace outputs

Terminal output now includes per-frame trace lines like:

- frame index
- segment id
- class-change fired or not
- disagreement fired or not
- IoU
- foreground IoU
- boundary distance if enabled
- bad-disagreement counter
- whether a re-prompt executed
- reason for re-prompt

If `--analysis_output_dir` is set, SASVi also writes:

- `analysis_output/inference/inference_metadata.csv`
- `analysis_output/inference/disagreement_gate_report.md`
- optional visuals under `analysis_output/inference/disagreement_visuals/<video_name>/`

## Exact commands

### Activate environment

```bash
conda activate sasvi
cd /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2
```

### Baseline run on one short clip range

This keeps disagreement gating off and gives you the original behavior with the new trace/report outputs.

```bash
python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/dataset \
  --video_name video01_00080 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/output_masks_baseline \
  --analysis_output_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_output_baseline
```

### Disagreement-gated run on the same short range

```bash
python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/dataset \
  --video_name video01_00080 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/output_masks_disagreement \
  --analysis_output_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_output_disagreement \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2 \
  --save_disagreement_visuals \
  --max_disagreement_visuals 6
```

### Disagreement-gated run with optional boundary trigger

```bash
python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/dataset \
  --video_name video01_00080 \
  --start_frame 100 \
  --end_frame 110 \
  --output_mask_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/output_masks_disagreement_bd \
  --analysis_output_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_output_disagreement_bd \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2 \
  --enable_boundary_distance_gate \
  --boundary_distance_threshold 20.0
```

### Smallest meaningful temporal test

A single frame is not meaningful for disagreement-gated temporal logic because the gate needs consecutive bad frames. Use a tiny range instead, for example:

```bash
--video_name video01_00080 --start_frame 100 --end_frame 103
```

## How to verify it worked

### In terminal

Look for lines like:

- `[trace] ... iou=... bad=True counter=2 reprompt=True reason=disagreement`
- `[reprompt] ... reason=disagreement ...`

If you only see `reason=class-change`, then the new gate did not fire on that test range.

### In CSV

Check:

- `class_change_trigger`
- `disagreement_trigger`
- `reprompt_executed`
- `reprompt_reason`
- `sam2_vs_overseer_iou`
- `disagreement_counter`

### In markdown report

Open:

- `analysis_output_*/inference/disagreement_gate_report.md`

This summarizes:

- configuration used
- frames processed
- class-change re-prompts
- disagreement re-prompts
- sampled IoU trace

## Assumptions and limitations

- The gate compares the current propagated SAM2 mask to the current Overseer prediction for the same frame.
- The primary trigger metric is mean class-wise IoU over the union of labels present in either mask.
- Foreground IoU is also logged for debugging.
- Boundary distance is optional and uses a foreground contour comparison.
- Single-frame runs are supported by the entrypoint, but they are not useful for testing this temporal feature.
- This implementation is intentionally conservative and modular. It adds a second trigger path rather than redesigning the full SASVi control flow.
