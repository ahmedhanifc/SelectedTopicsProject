# Disagreement-Gated Re-prompting for Reliable Surgical Video Segmentation: A Hybrid Prompt-and-Propagate Framework

Research codebase for a hybrid prompt-and-propagate framework for surgical video segmentation, centered on three extensions to baseline SASVi:

- **disagreement-gated corrective re-prompting**
- **offline error analysis with confidence-aware visual artifacts**
- **parallelized disagreement inference**

The repository is organized as a research workspace rather than a minimal package. It contains the baseline SASVi pipeline, but the main focus of this codebase is the study, implementation, and evaluation of the extensions listed above.

## Overview

This repository investigates how a baseline prompt-and-propagate segmentation pipeline can be made more reliable, more analyzable, and more practical to run at scale.

Baseline SASVi is used as the underlying segmentation framework. On top of it, this repository adds:

- a disagreement module that can trigger corrective re-prompting when SAM2 and the overseer diverge
- an offline error-analysis module that evaluates saved masks against ground truth and visualizes confidence patterns on the predicted classes
- parallel and streaming-oriented variants of the disagreement pipeline

In other words, the baseline SASVi code is the foundation, but the research contribution represented by this repository is the extension layer built around it.

## Main Components

### 1. Disagreement-Gated Re-Prompting

The primary extension implemented in this repository is disagreement-gated corrective re-prompting.

This mechanism compares SAM2 predictions against Overseer predictions and can trigger corrective re-prompting when disagreement persists over time. The goal is to improve reliability by identifying cases where propagation is drifting away from the Overseer signal.

Current logic includes:

- semantic disagreement using class-aware IoU
- an additional foreground-overlap constraint
- optional boundary-distance triggering
- configurable consecutive-bad-frame thresholds
- optional disagreement visualizations and trace reports

### 2. Error Analysis

The offline error-analysis module lives in:

- [analysis_tools](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools)

It reads saved predictions, confidence maps, and ground-truth masks to produce:

- frame-level and video-level segmentation metrics
- confidence summaries
- error breakdowns
- patterned confidence artifacts over predicted masks

Confidence is computed in a **final-label-aligned** way: the confidence at each pixel corresponds to the class that actually appears in the final saved mask at that location.

### 3. Parallelization

The repository includes parallelized variants of the disagreement pipeline under:

- [parallelization_and_streaming](/Users/sarrachouk/Desktop/SelectedTopicsProject/parallelization_and_streaming)

These variants improve throughput by parallelizing the surrounding pipeline, including:

- Overseer precomputation
- asynchronous output export
- batched preprocessing and caching

The disagreement trigger itself remains sequential. The speedup comes from reducing surrounding overhead, not from changing the decision logic.

### 4. Baseline SASVi

The core inference pipeline lives in:

- [eval_sasvi.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/eval_sasvi.py)

This script runs SAM2 over frame folders using Overseer-generated prompts and writes:

- raw and smoothed segmentation masks
- optional binary masks
- optional inference metadata and confidence maps

## Repository Layout

Key directories:

- [src](/Users/sarrachouk/Desktop/SelectedTopicsProject/src): core Python code, including SAM2 integration and SASVi inference
- [analysis_tools](/Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_tools): offline error-analysis tooling
- [parallelization_and_streaming](/Users/sarrachouk/Desktop/SelectedTopicsProject/parallelization_and_streaming): parallel and streaming pipeline variants
- [train_scripts](/Users/sarrachouk/Desktop/SelectedTopicsProject/train_scripts): overseer training scripts
- [eval_scripts](/Users/sarrachouk/Desktop/SelectedTopicsProject/eval_scripts): evaluation utilities
- [helper_scripts](/Users/sarrachouk/Desktop/SelectedTopicsProject/helper_scripts): dataset preparation and convenience utilities
- [checkpoints](/Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints): overseer checkpoints
- [src/sam2/sam2/checkpoints](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints): SAM2 checkpoints

User-facing runner:

- [main.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/main.py)

## Setup

Recommended setup:

1. Create and activate a Python environment.
2. Install PyTorch and torchvision appropriate for your device.
3. Install the repository requirements.
4. Install the local SAM2 package in editable mode.
5. Place the required checkpoints in the expected locations.

Example:

```bash
conda create -n surgical-seg python=3.11
conda activate surgical-seg
pip install -r requirements.txt
cd src/sam2 && pip install -e .
cd ../..
```

Additional dependency:

- `SDS_Playground` is required for several dataset and visualization utilities used by this codebase.

## Expected Data Layout

Inference expects videos as frame folders:

```text
<base_video_dir>/
  <video_name>/
    0001.jpg
    0002.jpg
    ...
```

Ground-truth masks for error analysis are typically prepared into a parallel root such as:

```text
<gt_root>/
  <video_name>/
    0001_rgb_mask.png
    0002_rgb_mask.png
    ...
```

## Checkpoints

The repository currently expects:

- a SAM2 checkpoint under [src/sam2/sam2/checkpoints](/Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints)
- an Overseer checkpoint under [checkpoints](/Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints)

Example files:

- `src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt`
- `checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth`

## Recommended Entrypoint

Use [main.py](/Users/sarrachouk/Desktop/SelectedTopicsProject/main.py) as the top-level entrypoint.

List presets:

```bash
python main.py list-presets
```

### Baseline inference

```bash
python main.py run \
  --mode baseline \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --video-name video01_00160 \
  --run-name video01_baseline
```

### Disagreement-enabled inference

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
  --run-name video01_disagreement
```

### Parallel disagreement pipeline

```bash
python main.py run \
  --mode parallel-disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --disagreement-threshold 0.90 \
  --bad-frames 2 \
  --run-name parallel_disagreement_090
```

### Original timed disagreement pipeline

```bash
python main.py run \
  --mode original-timed-disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --disagreement-threshold 0.90 \
  --bad-frames 2 \
  --run-name original_timed_disagreement_090
```

## Error Analysis Workflow

To export confidence maps during inference, provide an analysis output directory through the normal inference workflow. Then run offline error analysis on the saved masks.

Example:

```bash
python main.py error-analysis \
  --dataset CHOLECSEG8K \
  --frames-root frame_root \
  --pred-root output_masks_video01_disagreement/raw \
  --gt-root gt_root \
  --confidence-root analysis_output_video01_disagreement/inference/confidence_maps \
  --run-name video01_disagreement_raw
```

This writes a report bundle under `reports/`, including:

- `analysis.csv`
- `per_video_summary.csv`
- `per_class_summary.csv`
- `worst_frames.csv`
- `lowest_confidence_frames.csv`
- `summary.json`
- `report.md`

The artifact directory also contains confidence-pattern visualizations such as:

- `*_pred_confidence_pattern.png`
- `*_overlay_mask.png`
- `*_error_map_mask.png`

## Training

The repository includes training scripts for multiple overseer architectures and datasets.

Examples:

```bash
python train_scripts/train_MaskRCNN_<DATASET>.py
python train_scripts/train_DETR_<DATASET>.py
python train_scripts/train_Mask2Former_<DATASET>.py
```

## Evaluation

Evaluation utilities are provided under [eval_scripts](/Users/sarrachouk/Desktop/SelectedTopicsProject/eval_scripts).

Typical usage includes:

- frame-wise Overseer evaluation
- full-video prediction evaluation
- video-level segmentation metrics

## Notes On Confidence

The confidence values used by the error-analysis module should be interpreted as **model support for the final assigned class at each pixel**, not as a calibrated clinical probability.

The default confidence bands currently used are:

- low: `< 0.50`
- medium: `0.50` to `< 0.80`
- high: `>= 0.80`

Pattern mapping currently used in the error-analysis artifacts:

- high confidence: polka dots
- medium confidence: triangles
- low confidence: stripes

## Practical Outputs

Across the different modules, the repository can produce:

- segmentation masks
- smoothed masks
- binary masks
- Overseer masks
- confidence maps
- disagreement-gating reports
- error-analysis reports
- parallel timing and throughput outputs
