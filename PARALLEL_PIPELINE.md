# Parallel Pipeline

## Overview

This repository includes a parallelized version of the disagreement-gated SASVI pipeline together with a timed wrapper for the original implementation.

The purpose of this work is not to change the disagreement algorithm itself. The purpose is to preserve the same disagreement-triggered re-prompting behavior while reducing runtime and exporting results in a format that makes timing comparison straightforward.

In practical terms:

- the original pipeline remains the reference behavior
- the parallel pipeline keeps the same decision logic
- the comparison focuses on runtime, throughput, and exported outputs

## Files

The user-facing entrypoint is:

- `main.py`

The underlying implementation lives in:

- `parallelization_and_streaming/parallel_disagreement_sasvi.py`
- `parallelization_and_streaming/time_original_disagreement_sasvi.py`

Supporting context also exists in:

- `parallelization_and_streaming/README.md`
- `src/sam2/eval_sasvi.py`

## Core Logic

The disagreement-gated workflow is still based on the same sequence:

1. Run Overseer and SAM2 on the same frame sequence.
2. Compare SAM2 and Overseer masks at each propagated frame.
3. Track disagreement across consecutive frames.
4. Trigger a corrective re-prompt when the disagreement rule is satisfied.
5. Resume propagation from the new segment start.

That control flow remains sequential in both the original and parallel versions.

## Parallelization Strategy

Several possible strategies were considered:

1. Fully parallelize frame processing.
2. Parallelize only saving and export work.
3. Precompute Overseer predictions in batches and move saving to a background worker while keeping disagreement control flow sequential.

The implemented strategy is the third one.

This was chosen because the disagreement pipeline is inherently sequential:

- propagation state depends on previous frames
- disagreement counters depend on earlier decisions
- a re-prompt can restart the segment from the current frame

Because of that, frame-by-frame parallelization would risk changing behavior. The implemented design instead parallelizes only the parts that are independent of the disagreement decision itself.

## What Was Parallelized

### 1. Batched Overseer Precomputation

In the original pipeline, Overseer predictions are fetched during the main SASVI loop as frames are processed.

In the parallel pipeline:

- Overseer predictions for the selected frames are computed up front
- they are computed in batches
- the outputs are cached and reused during the SASVI loop

This reduces repeated inference stalls inside the main propagation loop.

### 2. Asynchronous Saving

Saving masks, confidence maps, metadata rows, and disagreement visuals is moved out of the main inference path.

In the parallel pipeline:

- the main loop creates save tasks
- tasks are pushed into a queue
- a background worker writes files asynchronously

This overlaps disk I/O with ongoing model work and reduces time spent blocking on exports.

### 3. Standardized Timing and Output Structure

The parallel runner writes outputs in a structured, timestamped layout. The timed wrapper for the original pipeline writes outputs in a comparable layout so the two versions can be benchmarked fairly.

## High-Level Execution Flow

The parallel pipeline can be understood in six stages:

### Stage A: Setup

- parse CLI arguments
- discover video folders
- build the SAM2 predictor
- build the Overseer model
- create timestamped run folders
- start the asynchronous save worker

### Stage B: Frame Preparation

For each selected video:

- list available frames
- optionally filter by video name, frame name, or frame range
- optionally build a temporary subset directory when needed
- initialize video state for SAM2

### Stage C: Overseer Precomputation

Before the main SASVI loop:

- run Overseer on the selected frames in batches
- store predictions in memory
- record the time spent in this stage

### Stage D: Sequential SASVI Inference

The main disagreement logic remains sequential:

- start a segment
- initialize prompts
- propagate SAM2
- compare SAM2 and Overseer outputs
- update disagreement counters
- trigger re-prompting when needed

### Stage E: Background Export

For each processed frame:

- package masks and metadata into save tasks
- queue them for the background worker
- continue inference without waiting for each write to complete

### Stage F: Finalization

After all videos are processed:

- wait for queued saves to finish
- write metadata CSVs
- write the markdown report
- write `timing_summary.json`

## Why The Original Wrapper Exists

The original disagreement pipeline already existed, but it did not naturally produce the same run-folder structure and timing summary format as the parallel version.

The timed-original runner exists so the original pipeline can be timed and exported in a directly comparable way.

It does not redesign the original disagreement logic. It only adds:

- timestamped run directories
- timing collection
- standardized metadata export
- `timing_summary.json`

## Output Layout

### Parallel Pipeline

Default output roots:

- `parallel_disagreement_frame_outputs_<run_name>/run_<timestamp>/`
- `analysis_output_disagreement_parallel_<run_name>/run_<timestamp>/`

Typical contents include:

- `output_masks/`
- optional `overseer_masks/`
- `inference/inference_metadata.csv`
- `inference/disagreement_gate_report.md`
- `run_parameters.csv`
- `timing_summary.json`

### Original Timed Pipeline

Default output roots:

- `disagreement_frame_outputs_<run_name>/run_<timestamp>/`
- `analysis_output_disagreement_original_<run_name>/run_<timestamp>/`

These are intended to mirror the same general structure so comparison is easier.

## Recommended Commands

Run from the repository root.

The recommended entrypoint is `main.py`.

### Parallel Disagreement Pipeline

```bash
python main.py run \
  --mode parallel-disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --disagreement-threshold 0.90 \
  --bad-frames 2
```

### Original Disagreement Pipeline With Timing

```bash
python main.py run \
  --mode original-timed-disagreement \
  --device cpu \
  --dataset CHOLECSEG8K \
  --base-video-dir frame_root \
  --gt-root-dir gt_root \
  --disagreement-threshold 0.90 \
  --bad-frames 2
```

To run a single clip, add:

```bash
--video-name video01_00160
```

To run a single frame, add:

```bash
--frame-name 0001
```

To give the run a clearer output root name, add:

```bash
--run-name parallel_video01_test
```

## Important Arguments

### Common Disagreement Arguments

- `--enable-disagreement-gate`
  Enables disagreement-triggered corrective re-prompting.

- `--disagreement-threshold`
  Marks a frame as bad when SAM2 vs Overseer IoU drops below the threshold.

- `--bad-frames`
  Number of consecutive bad frames needed before triggering a disagreement re-prompt.

### Optional Boundary Arguments

- `--enable-boundary-distance-gate`
  Adds foreground boundary distance as an additional trigger signal.

- `--boundary-threshold`
  Boundary-distance threshold in pixels.

### Parallel-Specific Arguments

- `--overseer-batch-size`
  Batch size for Overseer precomputation.

- `--save-queue-size`
  Queue size for asynchronous save tasks.

- `--save-overseer-masks`
  Also exports Overseer masks.

- `--save_binary_mask`
  Also exports binary masks.

- `--save_disagreement_visuals`
  Saves disagreement debug overlays.

- `--max-visuals`
  Maximum number of disagreement visuals to save per video.

### Filtering Arguments

Both scripts support:

- `--video_name`
- `--video_names`
- `--frame_name`
- `--start_frame`
- `--end_frame`

## How To Compare Results

After both runs finish, compare:

- `timing_summary.json`
- total elapsed time
- FPS
- total processed frames
- disagreement re-prompt counts
- metadata CSVs
- `inference/disagreement_gate_report.md`

The basic comparison formula is:

```text
speedup = original_time / parallel_time
```

A value greater than `1.0` means the parallel version is faster.

## Notes

- The parallel version is not a new disagreement method.
- The disagreement trigger logic is intentionally preserved.
- The speedup comes from reorganizing independent work around the sequential control flow.
- Small differences in processed-frame totals can still occur if a segment is restarted during re-prompting.

## Summary

The parallel pipeline keeps the same disagreement-gated SASVI behavior but improves execution efficiency by:

- precomputing Overseer masks in batches
- offloading file export to a background worker
- writing standardized timing and report outputs

This makes it suitable for fair benchmarking against the original pipeline without changing the core algorithmic behavior.
