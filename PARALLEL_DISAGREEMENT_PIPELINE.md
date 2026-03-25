# Parallel Disagreement Pipeline

## What changed

This adds a parallelized version of the disagreement-gated SASVI workflow, plus a timed wrapper for the original disagreement pipeline so both versions can be compared fairly.

The goal is not to change the disagreement logic itself. The goal is to keep the same disagreement-triggered re-prompting behavior while speeding up the surrounding work and exporting comparable timing summaries.

Two scripts were added:

- `parallelization_and_streaming/parallel_disagreement_sasvi.py`
  - parallel disagreement pipeline
  - saves masks, confidence maps, metadata, report, and timing summary
- `parallelization_and_streaming/time_original_disagreement_sasvi.py`
  - original disagreement pipeline with timing/export wrapper
  - used for timing comparison against the parallel version

## Main idea

The disagreement-gated SASVI logic is still the same core idea:

1. Run Overseer and SAM2 on the same frame.
2. Compare their masks.
3. If disagreement stays bad for enough consecutive frames, trigger a corrective re-prompt.

The difference is in how the work is organized:

- the original path runs the disagreement pipeline in its normal sequential form
- the parallel path keeps the disagreement decisions sequential, but speeds up independent work around them

## Parallelization strategy used

Several parallelization ideas were considered:

1. Fully parallelize frame processing
2. Parallelize only saving/output work
3. Precompute Overseer predictions in batches and save outputs asynchronously while keeping disagreement control flow sequential

The implemented strategy is the third one, because it is the safest way to get real speedup without breaking the disagreement logic.

Why this was chosen:

- the disagreement pipeline is segment-based and sequential by nature
- a re-prompt can restart propagation from the current frame
- that means the trigger logic itself cannot be safely parallelized frame-by-frame

So the implemented speedup comes from:

- batched Overseer prediction precomputation
- asynchronous mask/confidence-map/report-related saving
- preserving the original disagreement trigger behavior

## Files added

- `parallelization_and_streaming/parallel_disagreement_sasvi.py`
- `parallelization_and_streaming/time_original_disagreement_sasvi.py`

## Output layout

### Parallel script

Creates timestamped output folders like:

- `parallel_disagreement_frame_outputs/run_<timestamp>/`
- `analysis_output_disagreement_parallel/run_<timestamp>/`

These contain:

- output masks
- optional Overseer masks
- confidence maps
- `inference_metadata.csv`
- per-video metadata CSVs
- `disagreement_gate_report.md`
- `run_parameters.csv`
- `timing_summary.json`

### Original timed script

Creates timestamped output folders like:

- `disagreement_frame_outputs/run_<timestamp>/`
- `analysis_output_disagreement_original/run_<timestamp>/`

These contain the same style of timing and analysis outputs so the comparison is easier.

## Timed comparison script

The original disagreement pipeline already existed, but it did not produce the same run-folder structure and timing summary style as the parallel version.

So a timed wrapper script was created:

- `parallelization_and_streaming/time_original_disagreement_sasvi.py`

This script does not redesign the original logic. It simply:

- runs the original disagreement pipeline
- records elapsed time and FPS
- writes timing summaries in a comparable format
- saves metadata and report outputs in the same general structure as the parallel script

## Exact commands

Run these from inside:

```bash
cd C:\Users\Test\Desktop\SelectedTopicsProject\parallelization_and_streaming
```

### Parallel disagreement pipeline

```bash
python parallel_disagreement_sasvi.py \
  --device cuda \
  --sam2_cfg "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\configs\sam2.1_hiera_l.yaml" \
  --sam2_checkpoint "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\checkpoints\sam2.1_hiera_large.pt" \
  --overseer_checkpoint "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\checkpoints\cholecseg8k_maskrcnn_best_val_f1.pth" \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir "C:\Users\Test\Desktop\SelectedTopicsProject\frame_root" \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2
```

### Original disagreement pipeline with timing

```bash
python time_original_disagreement_sasvi.py \
  --device cuda \
  --sam2_cfg "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\configs\sam2.1_hiera_l.yaml" \
  --sam2_checkpoint "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\checkpoints\sam2.1_hiera_large.pt" \ 
  --overseer_checkpoint "C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\checkpoints\cholecseg8k_maskrcnn_best_val_f1.pth" \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir "C:\Users\Test\Desktop\SelectedTopicsProject\frame_root" \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.5 \
  --disagreement_bad_frames 2
```

## Side note on extra arguments

Some extra arguments are optional and let you test stricter or different behavior.

### Common disagreement arguments

- `--enable_disagreement_gate`
  - turns on disagreement-triggered re-prompting
- `--disagreement_iou_threshold 0.5`
  - marks a frame as bad if SAM2 vs Overseer IoU drops below this value
- `--disagreement_bad_frames 2`
  - triggers disagreement re-prompting after this many consecutive bad frames

### Optional boundary-distance arguments

- `--enable_boundary_distance_gate`
  - also use mask-boundary mismatch as a disagreement condition
- `--boundary_distance_threshold 20.0`
  - marks a frame as bad when the SAM2 and Overseer foreground boundaries are farther apart than this threshold

If these boundary-distance arguments are not used:

- only the IoU-based disagreement check is active
- the behavior is simpler and more conservative

### Parallel-script-specific arguments

- `--save_overseer_masks`
  - also export Overseer masks to the output folder
- `--save_binary_mask`
  - also save binary masks in addition to PNG masks
- `--save_disagreement_visuals`
  - save disagreement overlay images for bad or triggered frames
- `--max_disagreement_visuals 10`
  - maximum number of disagreement visuals saved per video
- `--overseer_batch_size 8`
  - batch size used for precomputing Overseer predictions
  - larger values can improve speed if memory allows
- `--save_queue_size 32`
  - queue size for asynchronous output saving

### Filtering arguments

These can be used with both scripts:

- `--video_name <name>`
  - run one video only
- `--video_names <name1> <name2> ...`
  - run a selected list of videos
- `--frame_name <frame_stem>`
  - run one frame only
- `--start_frame <idx>`
  - start from a chosen frame index
- `--end_frame <idx>`
  - stop at a chosen frame index

## How to compare results

After both runs finish, compare:

- `timing_summary.json`
- total elapsed time
- FPS
- total processed frames
- disagreement/class-change re-prompt counts
- metadata CSVs

Speedup is computed as:

```text
speedup = original_time / parallel_time
```

Example:

```text
speedup = 2019.747907500001 / 1813.2702166 = 1.1139x
```

That means the parallel version is about `1.11x` faster.

## Notes

- The parallel script speeds up the pipeline without redesigning the disagreement trigger logic.
- The timed original script exists only to make timing comparison cleaner and more consistent.
- A small difference in `total_frames` between runs can happen if a frame is reprocessed at a segment boundary due to re-prompting.
