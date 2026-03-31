# Parallel Disagreement Pipeline

## Overview

This folder contains the parallelized version of the disagreement-gated SASVI pipeline, along with a timing wrapper for the original version.

The key idea is simple:

- keep the actual SASVI decision logic the same
- speed up the work around that logic
- measure the speedup fairly against the original pipeline

In other words, the goal was not to redesign when re-prompting happens. The goal was to preserve the same pipeline behavior while reducing runtime.

## Files In This Folder

- `parallel_disagreement_sasvi.py`
  - the parallelized disagreement-gated pipeline
- `time_original_disagreement_sasvi.py`
  - a wrapper around the original pipeline that records timing and writes outputs in the same run-folder style
- `_subset_cache/`
  - temporary cached frame subsets used when running on a selected frame range or single frame

## High-Level Pipeline

At a high level, the pipeline still works like this:

1. Load a video or frame sequence.
2. Use the Overseer model to provide prompt masks.
3. Initialize SAM2 with those prompts.
4. Propagate SAM2 through the video.
5. For each propagated frame, compare the current SAM2 mask with the current Overseer mask.
6. If the disagreement stays bad for enough consecutive frames, stop propagation and re-prompt from that frame.
7. Save masks, confidence maps, metadata, and reports.

That logic is still the same in the parallel version.

## What Was Parallelized

The disagreement logic itself is still sequential.

That part must stay sequential because:

- each frame depends on the previous propagation state
- a re-prompt can restart the segment from the current frame
- future decisions depend on whether a trigger fired earlier

So instead of parallelizing the actual trigger logic, the parallel pipeline speeds up the surrounding work.

### Change 1: Overseer Predictions Are Precomputed In Batches

In the original pipeline, Overseer predictions are fetched as needed during the SASVI loop.

In the parallel pipeline:

- Overseer predictions for the selected frames are computed up front
- they are computed in batches
- the results are stored in a cache

Simple interpretation:

- original version: "ask Overseer for masks whenever we need them"
- parallel version: "prepare all Overseer masks first, then reuse them quickly"

Why this helps:

- running the Overseer in batches is usually faster than running it frame by frame
- the SASVI loop no longer has to wait for repeated Overseer inference calls during propagation

### Change 2: Saving Is Moved To A Background Worker

Saving masks, confidence maps, and disagreement visuals can take noticeable time.

In the original pipeline, these outputs are written immediately inside the main loop.

In the parallel pipeline:

- the main loop creates save tasks
- those tasks are pushed into a queue
- a background worker writes files asynchronously

Simple interpretation:

- original version: "predict, then stop and save"
- parallel version: "predict, hand off saving, keep going"

Why this helps:

- disk I/O is overlapped with ongoing inference work
- the main loop spends less time blocked on file writes

### Change 3: Timing And Export Structure Were Standardized

The parallel script writes outputs into timestamped run folders:

- `parallel_disagreement_frame_outputs/run_<timestamp>/`
- `analysis_output_disagreement_parallel/run_<timestamp>/`

Each run stores:

- output masks
- optional Overseer masks
- confidence maps
- per-frame metadata CSVs
- markdown report
- run parameters
- timing summary JSON

This makes runs easier to compare and easier to keep organized.

## Simple Outline Of The Parallel Script

This is the easiest way to think about `parallel_disagreement_sasvi.py`.

### Stage A: Setup

- parse arguments
- discover videos
- build the SAM2 predictor
- build the Overseer model
- create timestamped output folders
- start the async save worker

### Stage B: Prepare Frames

For each selected video:

- list frames
- optionally filter by frame name or frame range
- optionally create a temporary subset directory
- initialize SAM2 video state

### Stage C: Precompute Overseer Masks

Before the main SASVI loop starts:

- run Overseer on the selected frames in batches
- store the results in an in-memory cache
- record how long that stage took

### Stage D: Run SASVI Sequentially

Then the main sequential logic runs:

- start a segment
- reset predictor state
- inject prompt masks and point prompts
- propagate SAM2 frame by frame
- compare SAM2 output with Overseer output
- update disagreement counters
- decide whether to re-prompt

This is still the core SASVI logic.

### Stage E: Queue Output Saving

For each processed frame:

- package masks and metadata into a save task
- push the task to the background save queue
- continue inference while saving happens in parallel

### Stage F: Finalize Run

After all videos finish:

- wait for the save worker to complete
- write combined metadata CSVs
- write the markdown report
- write `timing_summary.json`

## Why A Wrapper Was Created For The Original Pipeline

The original disagreement pipeline already existed, but it was not ideal for a fair speed comparison.

It did not naturally produce the same run-folder organization and timing-summary format as the parallel script.

So `time_original_disagreement_sasvi.py` was created as a wrapper.

### What The Wrapper Does

It keeps the original SASVI logic intact, but adds:

- timestamped `run_<timestamp>` output folders
- timing collection
- comparable metadata/report export
- a `timing_summary.json` file in the same style as the parallel script

### Why This Matters

Without the wrapper, comparing runtime would be messy because:

- outputs would be organized differently
- timing information would not be captured in the same way
- it would be harder to compare total runtime and FPS side by side

So the wrapper exists mainly to answer this question cleanly:

"How much faster is the parallelized pipeline than the original one?"

## Practical Summary In Plain Language

If you want the simplest explanation, it is this:

- the original pipeline mixes inference, Overseer calls, and saving inside one main loop
- the parallel pipeline keeps the same decisions, but moves reusable work earlier and pushes saving into the background
- the timing wrapper exists so the old and new pipelines can be benchmarked fairly

## Short Takeaway

The parallel pipeline is not a new algorithm.

It is the same disagreement-gated SASVI control flow, reorganized so that:

- Overseer work is batched ahead of time
- saving happens in the background
- timing is recorded cleanly
- speedup can be measured against the original pipeline using the wrapper script
