# Analysis Tools

This folder contains the isolated MVP for SASVi error analysis.

## What it does

1. Exports per-frame confidence maps during SASVi inference if `--analysis_output_dir` is provided.
2. Builds an offline report from saved predictions and ground-truth masks.
3. Writes:
   - `analysis.csv`
   - `summary.json`
   - per-frame artifacts under `artifacts/<video_name>/`

## Files

- `run_error_analysis.py`: offline report generator
- `config.py`: dataset palette / ignore-index configuration
- `error_analysis.py`: metric computation and artifact generation
- `inference_export.py`: confidence-map export helpers and CSV writing

## Inference export

Add this optional argument to the existing SASVi command:

```bash
--analysis_output_dir /path/to/analysis_output
```

This creates:

```text
analysis_output/
  inference/
    inference_metadata.csv
    confidence_maps/
      <video_name>/
        <frame_name>_confidence.png
```

## Offline report generation

```bash
python analysis_tools/run_error_analysis.py \
  --frames_root /path/to/frame_root \
  --pred_root /path/to/output_masks \
  --gt_root /path/to/ground_truth_masks \
  --output_root /path/to/analysis_output/report \
  --dataset_type CHOLECSEG8K \
  --confidence_root /path/to/analysis_output/inference/confidence_maps
```

Expected predicted mask filenames: `*_rgb_mask.png`

Expected ground-truth mask filenames: `*_rgb_mask.png`

The report generator matches by:
- video directory name
- frame stem

When confidence maps are available, the error map applies pattern bands only on
the `Correct / no error` region so you can distinguish low-, medium-, and
high-confidence correct predictions without changing the existing error colors.

## Output artifacts

For each analyzed frame, the report saves:

- `<frame>_image.png`
- `<frame>_pred_rgb.png`
- `<frame>_gt_rgb.png`
- `<frame>_error_map.png`
- `<frame>_confidence.png` if available
- `<frame>_overlay.png`
