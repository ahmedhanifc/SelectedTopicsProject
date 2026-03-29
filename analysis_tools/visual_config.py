"""Tunable visualization settings for offline error-analysis artifacts.

Edit the values in this file, then rerun:

python analysis_tools/run_error_analysis.py \
  --frames_root ./frame_root \
  --pred_root ./output_masks \
  --gt_root ./gt_root \
  --output_root ./analysis_output/report \
  --dataset_type CHOLECSEG8K \
  --confidence_root ./analysis_output/inference/confidence_maps \
  --confidence_low_threshold 0.35 \
  --confidence_medium_threshold 0.60

Most useful knobs:
- BACKGROUND_DIM_FACTOR:
  Lower values make the source image darker.
  Higher values make the source image more visible behind the error map.
- BACKGROUND_BLUR_RADIUS:
  Higher values blur the source image more.
- ERROR_OVERLAY_ALPHA:
  Higher values make red/blue/yellow error colors stronger.
- LOW_CONFIDENCE_SPACING / MEDIUM_CONFIDENCE_SPACING / HIGH_CONFIDENCE_SPACING:
  Smaller spacing means denser lines.
"""

# Source-image background used in error_map and error overlay panels.
BACKGROUND_BLUR_RADIUS = 3.0
BACKGROUND_DIM_FACTOR = 0.8

# Strength of error colors when blended over the softened source image.
ERROR_OVERLAY_ALPHA = 0.75

# Confidence hatch settings.
LOW_CONFIDENCE_SPACING = 7
MEDIUM_CONFIDENCE_SPACING = 11
HIGH_CONFIDENCE_SPACING = 16
PATTERN_LINE_WIDTH_MIN = 1
PATTERN_LINE_WIDTH_DIVISOR = 4

# Mask/prediction overlay transparency in the combined panel.
PREDICTION_OVERLAY_ALPHA = 0.45
GROUND_TRUTH_OVERLAY_ALPHA = 0.45
