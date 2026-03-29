conda create -n sasvi python

conda activate sasvi

pip install requirements.txt

mkdir -p frame_root gt_root output_masks_disagreement_08 analysis_output_disagreement_08

python helper_scripts/dataset_prep.py \
  --src-root ./dataset \
  --dst-root ./frame_root \
  --clip video01_00160

python helper_scripts/prepare_cholecseg8k_gt_for_error_analysis.py \
  --src-root ./dataset \
  --dst-root ./gt_root \
  --clip video01_00160 \
  --symlink

cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint ./sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint ../../checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir ../../frame_root \
  --gt_root_dir ../../gt_root \
  --video_name video01_00160 \
  --output_mask_dir ../../output_masks_disagreement_08 \
  --analysis_output_dir ../../analysis_output_disagreement_08 \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.95 \
  --disagreement_bad_frames 2 \
  --save_disagreement_visuals \
  --max_disagreement_visuals 6

cd ../..

python analysis_tools/run_error_analysis.py \
  --frames_root ./frame_root \
  --pred_root ./output_masks_disagreement_08/raw \
  --gt_root ./gt_root \
  --output_root ./analysis_output_disagreement_08/report_raw \
  --dataset_type CHOLECSEG8K \
  --confidence_root ./analysis_output_disagreement_08/inference/confidence_maps \
  --confidence_low_threshold 0.35 \
  --confidence_medium_threshold 0.60

python analysis_tools/run_error_analysis.py \
  --frames_root ./frame_root \
  --pred_root ./output_masks_disagreement_08/smoothed \
  --gt_root ./gt_root \
  --output_root ./analysis_output_disagreement_08/report_smoothed \
  --dataset_type CHOLECSEG8K \
  --confidence_root ./analysis_output_disagreement_08/inference/confidence_maps \
  --confidence_low_threshold 0.35 \
  --confidence_medium_threshold 0.60
