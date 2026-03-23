# Environment Setup
conda create -n sasvi python=3.11

conda activate sasvi

pip install torch torchvision

pip install -r requirements.txt

pip install git+https://github.com/MECLabTUDA/SDS_Playground.git

cd src/sam2
pip install -e .
cd ../..

# Download and place checkpoints:
Note: You will need to create the directories yourself. So go to src/sam2/sam2 and create checkpoints folder for example.
Then download the model from this link and place it there: https://drive.google.com/file/d/1MSdA_mE4CF2aBtnQY7vEEVI2lMeejuDq/view?usp=sharing
SAM2:
src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt

Same for overseer: https://drive.google.com/file/d/1Y6f2OHOJalLei6dA1B1NHYJe8UHa3Rdb/view?usp=sharing
Overseer:
checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth

# Dataset Preparation

1) Download Cholecsef8k dataset: https://drive.google.com/drive/folders/1GgFfnAQ__LlSZjaHwZbV5xlCgJWMH5-G?usp=sharing

The dataset in its original form cannot be used and needs to be prepared.
Run the following command:

In the root of the project, run: mkdir frame_root
For one clip only:
python helper_scripts/dataset_prep.py \
  --src-root ./dataset \
  --dst-root ./frame_root \
  --clip video01_28660

For all clips:
python helper_scripts/dataset_prep.py \
  --src-root ./dataset \
  --dst-root ./frame_root


# Run Inference (Assuming you are currently at root)
cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint ./sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint ../../checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir ../../frame_root \
  --output_mask_dir ../../output_masks \
  --analysis_output_dir ../../analysis_output


  # Output:
output_masks/video01_28660/

# Error Report

Prepare the ground-truth masks first:
(optional) rm -rf ./gt_root/video01_28660

python helper_scripts/prepare_cholecseg8k_gt_for_error_analysis.py \
  --src-root ./dataset \
  --dst-root ./gt_root \
  --clip video01_28660 \
  --symlink

Then run the report:

python analysis_tools/run_error_analysis.py \
  --frames_root ./frame_root \
  --pred_root ./output_masks \
  --gt_root ./gt_root \
  --output_root ./analysis_output/report \
  --dataset_type CHOLECSEG8K \
  --confidence_root ./analysis_output/inference/confidence_maps \
  --confidence_low_threshold 0.35 \
  --confidence_medium_threshold 0.60

**Setup**

mkdir -p frame_root gt_root output_masks analysis_output

python helper_scripts/dataset_prep.py \
  --src-root ./dataset \
  --dst-root ./frame_root \
  --clip video01_28660

python helper_scripts/prepare_cholecseg8k_gt_for_error_analysis.py \
  --src-root ./dataset \
  --dst-root ./gt_root \
  --clip video01_28660 \
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
  --output_mask_dir ../../output_masks \
  --analysis_output_dir ../../analysis_output

cd ../..


Now Generate Report

python analysis_tools/run_error_analysis.py \
  --frames_root ./frame_root \
  --pred_root ./output_masks \
  --gt_root ./gt_root \
  --output_root ./analysis_output/report \
  --dataset_type CHOLECSEG8K \
  --confidence_root ./analysis_output/inference/confidence_maps \
  --confidence_low_threshold 0.35 \
  --confidence_medium_threshold 0.60


