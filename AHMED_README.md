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
SAM2:
src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt

Overseer:
checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth

# Run Inference (Assuming you are currently at root)
cd src/sam2

python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/src/sam2/sam2/checkpoints/sam2 1_hiera_large.pt \
  --overseer_checkpoint /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/frame_root \
  --output_mask_dir /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/output_masks

  # Output:
output_masks/video01_28660/
