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
python prepare_cholecseg8k_for_sasvi.py \
  --src-root /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/dataset \
  --dst-root /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/frame_root \
  --clip video01_28660

For all clips:
python prepare_cholecseg8k_for_sasvi.py \
  --src-root /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/dataset \
  --dst-root /Users/ahmedhanif/Desktop/dev/SelectedTopics/project/SASVi/frame_root


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
