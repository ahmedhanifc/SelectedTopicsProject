Sarra Run Commands

baseline:

cd /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2 && /opt/miniconda3/envs/selected/bin/python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/dataset \
  --video_name video01_00160 \
  --output_mask_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/output_masks_baseline \
  --analysis_output_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_output_baseline


  disagreement:

cd /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2 && /opt/miniconda3/envs/selected/bin/python eval_sasvi.py \
  --device cpu \
  --sam2_cfg configs/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/src/sam2/sam2/checkpoints/sam2.1_hiera_large.pt \
  --overseer_checkpoint /Users/sarrachouk/Desktop/SelectedTopicsProject/checkpoints/cholecseg8k_maskrcnn_best_val_f1.pth \
  --overseer_type MaskRCNN \
  --dataset_type CHOLECSEG8K \
  --base_video_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/dataset \
  --video_name video01_00160 \
  --output_mask_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/output_masks_disagreement_08 \
  --analysis_output_dir /Users/sarrachouk/Desktop/SelectedTopicsProject/analysis_output_disagreement_08 \
  --enable_disagreement_gate \
  --disagreement_iou_threshold 0.95 \
  --disagreement_bad_frames 2 \
  --save_disagreement_visuals \
  --max_disagreement_visuals 6
  