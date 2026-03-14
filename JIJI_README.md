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

# Run Stream Pipeline

Use this from the project root in `cmd`:

python stream_sasvi_video.py ^
    --input_video C:\Users\Test\Desktop\SelectedTopicsProject\videos\test_input_video.mp4 ^
    --output_root C:\Users\Test\Desktop\SelectedTopicsProject\stream_outputs ^
    --sam2_cfg C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\configs\sam2.1_hiera_l.yaml ^
    --sam2_checkpoint C:\Users\Test\Desktop\SelectedTopicsProject\src\sam2\sam2\checkpoints\sam2.1_hiera_large.pt ^
    --overseer_checkpoint C:\Users\Test\Desktop\SelectedTopicsProject\checkpoints\cholecseg8k_maskrcnn_best_val_f1.pth ^
    --overseer_type MaskRCNN ^
    --dataset_type CHOLECSEG8K ^
    --device cuda ^
    --sam2_window 64

Output will be created automatically in:

stream_outputs\<date_and_time>\
  output_video.mp4
  output_masks\

# Side Notes

Other useful optional arguments:

* `--frame_stride 2` or `--frame_stride 3` to process every 2nd/3rd frame for faster runs
* `--offload_state_to_cpu` if GPU memory gets tight
* `--progress_interval 10` to refresh the progress stats more often
* `--max_frames 100` for a short test run
* `--score_thresh 0.0` to change the SAM2 mask threshold if needed
* `--render_alpha 0.35` to change overlay transparency
* `--save_binary_mask` if you also want `.npz` binary masks
