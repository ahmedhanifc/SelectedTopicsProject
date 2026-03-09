conda create -n sasvi python=3.11

conda activate sasvi

pip install torch torchvision

pip install -r requirements.txt

pip install git+https://github.com/MECLabTUDA/SDS_Playground.git


cd src/sam2
pip install -e .
cd ../..
