#!/usr/bin/env bash
set -e  # stop immediately if any command fails

# # 1. Clone repo
git clone -b denoise --single-branch https://github.com/shyyshawarma/Restormer-.git
cd Restormer2

# # 2. Create experiment folder structure
mkdir -p experiments/Deraining_Restormer/training_states
mkdir -p experiments/Deraining_Restormer/models

# # 3. Download pretrained model checkpoint
cd experiments/Deraining_Restormer/models
pip install gdown
gdown "https://drive.google.com/file/d/17EnYYBElQEDhmH_wgjCVz5P8sEByqozZ/view?usp=sharing" 

# # 4. Download training state
cd ../training_states
gdown "https://drive.google.com/file/d/1NKJrT9ICYHO_yHgV_BR_TnmyY4VGzZ7X/view?usp=sharing" 

# # 5. Back to repo root
cd ../../../

# # 6. Download train/test data
cd Deraining
python download_data.py --data train-test
cd ../

# 7. Environment setup


bash setup_env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

# 8. Start training
bash ./train.sh Denoising/Options/GaussianGrayDenoising_Restormer.yml
