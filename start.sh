#!/usr/bin/env bash
set -e  # stop immediately if any command fails

# # 1. Clone repo
# git clone https://github.com/shyyshawarma/Restormer2.git
# cd Restormer2

# # 2. Create experiment folder structure
# mkdir -p experiments/Deraining_Restormer/training_states
# mkdir -p experiments/Deraining_Restormer/models

# # 3. Download pretrained model checkpoint
# cd experiments/Deraining_Restormer/models
# pip install gdown
# gdown "https://drive.google.com/file/d/1mKN0Zoud4bEKV6P6xgXnLb3A7xLNoUQZ/view?usp=sharing" 

# # 4. Download training state
# cd ../training_states
# gdown "https://drive.google.com/file/d/1u6ZLW6kwE-1NGRPILOvNJt65AwnoGXOr/view?usp=sharing" 

# # 5. Back to repo root
# cd ../../../

# # 6. Download train/test data
# cd Deraining
# python download_data.py --data train-test
# cd ../

# 7. Environment setup


bash setup_env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

# 8. Start training
bash ./train.sh Denoising/Options/GaussianColorDenoising_RestormerSigma25.yml
