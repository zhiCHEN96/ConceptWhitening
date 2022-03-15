#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint="v100|p100"

python3 ../train_isic.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --batch-size 1 --lr 0.005 --whitened_layers 8 --concepts age_le_20,size_geeq_10 --prefix resnet18_isic /usr/xtmp/zhichen/ISIC_data/ --evaluate plot_top50
