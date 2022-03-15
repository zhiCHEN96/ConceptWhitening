#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

python3 ../train_isic.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --epochs 10 --batch-size 32 --lr 0.005 --whitened_layers 8 --concepts age_le_20,size_geeq_10 --prefix resnet18_isic /usr/xtmp/zhichen/ISIC_data/
