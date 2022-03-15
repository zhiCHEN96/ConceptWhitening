#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

python3 ../train_isic.py --ngpu 1 --workers 4 --arch resnet_original --depth 18 --epochs 30 --batch-size 64 --lr 0.005 --prefix resnet18_isic /usr/xtmp/zhichen/ISIC_data/
