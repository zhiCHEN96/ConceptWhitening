#!/bin/sh
python3 ../train_isic.py --ngpu 1 --workers 4 --arch resnet_original --depth 18 --epochs 200 --batch-size 64 --lr 0.001 --whitened_layers 8 --concepts age_le_20,size_geeq_10 --prefix resnet18_isic /usr/xtmp/zhichen/ISIC_data_2/
