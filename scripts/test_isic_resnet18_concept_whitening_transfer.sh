#!/bin/sh
python3 ../train_isic.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --epochs 200 --batch-size 1 --lr 0.03 --whitened_layers 1 --concepts age_le_20,size_geeq_10 --prefix resnet18_isic /usr/xtmp/zhichen/ISIC_data_2/ --evaluate
