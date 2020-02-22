#!/bin/sh
python3 ../train_imagenet.py --ngpu 1 --workers 4 --arch resnet_transfer --depth 50 --epochs 200 --batch-size 32 --lr 0.05 --whitened_layers 16 --concepts airplane,bed,person --prefix RESNET50_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
