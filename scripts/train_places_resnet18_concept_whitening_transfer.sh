#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.05 --whitened_layers 7 --concepts bicycle,bird,boat,cat,chair,knife,microwave,pizza,skis --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
