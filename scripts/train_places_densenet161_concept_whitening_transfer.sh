#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch densenet_cw --depth 161 --epochs 200 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts airplane,bed,person --prefix DENSENET161_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
