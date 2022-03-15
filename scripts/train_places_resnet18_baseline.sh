#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_baseline --depth 18 --epochs 100 --batch-size 64 --lr 0.5 --whitened_layers 8 --concepts airplane,bed,bench,boat,book,horse,person --prefix RESNET18_PLACES365_BASELINE /usr/xtmp/zhichen/data_256/
