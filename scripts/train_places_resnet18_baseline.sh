#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_baseline --depth 18 --epochs 200 --batch-size 64 --lr 0.5 --whitened_layers 1 --concepts bus,car,dining_table,potted_plant,sink,umbrella,wine_glass --prefix RESNET18_PLACES365_BASELINE /usr/xtmp/zhichen/data_256/
