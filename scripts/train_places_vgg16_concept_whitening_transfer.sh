#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch vgg16_cw --depth 16 --epochs 3 --batch-size 64 --lr 0.05 --whitened_layers 13 --concepts airplane,bed,bench,boat,book,horse,person --prefix VGG16_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
