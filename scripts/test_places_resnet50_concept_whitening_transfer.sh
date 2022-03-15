#!/bin/sh
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_cw --depth 50 --batch-size 32 --lr 0.05 --whitened_layers 16 --concepts airplane,bed,bench,boat,book,horse,person --prefix RESNET50_PLACES365_CPT_WHITEN_TRANSFER --resume ./checkpoints/RESNET50_PLACES365_CPT_WHITEN_TRANSFER_model_best.pth.tar /usr/xtmp/zhichen/data_256/ --evaluate plot_top50