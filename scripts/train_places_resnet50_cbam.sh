python3 ../train_imagenet.py \
			--ngpu 1 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 16 --lr 0.003 \
			--att-type CBAM \
			--prefix RESNET50_PLACES_CBAM \
			/usr/xtmp/zhichen/image_data_new/scene/