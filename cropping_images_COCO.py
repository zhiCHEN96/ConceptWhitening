import numpy as np
import json
from PIL import Image
from pathlib import Path
import argparse
import os

def crop_images_bbox(input_dir,output_dir,anno_path, double_path = False):
    with open(anno_path) as f:
        anno  = json.load(f)

    imageid2filename = {}
    for item in anno['images']:
        imageid2filename[item['id']] = item['file_name']

    label2conceptname = {}
    for item in anno['categories']:
        name = item['name'].replace(' ','_')
        label2conceptname[item['id']] = name
        if double_path == True:
            (output_dir/name/name).mkdir(parents=True, exist_ok = True)
        else:
            (output_dir/name).mkdir(parents=True, exist_ok = True)

    for item in anno['annotations']:
        bbox = item['bbox']
        if bbox[2] < 30  or bbox[3] < 30:
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        img = Image.open(input_dir / imageid2filename[item['image_id']])
        img_cropped = img.crop(bbox)
        concept_name = label2conceptname[item['category_id']]
        if double_path == True:
            img_cropped.save(output_dir / f"{concept_name}/{concept_name}/{item['id']}.jpg")
        else:
            img_cropped.save(output_dir / f"{concept_name}/{item['id']}.jpg")
    
    return

parser = argparse.ArgumentParser('Cropping COCO images with bounding boxs')
parser.add_argument('-coco-path', type=str, default = 'data/coco')
parser.add_argument('-concept-path', type=str, default = 'data_256')
args = parser.parse_args()
print(args)

coco_path = Path(args.coco_path)
train_dir = coco_path / 'train2017'
val_dir = coco_path / 'val2017'
train_anno_path = coco_path / 'annotations/instances_train2017.json'
val_anno_path = coco_path / 'annotations/instances_val2017.json'

concept_path = Path(args.concept_path)
concept_path.mkdir(parents=True, exist_ok = True)
concept_train_dir = concept_path / 'concept_train'
concept_train_dir.mkdir(parents=True, exist_ok = True)
concept_val_dir = concept_path / 'concept_test'
concept_val_dir.mkdir(parents=True, exist_ok = True)

crop_images_bbox(val_dir, concept_val_dir, val_anno_path)
crop_images_bbox(train_dir, concept_train_dir, train_anno_path, double_path=True)

