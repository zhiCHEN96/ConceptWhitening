import os
import shutil
import json
import random

raw_data_folder = './Data' # path of raw ISIC data
description_folder = os.path.join(raw_data_folder,'Descriptions')
image_folder = os.path.join(raw_data_folder,'Images')
all_images = set(os.listdir(image_folder))

target_data_folder = '../ISIC_data' # path of preprocessed ISIC data
train_folder = os.path.join(target_data_folder,'train')
test_folder = os.path.join(target_data_folder,'test')
concept_train_folder = os.path.join(target_data_folder,'concept_train')
concept_test_folder = os.path.join(target_data_folder,'concept_test')

train_size = 0.8 # size of training set, value between 0.0 and 1.0

random.seed(42) # set random seed

n = len(os.listdir(description_folder))
print(n)

for filename_des in os.listdir(description_folder):
    with open(os.path.join(description_folder,filename_des)) as f:
        meta_data = json.load(f)
    # we only consider histopathology images in our experiments
    # to avoid visual differences from the type of diagonisis 
    if 'diagnosis_confirm_type' not in meta_data['meta']['clinical'] or meta_data['meta']['clinical']['diagnosis_confirm_type']!='histopathology':
        continue

    if filename_des+'.jpeg' in all_images:
        filename_img = filename_des+'.jpeg'
    elif filename_des+'.png' in all_images:
        filename_img = filename_des+'.png'
    else:
        continue
    path_img = os.path.join(image_folder,filename_img)
    print(path_img)

    is_train = (random.random()<train_size) # random train test split

    # binary label: 0 - benign, 1 - malignant
    if 'benign_malignant' not in meta_data['meta']['clinical']:
        continue
    if meta_data['meta']['clinical']["benign_malignant"]=='benign':
        if is_train:
            shutil.copy(path_img, os.path.join(train_folder,'benign',filename_img))
        else:
            shutil.copy(path_img, os.path.join(test_folder,'benign',filename_img))
    elif meta_data['meta']['clinical']["benign_malignant"]=='malignant':
        if is_train:
            shutil.copy(path_img, os.path.join(train_folder,'malignant',filename_img))
        else:
            shutil.copy(path_img, os.path.join(test_folder,'malignant',filename_img))
    
    # first concept age<20
    if 'age_approx' in meta_data['meta']['clinical'] and meta_data['meta']['clinical']['age_approx'] and int(meta_data['meta']['clinical']['age_approx'])<20:
        if is_train:
            shutil.copy(path_img, os.path.join(concept_train_folder,'age_le_20/age_le_20',filename_img))
        else:
            shutil.copy(path_img, os.path.join(concept_test_folder,'age_le_20',filename_img))
    # second concept lesion size>=10.0
    if 'clin_size_long_diam_mm' in meta_data['meta']['clinical'] and meta_data['meta']['clinical']['clin_size_long_diam_mm'] and float(meta_data['meta']['clinical']['clin_size_long_diam_mm'])>=10.0:
        if is_train:
            shutil.copy(path_img, os.path.join(concept_train_folder,'size_geeq_10/size_geeq_10',filename_img))
        else:
            shutil.copy(path_img, os.path.join(concept_test_folder,'size_geeq_10',filename_img))
