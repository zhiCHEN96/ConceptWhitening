# Concept Whitening for Interpretable Image Recognition

This repository contains the code for experiments in following paper

> Zhi Chen, Yijie Bei, Cynthia Rudin
>
> [Concept Whitening for Interpretable Image Recognition](https://rdcu.be/cbOKj)
> 
> Nature Machine Intelligence 2, 772–782 (2020). https://doi.org/10.1038/s42256-020-00265-z

arXiv [pre-publication version](https://arxiv.org/abs/2002.01650)

## Short Introduction Video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/D_LBBQEsQYw/0.jpg)](http://www.youtube.com/watch?v=D_LBBQEsQYw)

## Code Details

### Recognition
We have adapted skeleton code from https://github.com/Jongchan/attention-module in train_places.py, and adapted IterNrom implementation from https://github.com/huangleiBuaa/IterNorm as part of the implementation

### Code
Most of our contributions can be found in the implementation of **IterNormRotation** class from */MODELS/iterative_normalization.py*. Useful code for producing experimental results and visualizations are found in */train_imagenet.py* and */plot_functions.py*. 
*/scripts* folder contains shell scripts using which experiments are done. Also under the */scripts* folder, there are example checkpoints we saved during experiments.

### Dependencies
PyTorch (1.1.0), torchvision (0.3.0), NumPy (1.18.1), sklearn (0.20.3), matplotlib (3.1.3), PIL (6.2.1), Seaborn (0.9.0), skimage (0.15.0).

### Recommended hardware
NVIDIA Tesla P-100 GPUs or NVIDIA Tesla K-80 GPUs

### Dataset Structure
```
data_256
├── concept_train
│   ├── airplane
│   │   ├── airplane
│   ├── bed
│   │   ├── bed
│   ├── desk
│   │   ├── desk
│   ├── fridge
│   │   ├── fridge
│   ├── lamp
│   │   ├── lamp
│   ├── person
│   │   ├── person
│   ├── sofa
│   │   ├── sofa
│   └── ......
├── concept_test
│   ├── airplane
│   ├── bed
│   ├── desk
│   ├── fridge
│   ├── lamp
│   ├── person
│   ├── sofa
│   └── ......
├── test
│   ├── airfield
│   ├── airplane_cabin
│   ├── airport_terminal
│   ├── alcove
│   ├── alley
│   ├── amphitheater
│   ├── amusement_arcade
│   ├── amusement_park
│   ├── apartment_building_outdoor
│   ├── aquarium
│   ├── yard
│   ├── youth_hostel
│   └── ......
├── train
│   ├── airfield
│   ├── airplane_cabin
│   ├── airport_terminal
│   ├── alcove
│   ├── alley
│   ├── amphitheater
│   ├── amusement_arcade
│   ├── amusement_park
│   ├── apartment_building_outdoor_outdoor
│   ├── aquarium
│   ├── aqueduct
│   ├── arcade
│   ├── arch
│   ├── archaelogical_excavation
│   ├── archive
│   ├── arena_hockey
│   ├── arena_performance
│   ├── bamboo_forest
│   ├── bank_vault
│   ├── banquet_hall
│   ├── bar
│   ├── barn
│   ├── barndoor
│   ├── baseball_field
│   ├── basement
│   ├── basketball_court_indoor
│   ├── bathroom
│   ├── bazaar_indoor
│   ├── bazaar_outdoor
│   ├── youth_hostel
│   └── ......
└── val
    ├── airfield
    ├── airplane_cabin
    ├── airport_terminal
    ├── alcove
    ├── alley
    ├── amphitheater
    ├── amusement_arcade
    ├── amusement_park
    ├── apartment_building_outdoor_outdoor
    ├── aquarium
    ├── aqueduct
    ├── arcade
    ├── arch
    ├── archaelogical_excavation
    ├── archive
    ├── arena_hockey
    ├── arena_performance
    ├── arena_rodeo
    ├── army_base
    ├── art_gallery
    ├── artists_loft
    ├── art_school
    ├── ......
```
An example dataset folder structure looks like the above, where in each bottom level folder, there are images. Notice that during training, a sampler will randomly sample from dataset folders for images. Notice that in the above folder structure, the list of subfolders has been cut short and the complete dataset is more extensive.  

There generally two types of dataset, main objective dataset and auxiliary concept dataset.

Main dataset: We mainly use Places365 as the main dataset, and it can be downloaded from [Here](http://places2.csail.mit.edu/download.html). It should be divided into train, test, validation sets and stored in corresponding folders shown by the example dataset folder structure above.

Concept dataset: We mainly use objects in MS COCO as our auxiliary concept dataset, and it can be downloaded from [Here](https://cocodataset.org/#download). Each annotation, e.g., “person” in MS COCO, was used as one concept, and we selected all the images with this annotation (images having “person” in it), cropped them using bounding boxes and used the cropped images as the data representing the concept. The preprocessing code of the COCO dataset is provided in *cropping_images_COCO.py*. After downloading the 2017 COCO dataset, one can extract the concept images by running
```
python3 cropping_images_COCO.py -coco-path <coco_dataset_folder> -concept-path <folder_containing_concept_datasets>
```
. However, our model generalize to various concepts as well. It should be divided into train and test and stored in *concept_train/* and */concept_test* as is shown by the example dataset folder structure above. Note that in order to load data easily, the structures of the two folders are different: */concept_train* allows loading images of one concept while */concept_test* allows loading images of all concepts.

We also use the ISIC dataset in the experiments, and it can be downloaded from [Here](https://www.isic-archive.com). The attributes of the lesion images, such as "age<20", are used to define the concepts.

### Pretrained weights
#### Standard PlacesCNNs
To accelerate training process of our model, we utilize the pretrained weights of standard CNNs trained on Places365. The weights of those models can be downloaded from [Here](http://places2.csail.mit.edu/download.html)

Once the weights are downloaded, please put them under the folder *checkpoints/*
#### CNNs with CW
The weights of models trained with CW can be downloaded from [Here](https://drive.google.com/drive/folders/1iWtusRq9eWIEsuaRPEBGN3QNFQLRRTzR?usp=sharing). The folders are named by concepts trained in CW. For example *airplane_bed_person/* contains weights when CW is trained with concepts "airplane", "bed" and "person".

Once the weights are downloaded, please put the folders of weights under *checkpoints/*
### Usage
train_places.py has many arguments that one can use, here are some interesting ones:  
--arch: resnet_cw, resnet_original, densnet_cw, densnet_original, vgg16_cw, vgg16_bn_original  
--whitened_layers: refers to where the concept whitening module is added in the architecture (see explanation in example) 
--concepts: comma delimited list of concepts that needs to be disentangled (see example below)  
--act_mode: mean, max, pos_mean, pool_max (refer to paper for explanation)  

### Example
#### Train: 
Example training invocation scripts are inside the */scripts* folder. A typical configuration looks like:
```
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.05 --whitened_layers 5 --concepts airplane,bed,person --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER /data_256
```
This will start a training that adds concept whitening module to only the fifth residual block of the of the resnet 18. Note that a resnet 18 structure only has 8 residual blocks, and therefore the valid configuration numbers for resnet18 architecure are only 1 through 8. Note that when configuration number is 8, it corresponds to the 16th layer of the ResNet18 instead of 8th layer. Similarly for resnet 50, the valid configuration numbers are only 1 through 16.  
It is also possible to add concept whitening module to multiple layers by doing
```
--whitened_layers 1,2,3,8
```
The concepts to be disentangled will be specified through
```
--concepts airplane,bed,person
```
Note that these corresponds to directories in the */dataset_256/concept_train* and */dataset_256/concept_test* folder.

#### Test: 
Similarly, example testing invocation scripts are inside the */scripts* folder, and testing is done by invoking:
```
python3 train_places.py --ngpu 1 --workers 2 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.1 --whitened_layers 5 --concepts airplane,bed,person --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER --resume ./checkpoints/RESNET18_PLACES365_CPT_WHITEN_TRANSFER_model_best.pth.tar /data_256 --evaluate
```
You can reproduce all the experiment figures by running *./scripts/test_places_resnet18_concept_whitening_transfer.sh*. Note that figures may not be exactly the same since randomness involved in loading data.
