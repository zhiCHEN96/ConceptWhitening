## This repository contains the code for experiments done in paper [Concept Whitening for Interpretable Image Recognition](https://arxiv.org/abs/2002.01650)

### Recognition
We have adapted skeleton code from https://github.com/Jongchan/attention-module in train_places.py, and adapted IterNrom implementation from https://github.com/huangleiBuaa/IterNorm as part of the implementation

### Code
Most of our contributions can be found in the implementation of **IterNormRotation** class from */MODELS/iterative_normalization.py*. Useful code for producing experimental results and visualizations are found in */train_imagenet.py* and */plot_functions.py*. 
*/scripts* folder contains shell scripts using which experiments are done. Also under the */scripts* folder, there are example checkpoints we saved during experiments.

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

Main dataset: We have been using Places 365 as the main dataset, and it can be downloaded from [Here](http://places2.csail.mit.edu/download.html). It should be devided into train, test, validation sets and stored in corresponding folders shown by the example dataset folder structure above.

Concept dataset: We have been using images in ImageNet as our auxiliary concept dataset, and it can be downloaded from [Here](http://image-net.org/download). The concepts we used are "airplane", "bed", "desk", "fridge", "lamp", "person", "sofa". However, our model generalize to various concepts as well. It should be devided in to train and test and stored in concept_train/ and concept_test/ as is shown by the example dataset folder structure above. Note that in order to load data easily, the structures of the two folders are different: concept_train/ allows loading images of one concept while concept_test/ allows loading images of all concepts.

### Pretrained weights of standard PlacesCNNs
To accelerate training process of our model, we utilize the pretrained weights of standard CNNs trained on Places365. The weights of those models can be downloaded from [Here](http://places2.csail.mit.edu/download.html)

Once the weights are downloaded, please put them under the folder ./checkpoints/ 

### Usage
train_places.py has many arguments that one can use, here are some interesting ones:  
--arch: resnet_cw, resnet_original, densnet_cw, densnet_original, vgg16_cw, vgg16_bn_original  
--whitened_layers: refers to where the concept whitening module is added in the architecture (see explanation in example) 
--concepts: comma delimited list of concepts that needs to be disentablged (see example below)  
--act_mode: mean, max, pos_mean, pool_max (refer to paper for explanation)  

### Example
#### Train: 
Example training invocation scripts are inside the */scripts* folder. A typical configuration looks like:
```
python3 ../train_places.py --ngpu 1 --workers 4 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.05 --whitened_layers 5 --concepts airplane,bed,person --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER /data_256
```
This will start a training that adds concept whitening module to only the fifth residual block of the of the resnet 18. Note that a resnet 18 structure only has 8 residual blocks, and therefore the valid configuration numbers for resnet18 architecure are only 1 through 8. Similarly for resnet 50, the valid configuration numbers are only 1 through 16.  
It is also possible to add concept whitening module to multiple layers by doing
```
--whitened_layers 1,2,3,8
```
The concepts to be disentangled will be specified through
```
--concepts airplane,bed,person
```
Note that these corresponds to directories in the */dataset_256/concept* folder.

#### Test: 
Similarly, testing is done by invoking:
```
python3 train_places.py --ngpu 1 --workers 16 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.1 --whitened_layers 5 --concepts airplane,bed,person --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER --resume ./checkpoints/RESNET18_PLACES365_CPT_WHITEN_TRANSFER_model_best.pth.tar /data_256
```

