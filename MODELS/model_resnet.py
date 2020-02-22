import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.nn import init
from .iterative_normalization import IterNormRotation as cw_layer

class ResidualNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch = 'resnet18', layers = [2,2,2,2], model_file='resnet18_places365.pth.tar'):

        super(ResidualNetTransfer, self).__init__()
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)
        #from functools import partial
        #import pickle
        #import os
        #pickle.load = partial(pickle.load, encoding="latin1")
        #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        #if not os.path.exists(model_file):
        #     weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        #     os.system('wget ' + weight_url)
        # checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(model_file)
        #print(checkpoint['epoch'], checkpoint['best_prec1'])
        args.start_epoch = checkpoint['epoch']
        args.best_prec1 = checkpoint['best_prec1']
        print(checkpoint['best_prec1'])
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k,'bw','bn'): v for k,v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        #self.model.load_state_dict(state_dict)

        #self.model.fc = nn.Linear(512, num_classes)

        self.whitened_layers = whitened_layers

        for whitened_layer in whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer-1].bn1 = cw_layer(64, activation_mode = args.act_mode)
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1 = cw_layer(128, activation_mode = args.act_mode)
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1 = cw_layer(256, activation_mode = args.act_mode)
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1 = cw_layer(512, activation_mode = args.act_mode)
    
    def change_mode(self, mode):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1.mode = mode
    
    def update_rotation_matrix(self):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)

class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch = 'densenet161', model_file='densenet161_places365.pth.tar'):

        super(DenseNetTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        checkpoint = torch.load(model_file)
        args.start_epoch = checkpoint['epoch']
        args.best_prec1 = checkpoint['best_prec1']
        import re
        def repl(matchobj):
            return matchobj.group(0)[1:]
        state_dict = {re.sub('\.\d+\.',repl,str.replace(k,'module.','')): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers
        for whitened_layer in whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0 = cw_layer(64, activation_mode = args.act_mode)
            elif whitened_layer == 2:
                self.model.features.transition1.norm = cw_layer(384, activation_mode = args.act_mode)
            elif whitened_layer == 3:
                self.model.features.transition2.norm = cw_layer(768, activation_mode = args.act_mode)
            elif whitened_layer == 4:
                self.model.features.transition3.norm = cw_layer(2112, activation_mode = args.act_mode)
            elif whitened_layer == 5:
                self.model.features.norm5 = cw_layer(2208, activation_mode = args.act_mode)
    
    def change_mode(self, mode):
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0.mode = mode
            elif whitened_layer == 2:
                self.model.features.transition1.norm.mode = mode
            elif whitened_layer == 3:
                self.model.features.transition2.norm.mode = mode
            elif whitened_layer == 4:
                self.model.features.transition3.norm.mode = mode
            elif whitened_layer == 5:
                self.model.features.norm5.mode = mode
    
    def update_rotation_matrix(self):
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0.update_rotation_matrix()
            elif whitened_layer == 2:
                self.model.features.transition1.norm.update_rotation_matrix()
            elif whitened_layer == 3:
                self.model.features.transition2.norm.update_rotation_matrix()
            elif whitened_layer == 4:
                self.model.features.transition3.norm.update_rotation_matrix()
            elif whitened_layer == 5:
                self.model.features.norm5.update_rotation_matrix()
    
    def forward(self, x):
        return self.model(x)

class VGGBNTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch = 'vgg16_bn', model_file='./checkpoints/vgg16_bn_places365.pth.tar'):
        super(VGGBNTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        checkpoint = torch.load(model_file)
        args.start_epoch = 115#checkpoint['epoch']
        args.best_prec1 = checkpoint['best_prec1']
        state_dict = {str.replace(k,'module.model.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers
        self.layers = [1,4,8,11,15,18,21,25,28,31,35,38,41]
        for whitened_layer in whitened_layers:
            whitened_layer -= 1
            if whitened_layer in range(0,2):
                channel = 64
            elif whitened_layer in range(2,4):
                channel = 128
            elif whitened_layer in range(4,7):
                channel = 256
            else:
                channel = 512
            self.model.features[self.layers[whitened_layer]] = cw_layer(channel, activation_mode = args.act_mode)

    def change_mode(self, mode):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer-1]].mode = mode
    
    def update_rotation_matrix(self):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer-1]].update_rotation_matrix()

    def forward(self, x):
        return self.model(x)

class ResidualNetBN(nn.Module):
    def __init__(self, num_classes, args, arch = 'resnet18', layers = [2,2,2,2], model_file='resnet18_places365.pth.tar'):

        super(ResidualNetBN, self).__init__()
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)
        checkpoint = torch.load(model_file)
        args.start_epoch = checkpoint['epoch']
        args.best_prec1 = checkpoint['best_prec1']
        print(checkpoint.keys())
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

class DenseNetBN(nn.Module):
    def __init__(self, num_classes, args, arch = 'densenet161', model_file='densenet161_places365.pth.tar'):
        super(DenseNetBN, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        checkpoint = torch.load(model_file)
        args.start_epoch = checkpoint['epoch']
        args.best_prec1 = checkpoint['best_prec1']
        import re
        def repl(matchobj):
            return matchobj.group(0)[1:]
        state_dict = {re.sub('\.\d+\.',repl,str.replace(k,'module.','')): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

class VGGBN(nn.Module):
    def __init__(self, num_classes, args, arch = 'vgg16_bn', model_file=None):
        super(VGGBN, self).__init__()
        self.model = models.__dict__[arch](num_classes = 365)
        if model_file == 'vgg16_bn_places365.pt':
            state_dict = torch.load(model_file)
            args.start_epoch = 0
            args.best_prec1 = 40
            d = self.model.state_dict()
            new_state_dict = {k: state_dict[k] if k in state_dict.keys() else d[k] for k in d.keys()}
            self.model.load_state_dict(new_state_dict)
        elif model_file != None:
            checkpoint = torch.load(model_file)
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k,'module.model.',''): v for k,v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)