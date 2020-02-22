import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.nn import init
from .cbam import *
from .bam import *
from .dbn import *
from .iterative_normalization import IterNorm, IterNormRotation
from .iterative_normalization import IterNormRotation as cw_layer
#from .ZCANorm import ZCANormSVDPIRotation as cw_layer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if use_cbam:
            self.bw1 = IterNorm(planes, num_groups = 8, momentum = 0.1) # nn.BatchNorm2d(planes)
        else:
            self.bw1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if use_cbam:
            self.bw2 = IterNorm(planes, num_groups = 8, momentum = 0.1) # nn.BatchNorm2d(planes)
        else:
            self.bw2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bw1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bw2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bw1 = DBN(planes, num_groups = 1) # BatchWhitening(planes)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bw2 = DBN(planes, num_groups = 1) # BatchWhitening(planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bw3 = DBN(planes * 4, num_groups = 1) # BatchWhitening(planes * 4)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bw1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bw2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bw3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None, whitened_layers=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bw1 = IterNorm(64, num_groups = 1, momentum = 0.1)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layers = layers

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        
        self.whitened_layers = whitened_layers

        for whitened_layer in whitened_layers:
            if whitened_layer <= layers[0]:
                self.layer1[whitened_layer-1].bn1 = IterNormRotation(64)
            elif whitened_layer <= layers[0] + layers[1]:
                self.layer2[whitened_layer-layers[0]-1].bn1 = IterNormRotation(128)
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.layer3[whitened_layer-layers[0]-layers[1]-1].bn1 = IterNormRotation(256)
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1 = IterNormRotation(512)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def change_mode(self, mode):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.layer1[whitened_layer-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1]:
                self.layer2[whitened_layer-layers[0]-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.layer3[whitened_layer-layers[0]-layers[1]-1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1.mode = mode
    
    def update_rotation_matrix(self):
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.layer1[whitened_layer-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1]:
                self.layer2[whitened_layer-layers[0]-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.layer3[whitened_layer-layers[0]-layers[1]-1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1.update_rotation_matrix()

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))#, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))#, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bw1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResidualNet(network_type, depth, num_classes, att_type, whitened_layers):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, whitened_layers)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model

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
                self.model.layer1[whitened_layer-1].bn1 = cw_layer(64)
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1 = cw_layer(128)
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1 = cw_layer(256)
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1 = cw_layer(512)
    
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
                self.model.features.norm0 = cw_layer(64)
            elif whitened_layer == 2:
                self.model.features.transition1.norm = cw_layer(384)
            elif whitened_layer == 3:
                self.model.features.transition2.norm = cw_layer(768)
            elif whitened_layer == 4:
                self.model.features.transition3.norm = cw_layer(2112)
            elif whitened_layer == 5:
                self.model.features.norm5 = cw_layer(2208)
    
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
            self.model.features[self.layers[whitened_layer]] = cw_layer(channel)

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