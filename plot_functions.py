"""
Plot functions to generate figures in the paper

Note that these functions only work for ResNet, as claimed
in the paper, although one can adapt this code for other
type of architectures without too many efforts.

"""


import os
import shutil
import numpy as np
from numpy import linalg as LA
import seaborn as sns
from PIL import ImageFile, Image
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import matplotlib.pyplot as plt
import matplotlib
import skimage.measure
import random
import cv2
matplotlib.use('Agg')

from train_places import AverageMeter, accuracy

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from MODELS.model_resnet import *

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


'''
    This function finds the top 50 images that gets the greatest activations with respect to the concepts.
    Concept activation values are obtained based on iternorm_rotation module outputs.
    Since concept corresponds to channels in the output, we look for the top50 images whose kth channel activations
    are high.
'''
def plot_concept_top50(args, val_loader, model, whitened_layers, print_other = False, activation_mode = 'pool_max'):
    # switch to evaluate mode
    model.eval()
    from shutil import copyfile
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    layer_list = whitened_layers.split(',')
    folder = dst + '_'.join(layer_list) + '_rot/'
    # print(folder)
    if print_other:
        folder = dst + '_'.join(layer_list) + '_rot_otherdim/'
    if args.arch == "resnet_cw":
        folder = dst + '_'.join(layer_list) + '_rot_cw/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model

    outputs= []
    def hook(module, input, output):
        from MODELS.iterative_normalization import iterative_normalization_py
        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                 module.eps, module.momentum, module.training)
        size_X = X_hat.size()
        size_R = module.running_rot.size()
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
        X_hat = X_hat.view(*size_X)

        outputs.append(X_hat.cpu().numpy())
    
    for layer in layer_list:
        layer = int(layer)
        if layer <= layers[0]:
            model.layer1[layer-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

    begin = 0
    end = len(args.concepts.split(','))
    if print_other:
        # begin = len(args.concepts.split(','))
        # end = begin+30
        begin = print_other
        end = begin + 1
    concepts = args.concepts.split(',')
    with torch.no_grad():
        for k in range(begin, end):
            print(k)
            if k < len(concepts):
                output_path = os.path.join(folder, concepts[k])
            else:
                output_path = os.path.join(folder, 'other_dimension_'+str(k))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            paths = []
            vals = None
            for i, (input, _, path) in enumerate(val_loader):
                paths += list(path)
                input_var = torch.autograd.Variable(input).cuda()
                outputs = []
                model(input_var)
                val = []
                for output in outputs:
                    if activation_mode == 'mean':
                        val = np.concatenate((val,output.mean((2,3))[:,k]))
                    elif activation_mode == 'max':
                        val = np.concatenate((val,output.max((2,3))[:,k]))
                    elif activation_mode == 'pos_mean':
                        pos_bool = (output > 0).astype('int32')
                        act = (output * pos_bool).sum((2,3))/(pos_bool.sum((2,3))+0.0001)
                        val = np.concatenate((val,act[:,k]))
                    elif activation_mode == 'pool_max':
                        kernel_size = 3
                        r = output.shape[3] % kernel_size
                        if r == 0:
                            val = np.concatenate((val,skimage.measure.block_reduce(output[:,:,:,:],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,k]))
                        else:
                            val = np.concatenate((val,skimage.measure.block_reduce(output[:,:,:-r,:-r],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,k]))
                    elif activation_mode == 'pool_max_s1':
                        X_test = torch.Tensor(output)
                        maxpool_value, maxpool_indices = nn.functional.max_pool2d(X_test, kernel_size=3, stride=1, return_indices=True)
                        X_test_unpool = nn.functional.max_unpool2d(maxpool_value,maxpool_indices, kernel_size=3, stride =1)
                        maxpool_bool = X_test == X_test_unpool
                        act = (X_test_unpool.sum((2,3)) / maxpool_bool.sum((2,3)).float()).numpy()
                        val = np.concatenate((val,act[:,k]))

                val = val.reshape((len(outputs),-1))
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals,val),1)

            for i, layer in enumerate(layer_list):
                arr = list(zip(list(vals[i,:]),list(paths)))
                arr.sort(key = lambda t: t[0], reverse = True)
                # arr.sort(key = lambda t: t[0], reverse = False)
                # with open('76dim.txt', 'w') as f:
                #     for item in arr:
                #         f.write(item[1]+'\n')

                for j in range(5):
                    src = arr[j][1]
                    copyfile(src, output_path+'/'+'layer'+layer+'_'+str(j+1)+'.jpg')  
                    # copyfile(src, output_path+'/'+'layer'+layer+'_'+str(j+1)+'_reversed.jpg')  

    return 0

'''
    This method gets the activations of output from iternorm_rotation for images (from val_loader) at channel (cpt_idx)
'''
def get_layer_representation(args, val_loader, layer, cpt_idx):
    model = load_resnet_model(args, arch='resnet_cw', depth=18, whitened_layer=layer)
    with torch.no_grad():        
        model.eval()
        model = model.module
        layers = model.layers
        if args.arch == "resnet_cw":
            model = model.model
        outputs= []
    
        def hook(module, input, output):
            from MODELS.iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            #print(size_X)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())
            
        layer = int(layer)
        if layer <= layers[0]:
            model.layer1[layer-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

        paths = []
        vals = None
        for i, (input, _, path) in enumerate(val_loader):
            paths += list(path)
            input_var = torch.autograd.Variable(input).cuda()
            outputs = []
            model(input_var)
            val = []
            for output in outputs:
                val.append(output.sum((2,3))[:, cpt_idx])
            val = np.array(val)
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals,val),1)
    del model
    return paths, vals


# This method obtains the vector length of a representation (distance to origin)
# Can choose resnet_cw or resnet_original
def get_representation_distance_to_center(args, val_loader, layer, arch='resnet_cw'):
    # dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + arch + str(args.depth) + '/distance_to_center/'
    # if not os.path.exists(dst):
    #     os.mkdir(dst)
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer)
    #print(model)
    with torch.no_grad():
        model.eval()
        model = model.module
        layers = model.layers
        if args.arch == "resnet_cw":
            model = model.model
        outputs= []
    
        def hook(module, input, output):
            if arch == 'resnet_original':
                #outputs.append(input[0].cpu().numpy())
                outputs.append(output.cpu().numpy())
            else:
                from MODELS.iterative_normalization import iterative_normalization_py
                #print(input)
                X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
                size_X = X_hat.size()
                size_R = module.running_rot.size()
                X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                #print(size_X)
                X_hat = X_hat.view(*size_X)

                outputs.append(X_hat.cpu().numpy())
            
        layer = int(layer)
        if layer <= layers[0]:
            model.layer1[layer-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

        paths = []
        vals = []
        for i, (input, _) in enumerate(val_loader):
            #paths += list(path)
            # if i==500:
            #     break
            input_var = torch.autograd.Variable(input).cuda()
            outputs = []
            model(input_var)
            for output in outputs:
                # output_shape = output.size() #NCHW
                # reshaped = output.reshape((output_shape[0], output_shape[1], -1))
                # norms = LA.norm(reshaped, axis=2).flatten().tolist()
                output_shape = output.shape #NCHW
                # reshaped = output.transpose((0,2,3,1)).reshape((-1, output_shape[1]))
                reshaped = output.mean((2,3))
                norms = LA.norm(reshaped, axis=1).flatten().tolist()
                vals.extend(norms)
    del model
    #return paths, vals
    print(np.mean(vals),np.std(vals))
    # plt.figure()
    # plt.hist(vals, bins=100)
    # plt.xlim(left=0,right=30)
    # plt.savefig(dst+'layer'+str(layer)+'_mean.jpg')

    return vals


# This method compares the intra concept group dot product with inter concept group dot product
def intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir, layer, plot_cpt = ['airplane','bed','person'], activation_mode = 'mean', arch='resnet_cw', dataset = 'places365'):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/inner_product/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(conceptdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    concept_list = os.listdir(conceptdir)
    concept_list.sort()
    
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset = dataset)
    model.eval()
    model = model.module
    layers = model.layers
    # if arch == "resnet_cw" or arch == "resnet_baseline":
    model = model.model
    
    representations = {}
    for cpt in plot_cpt:
        representations[cpt] = []
    
    for c, cpt in enumerate(plot_cpt):
        with torch.no_grad():
            
            outputs= []
        
            def hook(module, input, output):
                if arch == 'resnet_original' or arch == "resnet_baseline":
                    #outputs.append(input[0].cpu().numpy())
                    outputs.append(output.cpu().numpy())
                else:
                    from MODELS.iterative_normalization import iterative_normalization_py
                    #print(input)
                    X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                            module.eps, module.momentum, module.training)
                    size_X = X_hat.size()
                    size_R = module.running_rot.size()
                    X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                    X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                    #print(size_X)
                    X_hat = X_hat.view(*size_X)

                    outputs.append(X_hat.cpu().numpy())
                
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

            print(c, cpt)
            for j, (input, y, path) in enumerate(concept_loader):
                labels = y.cpu().numpy().flatten().astype(np.int32).tolist()
                input_var = torch.autograd.Variable(input).cuda()
                outputs = []
                model(input_var)
                for instance_index in range(len(labels)): # batch size
                    instance_concept_index = labels[instance_index]
                    if concept_list[instance_concept_index] in plot_cpt: # only get the representation of concepts of instances from plot_cpt list
                        representation_concept_index = plot_cpt.index(concept_list[instance_concept_index])
                        output_shape = outputs[0].shape
                        representation_mean = outputs[0][instance_index:instance_index+1, :, :, :].transpose((0,2,3,1)).reshape((-1, output_shape[1])).mean(axis=0) # mean of all pixels of that instance
                        representations[concept_list[instance_concept_index]].append(representation_mean) # get the cpt_index channel of the output

    # representation of concepts in matrix form
    dot_product_matrix = np.zeros((len(plot_cpt),len(plot_cpt))).astype('float')
    m_representations = {}
    m_representations_normed = {}
    intra_dot_product_means = {}
    intra_dot_product_means_normed = {}
    for i, concept in enumerate(plot_cpt):
        m_representations[concept] = np.stack(representations[concept], axis=0) # n * (h*w)
        m_representations_normed[concept] = m_representations[concept]/LA.norm(m_representations[concept], axis=1, keepdims=True)
        intra_dot_product_means[concept] = np.matmul(m_representations[concept], m_representations[concept].transpose()).mean()
        intra_dot_product_means_normed[concept] = np.matmul(m_representations_normed[concept], m_representations_normed[concept].transpose()).mean()
        dot_product_matrix[i,i] = 1.0

    inter_dot_product_means = {}
    inter_dot_product_means_normed = {}
    for i in range(len(plot_cpt)):
        for j in range(i+1, len(plot_cpt)):
            cpt_1 = plot_cpt[i]
            cpt_2 = plot_cpt[j]
            inter_dot_product_means[cpt_1 + '_' + cpt_2] = np.matmul(m_representations[cpt_1], m_representations[cpt_2].transpose()).mean()
            inter_dot_product_means_normed[cpt_1 + '_' + cpt_2] = np.matmul(m_representations_normed[cpt_1], m_representations_normed[cpt_2].transpose()).mean()
            dot_product_matrix[i,j] = abs(inter_dot_product_means_normed[cpt_1 + '_' + cpt_2]) / np.sqrt(abs(intra_dot_product_means_normed[cpt_1]*intra_dot_product_means_normed[cpt_2]))
            dot_product_matrix[j,i] = dot_product_matrix[i,j]
    
    print(intra_dot_product_means, inter_dot_product_means)
    print(intra_dot_product_means_normed, inter_dot_product_means_normed)
    print(dot_product_matrix)
    plt.figure()
    ticklabels  = [s.replace('_',' ') for s in plot_cpt]
    sns.set(font_scale=1.4)
    ax = sns.heatmap(dot_product_matrix, vmin = 0, vmax = 1, xticklabels = ticklabels, yticklabels = ticklabels)
    ax.figure.tight_layout()
    plt.savefig(dst + arch + '_' + str(layer) +'.jpg')

    return intra_dot_product_means, inter_dot_product_means, intra_dot_product_means_normed, inter_dot_product_means_normed

'''
    This function plots the relative activations of a image on two different concepts. 
'''
def plot_trajectory(args, val_loader, whitened_layers, plot_cpt = ['airplane','bed']):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/trajectory_all/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    concepts = args.concepts.split(',')
    cpt_idx = [concepts.index(plot_cpt[0]),concepts.index(plot_cpt[1])]
    vals = None 
    layer_list = whitened_layers.split(',')
    for i, layer in enumerate(layer_list):
        #print(i)
        if i == 0:
            paths, vals = get_layer_representation(args, val_loader, layer, cpt_idx)
        else:
            _, val = get_layer_representation(args, val_loader, layer, cpt_idx)
            vals = np.concatenate((vals,val),0)
        #print(vals.shape)
    try:
        os.mkdir('{}{}'.format(dst,'_'.join(plot_cpt)))
    except:
        pass

    num_examples = vals.shape[1]
    num_layers = vals.shape[0]
    max_vals = np.amax(vals, axis=1)
    min_vals = np.amin(vals, axis=1)
    vals = vals.transpose((1,0,2))
    # vals = (vals - min_vals)/(max_vals-min_vals)
    sort_idx = vals.argsort(0)
    for i in range(num_layers):
        for j in range(2):
            vals[sort_idx[:,i,j],i,j] = np.arange(num_examples)/num_examples
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    for k, i in enumerate(idx):
        #print(k)
        if k==300:
            break
        fig = plt.figure(figsize=(10,5))
        ax2 = plt.subplot(1,2,2)
        ax2.set_xlim([0.0,1.0])
        ax2.set_ylim([0.0,1.0])
        ax2.set_xlabel(plot_cpt[0])
        ax2.set_ylabel(plot_cpt[1])
        plt.scatter(vals[i,:,0],vals[i,:,1])
        start_x = vals[i,0,0]
        start_y = vals[i,0,1]
        for j in range(1, num_layers):
            dx, dy = vals[i,j,0]-vals[i,j-1,0],vals[i,j,1]-vals[i,j-1,1]
            plt.arrow(start_x, start_y, dx, dy, length_includes_head=True, head_width=0.01, head_length=0.02)
            start_x, start_y = vals[i,j,0], vals[i,j,1]
        ax1 = plt.subplot(1,2,1)
        ax1.axis('off')
        I = Image.open(paths[i]).resize((300,300),Image.ANTIALIAS)
        plt.imshow(np.asarray(I).astype(np.int32))
        plt.savefig('{}{}/{}.jpg'.format(dst,'_'.join(plot_cpt), k))

'''
    For each layer and each concept, using activation value as the predicted probability of being a certain concept,
    auc score is computed with respect to label
'''
def plot_auc_cw(args, conceptdir, whitened_layers, plot_cpt = ['airplane','bed','person'], activation_mode = 'pool_max', dataset = 'places365'):
    # dst = './plot/' + args.arch + str(args.depth) + '/auc/cw/'
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) +'/auc/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    dst += 'cw/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    concept_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(conceptdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    layer_list = whitened_layers.split(',')
    concept_list = os.listdir(conceptdir)
    concept_list.sort()
    #print(concept_list)
    aucs = np.zeros((len(plot_cpt),len(layer_list)))
    aucs_err = np.zeros((len(plot_cpt),len(layer_list)))
    #print(aucs.shape)
    for c, cpt in enumerate(plot_cpt):
        #print(cpt)
        cpt_idx_2 = concept_list.index(cpt)
        cpt_idx = plot_cpt.index(cpt)
        #print(cpt_idx, cpt_idx_2)
        for i, layer in enumerate(layer_list):
            model = load_resnet_model(args, arch='resnet_cw', depth=18, whitened_layer=layer, dataset = dataset)
            with torch.no_grad():
                model.eval()
                model = model.module
                layers = model.layers
                model = model.model
                outputs= []
            
                def hook(module, input, output):
                    from MODELS.iterative_normalization import iterative_normalization_py
                    #print(input)
                    X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                            module.eps, module.momentum, module.training)
                    size_X = X_hat.size()
                    size_R = module.running_rot.size()
                    X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

                    X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
                    #print(size_X)
                    X_hat = X_hat.view(*size_X)

                    outputs.append(X_hat.cpu().numpy())
                    
                layer = int(layer)
                if layer <= layers[0]:
                    model.layer1[layer-1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1]:
                    model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2]:
                    model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
                elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                    model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

                labels = []
                vals = []
                for j, (input, y, path) in enumerate(concept_loader):
                    #print(y, path)
                    labels += list(y.cpu().numpy())
                    input_var = torch.autograd.Variable(input).cuda()
                    outputs = []
                    model(input_var)
                    for output in outputs:
                        if activation_mode == 'mean':
                            vals += list(output.mean((2,3))[:, cpt_idx])
                        elif activation_mode == 'max':
                            vals += list(output.max((2,3))[:, cpt_idx])
                        elif activation_mode == 'pos_mean':
                            pos_bool = (output > 0).astype('int32')
                            act = (output * pos_bool).sum((2,3))/(pos_bool.sum((2,3))+0.0001)
                            vals += list(act[:, cpt_idx])
                        elif activation_mode=='pool_max':
                            kernel_size = 3
                            r = output.shape[3] % kernel_size
                            if r == 0:
                                vals += list(skimage.measure.block_reduce(output[:,:,:,:],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,cpt_idx])
                            else:
                                vals += list(skimage.measure.block_reduce(output[:,:,:-r,:-r],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,cpt_idx])
                        elif activation_mode == 'pool_max_s1':
                            X_test = torch.Tensor(output)
                            maxpool_value, maxpool_indices = nn.functional.max_pool2d(X_test, kernel_size=3, stride=1, return_indices=True)
                            X_test_unpool = nn.functional.max_unpool2d(maxpool_value,maxpool_indices, kernel_size=3, stride =1)
                            maxpool_bool = X_test == X_test_unpool
                            act = (X_test_unpool.sum((2,3)) / maxpool_bool.sum((2,3)).float()).numpy()
                            vals += list(act[:, cpt_idx])
                del model
            vals = np.array(vals)
            labels = np.array(labels)
            labels = (labels == cpt_idx_2).astype('int32')
            n_samples = labels.shape[0]
            t = 5
            idx = np.array_split(np.random.permutation(n_samples),t)
            auc_t = []
            for j in range(t):
                # idx = np.random.permutation(n_samples)[:n_samples//2]
                # auc_t.append(roc_auc_score(labels[idx], vals[idx]))
                auc_t.append(roc_auc_score(labels[idx[j]], vals[idx[j]]))
            aucs[c,i] = np.mean(auc_t)
            aucs_err[c,i] = np.std(auc_t)
            print(aucs[c,i])
            print(aucs_err[c,i])
    
    print('AUC-CW', aucs)
    print('AUC-CW-err', aucs_err)
    np.save(dst + 'aucs_cw.npy', aucs)
    np.save(dst + 'aucs_cw_err.npy', aucs_err)
    return aucs

'''
    Attempt to predict concept class using activation values. This is a measure of separability of concept representation in the latent space.
    Better separated concept class representations (output of BN in resnet blocks) should produce greater AUC.
'''
def plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir, whitened_layers, plot_cpt = ['airplane', 'bed', 'person'], model_type = 'svm'):
    # dst = './plot/' + 'resnet_cw' + str(args.depth) + '/auc/tcav/'
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) +'/auc/tcav/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    layer_list = whitened_layers.split(',')
    aucs = np.zeros((len(plot_cpt),len(layer_list)))
    aucs_err = np.zeros((len(plot_cpt),len(layer_list)))

    model.eval()
    model = model.module
    layers = model.layers
    model = model.model

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    concept_loader_test = torch.utils.data.DataLoader(
        datasets.ImageFolder(conceptdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    concept_list = os.listdir(conceptdir)
    concept_list.sort()

    n_batch = 9
    with torch.no_grad(): 
        outputs= []
        def hook(module, input, output):
            outputs.append(output.cpu().numpy())
        
        for layer in layer_list:
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)
        
        labels = []
        activation_test = None
        for i, (input, y) in enumerate(concept_loader_test):
            labels += list(y.cpu().numpy())
            outputs = []
            input_var = torch.autograd.Variable(input).cuda()
            model(input_var)
            if i == 0:
                activation_test = outputs
            else:
                for k in range(len(outputs)):
                    activation_test[k] = np.concatenate((activation_test[k], outputs[k]),0)
        labels = np.array(labels).astype('int32')

        for c, cpt in enumerate(plot_cpt):
            cpt_idx_2 = concept_list.index(cpt)
            concept_loader_train = concept_loaders[c]

            activation = None
            for i, (input, _) in enumerate(concept_loader_train):
                if i == n_batch:
                    break
                outputs = []
                input_var = torch.autograd.Variable(input).cuda()
                model(input_var)
                if i == 0:
                    activation = outputs
                else:
                    for k in range(len(outputs)):
                        activation[k] = np.concatenate((activation[k], outputs[k]),0)

            num_positive = activation[0].shape[0]

            for i, (input, _) in enumerate(train_loader):
                if i == n_batch:
                    break
                outputs = []
                input_var = torch.autograd.Variable(input).cuda()
                model(input_var)
                for k in range(len(outputs)):
                    activation[k] = np.concatenate((activation[k], outputs[k]),0)
            
            y_train = np.ones(activation[0].shape[0])
            y_train[num_positive:] = 0

            for i in range(len(layer_list)):
                x_train = activation[i].reshape((len(y_train),-1))
                y_train = y_train

                if model_type == 'svm':
                    lm = SGDClassifier(loss='hinge')
                elif model_type == 'lr':
                    lm = LogisticRegression()
                lm.fit(x_train, y_train)

                x_test = activation_test[i].reshape((len(labels),-1))
                y_test = (labels == cpt_idx_2).astype('int32')
                cav = lm.coef_
                score = (x_test*cav).sum(1)
                n_samples = labels.shape[0]
                t = 5
                idx = np.array_split(np.random.permutation(n_samples),t)
                auc_t = []
                for j in range(t):
                    auc_t.append(roc_auc_score(y_test[idx[j]], score[idx[j]]))
                aucs[c,i] = np.mean(auc_t)
                aucs_err[c,i] = np.std(auc_t)
                print(aucs[c,i])
                print(aucs_err[c,i])
    
    print('AUC-'+model_type, aucs)
    print('AUC-'+model_type + '-err', aucs_err)
    np.save(dst + 'aucs_' + model_type + '.npy', aucs)
    np.save(dst + 'aucs_' + model_type + '_err.npy', aucs_err)

    return aucs


'''
    For each concept and output of each layer, look for the channel in the output that best predicts the concept class (binary classification).
    The activation of a channel that does the best when used as the predicted probability (measured using AUC) is assumed to be the channel that
    best represents the concept.
'''
def plot_auc_filter(args, model, conceptdir, whitened_layers, plot_cpt = ['airplane', 'bed', 'person'], activation_mode = 'pool_max'):
    # dst = './plot/' + 'resnet_cw' + str(args.depth) + '/auc/filter/'
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) +'/auc/filter/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    layer_list = whitened_layers.split(',')
    aucs = np.zeros((len(plot_cpt),len(layer_list)))
    aucs_err = np.zeros((len(plot_cpt),len(layer_list)))

    model.eval()
    model = model.module
    layers = model.layers
    model = model.model

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    concept_loader_test = torch.utils.data.DataLoader(
        datasets.ImageFolder(conceptdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    concept_list = os.listdir(conceptdir)
    concept_list.sort()

    aucs = np.zeros((len(plot_cpt),len(layer_list)))

    with torch.no_grad(): 
        outputs= []
        def hook(module, input, output):
            outputs.append(output.cpu().numpy())
        
        for layer in layer_list:
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)
        
        labels = []
        activation_test = None
        for i, (input, y) in enumerate(concept_loader_test):
            labels += list(y.cpu().numpy())
            outputs = []
            input_var = torch.autograd.Variable(input).cuda()
            model(input_var)
            if i == 0:
                activation_test = outputs
            else:
                for k in range(len(outputs)):
                    activation_test[k] = np.concatenate((activation_test[k], outputs[k]),0)
        labels = np.array(labels).astype('int32')

        for c, cpt in enumerate(plot_cpt):
            cpt_idx_2 = concept_list.index(cpt)
            for i in range(len(layer_list)):
                if activation_mode == 'mean':
                    x_test = activation_test[i].mean((2,3))
                elif activation_mode == 'max':
                    x_test = activation_test[i].max((2,3))
                elif activation_mode == 'pos_mean':
                    pos_bool = (activation_test[i] > 0).astype('int32')
                    x_test = (activation_test[i] * pos_bool).sum((2,3))/(pos_bool.sum((2,3))+0.0001)
                elif activation_mode == 'pool_max':
                    kernel_size = 3
                    r = activation_test[i].shape[3] % kernel_size
                    if r == 0:
                        x_test = skimage.measure.block_reduce(activation_test[i][:,:,:,:],(1,1,kernel_size,kernel_size),np.max).mean((2,3))
                    else:
                        x_test = skimage.measure.block_reduce(activation_test[i][:,:,:-r,:-r],(1,1,kernel_size,kernel_size),np.max).mean((2,3))
                elif activation_mode == 'pool_max_s1':
                    X_test = torch.Tensor(activation_test[i])
                    maxpool_value, maxpool_indices = nn.functional.max_pool2d(X_test, kernel_size=3, stride=1, return_indices=True)
                    X_test_unpool = nn.functional.max_unpool2d(maxpool_value,maxpool_indices, kernel_size=3, stride =1)
                    maxpool_bool = X_test == X_test_unpool
                    x_test = (X_test_unpool.sum((2,3)) / maxpool_bool.sum((2,3)).float()).numpy()
                y_test = (labels == cpt_idx_2).astype('int32')
                t = 5
                auc_t = np.zeros([x_test.shape[1], t])
                n_samples = labels.shape[0]
                idx = np.array_split(np.random.permutation(n_samples),t)
                for j in range(x_test.shape[1]):
                    score = x_test[:,j]
                    for k in range(t):
                        #aucs[c,i] = max(roc_auc_score(y_test, score),aucs[c,i])
                        # auc_t[k] = max(roc_auc_score(y_test[idx[k]], score[idx[k]]),auc_t[k])
                        auc_t[j,k] = roc_auc_score(y_test[idx[k]], score[idx[k]])
                
                filter_i = auc_t.mean(1).argmax()
                aucs[c,i] = np.mean(auc_t[filter_i])
                aucs_err[c,i] = np.std(auc_t[filter_i])
                print(aucs[c,i])
                print(aucs_err[c,i])
    print('AUC-best_filter',aucs)
    print('AUC-best_filter-err',aucs_err)
    np.save(dst + 'aucs_filter.npy', aucs)
    np.save(dst + 'aucs_filter_err.npy', aucs_err)
    return aucs

def plot_auc(args, aucs_cw, aucs_svm, aucs_lr, aucs_filter, plot_cpt = ['airplane', 'bed', 'person']):
    folder = './plot/' + '_'.join(args.concepts.split(',')) + '/' + 'resnet_cw' + str(args.depth) + '/auc/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    aucs_cw = np.load(folder + 'cw/' + 'aucs_cw.npy')
    aucs_svm = np.load(folder + 'tcav/' + 'aucs_svm.npy')
    aucs_lr = np.load(folder + 'tcav/' + 'aucs_lr.npy')
    aucs_filter = np.load(folder + 'filter/' + 'aucs_filter.npy')
    aucs_cw_err = np.load(folder + 'cw/' + 'aucs_cw_err.npy')
    aucs_svm_err = np.load(folder + 'tcav/' + 'aucs_svm_err.npy')
    aucs_lr_err = np.load(folder + 'tcav/' + 'aucs_lr_err.npy')
    aucs_filter_err = np.load(folder + 'filter/' + 'aucs_filter_err.npy')

    for c, cpt in enumerate(plot_cpt):
        fig = plt.figure(figsize=(5,5))
        # plt.plot([2,4,6,8,10,12,14,16], aucs_cw[c], label = 'CW')
        # plt.plot([2,4,6,8,10,12,14,16], aucs_svm[c], label = 'SVM (CAV)', )
        # plt.plot([2,4,6,8,10,12,14,16], aucs_lr[c], label = 'LR (IBD,CAV)')
        # plt.plot([2,4,6,8,10,12,14,16], aucs_filter[c], label = 'Best filter')
        plt.errorbar([2,4,6,8,10,12,14,16], aucs_cw[c], yerr=aucs_cw_err[c], label = 'CW')
        plt.errorbar([2,4,6,8,10,12,14,16], aucs_svm[c], yerr=aucs_svm_err[c], label = 'SVM (CAV)', )
        plt.errorbar([2,4,6,8,10,12,14,16], aucs_lr[c], yerr=aucs_lr_err[c], label = 'LR (IBD,CAV)')
        plt.errorbar([2,4,6,8,10,12,14,16], aucs_filter[c], yerr=aucs_filter_err[c], label = 'Best filter')
        plt.xlabel('layer', fontsize = 16)
        plt.ylabel('auc', fontsize = 16)
        plt.legend(fontsize = 13)
        plt.savefig('{}/{}.jpg'.format(folder,cpt))    

def plot_top10(args, plot_cpt = ['airplane', 'bed', 'person'], layer = 1):
    folder = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/' + str(layer) + '_rot_cw/'

    fig, axes = plt.subplots(figsize=(30, 3*len(plot_cpt)) , nrows=len(plot_cpt), ncols=10)

    import matplotlib.image as mpimg
    for c, cpt in enumerate(plot_cpt):
        for i in range(10):
            axes[c,i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' +str(i+1)+'.jpg'))
            axes[c,i].set_yticks([])
            axes[c,i].set_xticks([])

    for ax, row in zip(axes[:,0], plot_cpt):
        ax.set_ylabel(row.replace('_','\n'), rotation=90, size='large', fontsize = 40, wrap=False)

    fig.tight_layout()
    plt.show()
    fig.savefig(folder+'layer'+str(layer)+'.jpg')


def plot_concept_representation(args, val_loader, model, whitened_layers, plot_cpt = ['airplane','bed'], activation_mode = 'mean'):    
    with torch.no_grad():
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) +'/representation/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        layers = model.layers
        layer_list = whitened_layers.split(',')
        dst = dst + '_'.join(layer_list) + '/'
        if args.arch == "resnet_cw":
            model = model.model
        if not os.path.exists(dst):
            os.mkdir(dst)
        outputs= []
    
        def hook(module, input, output):
            from MODELS.iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            #print(size_X)
            X_hat = X_hat.view(*size_X)

            outputs.append(X_hat.cpu().numpy())
            
        for layer in layer_list:
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

        concepts = args.concepts.split(',')
        cpt_idx = [concepts.index(plot_cpt[0]),concepts.index(plot_cpt[1])]


        paths = []
        vals = None
        for i, (input, _, path) in enumerate(val_loader):
            paths += list(path)
            input_var = torch.autograd.Variable(input).cuda()
            outputs = []
            model(input_var)
            val = []
            for output in outputs:
                #val.append(output.sum((2,3))[:,cpt_idx])
                if activation_mode == 'mean':
                    val.append(output.mean((2,3))[:,cpt_idx])
                elif activation_mode == 'max':
                    val.append(output.max((2,3))[:,cpt_idx])
                elif activation_mode == 'pos_mean':
                    pos_bool = (output > 0).astype('int32')
                    act = (output * pos_bool).sum((2,3))/(pos_bool.sum((2,3))+0.0001)
                    val.append(act[:,cpt_idx])
                elif activation_mode=='pool_max':
                    kernel_size = 3
                    r = output.shape[3] % kernel_size
                    if r == 0:
                        val.append(skimage.measure.block_reduce(output[:,:,:,:],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,cpt_idx])
                    else:
                        val.append(skimage.measure.block_reduce(output[:,:,:-r,:-r],(1,1,kernel_size,kernel_size),np.max).mean((2,3))[:,cpt_idx])
                elif activation_mode == 'pool_max_s1':
                    X_test = torch.Tensor(output)
                    maxpool_value, maxpool_indices = nn.functional.max_pool2d(X_test, kernel_size=3, stride=1, return_indices=True)
                    X_test_unpool = nn.functional.max_unpool2d(maxpool_value,maxpool_indices, kernel_size=3, stride =1)
                    maxpool_bool = X_test == X_test_unpool
                    act = (X_test_unpool.sum((2,3)) / maxpool_bool.sum((2,3)).float()).numpy()
                    val.append(act[:,cpt_idx])
            val = np.array(val)
            if i == 0:
                vals = val
            else:
                vals = np.concatenate((vals,val),1)
        
        for l, layer in enumerate(layer_list):
            n_grid = 20
            img_size = 100
            img_merge = np.zeros((img_size*n_grid,img_size*n_grid,3))
            idx_merge = -np.ones((n_grid+1,n_grid+1))
            cnt = np.zeros((n_grid+1,n_grid+1))
            arr = vals[l,:]
            for j in range(len(paths)):
                index = np.floor((arr[j,:]-arr.min(0))/(arr.max(0)-arr.min(0))*n_grid).astype(np.int32)
                idx_merge[index[0],index[1]] = j
                cnt[index[0],index[1]] += 1

            for i in range(n_grid):
                for j in range(n_grid):
                    index = idx_merge[i,j].astype(np.int32)
                    if index >= 0:
                        path = paths[index]
                        I = Image.open(path).resize((img_size,img_size),Image.ANTIALIAS).convert("RGB")
                        img_merge[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size,:] = np.asarray(I)
            plt.figure()
            plt.imshow(img_merge.astype(np.int32))
            plt.xlabel(plot_cpt[1])
            plt.ylabel(plot_cpt[0])
            plt.savefig(dst+'layer'+layer+'_'+'_'.join(plot_cpt)+'.jpg',dpi=img_size*n_grid//4)
            plt.figure()
            ax = sns.heatmap(cnt/cnt.sum(), linewidth=0.5)
            plt.xlabel(plot_cpt[1])
            plt.ylabel(plot_cpt[0])
            plt.savefig(dst+'density_layer'+layer+'_'+'_'.join(plot_cpt)+'.jpg')
    
    return 0

def plot_correlation(args, val_loader, model, layer):
    with torch.no_grad():
        dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        dst = dst + 'correlation_matrix/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        layers = model.layers
        model = model.model
        outputs= []
    
        def hook(module, input, output):
            size_X = output.size()
            X = output.transpose(0,1).reshape(size_X[1],-1).transpose(0,1)
            M = X.cpu().numpy()
            outputs.append(M)

        layer = int(layer)
        if layer <= layers[0]:
            model.layer1[layer-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

        for i, (input, _, path) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input).cuda()
            model(input_var)
            if i==50:
                break
        # print(np.shape(outputs),np.shape(outputs[0]))
        #activation = np.array(outputs).reshape((-1,np.shape(outputs)[2]))
        activation = np.vstack(outputs)
        activation -= activation.mean(0)
        activation = activation / activation.std(0)
        Sigma = np.dot(activation.transpose((1,0)),activation) / activation.shape[0]
        #print(Sigma)
        plt.figure()
        sns.heatmap(np.abs(Sigma),cmap='hot')
        plt.tight_layout()
        plt.savefig(dst + str(layer) + '.jpg')

# will compute the concept importance of the top {num_concepts} concepts in the given layer
def concept_permutation_importance(args, val_loader, layer, criterion, arch='resnet_cw', dataset='isic', num_concepts=7):
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)

    base_loss = 0
    base_accuracy = 0
    permutation_loss = [] # permutation_loss[i] represents the loss obtained when concept i is shuffled
    permutation_accuracy = []

    with torch.no_grad():
        model.eval()
        model = model.module
        layers = model.layers
        if args.arch == "resnet_cw":
            model = model.model

        loss_avg = AverageMeter()
        accuracy_avg = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)
            loss_avg.update(loss.data, input.size(0))

            [accuracy_batch] = accuracy(output.data, target_var, topk=(1,))
            accuracy_avg.update(accuracy_batch, input.size(0))
        
        base_loss = loss_avg.avg
        print('base loss', base_loss)
        base_accuracy = accuracy_avg.avg

        for axis_to_permute in range(num_concepts):
            loss_avg = AverageMeter()
            accuracy_avg = AverageMeter()

            def hook(module, input, output):
                batch_size = output.size()[0]
                idx = list(range(batch_size))
                random.shuffle(idx)
                new_output = output.clone()
                new_output[:,axis_to_permute,:,:] = new_output[idx,axis_to_permute,:,:]
                return new_output
                
            layer = int(layer)
            if layer <= layers[0]:
                model.layer1[layer-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1]:
                model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2]:
                model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

            for i, (input, target) in enumerate(val_loader):
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                output = model(input_var)
                loss = criterion(output, target_var)
                loss_avg.update(loss.data, input.size(0))
                [accuracy_batch] = accuracy(output.data, target_var, topk=(1,))
                accuracy_avg.update(accuracy_batch, input.size(0))
            
            print(axis_to_permute, loss_avg.avg)
            permutation_loss.append(loss_avg.avg)
            permutation_accuracy.append(accuracy_avg.avg)
        
    print('max_i', np.argmax(permutation_loss), np.max(permutation_loss))
    print('min_i', np.argmin(permutation_loss), np.min(permutation_loss))
    print(permutation_loss)
    print(base_loss)
    print(permutation_accuracy)
    print(base_accuracy)


def concept_gradient_importance(args, val_loader, layer, criterion, arch='resnet_cw', dataset='isic', num_classes=2):
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)
    # model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model
    
    print(model)
    outputs = []

    def hook(module, grad_input, grad_output):
        outputs.append(grad_input[0])
    
    layer = int(layer)
    if layer <= layers[0]:
        model.layer1[layer-1].relu.register_backward_hook(hook)
    elif layer <= layers[0] + layers[1]:
        model.layer2[layer-layers[0]-1].relu.register_backward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2]:
        model.layer3[layer-layers[0]-layers[1]-1].relu.register_backward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
        model.layer4[layer-layers[0]-layers[1]-layers[2]-1].relu.register_backward_hook(hook)

    class_count = [0] * num_classes
    concept_importance_per_class = [None] * num_classes

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        output = model(input_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        class_count[prediction_result] += 1
        output[:,prediction_result].backward()
        directional_derivatives = outputs[1].mean(dim=1).flatten().cpu().numpy()
        is_positive = (directional_derivatives > 0).astype(np.int64)
        if concept_importance_per_class[prediction_result] is None:
            concept_importance_per_class[prediction_result] = is_positive 
        else:
            concept_importance_per_class[prediction_result] += is_positive
        outputs = []
    
    for i in range(num_classes):
        concept_importance_per_class[i] = concept_importance_per_class[i].astype(np.float32)
        concept_importance_per_class[i] /= class_count[i]
        print(concept_importance_per_class[i])
        print(concept_importance_per_class[i].mean())


def saliency_map_class(args, val_loader, layer, arch='resnet_cw', dataset='isic'):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map/'
    # dst = '/usr/xtmp/zhichen/temp_plots/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)
    # model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model
    
    print(model)

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        input_var.requires_grad = True
        output = model(input_var)
        model.zero_grad()
        prediction_result = torch.argmax(output, dim=1).flatten().tolist()[0]
        output[:,prediction_result].backward()
        
        save_folder = os.path.join(dst, "class_"+str(prediction_result))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(input[0].permute(1,2,0))
        plt.subplot(1,2,2)
        grad = input_var.grad[0].permute(1,2,0).abs().cpu().numpy()
        grad = (grad/grad.max() * 255).astype(np.int8).max(axis=2)
        plt.imshow(grad, cmap='hot', interpolation='nearest')
        plt.savefig(os.path.join(save_folder, str(i)+'.png'))
        plt.close()

def saliency_map_concept(args, val_loader, layer, arch='resnet_cw', dataset='isic', num_concepts=7):
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map_concept/'
    # dst = '/usr/xtmp/zhichen/temp_plots/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)
    # model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model
    
    print(model)

    outputs = []
    
    def hook(module, input, output):
        outputs.append(output)
    
    layer = int(layer)
    if layer <= layers[0]:
        model.layer1[layer-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1]:
        model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2]:
        model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
        model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

    for j in range(num_concepts):
        save_folder = os.path.join(dst, "concept_"+str(j))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        input_var.requires_grad = True
        for j in range(num_concepts):
            output = model(input_var)
            model.zero_grad()
            outputs[0][0,j,:,:].mean().backward()
            save_folder = os.path.join(dst, "concept_"+str(j))    
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(input[0].permute(1,2,0))
            plt.subplot(1,2,2)
            grad = input_var.grad[0].permute(1,2,0).abs().cpu().numpy()
            grad = (grad/grad.max() * 255).astype(np.int8).max(axis=2)
            plt.imshow(grad, cmap='hot', interpolation='nearest')
            plt.savefig(os.path.join(save_folder, str(i)+'.png'))
            plt.close()
            outputs = []

def saliency_map_concept_cover(args, val_loader, layer, arch='resnet_cw', dataset='isic', num_concepts=7):
    # dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map_concept_cover_fine_grain_2/'
    dst = '/usr/xtmp/zhichen/temp_plots_layer1_3/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)
    model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model
    
    print(model)

    outputs = []
    
    def hook(module, input, output):
        outputs.append(output)
    
    layer = int(layer)
    if layer <= layers[0]:
        model.layer1[layer-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1]:
        model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2]:
        model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
        model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

    for j in range(num_concepts):
        save_folder = os.path.join(dst, "concept_"+str(j))
        try:
            os.mkdir(save_folder)
        except:
            pass
    
    from matplotlib import cm
    cover_size = 32
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            random_patch = torch.tensor(np.random.normal(size=(3,cover_size,cover_size)))
            input_var = torch.autograd.Variable(input).cuda()
            output = model(input_var)
            base_activations = nn.functional.max_pool2d(outputs[0], kernel_size=3, stride=3)
            base_activations = base_activations[0,:,:,:].clamp(min=0.0).mean(dim=(1,2))
            outputs = []

            input_size = input.size()
            saliency = np.zeros((num_concepts, input_size[2], input_size[3]))
            counter = np.zeros((input_size[2], input_size[3])) + 0.00001
            for p in range(0, input_size[2]-cover_size+1, 5):
                print("p={}\n".format(p))
                for q in range(0, input_size[3]-cover_size+1, 5):
                    new_input = input.clone()
                    new_input[0,:,p:p+cover_size, q:q+cover_size] = random_patch
                    input_var = torch.autograd.Variable(new_input).cuda()
                    output = model(input_var)
                    new_activations = nn.functional.max_pool2d(outputs[0], kernel_size=3, stride=3)
                    new_activations = new_activations[0,:,:,:].clamp(min=0.0).mean(dim=(1,2))
                    # new_activations = outputs[0][0,:,:,:].clamp(min=0.0).max(dim=(1,2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    for j in range(num_concepts):
                        saliency[j, p:p+cover_size, q:q+cover_size] += decrease_in_activations[j].cpu().numpy()
                    counter[p:p+cover_size, q:q+cover_size] += 1.0
            
            saliency = saliency/counter
            # print(saliency)
            
            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)
            
            saliency = (saliency - saliency.min())/saliency.max()
            lower_limit = np.percentile(saliency, 96)
            saliency[saliency<lower_limit] = 0.3
            saliency[saliency>=lower_limit] = 1.0

            input_image = input[0,:,:,:].permute(1,2,0).cpu().numpy()
            input_image *= std
            input_image += mean
            
            for j in range(num_concepts):
                save_folder = os.path.join(dst, "concept_"+str(j))
                image = Image.fromarray(np.uint8(input_image*255)).convert('RGBA')
                image = np.array(image)
                image[:,:,3] = (saliency[j,:,:]*255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(save_folder, str(i)+'.png'), 'PNG')
                print("saved: " + str(j))

def saliency_map_concept_cover_2(args, val_loader, layer, arch='resnet_cw', dataset='isic', num_concepts=7):
    # dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + str(args.depth) + '/saliency_map_concept_cover_fine_grain_2/'
    dst = '/usr/xtmp/zhichen/temp_plots_isic_3/'
    try:
        os.mkdir(dst)
    except:
        pass
    model = load_resnet_model(args, arch=arch, depth=18, whitened_layer=layer, dataset=dataset)
    #print(model)
    model.eval()
    model = model.module
    layers = model.layers
    if args.arch == "resnet_cw":
        model = model.model
    
    print(model)

    outputs = []
    
    def hook(module, input, output):
        outputs.append(output)
    
    layer = int(layer)
    if layer <= layers[0]:
        model.layer1[layer-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1]:
        model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2]:
        model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
    elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
        model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)

    for j in range(num_concepts,num_concepts+1):
        save_folder = os.path.join(dst, "concept_"+str(j))
        try:
            os.mkdir(save_folder)
        except:
            pass
    
    from matplotlib import cm
    cover_size = 32
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            random_patch = torch.tensor(np.random.normal(size=(3,cover_size,cover_size)))
            input_var = torch.autograd.Variable(input).cuda()
            output = model(input_var)
            base_activations = nn.functional.max_pool2d(outputs[0], kernel_size=3, stride=3)
            base_activations = base_activations[0,:,:,:].clamp(min=0.0).mean(dim=(1,2))
            outputs = []

            input_size = input.size()
            saliency = np.zeros((1, input_size[2], input_size[3]))
            counter = np.zeros((input_size[2], input_size[3])) + 0.00001
            for p in range(0, input_size[2]-cover_size+1, 5):
                print("p={}\n".format(p))
                for q in range(0, input_size[3]-cover_size+1, 5):
                    new_input = input.clone()
                    new_input[0,:,p:p+cover_size, q:q+cover_size] = random_patch
                    input_var = torch.autograd.Variable(new_input).cuda()
                    output = model(input_var)
                    new_activations = nn.functional.max_pool2d(outputs[0], kernel_size=3, stride=3)
                    new_activations = new_activations[0,:,:,:].clamp(min=0.0).mean(dim=(1,2))
                    # new_activations = outputs[0][0,:,:,:].clamp(min=0.0).max(dim=(1,2))
                    outputs = []
                    decrease_in_activations = base_activations - new_activations
                    for j in range(num_concepts,num_concepts+1):
                        saliency[0, p:p+cover_size, q:q+cover_size] += decrease_in_activations[j].cpu().numpy()
                    counter[p:p+cover_size, q:q+cover_size] += 1.0
            
            saliency = saliency/counter
            # print(saliency)
            
            u_limit = np.percentile(saliency, 99.99)
            l_limit = np.percentile(saliency, 0.01)
            saliency = saliency.clip(l_limit, u_limit)
            
            saliency = (saliency - saliency.min())/saliency.max()
            lower_limit = np.percentile(saliency, 94)
            saliency[saliency<lower_limit] = 0.3
            saliency[saliency>=lower_limit] = 1.0

            input_image = input[0,:,:,:].permute(1,2,0).cpu().numpy()
            input_image *= std
            input_image += mean
            
            for j in range(num_concepts,num_concepts+1):
                save_folder = os.path.join(dst, "concept_"+str(j))
                image = Image.fromarray(np.uint8(input_image*255)).convert('RGBA')
                image = np.array(image)
                image[:,:,3] = (saliency[0,:,:]*255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(os.path.join(save_folder, str(i)+'.png'), 'PNG')
                print("saved: " + str(j))


def load_resnet_model(args, arch = 'resnet_original', depth=18, checkpoint_folder="./checkpoints", whitened_layer=None, dataset = 'places365'):
    if dataset == 'places365':
        n_classes = 365
    elif dataset == 'isic':
        n_classes = 2
    prefix_name = args.prefix[:args.prefix.rfind('_')]
    
    model = None
    if arch == 'resnet_original':
        if depth == 50:
            model = ResidualNetBN(n_classes, args, arch = 'resnet50', layers = [3, 4, 6, 3], model_file=os.path.join(checkpoint_folder, 'resnet50_{}.pth.tar'.format(dataset)))
        elif depth == 18:
            model = ResidualNetBN(n_classes, args, arch = 'resnet18', layers = [2, 2, 2, 2], model_file=os.path.join(checkpoint_folder, 'resnet18_{}.pth.tar'.format(dataset)))
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()
    elif arch == 'resnet_cw':
        concept_names = '_'.join(args.concepts.split(','))
        if whitened_layer == None:
            raise Exception("whitened_layer argument is required")
        else:
            if depth == 50:
                model = ResidualNetTransfer(n_classes, args, [int(whitened_layer)], arch = 'resnet50', layers = [3, 4, 6, 3], model_file=os.path.join(checkpoint_folder, 'resnet50_{}.pth.tar'.format(dataset)))
                checkpoint_name = '{}_{}_model_best.pth.tar'.format(prefix_name, whitened_layer)
            elif depth == 18:
                model = ResidualNetTransfer(n_classes, args, [int(whitened_layer)], arch = 'resnet18', layers = [2, 2, 2, 2], model_file=os.path.join(checkpoint_folder, 'resnet18_{}.pth.tar'.format(dataset)))
                # model = ResidualNetTransfer(n_classes, args, [int(whitened_layer)], arch = 'resnet18', layers = [2, 2, 2, 2], model_file=None)
                checkpoint_name = '{}_{}_checkpoint.pth.tar'.format(prefix_name, whitened_layer)
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()
        checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            print(checkpoint['epoch'])
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception("checkpoint {} not found!".format(checkpoint_path))
    elif arch == 'resnet_baseline':
        concept_names = '_'.join(args.concepts.split(','))
        if whitened_layer == None:
            raise Exception("whitened_layer argument is required")
        else:
            if depth == 50:
                model = ResidualNetBN(n_classes, args, arch = 'resnet50', layers = [3, 4, 6, 3], model_file=os.path.join(checkpoint_folder, 'resnet50_{}.pth.tar'.format(dataset)))
                checkpoint_name = 'RESNET50_PLACES365_BASELINE_{}_model_best.pth.tar'.format(whitened_layer)
            elif depth == 18:
                model = ResidualNetBN(n_classes, args, arch = 'resnet18', layers = [2, 2, 2, 2], model_file=os.path.join(checkpoint_folder, 'resnet18_{}.pth.tar'.format(dataset)))
                checkpoint_name = 'RESNET18_PLACES365_BASELINE_{}_checkpoint.pth.tar'.format(whitened_layer)
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()
        checkpoint_path = os.path.join(checkpoint_folder, concept_names, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            print(checkpoint['epoch'])
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception("checkpoint {} not found!".format(checkpoint_path))
    return model
