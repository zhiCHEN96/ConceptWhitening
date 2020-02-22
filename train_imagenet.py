import argparse
import os
import shutil
import time
import random
import numpy as np

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile, Image
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import matplotlib.pyplot as plt
import matplotlib
import skimage.measure
matplotlib.use('Agg')

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--whitened_layers', default='1,2,3')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--concepts', type=str, required=True)
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default='CBAM')
best_prec1 = 0

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

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

def get_param_list(model, whitened_layers):
    param_list = list(model.model.fc.parameters())
    layers = model.layers
    for whitened_layer in whitened_layers:
        if whitened_layer <= layers[0]:
            param_list += list(model.model.layer1[whitened_layer-1].bn1.parameters())
        elif whitened_layer <= layers[0] + layers[1]:
            param_list += list(model.model.layer2[whitened_layer-layers[0]-1].bn1.parameters())
        elif whitened_layer <= layers[0] + layers[1] + layers[2]:
            param_list += list(model.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1.parameters())
        elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            param_list += list(model.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1.parameters())
    return param_list

def get_param_list_bn(model):
    param_list = list(model.parameters())
    new_param_list = []
    for i in range(52):
        if i % 4 == 2 or i % 4 == 3:
            new_param_list.append(param_list[i])
    return new_param_list

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.prefix += '_'+'_'.join(args.whitened_layers.split(','))

    #create model
    if args.arch == "resnet":
        model = ResidualNet( 'ImageNet', args.depth, 9, None, [int(x) for x in args.whitened_layers.split(',')])
    elif args.arch == "resnet_transfer":
        # model = ResidualNetTransfer(9, [int(x) for x in args.whitened_layers.split(',')], model_file ='./checkpoints/RESNET18_PLACES_VANILLA_model_best.pth.tar')
        if args.depth == 50:
            model = ResidualNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'resnet50', layers = [3, 4, 6, 3], model_file='resnet50_places365.pth.tar')
        elif args.depth == 18:
            model = ResidualNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'resnet18', layers = [2, 2, 2, 2], model_file='resnet18_places365.pth.tar')
    elif args.arch == "resnet_original":
        if args.depth == 50:
            model = ResidualNetBN(365, args, arch = 'resnet50', layers = [3, 4, 6, 3], model_file='resnet50_places365.pth.tar')
        if args.depth == 18:
            model = ResidualNetBN(365, args, arch = 'resnet18', layers = [2, 2, 2, 2], model_file='resnet18_places365.pth.tar')
    elif args.arch == "densenet_transfer":
        model = DenseNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'densenet161', model_file='densenet161_places365.pth.tar')
    elif args.arch == 'densenet_original':
        model = DenseNetBN(365, args, arch = 'densenet161', model_file='densenet161_places365.pth.tar')
    elif args.arch == "vgg16_bn_transfer":
        model = VGGBNTransfer(365, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'vgg16_bn', model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')
    elif args.arch == "vgg16_bn_original":
        model = VGGBN(365, args, arch = 'vgg16_bn', model_file = './checkpoints/vgg16_bn_places365_12_model_best.pth.tar') #'vgg16_bn_places365.pt')
    # define loss function (criterion) and optimizer
    print(args.start_epoch, args.best_prec1)
    best_prec1 = 0#args.best_prec1
    criterion = nn.CrossEntropyLoss().cuda()
    # param_list = get_param_list_bn(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(param_list, args.lr,
    #                         momentum=args.momentum,
    #                         weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
    #                         weight_decay=args.weight_decay)
                            
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print ("model")
    print (model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_path = args.resume[:-19] + '_' + '_'.join(args.whitened_layers.split(',')) + args.resume[-19:]
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    conceptdir = os.path.join(args.data, 'concept')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # import pdb
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    concept_loaders = [
        torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(conceptdir, concept), transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
        for concept in args.concepts.split(',')
    ]

    potential_concept_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(conceptdir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    val_loader_2 = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print("Start training")
    for epoch in range(args.start_epoch, args.start_epoch + 20):
    #for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        train(train_loader, concept_loaders, potential_concept_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.prefix)
    print(best_prec1)
    # print_concept_top5(val_loader_2, model, args.whitened_layers, activation_mode = 'pool_max')
    # print_concept_top5(val_loader_2, model, args.whitened_layers, print_other = True)
    # plot_concept_representation(val_loader_2, model, args.whitened_layers, plot_cpt = ['airplane','bed'], activation_mode = 'pool_max_s1')
    # plot_concept_representation(val_loader_2, model, args.whitened_layers, plot_cpt = ['airplane','person'], activation_mode = 'pool_max_s1')
    # plot_concept_representation(val_loader_2, model, args.whitened_layers, plot_cpt = ['bed','person'], activation_mode = 'pool_max_s1')
    # plot_trajectory_all(val_loader_2, args.whitened_layers, plot_cpt = ['airplane','bed'])
    # del model
    # plot_trajectory(val_loader_2, model, args.whitened_layers, plot_cpt = ['airplane','bed'])
    # check_correlation(val_loader_2, model, 8)
    # plot_auc_all('/usr/xtmp/zhichen/image_data/object/val/', '1,2,3,4,5,6,7,8', plot_cpt = ['airplane','bed','person'], activation_mode = 'pool_max')
    # plot_auc_all('/usr/xtmp/zhichen/image_data/object/val/', '1,8', plot_cpt = ['airplane','bed','person'], activation_mode = 'pool_max_s1')
    # plot_auc_lm(model, concept_loaders, train_loader, '/usr/xtmp/zhichen/image_data/object/val/', '1,2,3,4,5,6,7,8', plot_cpt = ['airplane', 'bed', 'person'])
    # plot_auc_filter(model, '/usr/xtmp/zhichen/image_data/object/val/', '1,2,3,4,5,6,7,8', plot_cpt = ['airplane', 'bed', 'person'], activation_mode = 'mean')
    # plot_auc()
    # plot_top10()
    # plot_top10(layer = 8)

def get_optimal_direction(concept_loader, model, whitened_layers):
    n = 0
    layer_list = whitened_layers.split(',')
    model = model.module
    layers = model.layers
    outputs = []
    def hook(module, input, output):
        from MODELS.iterative_normalization import iterative_normalization_py
        #print(input)
        X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                 module.eps, module.momentum, module.training)
        #print(X_hat.size())
        outputs.append(X_hat.mean((0,2,3)))
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

    with torch.no_grad():
        model.eval()
        for X, _ in concept_loader:
            X_var = torch.autograd.Variable(X).cuda()
            model(X_var)

        mean = torch.zeros((128,)).cuda()
        for item in outputs:
            mean += item
        mean /= len(outputs)

    return mean


def train(train_loader, concept_loaders, potential_concept_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if (i + 1) % 811 == 0:
            break
        if (i + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                for t in range(1):
                    for concept_index, concept_loader in enumerate(concept_loaders):
                        model.module.change_mode(concept_index)
                        for j, (X, _) in enumerate(concept_loader):
                            X_var = torch.autograd.Variable(X).cuda()
                            model(X_var)
                            break
                    # Load 5 batches
                    #potential_concept_batch_counter = 0
                    #model.module.change_mode(len(concept_loaders))
                    #for (potential_concept_image_batch, _) in potential_concept_loader:
                    #    potential_concept_batch_counter += 1
                    #    X_var = torch.autograd.Variable(potential_concept_image_batch).cuda()
                    #    model(X_var)
                    #    if potential_concept_batch_counter >= 10:
                    #        break
                    model.module.update_rotation_matrix()
                model.module.change_mode(-1)
            model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
  

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            
            # pred = output.argmax(1).cpu().numpy()
            # if i>=2 and i<4:
            #     x = input.cpu().numpy()
            #     y = target.cpu().numpy()
            #     for j, xx in enumerate(x):
            #         img = Image.fromarray(xx[0]*255).convert('L')
            #         img.save('/usr/xtmp/zhichen/attention-module/plot/valval'+str(i)+str(j)+'_'+str(y[j])+'_'+str(pred[j])+'.png')

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

def print_concept_top5(val_loader, model, whitened_layers, print_other = False, activation_mode = 'mean'):
    # switch to evaluate mode
    model.eval()
    from shutil import copyfile
    dst = '/usr/xtmp/zhichen/attention-module/plot3/' + args.arch + str(args.depth) + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    layer_list = whitened_layers.split(',')
    folder = dst + '_'.join(layer_list) + '_rot/'
    # print(folder)
    if print_other:
        folder = dst + '_'.join(layer_list) + '_rot_otherdim/'
    if args.arch == "resnet_transfer":
        folder = dst + '_'.join(layer_list) + '_rot_transfer/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    model = model.module
    layers = model.layers
    if args.arch == "resnet_transfer":
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
        begin = len(args.concepts.split(','))
        end = begin+30
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
                for j in range(50):
                    src = arr[j][1]
                    # print(src)
                    # print(folder+'layer'+layer+'_'+str(j+1)+'.jpg')
                    copyfile(src, output_path+'/'+'layer'+layer+'_'+str(j+1)+'.jpg')  

    return 0

def get_layer_representation(val_loader, layer, cpt_idx):
    model = ResidualNetTransfer(365, args, [int(layer)], arch = 'resnet18', layers = [2, 2, 2, 2], model_file='resnet18_places365.pth.tar')
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model.cuda()
    if args.resume:
        checkpoint_path = args.resume[:-19] + '_' + layer + args.resume[-19:]
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    
    with torch.no_grad():        
        model.eval()
        model = model.module
        layers = model.layers
        if args.arch == "resnet_transfer":
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

def plot_trajectory_all(val_loader, whitened_layers, plot_cpt = ['airplane','bed']):
    dst = '/usr/xtmp/zhichen/attention-module/plot2/' + args.arch + str(args.depth) + '/trajectory_all/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    concepts = args.concepts.split(',')
    cpt_idx = [concepts.index(plot_cpt[0]),concepts.index(plot_cpt[1])]
    vals = None 
    layer_list = whitened_layers.split(',')
    for i, layer in enumerate(layer_list):
        print(i)
        if i == 0:
            paths, vals = get_layer_representation(val_loader, layer, cpt_idx)
        else:
            _, val = get_layer_representation(val_loader, layer, cpt_idx)
            vals = np.concatenate((vals,val),0)
        print(vals.shape)
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
        print(k)
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
        I = Image.open(paths[i]).resize((100,100),Image.ANTIALIAS)
        plt.imshow(np.asarray(I).astype(np.int32))
        plt.savefig('{}{}/{}.jpg'.format(dst,'_'.join(plot_cpt), k))

def plot_auc_all(conceptdir, whitened_layers, plot_cpt = ['airplane','bed','person'], activation_mode = 'mean'):
    dst = '/usr/xtmp/zhichen/attention-module/plot2/' + args.arch + str(args.depth) + '/auc/cw/'
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
        num_workers=args.workers, pin_memory=True)

    layer_list = whitened_layers.split(',')
    concept_list = os.listdir(conceptdir)
    concept_list.sort()
    print(concept_list)
    aucs = np.zeros((len(plot_cpt),len(layer_list)))
    print(aucs.shape)
    for c, cpt in enumerate(plot_cpt):
        print(cpt)
        cpt_idx_2 = concept_list.index(cpt)
        cpt_idx = plot_cpt.index(cpt)
        print(cpt_idx, cpt_idx_2)
        for i, layer in enumerate(layer_list):
            model = ResidualNetTransfer(365, args, [int(layer)], arch = 'resnet18', layers = [2, 2, 2, 2], model_file='resnet18_places365.pth.tar')
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
            model.cuda()
            if args.resume :
                checkpoint_path = args.resume[:-19] + '_' + layer + args.resume[-19:]
                if os.path.isfile(checkpoint_path):
                    print("=> loading checkpoint '{}'".format(checkpoint_path))
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                        .format(checkpoint_path, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(checkpoint_path))
            
            with torch.no_grad():        
                model.eval()
                model = model.module
                layers = model.layers
                if args.arch == "resnet_transfer":
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
            aucs[c,i] = roc_auc_score(labels, vals)
            print(aucs[c,i])
        
        #fig = plt.figure(figsize=(5,5))
        #plt.plot([2,4,6,8,10,12,14,16], aucs[c])
        #plt.xlabel('layer')
        #plt.ylabel('auc')
        #plt.savefig('{}/{}.jpg'.format(dst,cpt))
    
    print(aucs)
    np.save(dst + 'aucs_cw.npy', aucs)
    return aucs

def plot_auc_lm(model, concept_loaders, train_loader, conceptdir, whitened_layers, plot_cpt = ['airplane', 'bed', 'person'], model_type = 'svm'):
    dst = '/usr/xtmp/zhichen/attention-module/plot2/' + 'resnet_transfer' + str(args.depth) + '/auc/tcav/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    layer_list = whitened_layers.split(',')
    aucs = np.zeros((len(plot_cpt),len(layer_list)))

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
        num_workers=args.workers, pin_memory=True)
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

            y_train = np.ones(activation[0].shape[0]*2)
            y_train[activation[0].shape[0]:] = 0

            for i, (input, _) in enumerate(train_loader):
                if i == n_batch:
                    break
                outputs = []
                input_var = torch.autograd.Variable(input).cuda()
                model(input_var)
                for k in range(len(outputs)):
                    activation[k] = np.concatenate((activation[k], outputs[k]),0)
            
            for i in range(len(layer_list)):
                x_train = activation[i].reshape((len(y_train),-1))
                y_train = y_train

                if model_type == 'svm':
                    lm = SGDClassifier()
                elif model_type == 'lr':
                    lm = LogisticRegression()
                lm.fit(x_train, y_train)

                x_test = activation_test[i].reshape((len(labels),-1))
                y_test = (labels == cpt_idx_2).astype('int32')
                cav = lm.coef_
                score = (x_test*cav).sum(1)
                aucs[c,i] = roc_auc_score(y_test, score)
                print(aucs[c,i])
    
    np.save(dst + 'aucs_' + model_type + '.npy', aucs)

    return aucs

def plot_auc_filter(model, conceptdir, whitened_layers, plot_cpt = ['airplane', 'bed', 'person'], activation_mode = 'mean'):
    dst = '/usr/xtmp/zhichen/attention-module/plot2/' + 'resnet_transfer' + str(args.depth) + '/auc/filter/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    layer_list = whitened_layers.split(',')
    aucs = np.zeros((len(plot_cpt),len(layer_list)))

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
        num_workers=args.workers, pin_memory=True)
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
                for j in range(x_test.shape[1]):
                    score = x_test[:,j]
                    aucs[c,i] = max(roc_auc_score(y_test, score),aucs[c,i])
                print(aucs[c,i])
    print(aucs)
    #np.save(dst + 'aucs_filter.npy', aucs)
    return 

def plot_auc(plot_cpt = ['airplane', 'bed', 'person']):
    folder = '/usr/xtmp/zhichen/attention-module/plot2/' + 'resnet_transfer' + str(args.depth) + '/auc/'
    aucs_cw = np.load(folder + 'cw/' + 'aucs_cw.npy')
    aucs_svm = np.load(folder + 'tcav/' + 'aucs_svm.npy')
    aucs_lr = np.load(folder + 'tcav/' + 'aucs_lr.npy')
    aucs_filter = np.load(folder + 'filter/' + 'aucs_filter.npy')

    for c, cpt in enumerate(plot_cpt):
        fig = plt.figure(figsize=(5,5))
        plt.plot([2,4,6,8,10,12,14,16], aucs_cw[c], label = 'CW')
        plt.plot([2,4,6,8,10,12,14,16], aucs_svm[c], label = 'SVM (CAV)', )
        plt.plot([2,4,6,8,10,12,14,16], aucs_lr[c], label = 'LR (IBD,CAV)')
        plt.plot([2,4,6,8,10,12,14,16], aucs_filter[c], label = 'Best filter')
        plt.xlabel('layer')
        plt.ylabel('auc')
        plt.legend()
        plt.savefig('{}/{}.jpg'.format(folder,cpt))    

def plot_top10(plot_cpt = ['airplane', 'bed', 'person'], layer = 1):
    folder = '/usr/xtmp/zhichen/attention-module/plot3/' + 'resnet_transfer' + str(args.depth) + '/' + str(layer) + '_rot_transfer/'

    fig, axes = plt.subplots(figsize=(30, 3*len(plot_cpt)) , nrows=len(plot_cpt), ncols=10)

    import matplotlib.image as mpimg
    for c, cpt in enumerate(plot_cpt):
        for i in range(10):
            axes[c,i].imshow(mpimg.imread(folder + cpt + '/layer' + str(layer) + '_' +str(i+1)+'.jpg'))
            #axes[c,i].axis('off')
            axes[c,i].set_yticks([])
            axes[c,i].set_xticks([])
            # axes[c,i].set_frame_on(False)

    for ax, row in zip(axes[:,0], plot_cpt):
        ax.set_ylabel(row, rotation=90, size='large', fontsize = 50)

    fig.tight_layout()
    plt.show()
    fig.savefig(folder+'layer'+str(layer)+'.jpg')


def plot_trajectory(val_loader, model, whitened_layers, plot_cpt = ['airplane','bed']):
    dst = '/usr/xtmp/zhichen/attention-module/plot2/' + args.arch + str(args.depth) + '/trajectory/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    concepts = args.concepts.split(',')
    cpt_idx = [concepts.index(plot_cpt[0]),concepts.index(plot_cpt[1])]
    with torch.no_grad():        
        model.eval()
        model = model.module
        layers = model.layers
        layer_list = whitened_layers.split(',')
        if args.arch == "resnet_transfer":
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


            paths = []
            vals = None
            for i, (input, _, path) in enumerate(val_loader):
                paths += list(path)
                input_var = torch.autograd.Variable(input).cuda()
                outputs = []
                model(input_var)
                val = []
                for output in outputs:
                    val.append(output.sum((2,3))[:,cpt_idx])
                val = np.array(val)
                if i == 0:
                    vals = val
                else:
                    vals = np.concatenate((vals,val),1)
        
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
        print(k)
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
        I = Image.open(paths[i]).resize((100,100),Image.ANTIALIAS)
        plt.imshow(np.asarray(I).astype(np.int32))
        plt.savefig('{}{}/{}.jpg'.format(dst,'_'.join(plot_cpt), k))

def plot_concept_representation(val_loader, model, whitened_layers, plot_cpt = ['airplane','bed'], activation_mode = 'mean'):    
    with torch.no_grad():
        dst = '/usr/xtmp/zhichen/attention-module/plot3/' + args.arch + str(args.depth) +'/representation/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        model.eval()
        model = model.module
        layers = model.layers
        layer_list = whitened_layers.split(',')
        dst = dst + '_'.join(layer_list) + '/'
        if args.arch == "resnet_transfer":
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
            plt.savefig(dst+'layer'+layer+'_'+'_'.join(plot_cpt)+'.jpg',dpi=img_size*n_grid)
            plt.figure()
            ax = sns.heatmap(cnt/cnt.sum(), linewidth=0.5)
            plt.xlabel(plot_cpt[1])
            plt.ylabel(plot_cpt[0])
            plt.savefig(dst+'density_layer'+layer+'_'+'_'.join(plot_cpt)+'.jpg')
    
    return 0

def check_correlation(val_loader, model, layer):
    with torch.no_grad():
        dst = '/usr/xtmp/zhichen/attention-module/plot2/' + args.arch + str(args.depth) +'/correlation_matrix/'
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
            #print(np.dot(M.transpose((1,0)),M)[:10,:10])
            #from scipy.linalg import svd
            #Sigma = np.dot(M.transpose((1,0)),M) / (64*49)
            #print(Sigma)
            #U,s,_ = svd(Sigma)
            #print(M.shape)
            #M = U.dot(np.diag(1/np.sqrt(s))).dot(U.transpose(1,0)).dot(M.transpose(1,0)).transpose(1,0)
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
        activation = np.array(outputs).reshape((-1,np.shape(outputs)[2]))
        activation -= activation.mean(0)
        activation = activation / activation.std(0)
        Sigma = np.dot(activation.transpose((1,0)),activation) / activation.shape[0]
        #print(np.sum(Sigma<0))
        #print(Sigma)
        sns.heatmap(np.abs(Sigma),cmap='hot')
        plt.tight_layout()
        plt.savefig(dst + str(layer) + '.jpg')
        #from scipy.linalg import svdvals
        #s = svdvals(activation)
        #print(s)


def save_checkpoint(state, is_best, prefix):
    #pass
    filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
    torch.save(state, filename)
    if is_best:
         shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
