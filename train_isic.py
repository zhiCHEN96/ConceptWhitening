import argparse
import os
import sys
import gc
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from plot_functions import *
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--whitened_layers', default='8')
parser.add_argument('--act_mode', default='pool_max')
parser.add_argument('--depth', default=18, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--concepts', type=str, default=None)
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', type=str, default=None, help='type of evaluation')
best_prec1 = 0

os.chdir(sys.path[0])

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def main():
    global args, best_prec1
    args = parser.parse_args()

    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.arch == "resnet_cw" or args.arch == "densenet_cw" or args.arch == "vgg16_cw":
        args.prefix += '_'+'_'.join(args.whitened_layers.split(','))

    #create model
    if args.arch == "resnet_cw":
        if args.depth == 50:
            model = ResidualNetTransfer(2, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'resnet50', layers = [3, 4, 6, 3], model_file='./checkpoints/resnet50_isic.pth.tar')
        elif args.depth == 18:
            model = ResidualNetTransfer(2, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'resnet18', layers = [2, 2, 2, 2], model_file='./checkpoints/resnet18_isic_model_best.pth.tar')
            print(args.start_epoch)
    elif args.arch == "resnet_original" or args.arch == "resnet_baseline":
        if args.depth == 50:
            model = ResidualNetBN(2, args, arch = 'resnet50', layers = [3, 4, 6, 3], model_file='./checkpoints/resnet50_isic.pth.tar')
            # model = models.resnet50(num_classes = 2)
        if args.depth == 18:
            model = models.resnet18(num_classes = 2)
            # model = ResidualNetBN(2, args, arch = 'resnet18', layers = [2, 2, 2, 2], model_file='./checkpoints/resnet18_isic.pth.tar')
    elif args.arch == "densenet_cw":
        model = DenseNetTransfer(2, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'densenet161', model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == 'densenet_original':
        model = DenseNetBN(2, args, arch = 'densenet161', model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == "vgg16_cw":
        model = VGGBNTransfer(2, args, [int(x) for x in args.whitened_layers.split(',')], arch = 'vgg16_bn', model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')
    elif args.arch == "vgg16_bn_original":
        model = VGGBN(2, args, arch = 'vgg16_bn', model_file = './checkpoints/vgg16_bn_places365_12_model_best.pth.tar') #'vgg16_bn_places365.pt')
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # param_list = get_param_list_bn(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)
                            
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()
    print("model")
    print(model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    conceptdir_train = os.path.join(args.data, 'concept_train')
    conceptdir_test = os.path.join(args.data, 'concept_test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_loader = balanced_data_loader(args, traindir)
    test_loader_2 = balanced_data_loader(args, testdir, 'test')

    if args.arch == "resnet_cw" or args.arch == "densenet_cw" or args.arch == "vgg16_cw":
        concept_loaders = [
                torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
                    transforms.Scale(256),
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
                for concept in args.concepts.split(',')
            ]
    else:
        concept_loaders = [None]
    
    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
    args.test_weights = get_class_weights(test_dataset.imgs, len(test_dataset.classes))
    print('Class weights:',args.test_weights)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    test_loader_with_path = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    if args.evaluate is None:
        print("Start training")
        best_prec1 = 0
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            adjust_learning_rate(optimizer, epoch)
            
            # training
            if args.arch == "resnet_cw" or args.arch == "resnet_original":
                train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            elif args.arch == "resnet_baseline":
                train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = validate(test_loader, model, criterion, epoch)
            
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
        validate(test_loader, model, criterion, epoch)
    else:
        plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir_test)
        
def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # update the CW parameters, not used when training standard network
        if args.arch == "resnet_cw":
            if (i + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    # update the gradient matrix G
                    for concept_index, concept_loader in enumerate(concept_loaders):
                        # print(concept_index)
                        model.module.change_mode(concept_index)
                        for j, (X, _) in enumerate(concept_loader):
                            X_var = torch.autograd.Variable(X).cuda()
                            model(X_var)
                            break
                    model.module.update_rotation_matrix()
                    # change to ordinary mode
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
        [prec1] = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1.item(), input.size(0))
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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
  

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    baccs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            [prec1] = accuracy(output.data, target, topk=(1,))
            [bacc] = weighted_accuracy(output.data, target)
            losses.update(loss.data, input.size(0))
            top1.update(prec1.item(), input.size(0))
            baccs.update(bacc.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Balance-acc {baccs.val:.3f} ({baccs.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, baccs=baccs))
        
    print(' * Prec@1 {top1.avg:.3f}\t'
        'Balance-acc {baccs.avg:.3f}'.format(top1=top1, baccs=baccs))
    return baccs.avg


'''
This function train a baseline with auxiliary concept loss jointly
train with main objective
'''
def train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch, activation_mode = 'pool_max'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_aux = AverageMeter()
    top1_cpt = AverageMeter()

    n_cpt = len(concept_loaders)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # switch to train mode
    model.train()
    
    end = time.time()

    inter_feature = []
    def hookf(module, input, output):
        inter_feature.append(output[:,:n_cpt,:,:])
    for i, (input, target) in enumerate(train_loader):
        if (i + 1) % 10 == 0:

            #model.eval()
            
            layer = int(args.whitened_layers)
            layers = model.module.layers
            if layer <= layers[0]:
                hook = model.module.model.layer1[layer-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1]:
                hook = model.module.model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2]:
                hook = model.module.model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                hook = model.module.model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hookf)
            
            y = []
            inter_feature = []
            for concept_index, concept_loader in enumerate(concept_loaders):
                for j, (X, _) in enumerate(concept_loader):
                    y += [concept_index] * X.size(0)
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break
            
            inter_feature = torch.cat(inter_feature,0)
            y_var = torch.Tensor(y).long().cuda()
            f_size = inter_feature.size()
            if activation_mode == 'mean':
                y_pred = F.avg_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'max':
                y_pred = F.max_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'pos_mean':
                y_pred = F.avg_pool2d(F.relu(inter_feature),f_size[2:]).squeeze()
            elif activation_mode == 'pool_max':
                kernel_size = 3
                y_pred = F.max_pool2d(inter_feature, kernel_size)
                y_pred = F.avg_pool2d(y_pred,y_pred.size()[2:]).squeeze()
            
            loss_cpt = criterion(y_pred, y_var)
            # measure accuracy and record loss
            [prec1_cpt] = accuracy(y_pred.data, y_var, topk=(1,))
            loss_aux.update(loss_cpt.data, f_size[0])
            top1_cpt.update(prec1_cpt[0], f_size[0])
            
            optimizer.zero_grad()
            loss_cpt.backward()
            optimizer.step()

            hook.remove()
            #model.train()
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
                  'Loss_aux {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Prec_cpt@1 {top1_cpt.val:.3f} ({top1_cpt.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_a=loss_aux, top1=top1, top5=top5, top1_cpt=top1_cpt))

def plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir):
    concept_name = args.concepts.split(',')

    if not os.path.exists('./plot/'+'_'.join(concept_name)):
        os.mkdir('./plot/'+'_'.join(concept_name))
    
    if args.evaluate == 'plot_top50':
        print("Plot top50 activated images")
        model = load_resnet_model(args, arch = 'resnet_cw', depth=18, whitened_layer='8', dataset = 'isic')
        plot_concept_top50(args, test_loader_with_path, model, '8', activation_mode = args.act_mode)
        print("End plotting")
    elif args.evaluate == 'plot_auc':
        print("Plot AUC-concept_purity")
        print("Note: this requires multiple models trained with CW on different layers")
        aucs_cw = plot_auc_cw(args, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, activation_mode = args.act_mode, dataset = 'isic')
        print("Running AUCs svm")
        model = load_resnet_model(args, arch='resnet_original', depth=18, dataset = 'isic')
        aucs_svm = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, model_type = 'svm')
        print("Running AUCs lr")
        model = load_resnet_model(args, arch='resnet_original', depth=18, dataset = 'isic')
        aucs_lr = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, model_type = 'lr')
        print("Running AUCs best filter")
        model = load_resnet_model(args, arch='resnet_original', depth=18, dataset = 'isic')
        aucs_filter = plot_auc_filter(args, model, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name)
        print("AUC plotting")
        plot_auc(args, 0, 0, 0, 0, plot_cpt = concept_name)
        print("End plotting")

def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    if args.arch == "resnet_cw" or args.arch == "densenet_cw" or args.arch == "vgg16_cw":
        # save checkpoints for model with CW layer
        concept_name = '_'.join(args.concepts.split(','))
        if not os.path.exists(os.path.join(checkpoint_folder,concept_name)):
            os.mkdir(os.path.join(checkpoint_folder,concept_name))
        filename = os.path.join(checkpoint_folder,concept_name,'%s_checkpoint.pth.tar'%prefix)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_folder,concept_name,'%s_model_best.pth.tar'%prefix))
    elif args.arch == "resnet_original" or args.arch == "densenet_original" or args.arch == "vgg16_original":
        # save checkpoints for model without CW layer
        filename = os.path.join(checkpoint_folder,'%s_checkpoint.pth.tar'%prefix)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_folder,'%s_model_best.pth.tar'%prefix))
    

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
    lr = args.lr * (0.1 ** (epoch // 10))
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

def weighted_accuracy(output, target):
    """Computes the weighted accuracy"""
    weights = args.test_weights
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).long()
    weighted_correct = ((correct * target).float()*weights[1] \
        + (correct*(1 - target)).float()*weights[0]).sum()

    res = []
    res.append(weighted_correct.mul_(100.0 / batch_size))

    return res

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def get_class_weights(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/(2*float(count[i]))
    
    return weight_per_class

def balanced_data_loader(args, dataset_dir, loader_type = 'train'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0.7044832 0.5509753 0.5327961],
    #                              std=[0.09999301 0.12828484 0.14426188])
    if loader_type == 'train':
        dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([
                # transforms.Scale(256),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    elif loader_type == 'test':
        dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]))

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle = False,
        sampler = sampler, num_workers=args.workers, pin_memory=False)

    return loader

if __name__ == '__main__':
    main()
