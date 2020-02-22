import sys
import torch
import numbers
from torch.nn.parameter import Parameter
from time import sleep
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.functional import normalize
import os
import numpy as np
import math

class power_iteration_unstable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        vk_list = []
        vk_list.append(v_k)
        ctx.num_iter = num_iter
        for _ in range(int(ctx.num_iter)):
            v_k = M.mm(v_k)
            v_k /= torch.norm(v_k).clamp(min=1.e-5)
            vk_list.append(v_k)

        ctx.save_for_backward(M, *vk_list)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M = ctx.saved_tensors[0]
        vk_list = ctx.saved_tensors[1:]
        dL_dvk1 = grad_output
        dL_dM = 0
        for i in range(1, ctx.num_iter + 1):
            v_k1 = vk_list[-i]
            v_k = vk_list[-i - 1]
            mid = calc_mid(M, v_k, v_k1, dL_dvk1)
            dL_dM += mid.mm(torch.t(v_k))
            dL_dvk1 = M.mm(mid)
        return dL_dM, dL_dvk1


def calc_mid(M, v_k, v_k1, dL_dvk1):
    I = torch.eye(M.shape[-1], out=torch.empty_like(M))
    mid = (I - v_k1.mm(torch.t(v_k1)))/torch.norm(M.mm(v_k)).clamp(min=1.e-5)
    mid = mid.mm(dL_dvk1)
    return mid


class power_iteration_once(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = torch.eye(M.shape[-1], out=torch.empty_like(M))
        numerator = I - v_k.mm(torch.t(v_k))
        denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak

class ZCANormSVDPI(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormSVDPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            #print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            #print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        #print('eigenvalues: ', e)
                        #print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormSVDPIRotation(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.05, affine=False, mode = -1, num_selected_concept = 3, activation_mode = 'mean' ):
        super(ZCANormSVDPIRotation, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.mode = mode
        self.k = num_selected_concept
        self.activation_mode = activation_mode

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

        # running rotation matrix
        self.register_buffer('running_rot', torch.eye(num_features).expand(groups, num_features, num_features))
        # sum Gradient, need to take average later
        self.register_buffer('sum_G', torch.zeros(groups, num_features, num_features))
        # counter, number of gradient for each concept
        self.register_buffer("counter", torch.ones(num_features)*0.001)

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            #print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            #print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        #print('eigenvalues: ', e)
                        #print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            X_hat = torch.cat(xgr_list, dim=0)

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            X_hat = torch.cat(xg, dim=0)

        # nchw
        X_hat = X_hat.view(C, N, H, W).transpose(0, 1)
        size_X = X_hat.size()
        size_R = self.running_rot.size()
        # ngchw
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        with torch.no_grad():
            if self.mode>=0 and self.mode<self.k:
                if self.activation_mode=='mean':
                    self.sum_G[:,self.mode,:] = self.momentum * -X_hat.mean((0,3,4)) + (1. - self.momentum) * self.sum_G[:,self.mode,:]
                    self.counter[self.mode] += 1
                elif self.activation_mode=='max':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    max_values = torch.max(torch.max(X_test, 3, keepdim=True)[0], 4, keepdim=True)[0]
                    max_bool = max_values==X_test
                    grad = (X_hat * max_bool.to(X_hat)).sum((0,3,4))/max_bool.to(X_hat).sum((0,3,4))
                    self.sum_G[:,self.mode,:] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:,self.mode,:]

            elif self.mode>=0 and self.mode>=self.k:
                X_dot = torch.einsum('ngchw,gdc->ngdhw', X_hat, self.running_rot)
                X_dot = (X_dot == torch.max(X_dot, dim=2,keepdim=True)[0]).float().cuda()
                X_dot_unity = torch.clamp(torch.ceil(X_dot), 0.0, 1.0)
                X_G = torch.einsum('ngchw,ngdhw->gdchw', X_hat, X_dot_unity).mean((3,4))
                X_G[:,:self.k,:] = 0.0
                self.sum_G[:,:,:] += -X_G/size_X[0]
                self.counter[self.k:] += 1


        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
        X_hat = X_hat.view(*size_X)
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def update_rotation_matrix(self):
        size_R = self.running_rot.size()
        with torch.no_grad():
            G = self.sum_G/self.counter.reshape(-1,1)
            R = self.running_rot.clone()
            for i in range(2):
                tau = 1000
                alpha = 0
                beta = 100000000
                c1 = 1e-4
                c2 = 0.9
                
                A = torch.einsum('gin,gjn->gij', G, R) - torch.einsum('gin,gjn->gij', R, G) # GR^T - RG^T
                I = torch.eye(size_R[2]).expand(*size_R).cuda()
                dF_0 = -0.5 * (A ** 2).sum()
                while True:
                    Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                    Y_tau = torch.bmm(Q, R)
                    F_X = (G[:,:,:] * R[:,:,:]).sum()
                    F_Y_tau = (G[:,:,:] * Y_tau[:,:,:]).sum()
                    dF_tau = -torch.bmm(torch.einsum('gni,gnj->gij', G, (I + 0.5 * tau * A).inverse()), torch.bmm(A,0.5*(R+Y_tau)))[0,:,:].trace()
                    #print(F_Y_tau - F_X - c1*tau*dF_0)
                    #print(c2*dF_0 - dF_tau)
                    if F_Y_tau > F_X + c1*tau*dF_0 + 1e-18:
                        beta = tau
                        tau = (beta+alpha)/2
                    elif dF_tau  + 1e-18 < c2*dF_0:
                        alpha = tau
                        tau = (beta+alpha)/2
                    else:
                        break
                print(tau, F_Y_tau)
                Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                R = torch.bmm(Q, R)
            
            self.running_rot = R
            self.counter = (torch.ones(size_R[-1]) * 0.001).cuda()

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
