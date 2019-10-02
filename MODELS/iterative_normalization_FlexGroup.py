"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm

*****
This implementation allows the number of featur maps is not divided by the channel number of per Group. E,g. one can use group size of 64 when the channel number is 80. (64 + 16)

"""
import torch.nn
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['iterative_normalization_FlexGroup', 'IterNorm']


#
# class iterative_normalization(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, *inputs):
#         result = bcnn.iterative_normalization_forward(*inputs)
#         ctx.save_for_backward(*result[:-1])
#         return result[-1]
#
#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         grad, = grad_outputs
#         grad_input = bcnn.iterative_normalization_backward(grad, ctx.saved_variables)
#         return grad_input, None, None, None, None, None, None, None


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # change NxCxHxW to Dx(NxHxW), i.e., d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(nc, -1)
        d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X)
            Sigma = torch.addmm(eps, P[0], 1. / m, xc, xc.transpose(0, 1))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((0, 1), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.addmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.mm(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].mm(P[k - 1])
            g_sn += P2.mm(P[k - 1]).mm(g_P)
            g_tmp = g_P.mm(sn)
            g_P.addmm_(1.5, -0.5, g_tmp, P2)
            g_P.addmm_(1, -0.5, P2, g_tmp)
            g_P.addmm_(1, -0.5, P[k - 1].mm(g_tmp), P[k - 1])
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.mm(g_sn) + g_wm.transpose(-2, -1).mm(wm)) * P[0]).sum((0, 1), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.addmm(wm.mm(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm_Single(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm_Single, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_features, self.T,  self.eps, self.momentum, self.training)
        return X_hat

class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1 
        self.num_groups = num_groups
        self.iterNorm_Groups = torch.nn.ModuleList(
            [IterNorm_Single(num_features = self.num_channels, eps=eps, momentum=momentum, T=T) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.iterNorm_Groups.append(IterNorm_Single(num_features = num_channels_last, eps=eps, momentum=momentum, T=T))
         
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.iterNorm_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = IterNorm(16, num_channels=4, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    #x = torch.randn(32, 64, 14, 14)
    x = torch.randn(32, 16)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
