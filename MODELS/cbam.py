import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelGate2(nn.Module):
    """ Channel attention module"""
    def __init__(self, gate_channels):
        super(ChannelGate2, self).__init__()
        self.gate_channels = gate_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def get_attention(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        return attention

    def forward(self,x):

        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        attention = self.get_attention(x)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        
        out = out * self.gamma + x

        return out

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate2(gate_channels)#, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class BatchWhitening(nn.Module):
    def __init__(self, gate_channels, eps=1e-7, alpha = 0.9):
        super(BatchWhitening, self).__init__()
        self.eps = torch.eye(gate_channels).cuda() * eps
        self.alpha = alpha
        self.gate_channels = gate_channels
        self.mu_E = torch.zeros(gate_channels).cuda()
        self.Sigma_invsqrt_E = torch.eye(gate_channels).cuda()
        self.Beta = nn.Parameter(torch.eye(gate_channels))
        self.gamma = nn.Parameter(torch.zeros(gate_channels))
    
    def forward(self, x):

        B, C, H, W = x.size()
        if self.training == True:

            mu = x.mean((0,2,3))
            x -= mu.reshape(1,C,1,1)
            x = x.permute(1,0,2,3).reshape(C,-1)
            Sigma = x.mm(x.t()) / (B*H*W) + self.eps
            
            [U,S,V] = Sigma.svd()
            Sigma_invsqrt = U.mm(S.rsqrt().diag().mm(U.t()))
            
            x = Sigma_invsqrt.mm(x)
            x = self.Beta.mm(x) + self.gamma.reshape(C,1)

            x = x.reshape(C,B,H,W).permute(1,0,2,3)

            self.mu_E = self.mu_E * self.alpha + mu * (1.0 - self.alpha)
            self.Sigma_invsqrt_E = self.Sigma_invsqrt_E * self.alpha + Sigma_invsqrt * (1.0 - self.alpha)

            return x

        else:
            
            x -= self.mu_E.reshape(1,C,1,1)
            x = x.permute(1,0,2,3).reshape(C,-1)
            x = self.Sigma_invsqrt_E.mm(x)
            x = x.reshape(C,B,H,W).permute(1,0,2,3)
            # x = self.Beta.mm(x) + self.gamma.reshape(C,1)

            return x
            