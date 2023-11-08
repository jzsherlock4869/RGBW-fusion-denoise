from copy import deepcopy

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights, make_layer
from basicsr.archs.rrdbnet_arch import RRDB
from basicsr.utils.registry import ARCH_REGISTRY

import sys
sys.path.append("..")

from utils.utils_bayer import Bayerize_Layer


class ResBlock(nn.Module):
    """Residual block with/without BN.

    It has a style of:
        ---Conv-(BN)-ReLU-Conv-(BN)-+-(ReLU)
         |__________________________|

    """

    def __init__(self, num_feat=64, res_scale=1, with_bn=True, last_relu=False, act_type='lrelu'):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.with_bn = with_bn
        self.last_relu = last_relu
        
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(num_feat)
            self.bn2 = nn.BatchNorm2d(num_feat)

    def forward(self, x):
        identity = x
        if self.with_bn:
            x = self.bn1(self.conv1(x))
            x = self.act(x)
            out = self.bn2(self.conv2(x))
        else:
            out = self.conv2(self.act(self.conv1(x)))
        merged_out = identity + out * self.res_scale
        if self.last_relu:
            return self.act(merged_out)
        else:
            return merged_out


def select_block_type(block_type):
    if block_type == 'resblock':
        return ResBlock
    elif block_type == 'rrdb':
        return RRDB
    else:
        raise NotImplementedError


@ARCH_REGISTRY.register()
class RGBWDualstreamV2Arch(nn.Module):
    """RGBW dual-stream architecture.

    Args:
        num_feat_b (int): Channel number of intermediate features net_b. Default: 64.
        num_feat_c (int): Channel number of intermediate features net_c. Default: 64.
        num_feat_dn (int): Channel number of intermediate features net_dn. Default: 64.
    """

    def __init__(self,
                nf_b=64, nf_c=64, nf_dn=64,
                nb_b=3, nb_c=3, nb_dn=16,
                block_b='resblock', block_c='resblock', block_dn='rrdb',
                net_b_args={}, net_c_args={}, net_dn_args={}
                ):
        super(RGBWDualstreamV2Arch, self).__init__()
        self.pus = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_b_in = nn.Conv2d(4, nf_b, 3, 1, 1, bias=True)
        self.conv_c_in = nn.Conv2d(1, nf_b, 3, 1, 1, bias=True)

        # construct net_b, net_c
        net_b_block = select_block_type(block_b)
        net_b_args_cp = deepcopy(net_b_args)
        net_b_args_cp.update({"num_feat": nf_b})
        self.net_b = make_layer(net_b_block, nb_b, **net_b_args_cp)

        net_c_block = select_block_type(block_c)
        net_c_args_cp = deepcopy(net_c_args)
        net_c_args_cp.update({"num_feat": nf_c})
        self.net_c = make_layer(net_c_block, nb_c, **net_c_args_cp)
        self.avgpool = nn.AvgPool2d(2)

        self.conv_fusion = nn.Conv2d(nf_b + nf_c, nf_dn, 3, 1, 1)

        # construct net_dn for merging binB and binC features
        net_dn_block = select_block_type(block_dn)
        net_dn_args_cp = deepcopy(net_dn_args)
        net_dn_args_cp.update({"num_feat": nf_dn})
        self.net_dn = make_layer(net_dn_block, nb_dn, **net_dn_args_cp)
        # self.conv_skip = nn.Conv2d(nf_b, nf_dn, 1, 1, 0)
        self.conv_out = nn.Sequential(
            nn.Conv2d(nf_dn, nf_dn, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf_dn, nf_dn * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(nf_dn, 4, 3, 1, 1)
        )

        # self.bayerize = Bayerize_Layer(device='cpu')  # uncomment this line for test arch
        self.bayerize = Bayerize_Layer()
        # end version 2 add

    def forward(self, x_b, x_c):
        x_b_pus = self.pus(x_b)
        feat_b_ = self.conv_b_in(x_b_pus)
        feat_b = self.net_b(feat_b_)
        feat_c = self.conv_c_in(x_c)
        feat_c = self.avgpool(self.net_c(feat_c))
        feat_fusion = self.conv_fusion(torch.cat((feat_b, feat_c), dim=1))
        feat_restore = self.net_dn(feat_fusion)
        # version 2 add
        # long_skip = self.conv_skip(feat_b_)
        # res = self.conv_out(feat_restore + long_skip)
        res = self.conv_out(feat_restore)
        output = self.bayerize(res) + x_b
        # end version 2 add
        return output


if __name__ == "__main__":

    from thop import profile

    model = RGBWDualstreamV2Arch()
    dummy_in_b = torch.rand(4, 1, 64, 64)
    dummy_in_c = torch.rand(4, 1, 64, 64)
    output = model(dummy_in_b, dummy_in_c)
    print(output.size())
    flops, params = profile(model, inputs=(dummy_in_b, dummy_in_c, ))
    print('GFLOPs and Params (M): {:.4f}, {:.4f}'.format(flops / 1024**3, params / 1024**2))