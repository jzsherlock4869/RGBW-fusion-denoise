import struct

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
from debayer import Debayer5x5, Debayer3x3, Debayer2x2
from debayer import Layout
from xml.etree import cElementTree as ET

def read_simpleISP_imgIno(path):
    tree = ET.parse(path)
    root = tree.getroot()

    r_gain = root.find('r_gain').text
    b_gain = root.find('b_gain').text
    ccm_00 = root.find('ccm_00').text
    ccm_01 = root.find('ccm_01').text
    ccm_02 = root.find('ccm_02').text
    ccm_10 = root.find('ccm_10').text
    ccm_11 = root.find('ccm_11').text
    ccm_12 = root.find('ccm_12').text
    ccm_20 = root.find('ccm_20').text
    ccm_21 = root.find('ccm_21').text
    ccm_22 = root.find('ccm_22').text
    ccm_matrix = np.array([ccm_00, ccm_01, ccm_02,
                           ccm_10, ccm_11, ccm_12,
                           ccm_20, ccm_21, ccm_22])
    ccm_matrix = np.asarray(ccm_matrix).astype(np.float32)
    ccm_matrix = torch.from_numpy(ccm_matrix.reshape((3, 3)))
    return float(r_gain), float(b_gain), ccm_matrix

def read_bin_file(filepath, norm_type='simple'):
    '''
    read '.bin' file to 3-d numpy array
    :param path_bin_file:
        path to '.bin' file
    :return:
        3-d image as numpy array (float32)  (h * w * 1)
    '''
    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]
    data_3d = data[2:].reshape((hh, ww, 1))
    data_3d = data_3d.astype(np.float32)

    if norm_type == 'none':
        pass
    elif norm_type == 'simple':
        data_3d = data_3d / 1023.0
    elif norm_type == 'clip_norm':
        data_3d = np.clip((data_3d.astype(np.float32) - 64) / (1023 - 64), 0, 1)
    elif norm_type == 'reinhard':
        # REMEMBER: reinhard mode first conduct clip norm to [0...1] and then tone_map
        data_3d = np.clip((data_3d.astype(np.float32) - 64) / (1023 - 64), 0, 1)
        data_3d = tone_map(data_3d, c=0.1)
    else:
        raise NotImplementedError
    return data_3d

def save_bin(filepath, arr):
    '''
    save 2-d numpy array to '.bin' files with uint16
    @param filepath: expected file path to store data
    @param arr: 2-d numpy array
    @return: None
    '''
    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape

    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

def tone_map(x, c=0.25):
    # Modified Reinhard tone mapping.
    mapped_x = x / (x + c)
    return mapped_x

def reversed_tone_map(x, c=0.25, maxi=1023):
    # Modified Reinhard tone mapping.
    mapped_x = c * x / (1 - x)
    if isinstance(x, torch.Tensor):
        mapped_x = torch.clamp(mapped_x, min=0, max=maxi)
    elif isinstance(x, np.ndarray):
        mapped_x = np.clip(mapped_x, a_min=0, a_max=maxi)
    return mapped_x


def ccm_corr_torch(rgb, CCM, fwd=True):
    '''
    rgb: torch.Tensor, 0-1, b3hw
    CCM: torch.Tensor, b33
    rgb_ccm: torch.Tensor, 0-1, b3hw, color corrected
    '''

    # rgb = torch.clamp(rgb, min=0, max=1)
    n, c, h, w = rgb.size()
    # assert n == 1 and c == 3, 'rgb need to be in shape of [1, 3, h, w]'
    assert c == 3, f'rgb need to be in shape of [b, 3, h, w] but got [{n}, {c}, {h}, {w}]'
    rgb = rgb.reshape(n, c, h * w)
    CCM.requires_grad = False
    if fwd:
        rgb_ccm = torch.matmul(CCM, rgb).reshape(n, c, h, w)
    # rgb_ccm = torch.clamp(rgb_ccm, min=0, max=1)
    return rgb_ccm


def bayer2rgb_numpy(img_np, r_gain, b_gain, CCM):
    """
    img_np: 0-1023|hw
    rgb_np: 0-1|hw3
    r_gain, b_gain: int
    CCM: list of float
    """
    bayer = np.clip((img_np.astype(np.float32) - 64) / (1023 - 64), 0, 1)
    ten_bayer = torch.from_numpy(bayer).unsqueeze(0).unsqueeze(0)

    f = Debayer3x3(Layout.GBRG)

    # a Bx1xHxW, [0..1], torch.float32 RGGB-Bayer tensor
    with torch.no_grad():
        # a Bx3xHxW, torch.float32 tensor of RGB images
        rgb = f(ten_bayer)

    rgb = torch.clamp(rgb, min=0, max=1)
    rgb[:,0,...] *= r_gain.view(-1, 1, 1)
    rgb[:,2,...] *= b_gain.view(-1, 1, 1)
    rgb = ccm_corr_torch(rgb, CCM, fwd=True)
    rgb = torch.clamp(rgb, min=0, max=1)
    rgb = torch.pow(rgb, 1/2.2)

    # -----torch----numpy-----borderline------
    rgb_np = rgb.numpy()[0].transpose((1,2,0))
    rgb_np = (rgb_np * 255).astype(np.uint8)

    return rgb_np


class BayerRGB_Layer(nn.Module):
    """
    img_ten: 0-1|b1hw
    rgb_torch: 0-1|b3hw
    r_gain, b_gain: FloatTensor (b), (b)
    CCM: FloatTensor(b, 3, 3)
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.debayer = Debayer3x3(Layout.GBRG).to(device)

    def forward(self, img_ten, r_gain, b_gain, CCM):
        rgb = self.debayer(img_ten)
        rgb[:,0,...] *= r_gain.view(-1, 1, 1)
        rgb[:,2,...] *= b_gain.view(-1, 1, 1)
        rgb = ccm_corr_torch(rgb, CCM, fwd=True)
        return rgb


class Bayerize_Layer(nn.Module):
    """
    turn [n,4,h,w] rgb to [n,1,h,w] bayer
    """
    def __init__(self):
        super().__init__()
        filt_1 = torch.FloatTensor([[1,0], [0, 0]])
        filt_2 = torch.FloatTensor([[0,1], [0, 0]])
        filt_3 = torch.FloatTensor([[0,0], [1, 0]])
        filt_4 = torch.FloatTensor([[0,0], [0, 1]])
        # self.weight: shape [4,1,2,2]
        self.weight = torch.stack((filt_1, filt_2, filt_3, filt_4), dim=0).unsqueeze(1)
        self.weight.requires_grad = False
        self.ps = nn.PixelShuffle(upscale_factor=2)
        # if device == 'cuda':
            # self.weight = self.weight.cuda()
        
    def forward(self, x):
        # x: [n, 4, h, w]
        # gbrg: [n, 4, h/2, w/2]
        # bayer: [n, 1, h, w]
        self.weight = self.weight.to(x.device)
        gbrg = F.conv2d(x, self.weight, bias=None, stride=2, padding=0, dilation=1, groups=4)
        bayer = self.ps(gbrg)
        return bayer