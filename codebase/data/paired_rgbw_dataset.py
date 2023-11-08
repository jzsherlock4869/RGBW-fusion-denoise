import cv2
import os
import os.path as osp
from xml.etree import cElementTree as ET
import random
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from glob import glob

import sys
sys.path.append("..")
from utils.utils_bayer import tone_map, reversed_tone_map, read_bin_file, read_simpleISP_imgIno


def paired_random_crop_bayer(img_gts, img_lqs1, img_lqs2, gt_patch_size):
    """Paired random crop. Support Numpy array and Tensor inputs.
    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs1, list):
        img_lqs1 = [img_lqs1]
    if not isinstance(img_lqs2, list):
        img_lqs2 = [img_lqs2]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs1[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs1[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]

    assert gt_patch_size % 4 == 0
    lq_patch_size = gt_patch_size

    if h_gt != h_lq or w_gt != w_lq:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not same as LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    # NOTE: bayer patch should with stride = 2
    top = random.randint(0, h_lq - lq_patch_size) // 2 * 2
    left = random.randint(0, w_lq - lq_patch_size) // 2 * 2

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs1 = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs1]
        img_lqs2 = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs2]
    else:
        img_lqs1 = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs1]
        img_lqs2 = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs2]

    # crop corresponding gt patch
    top_gt, left_gt = top, left
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs1) == 1:
        img_lqs1 = img_lqs1[0]
    if len(img_lqs2) == 1:
        img_lqs2 = img_lqs2[0]
    return img_gts, img_lqs1, img_lqs2


@DATASET_REGISTRY.register()
class PairedRGBWDataset(data.Dataset):
    """Paired RGBW dataset. Load binB, binC and corresponding clean bin
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedRGBWDataset, self).__init__()
        self.opt = opt
        self.norm_type = opt['norm_type']

        self.gt_folder = opt['dataroot_gt']
        self.lq24_folder = opt['dataroot_lq24']
        self.lq42_folder = opt['dataroot_lq42']
        self.info_folder = opt['dataroot_imginfo']

        val_ratio = 0.1
        val_num = int(70 * val_ratio)
        print(f'[PairedRGBWDataset] leave out {val_ratio * 100} percent for validation, which is {val_num}')

        self.gt_paths = sorted(glob(osp.join(self.gt_folder, '*halfres.bin')))
        self.lq24_rgb_paths = sorted(glob(osp.join(self.lq24_folder, '*DbinB.bin')))
        self.lq24_w_paths = sorted(glob(osp.join(self.lq24_folder, '*DbinC.bin')))
        self.lq42_rgb_paths = sorted(glob(osp.join(self.lq42_folder, '*DbinB.bin')))
        self.lq42_w_paths = sorted(glob(osp.join(self.lq42_folder, '*DbinC.bin')))

        if opt['phase'] == 'train':
            self.gt_paths = self.gt_paths[:-val_num]
            self.lq24_rgb_paths = self.lq24_rgb_paths[:-val_num]
            self.lq24_w_paths = self.lq24_w_paths[:-val_num]
            self.lq42_rgb_paths = self.lq42_rgb_paths[:-val_num]
            self.lq42_w_paths = self.lq42_w_paths[:-val_num]
        else:
            self.gt_paths = self.gt_paths[-val_num:]
            self.lq24_rgb_paths = self.lq24_rgb_paths[-val_num:]
            self.lq24_w_paths = self.lq24_w_paths[-val_num:]
            self.lq42_rgb_paths = self.lq42_rgb_paths[-val_num:]
            self.lq42_w_paths = self.lq42_w_paths[-val_num:]

        self.total_num = len(self.gt_paths)
        asc_order = list(range(self.total_num * 2))
        if opt['phase'] == 'train':
            random.shuffle(asc_order)
        self.order = asc_order

    def __getitem__(self, index):

        cur_id = self.order[index]
        noise_level = cur_id // self.total_num
        abs_id = cur_id % self.total_num
        if noise_level == 0:
            # 24dB noisy
            cur_rgb_path = self.lq24_rgb_paths[abs_id]
            cur_w_path = self.lq24_w_paths[abs_id]
            cur_gt_path = self.gt_paths[abs_id]
        else:
            # 42dB noisy
            cur_rgb_path = self.lq42_rgb_paths[abs_id]
            cur_w_path = self.lq42_w_paths[abs_id]
            cur_gt_path = self.gt_paths[abs_id]

        # if self.norm_type is 'simple', then is float in [0, 1]
        img_rgb = read_bin_file(cur_rgb_path, norm_type=self.norm_type)
        img_w = read_bin_file(cur_w_path, norm_type=self.norm_type)
        img_gt = read_bin_file(cur_gt_path, norm_type=self.norm_type)

        cur_info_path = osp.join(self.info_folder, osp.basename(cur_gt_path.replace('bin', 'xml')))
        r_gain, b_gain, ccm_matrix = read_simpleISP_imgIno(cur_info_path)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            assert gt_size % 2 == 0
            # random crop
            img_gt, img_rgb, img_w = paired_random_crop_bayer(img_gt, img_rgb, img_w, gt_size)

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        img_gt, img_rgb, img_w = img_gt.transpose((2,0,1)), img_rgb.transpose((2,0,1)), img_w.transpose((2,0,1))

        return {'lq_rgb': img_rgb, 'lq_w': img_w, 'gt': img_gt,
                'r_gain': r_gain, 'b_gain': b_gain, 'ccm_matrix': ccm_matrix, 
                'rgb_path': cur_rgb_path, 'w_path': cur_w_path, 'gt_path': cur_gt_path}

    def __len__(self):
        return self.total_num * 2



if __name__ == "__main__":
    opt = {
        "phase": 'train',
        "gt_size": 64,
        "norm_type": 'simple',
        "dataroot_gt": '/data/datasets/MIPI/003_rgbw/RGBW_train_dataset_halfres/'\
                'GT_bayer/train_bayer_half_gt',
        "dataroot_lq24": '/data/datasets/MIPI/003_rgbw/RGBW_train_dataset_halfres/'\
                'input/train_RGBW_half_input_24dB',
        "dataroot_lq42": '/data/datasets/MIPI/003_rgbw/RGBW_train_dataset_halfres/'\
                'input/train_RGBW_half_input_42dB',
        "dataroot_imginfo": '/data/datasets/MIPI/003_rgbw/RGBW_train_dataset_halfres/ImgInfo/train_RGBW_half_imgInfo/'
    }
    dataset = PairedRGBWDataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for bid, batch in enumerate(dataloader):
        print(bid)
        print(batch["lq_rgb"].size(), batch["lq_w"].size(), batch["gt"].size())
        print(batch["r_gain"], batch["b_gain"], batch["ccm_matrix"].size())
