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


@DATASET_REGISTRY.register()
class SingleRGBWDataset(data.Dataset):
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
        super(SingleRGBWDataset, self).__init__()
        self.opt = opt
        self.norm_type = opt['norm_type']

        self.lq24_folder = opt['dataroot_lq24']
        self.lq42_folder = opt['dataroot_lq42']
        self.info_folder = opt['dataroot_imginfo']

        self.lq24_rgb_paths = sorted(glob(osp.join(self.lq24_folder, '*DbinB.bin')))
        self.lq24_w_paths = sorted(glob(osp.join(self.lq24_folder, '*DbinC.bin')))
        self.lq42_rgb_paths = sorted(glob(osp.join(self.lq42_folder, '*DbinB.bin')))
        self.lq42_w_paths = sorted(glob(osp.join(self.lq42_folder, '*DbinC.bin')))
        self.info_paths = sorted(glob(osp.join(self.info_folder, '*.xml')))

        self.total_num = len(self.info_paths)
        print(f'[SingleRGBWDataset] loaded all, total {self.total_num} x2 (noise level) to be processed bins')


    def __getitem__(self, index):

        cur_id = index
        noise_level = cur_id // self.total_num
        abs_id = cur_id % self.total_num
        if noise_level == 0:
            # 24dB noisy
            cur_rgb_path = self.lq24_rgb_paths[abs_id]
            cur_w_path = self.lq24_w_paths[abs_id]
        else:
            # 42dB noisy
            cur_rgb_path = self.lq42_rgb_paths[abs_id]
            cur_w_path = self.lq42_w_paths[abs_id]

        # if self.norm_type is 'simple', then is float in [0, 1]
        img_rgb = read_bin_file(cur_rgb_path, norm_type=self.norm_type)
        img_w = read_bin_file(cur_w_path, norm_type=self.norm_type)

        cur_info_path = self.info_paths[abs_id]
        r_gain, b_gain, ccm_matrix = read_simpleISP_imgIno(cur_info_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        img_rgb, img_w = img_rgb.transpose((2,0,1)), img_w.transpose((2,0,1))

        return {'lq_rgb': img_rgb, 'lq_w': img_w,
                'r_gain': r_gain, 'b_gain': b_gain, 'ccm_matrix': ccm_matrix,
                'rgb_path': cur_rgb_path, 'w_path': cur_w_path}

    def __len__(self):
        return self.total_num * 2



if __name__ == "__main__":
    opt = {
        "gt_size": 64,
        "norm_type": 'simple',
        "dataroot_lq24": '/data/datasets/MIPI/003_rgbw/RGBW_validation_dataset_halfres/'\
                'input/valid_RGBW_half_input_24dB',
        "dataroot_lq42": '/data/datasets/MIPI/003_rgbw/RGBW_validation_dataset_halfres/'\
                'input/valid_RGBW_half_input_42dB',
        "dataroot_imginfo": '/data/datasets/MIPI/003_rgbw/RGBW_validation_dataset_halfres/ImgInfo/valid_RGBW_half_imgInfo/'
    }
    dataset = SingleRGBWDataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for bid, batch in enumerate(dataloader):
        print(bid)
        print(batch["lq_rgb"].size(), batch["lq_w"].size())
        print(batch["r_gain"], batch["b_gain"], batch["ccm_matrix"].size())
