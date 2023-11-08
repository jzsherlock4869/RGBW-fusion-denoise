from collections import OrderedDict
from doctest import OutputChecker
from operator import gt
from reprlib import recursive_repr

import os
from os import path as osp
from tqdm import tqdm
import torch
import numpy as np

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from debayer import Debayer5x5, Debayer3x3, Debayer2x2
from debayer import Layout
from copy import deepcopy

import sys
sys.path.append("..")

from utils.utils_bayer import bayer2rgb_numpy, reversed_tone_map, save_bin, BayerRGB_Layer

@MODEL_REGISTRY.register()
class BaseRGBWModel(SRModel):
    """Example model based on the SRModel class.

    In this example model, we want to implement a new model that trains with both L1 and L2 loss.

    New defined functions:
        init_training_settings(self)
        feed_data(self, data)
        optimize_parameters(self, current_iter)

    Inherited functions:
        __init__(self, opt)
        setup_optimizers(self)
        test(self)
        dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        nondist_validation(self, dataloader, current_iter, tb_logger, save_img)
        _log_validation_metric_values(self, current_iter, dataset_name, tb_logger)
        get_current_visuals(self)
        save(self, epoch, current_iter)
    """

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.loss_pix = build_loss(train_opt['pix_opt']).to(self.device) if 'pix_opt' in train_opt else None
        self.loss_rgb_pix = build_loss(train_opt['rgb_pix_opt']).to(self.device) if 'rgb_pix_opt' in train_opt else None
        self.loss_percep = build_loss(train_opt['perceptual_opt']).to(self.device) if 'perceptual_opt' in train_opt else None

        # if self.loss_rgb_pix is not None or self.loss_percep is not None:
        self.bayer2rgb = BayerRGB_Layer(device='cuda')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):

        self.lq_rgb = data['lq_rgb'].to(self.device)
        self.lq_w = data['lq_w'].to(self.device)
        self.r_gain = data['r_gain']
        self.b_gain = data['b_gain']
        self.ccm_matrix = data['ccm_matrix']

        # print(type(self.r_gain), type(self.ccm_matrix))
        # print(self.r_gain.size(), self.ccm_matrix.size())

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        # print(self.optimizer_g.param_groups[0]['params'][0].grad)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq_rgb, self.lq_w)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.loss_pix:
            l_pix = self.loss_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # self.output_rgb = bayer2rgb_torch(self.output, self.r_gain, self.b_gain, self.ccm_matrix)
        # self.gt_rgb = bayer2rgb_torch(self.gt, self.r_gain, self.b_gain, self.ccm_matrix)

        # self.output_rgb = self.bayer2rgb(self.output, self.r_gain, self.b_gain, self.ccm_matrix)
        # self.gt_rgb = self.bayer2rgb(self.gt, self.r_gain, self.b_gain, self.ccm_matrix)

        if self.opt['datasets']['train']['norm_type'] == 'none':
            # norm_type = none means no pre-process
            # clip to 64-1023 and then debayer
            norm_bayer_output = torch.clamp((self.output - 64) / (1023 - 64), 0, 1)
            norm_bayer_gt = torch.clamp((self.gt - 64) / (1023 - 64), 0, 1)

        if self.opt['datasets']['train']['norm_type'] == 'clip_norm':
            # norm_type = clip_norm means clip 64-1023 and make input scale [0...1]
            # direct use to debayer
            norm_bayer_output = self.output
            norm_bayer_gt = self.gt

        if self.opt['datasets']['train']['norm_type'] == 'simple':
            # norm_type = simple means div 1023 to make input scale [0...1]
            # clip to 64-1023, and rescale [0...1] and then debayer
            norm_bayer_output = torch.clamp((self.output * 1023 - 64) / (1023 - 64), 0, 1)
            norm_bayer_gt = torch.clamp((self.gt * 1023 - 64) / (1023 - 64), 0, 1)

        elif self.opt['datasets']['train']['norm_type'] == 'reinhard':
            # norm_type = reinhard means clip norm and tonemaping to make input scale [0...1]
            # direct debayer in tone-mapped domain
            norm_bayer_output = self.output
            norm_bayer_gt = self.gt

        # self.output_rgb = Debayer3x3(Layout.GBRG).cuda()(norm_bayer_output)
        # self.gt_rgb = Debayer3x3(Layout.GBRG).cuda()(norm_bayer_gt)

        self.output_rgb = self.bayer2rgb(norm_bayer_output, self.r_gain, self.b_gain, self.ccm_matrix)
        self.gt_rgb = self.bayer2rgb(norm_bayer_gt, self.r_gain, self.b_gain, self.ccm_matrix)

        # print('output', self.output_rgb.min(), self.output_rgb.max())
        # print('gt', self.gt_rgb.min(), self.gt_rgb.max())

        # pixel loss in rgb domain
        if self.loss_rgb_pix:
            l_rgb_pix = self.loss_rgb_pix(self.output_rgb, self.gt_rgb)
            l_total += l_rgb_pix
            loss_dict['l_rgb_pix'] = l_rgb_pix

        # vgg loss in rgb domain
        if self.loss_percep:
            l_g_percep, l_g_style = self.loss_percep(self.output_rgb, self.gt_rgb)
            if l_g_percep is not None:
                l_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        l_total.backward()
        # print('\n l_total', l_total)
        # print('\n l_total req grad ', l_total.requires_grad, l_total.grad)
        # print('\n ==================')
        # print('\n output_rgb', self.output_rgb)
        # print('\n ==================')

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # TODO: need to re-write more following functions

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq_rgb, self.lq_w)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq_rgb, self.lq_w)
            self.net_g.train()


    def test_tta(self):
        self.net_g.eval()
        with torch.no_grad():

            output_ori = self.net_g(self.lq_rgb, self.lq_w)

            # hflip aug
            hflip_rgb_ = torch.flip(self.lq_rgb, dims=[3])[:, :, :, 1:-1]
            hflip_w_ = torch.flip(self.lq_w, dims=[3])[:, :, :, 1:-1]
            output_hflip_ = torch.flip(self.net_g(hflip_rgb_, hflip_w_), dims=[3])
            output_hflip = deepcopy(output_ori)
            output_hflip[:, :, :, 1:-1] = output_hflip_

            # hflip aug
            vflip_rgb_ = torch.flip(self.lq_rgb, dims=[2])[:, :, 1:-1, :]
            vflip_w_ = torch.flip(self.lq_w, dims=[2])[:, :, 1:-1, :]
            output_vflip_ = torch.flip(self.net_g(vflip_rgb_, vflip_w_), dims=[2])
            output_vflip = deepcopy(output_ori)
            output_vflip[:, :, 1:-1, :] = output_vflip_

            self.output = (output_ori + output_hflip + output_vflip) / 3.0

        self.net_g.train()


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        is_tta = self.opt['val'].get('tta', False)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['rgb_path'][0]))[0]
            self.feed_data(val_data)

            if not is_tta:
                self.test()
            else:
                self.test_tta()
                
            # r_gain, b_gain, CCM = val_data['r_gain'][0], val_data['b_gain'][0], val_data['ccm_matrix'][0]
            r_gain, b_gain, CCM = val_data['r_gain'], val_data['b_gain'], val_data['ccm_matrix']

            visuals = self.get_current_visuals()

            if self.opt['datasets']['val']['norm_type'] == 'none':
                # in-net: 0-1023, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                sr_img = tensor2img([visuals['result']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1023))
                sr_img_rgb = bayer2rgb_numpy(sr_img * 1023, r_gain, b_gain, CCM)

            if self.opt['datasets']['val']['norm_type'] == 'clip_norm':
                # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                sr_img = tensor2img([visuals['result']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                sr_img_rgb = bayer2rgb_numpy(sr_img * (1023 - 64) + 64, r_gain, b_gain, CCM)

            if self.opt['datasets']['val']['norm_type'] == 'simple':
                # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                sr_img = tensor2img([visuals['result']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                sr_img_rgb = bayer2rgb_numpy(sr_img * 1023, r_gain, b_gain, CCM)

            if self.opt['datasets']['val']['norm_type'] == 'reinhard':
                # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                sr_img = tensor2img([visuals['result']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                sr_img_rgb = bayer2rgb_numpy(reversed_tone_map(sr_img, c=0.1, maxi=1) * (1023 - 64) + 64, r_gain, b_gain, CCM)

                # print('vis sr ', visuals['result'].min(), visuals['result'].max())
                # print('vis sr ', sr_img.min(), sr_img.max())
                # print('sr img rgb', sr_img_rgb.min(), sr_img_rgb.max())

            metric_data['img'] = sr_img_rgb

            if 'gt' in visuals:
                if self.opt['datasets']['val']['norm_type'] == 'none':
                    # in-net: 0-1023, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1023))
                    gt_img_rgb = bayer2rgb_numpy(gt_img * 1023, r_gain, b_gain, CCM)

                if self.opt['datasets']['val']['norm_type'] == 'clip_norm':
                    # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                    gt_img_rgb = bayer2rgb_numpy(gt_img * (1023 - 64) + 64, r_gain, b_gain, CCM)

                if self.opt['datasets']['val']['norm_type'] == 'simple':
                    # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                    gt_img_rgb = bayer2rgb_numpy(gt_img * 1023, r_gain, b_gain, CCM)

                if self.opt['datasets']['val']['norm_type'] == 'reinhard':
                    # in-net: 0-1, sr_img: 0-1, sr_img_rgb: 0-1023 -> 255
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=False, out_type=np.float32, min_max=(0, 1))
                    gt_img_rgb = bayer2rgb_numpy(reversed_tone_map(gt_img, c=0.1, maxi=1) * (1023 - 64) + 64, r_gain, b_gain, CCM)

                    # print('vis gt ', visuals['gt'].min(), visuals['gt'].max())
                    # print('gt img ', gt_img.min(), gt_img.max())
                    # print('gt img rgb', gt_img_rgb.min(), gt_img_rgb.max())

                metric_data['img2'] = gt_img_rgb

                del self.gt

            # tentative for out of GPU memory
            del self.lq_rgb
            del self.lq_w
            del self.output
            torch.cuda.empty_cache()

            save_binary = self.opt['val'].get('save_binary', None)

            if save_binary:
                if self.opt['is_train']:
                    save_bin_path = osp.join(self.opt['path']['visualization'], 'bin', img_name,
                                             f'{img_name}_{current_iter}.bin')
                else:
                    if self.opt['val']['suffix']:
                        save_bin_path = osp.join(self.opt['path']['visualization'], 'bin', dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.bin')
                    else:
                        save_bin_path = osp.join(self.opt['path']['visualization'], 'bin', dataset_name,
                                                 f'{img_name}.bin')

                os.makedirs(osp.dirname(save_bin_path), exist_ok=True)

                if self.opt['datasets']['val']['norm_type'] == 'none':
                    save_1023 = sr_img * 1023
                if self.opt['datasets']['val']['norm_type'] == 'clip_norm':
                    save_1023 = sr_img * (1023 - 64) + 64
                if self.opt['datasets']['val']['norm_type'] == 'simple':
                    save_1023 = sr_img * 1023
                if self.opt['datasets']['val']['norm_type'] == 'reinhard':
                    save_1023 = reversed_tone_map(sr_img, c=0.1, maxi=1) * (1023 - 64) + 64

                save_bin(save_bin_path, save_1023)

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'png', img_name,
                                             f'{img_name}_{current_iter}.png')
                                             
                    if current_iter == self.opt['val']['val_freq']:
                        save_ori_path = osp.join(self.opt['path']['visualization'], 'png', img_name,
                                             f'{img_name}_ori.png')
                        os.makedirs(osp.dirname(save_ori_path), exist_ok=True)
                        imwrite(gt_img_rgb[:,:,::-1].copy(), save_ori_path)
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'png', dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], 'png', dataset_name,
                                                 f'{img_name}.png')

                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                imwrite(sr_img_rgb[:,:,::-1].copy(), save_img_path)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_rgb'] = self.lq_rgb.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
