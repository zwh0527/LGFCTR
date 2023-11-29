# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:24:21 2023

@author: knight
"""

from .transformer import Backbone, Backbone_vis
from .coarse_matching import CoarseMatching
from .fine_matching import FineMatching
from .fine_preprocess import FinePreprocess
import torch.nn as nn
from einops import rearrange

class LGFCTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.backbone = Backbone(config)
        self.coarse_matching = CoarseMatching(config['MATCH_COARSE'])
        self.fine_preprocess = FinePreprocess(config['MATCH_FINE'])
        self.fine_matching = FineMatching(config['MATCH_FINE'])
        
    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        
        # 1. backbone -- encoder and decoder
        mask_i0 = mask_i1 = mask_c0 = mask_c1 = None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']
            mask_i0, mask_i1 = data['mask_i0'], data['mask_i1'] # mask with initial resolution for backbone transformers
            
        feat_c0, feat_f0, feat_c1, feat_f1 = self.backbone(data['image0'], data['image1'], mask_i0, mask_i1)
        
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        
        # 2. coarse-level matching
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        if mask_c0 is not None:
            mask_c0 = rearrange(mask_c0, 'n h w -> n (h w)')
        if mask_c1 is not None:
            mask_c1 = rearrange(mask_c1, 'n h w -> n (h w)')
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        
        # 3. preprocess before fine-level matching
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        
        # 4. fine-level matching
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class LGFCTR_vis(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = Backbone_vis(config)
        self.coarse_matching = CoarseMatching(config['MATCH_COARSE'])
        self.fine_preprocess = FinePreprocess(config['MATCH_FINE'])
        self.fine_matching = FineMatching(config['MATCH_FINE'])

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        # 1. backbone -- encoder and decoder
        mask_i0 = mask_i1 = mask_c0 = mask_c1 = None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']
            mask_i0, mask_i1 = data['mask_i0'], data[
                'mask_i1']  # mask with initial resolution for backbone transformers

        feat_c0, feat_f0, feat_c1, feat_f1, attention_vis_lists = self.backbone(data['image0'], data['image1'], mask_i0, mask_i1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
            'attention': attention_vis_lists
        })

        # 2. coarse-level matching
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
        if mask_c0 is not None:
            mask_c0 = rearrange(mask_c0, 'n h w -> n (h w)')
        if mask_c1 is not None:
            mask_c1 = rearrange(mask_c1, 'n h w -> n (h w)')
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 3. preprocess before fine-level matching
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)

        # 4. fine-level matching
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
        
        
        
        
        
        
        
        