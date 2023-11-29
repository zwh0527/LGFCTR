import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from einops import repeat, rearrange

from .stem import conv3x3, conv1x1
from .transformer import TransformerBlock


class Feat_Reg_Cls(nn.Module):
    """
    For feature regressor and classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config['FINE_WINDOW_SIZE']
        self.compact_dims = config['COMPACT_DIMS']
        self.mlp_dims = config['MLP_DIMS']
        self.compact = nn.Sequential(conv3x3(config['D_MODEL_F'] * 2, self.compact_dims[0], 2),
                                     nn.BatchNorm2d(self.compact_dims[0]),
                                     nn.ReLU(True),
                                     # CBAM(self.compact_dims[0], 3),
                                     conv3x3(self.compact_dims[0], self.compact_dims[1], 2))
        self.reg_mlp = nn.Sequential(
            nn.Linear(round(self.W / 4 + 0.5) ** 2 * self.compact_dims[-1], self.mlp_dims[0], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-3], self.mlp_dims[-2], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1], bias=False),
            nn.Tanh())
        self.cls_mlp = nn.Sequential(
            nn.Linear(round(self.W / 4 + 0.5) ** 2 * self.compact_dims[-1], self.mlp_dims[0], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-3], self.mlp_dims[-2], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-2], 1, bias=False),
            nn.Sigmoid())

    def forward(self, feat0, feat1):
        """
        Args:
            feat0 (torch.Tensor): [M, C, H, W]
            feat1 (torch.Tensor): [M, C, H, W]
        Return:
            reg: torch.Tensor: [M, 2]
            cls: torch.Tensor: [M, 2]
        """
        feat = torch.cat([feat0, feat1], dim=1)
        feat = self.compact(feat).contiguous().view(feat.size(0), -1)
        return self.reg_mlp(feat), self.cls_mlp(feat)


class Feat_Regressor(nn.Module):
    """
    For feature regressor
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config['FINE_WINDOW_SIZE']
        self.compact_dims = config['COMPACT_DIMS']
        self.mlp_dims = config['MLP_DIMS']
        self.compact = nn.Sequential(conv3x3(config['D_MODEL_F']*2, self.compact_dims[0], 2),
                                     nn.BatchNorm2d(self.compact_dims[0]),
                                     nn.ReLU(True),
                                     conv3x3(self.compact_dims[0], self.compact_dims[1], 2))
        self.mlp = nn.Sequential(nn.Linear(round(self.W/4+0.5)**2 * self.compact_dims[-1], self.mlp_dims[0], bias=False),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.mlp_dims[-3], self.mlp_dims[-2], bias=False),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1], bias=False),
                                 nn.Tanh())
                                 
    def forward(self, feat0, feat1):
        """
        Args:
            feat0 (torch.Tensor): [M, C, H, W]
            feat1 (torch.Tensor): [M, C, H, W]
        Return:
            torch.Tensor: [M, 2]
        """
        feat = torch.cat([feat0, feat1], dim=1)
        feat = self.compact(feat).contiguous().view(feat.size(0), -1)
        return self.mlp(feat)


class Feat_Regressor_Attn(nn.Module):
    """
    For feature regressor with attention
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config['FINE_WINDOW_SIZE']
        self.compact_dims = config['COMPACT_DIMS']
        self.mlp_dims = config['MLP_DIMS']
        self.self_attn = TransformerBlock(self.compact_dims[0], config['NHEAD'], FF_type=config['FF_TYPE'])
        self.cross_attn = TransformerBlock(self.compact_dims[0], config['NHEAD'], FF_type=config['FF_TYPE'])
        self.compact = nn.Sequential(conv3x3(self.compact_dims[0]*2, self.compact_dims[0], 2),
                                     nn.BatchNorm2d(self.compact_dims[0]),
                                     nn.ReLU(True),
                                     conv3x3(self.compact_dims[0], self.compact_dims[1], 2))
        self.mlp = nn.Sequential(
            nn.Linear(round(self.W / 4 + 0.5) ** 2 * self.compact_dims[-1], self.mlp_dims[0], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-3], self.mlp_dims[-2], bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1], bias=False),
            nn.Tanh())

    def forward(self, feat0, feat1):
        """
        Args:
            feat0 (torch.Tensor): [M, C, H, W]
            feat1 (torch.Tensor): [M, C, H, W]
        Return:
            torch.Tensor: [M, 2]
        """
        feat0, feat1 = self.self_attn(feat0, feat0), self.self_attn(feat1, feat1)
        feat0, feat1 = self.cross_attn(feat0, feat1), self.cross_attn(feat1, feat0)
        feat = torch.cat([feat0, feat1], dim=1)
        feat = self.compact(feat).contiguous().view(feat.size(0), -1)
        return self.mlp(feat)


class FineMatching(nn.Module):
    """
    FineMatching with s2d paradigm
    There are three matching types to choose:
        1) sim_expectation: compute 2D expectation of similarity matrix
        2) feat_reg_cls: regress a 2D coordinate and classification by concatenate two fine feature patches
        3) feat_regressor: regress a 2D coodinate by concatenate two fine feature patches
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config['FINE_WINDOW_SIZE']
        self.WW = self.W ** 2
        self.matching_type = config['MATCHING_TYPE']
        assert self.matching_type in ['sim_expectation', 'feat_regressor', 'feat_reg_cls', 'feat_regressor_attn']
        if self.matching_type == 'feat_reg_cls':
            self.matching = Feat_Reg_Cls(config)
        elif self.matching_type == 'feat_regressor':
            self.matching = Feat_Regressor(config)
        elif self.matching_type == 'feat_regressor_attn':
            self.matching = Feat_Regressor_Attn(config)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat_f0 (torch.Tensor): [M, WW, C]
            feat_f1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        assert self.W == W and self.WW == WW
        
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.C, self.scale = M, C, scale
        
        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            if self.matching_type == 'feat_reg_cls':
                data.update({'prob_f': torch.empty(0, 1, device=feat_f0.device)})
            return

        if self.matching_type == 'feat_reg_cls':
            coords_normalized, prob_f = self.matching(rearrange(feat_f0, 'n (h w) c -> n c h w', h=W, w=W),
                                                     rearrange(feat_f1, 'n (h w) c -> n c h w', h=W, w=W))
            std = torch.ones(coords_normalized.size(0)).to(coords_normalized.device)
            data.update({'prob_f': prob_f, 'cls_f': prob_f > 0.5})
        elif self.matching_type in ['feat_regressor', 'feat_regressor_attn']:
            coords_normalized = self.matching(rearrange(feat_f0, 'n (h w) c -> n c h w', h=W, w=W),
                                                      rearrange(feat_f1, 'n (h w) c -> n c h w', h=W, w=W))
            std = torch.ones(coords_normalized.size(0)).to(coords_normalized.device)
        else:
            # pick center feature and compute heatmap
            feat_f0_picked = feat_f0[:, WW//2, :]

            sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
            softmax_temp = 1. / C**.5
            heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)

            # compute coordinates from heatmap through three different methods
            grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]
            coords_normalized = dsnt.spatial_expectation2d(heatmap.view(-1, W, W)[None], True)[0]  # [M, 2]

            # compute std over <x, y>
            var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)}) # [M, 3]
        #data.update({'expec_f': coords_normalized}) # [M, 2] for fine loss type : l2

        # compute absolute kpt coords
        """feat_f0_picked = feat_f0[:, WW//2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
        self.get_fine_match_subpixel(coords_normalized, data, dsnt.spatial_expectation2d(heatmap.view(-1, W, W)[None], True)[0])"""
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match_subpixel(self, coords_normed, data, fake):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_c = data['mkpts0_c']
        mkpts1_c = data['mkpts1_c']
        m_bids = data['m_bids']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]
        mkpts1_f_fake = mkpts1_c + (fake * (W // 2) * scale1)[:len(data['mconf'])]
        if 'cls_f' in data:
            data.update({
                "mkpts0_f_i": mkpts0_f,
                "mkpts1_f_i": mkpts1_f,
                "mkpts0_c_i": mkpts0_c,
                "mkpts1_c_i": mkpts1_c,
                "mkpts1_f_fake_i": mkpts1_f_fake,
                "m_bids_i": m_bids
            })
            mask = data['cls_f'][:len(data['mconf']), 0]
            mkpts0_c, mkpts1_c = mkpts0_c[mask], mkpts1_c[mask]
            mkpts0_f, mkpts1_f, mkpts1_f_fake = mkpts0_f[mask], mkpts1_f[mask], mkpts1_f_fake[mask]
            m_bids = m_bids[mask]
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mkpts0_c": mkpts0_c,
            "mkpts1_c": mkpts1_c,
            "mkpts1_f_fake": mkpts1_f_fake, # for sub-pixel visualization
            "m_bids": m_bids
        })
        
    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_c = data['mkpts0_c']
        mkpts1_c = data['mkpts1_c']
        m_bids = data['m_bids']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]
        # filter mkpts according to classification from fine-level
        if 'cls_f' in data:
            data.update({
                "mkpts0_f_i": mkpts0_f,
                "mkpts1_f_i": mkpts1_f,
                "mkpts0_c_i": mkpts0_c,
                "mkpts1_c_i": mkpts1_c,
                "m_bids_i": m_bids
            })
            mask = data['cls_f'][:len(data['mconf']), 0]
            mkpts0_c, mkpts1_c = mkpts0_c[mask], mkpts1_c[mask]
            mkpts0_f, mkpts1_f = mkpts0_f[mask], mkpts1_f[mask]
            m_bids = m_bids[mask]
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mkpts0_c": mkpts0_c,
            "mkpts1_c": mkpts1_c,
            "m_bids": m_bids
        })
