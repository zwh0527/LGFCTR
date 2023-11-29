# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:58:50 2023

@author: knight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import LinearAttention, FullAttention, FullAttention_vis
from .position_encoding import PositionEncodingSine
from .stem import *
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 FF_type='mix'):
        super(TransformerBlock, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if FF_type == 'mix':
            self.norm3 = nn.LayerNorm(d_model)

        # feed-forward network
        assert FF_type in ['cnn', 'mlp', 'mix']
        self.FF_type = FF_type
        if FF_type == 'cnn':
            self.FF = CNNBlock3x3(d_model * 2, d_model)
        elif FF_type == 'mlp':
            self.FF = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
        elif FF_type == 'mix':
            self.FF_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
            self.FF_cnn = CNNBlock3x3(d_model, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H, W] -> [N, S, C]
            source (torch.Tensor): [N, C, H, W] -> [N, S, C]
            x_mask (torch.Tensor): [N, H, W] -> [N, L] (optional)
            source_mask (torch.Tensor): [N, H, W] -> [N, S] (optional)
        """
        bs, h = x.size(0), x.size(2)
        x = rearrange(x, 'b c h w -> b (h w) c')
        source = rearrange(source, 'b c h w -> b (h w) c')
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')
        if source_mask is not None:
            source_mask = rearrange(source_mask, 'b h w -> b (h w)')

        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        if self.FF_type == 'cnn':
            message = rearrange(self.FF(rearrange(torch.cat([x, message], dim=2), 'b (h w) c -> b c h w', h=h)),
                                'b c h w -> b (h w) c')
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mlp':
            message = self.FF(torch.cat([x, message], dim=2))
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mix':
            message = self.norm2(self.FF_mlp(torch.cat([x, message], dim=2)))
            message = rearrange(self.FF_cnn(rearrange(message, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')

            # CNN first
            # message = self.FF_mlp(self.norm2(rearrange(self.FF_cnn(rearrange(torch.cat([x, message], dim=2), 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')))

            message = rearrange(self.norm3(message), 'b (h w) c -> b c h w', h=h)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x + message


class SemiCNNTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 FF_type='mix'):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = conv3x3(d_model, d_model)
        self.k_proj = conv3x3(d_model, d_model)
        self.v_proj = conv3x3(d_model, d_model)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if FF_type == 'mix':
            self.norm3 = nn.LayerNorm(d_model)

        # feed-forward network
        assert FF_type in ['cnn', 'mlp', 'mix']
        self.FF_type = FF_type
        if FF_type == 'cnn':
            self.FF = CNNBlock3x3(d_model * 2, d_model)
        elif FF_type == 'mlp':
            self.FF = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
        elif FF_type == 'mix':
            self.FF_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
            self.FF_cnn = nn.Sequential(
                conv3x3(d_model, d_model),
                nn.ReLU(True),
                conv3x3(d_model, d_model)
            )
            self.FF_cnn = CNNBlock3x3(d_model, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H, W]
            source (torch.Tensor): [N, C, H, W]
            x_mask (torch.Tensor): [N, H, W] -> [N, L] (optional)
            source_mask (torch.Tensor): [N, H, W] -> [N, S] (optional)
        """
        bs, h = x.size(0), x.size(2)
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')
        if source_mask is not None:
            source_mask = rearrange(source_mask, 'b h w -> b (h w)')

        query, key, value = x, source, source

        # multi-head attention
        query = rearrange(self.q_proj(query), 'b c h w -> b (h w) c')
        query = query.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = rearrange(self.k_proj(key), 'b c h w -> b (h w) c')
        key = key.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = rearrange(self.v_proj(value), 'b c h w -> b (h w) c')
        value = value.view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.FF_type == 'cnn':
            message = rearrange(self.FF(rearrange(torch.cat([x, message], dim=2), 'b (h w) c -> b c h w', h=h)),
                                'b c h w -> b (h w) c')
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mlp':
            message = self.FF(torch.cat([x, message], dim=2))
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mix':
            message = self.norm2(self.FF_mlp(torch.cat([x, message], dim=2)))
            message = rearrange(self.FF_cnn(rearrange(message, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')
            message = rearrange(self.norm3(message), 'b (h w) c -> b c h w', h=h)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x + message


class PureCNNTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 FF_type='mix'):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = conv3x3(d_model, d_model)
        self.k_proj = conv3x3(d_model, d_model)
        self.v_proj = conv3x3(d_model, d_model)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = conv1x1(d_model, d_model)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if FF_type == 'mix':
            self.norm3 = nn.LayerNorm(d_model)

        # reset parameters
        self._reset_parameters()

        # feed-forward network
        assert FF_type in ['cnn', 'mlp', 'mix']
        self.FF_type = FF_type
        if FF_type == 'cnn':
            self.FF = CNNBlock3x3(d_model * 2, d_model)
        elif FF_type == 'mlp':
            self.FF = CNNBlock1x1(d_model * 2, d_model)
        elif FF_type == 'mix':
            self.FF_mlp = nn.Sequential(
                conv1x1(d_model * 2, d_model * 2),
                nn.ReLU(True),
                conv1x1(d_model * 2, d_model)
            )
            self.FF_cnn = nn.Sequential(
                conv3x3(d_model, d_model),
                nn.ReLU(True),
                conv3x3(d_model, d_model)
            )
            self.FF_mlp = CNNBlock1x1(d_model * 2, d_model)
            self.FF_cnn = CNNBlock3x3(d_model, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H, W]
            source (torch.Tensor): [N, C, H, W]
            x_mask (torch.Tensor): [N, H, W] -> [N, L] (optional)
            source_mask (torch.Tensor): [N, H, W] -> [N, S] (optional)
        """
        bs, h = x.size(0), x.size(2)
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')
        if source_mask is not None:
            source_mask = rearrange(source_mask, 'b h w -> b (h w)')

        query, key, value = x, source, source

        # multi-head attention
        query = rearrange(self.q_proj(query), 'b c h w -> b (h w) c')
        query = query.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = rearrange(self.k_proj(key), 'b c h w -> b (h w) c')
        key = key.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = rearrange(self.v_proj(value), 'b c h w -> b (h w) c')
        value = value.view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(rearrange(message.view(bs, -1, self.nhead * self.dim), 'b (h w) c -> b c h w',
                                       h=h))  # [N, L, C] -> [N, C, H, W]
        message = self.norm1(message.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # feed-forward network
        if self.FF_type != 'mix':
            message = self.norm2(self.FF(torch.cat([x, message], dim=1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            message = self.norm2(self.FF_mlp(torch.cat([x, message], dim=1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            message = self.norm3(self.FF_cnn(message).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + message


class MultiConv(nn.Module):
    def __init__(self, multi_size, d_model, dim):
        super().__init__()
        conv_dict = {1: conv1x1, 3: conv3x3, 5: conv5x5, 7: conv7x7}
        self.multiconv = []
        for conv_size in multi_size:
            self.multiconv.append(conv_dict[conv_size](d_model, dim))
        self.multiconv = nn.ModuleList(self.multiconv)

    def forward(self, feat):
        output = []
        for conv in self.multiconv:
            output.append(conv(feat))
        return torch.cat(output, dim=1)


class MultiCNNTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 FF_type='mix'):
        super().__init__()

        # config
        self.nhead = nhead
        multi_size = [1, 3, 5, 7]
        self.nhead = len(multi_size)
        self.dim = d_model // self.nhead

        # multi-head attention
        self.q_proj = conv3x3(d_model, d_model)
        self.k_proj = MultiConv(multi_size, d_model, self.dim)
        self.v_proj = MultiConv(multi_size, d_model, self.dim)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if FF_type == 'mix':
            self.norm3 = nn.LayerNorm(d_model)

        # feed-forward network
        assert FF_type in ['cnn', 'mlp', 'mix']
        self.FF_type = FF_type
        if FF_type == 'cnn':
            self.FF = CNNBlock3x3(d_model * 2, d_model)
        elif FF_type == 'mlp':
            self.FF = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
        elif FF_type == 'mix':
            self.FF_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
            self.FF_cnn = CNNBlock3x3(d_model, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H, W]
            source (torch.Tensor): [N, C, H, W]
            x_mask (torch.Tensor): [N, H, W] -> [N, L] (optional)
            source_mask (torch.Tensor): [N, H, W] -> [N, S] (optional)
        """
        bs, h = x.size(0), x.size(2)
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')
        if source_mask is not None:
            source_mask = rearrange(source_mask, 'b h w -> b (h w)')

        query, key, value = x, source, source

        # multi-head attention
        query = rearrange(self.q_proj(query), 'b c h w -> b (h w) c')
        query = query.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = rearrange(self.k_proj(key), 'b c h w -> b (h w) c')
        key = key.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = rearrange(self.v_proj(value), 'b c h w -> b (h w) c')
        value = value.view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.FF_type == 'cnn':
            message = rearrange(self.FF(rearrange(torch.cat([x, message], dim=2), 'b (h w) c -> b c h w', h=h)),
                                'b c h w -> b (h w) c')
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mlp':
            message = self.FF(torch.cat([x, message], dim=2))
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mix':
            message = self.norm2(self.FF_mlp(torch.cat([x, message], dim=2)))
            message = rearrange(self.FF_cnn(rearrange(message, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')
            message = rearrange(self.norm3(message), 'b (h w) c -> b c h w', h=h)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x + message


class MultiCNNTransformerBlock_vis(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 FF_type='mix'):
        super().__init__()

        # config
        self.nhead = nhead
        multi_size = [1, 3, 5, 7]
        self.nhead = len(multi_size)
        self.dim = d_model // self.nhead

        # multi-head attention
        self.q_proj = conv3x3(d_model, d_model)
        self.k_proj = MultiConv(multi_size, d_model, self.dim)
        self.v_proj = MultiConv(multi_size, d_model, self.dim)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.attention_vis = FullAttention_vis()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if FF_type == 'mix':
            self.norm3 = nn.LayerNorm(d_model)

        # feed-forward network
        assert FF_type in ['cnn', 'mlp', 'mix']
        self.FF_type = FF_type
        if FF_type == 'cnn':
            self.FF = CNNBlock3x3(d_model * 2, d_model)
        elif FF_type == 'mlp':
            self.FF = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
        elif FF_type == 'mix':
            self.FF_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(True),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self._reset_parameters()
            self.FF_cnn = CNNBlock3x3(d_model, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H, W]
            source (torch.Tensor): [N, C, H, W]
            x_mask (torch.Tensor): [N, H, W] -> [N, L] (optional)
            source_mask (torch.Tensor): [N, H, W] -> [N, S] (optional)
        """
        bs, h, h1 = x.size(0), x.size(2), source.size(2)
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b h w -> b (h w)')
        if source_mask is not None:
            source_mask = rearrange(source_mask, 'b h w -> b (h w)')

        query, key, value = x, source, source

        # multi-head attention
        query = rearrange(self.q_proj(query), 'b c h w -> b (h w) c')
        query = query.view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = rearrange(self.k_proj(key), 'b c h w -> b (h w) c')
        key = key.view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = rearrange(self.v_proj(value), 'b c h w -> b (h w) c')
        value = value.view(bs, -1, self.nhead, self.dim)
        attention_vis = self.attention_vis(query, key, value, q_mask=x_mask, kv_mask=source_mask)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.FF_type == 'cnn':
            message = rearrange(self.FF(rearrange(torch.cat([x, message], dim=2), 'b (h w) c -> b c h w', h=h)),
                                'b c h w -> b (h w) c')
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mlp':
            message = self.FF(torch.cat([x, message], dim=2))
            message = rearrange(self.norm2(message), 'b (h w) c -> b c h w', h=h)
        elif self.FF_type == 'mix':
            message = self.norm2(self.FF_mlp(torch.cat([x, message], dim=2)))
            message = rearrange(self.FF_cnn(rearrange(message, 'b (h w) c -> b c h w', h=h)), 'b c h w -> b (h w) c')
            message = rearrange(self.norm3(message), 'b (h w) c -> b c h w', h=h)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        attention_vis = rearrange(attention_vis, 'b (h0 w0) (h1 w1) c -> b c h0 w0 h1 w1', h0=h, h1=h1)
        return x + message, attention_vis


class Backbone(nn.Module):
    """
        Backbone including encoder and decoder part
    """

    def __init__(self, config):
        super().__init__()

        # 1. config
        self.config = config
        self.attention = self.config['ATTENTION']
        self.FF_type = self.config['FF_TYPE']
        self.transformer_type = self.config['TRANSFORMER_TYPE']
        self.positional_mode = self.config['POSITIONAL_MODE']
        self.positional_pos = self.config['POSITIONAL_POS']
        self.nhead = self.config['NHEAD']

        # 2. encoder
        self.en_layer_names = self.config['ENCODER']['LAYER_NAMES']
        self.en_layer_dims = self.config['ENCODER']['LAYER_DIMS']

        if self.positional_mode is not None:
            self.en_positional_encoding = nn.ModuleList(
                [PositionEncodingSine(self.en_layer_dims[idx], positional_mode=self.positional_mode)
                 for idx in range(len(self.en_layer_dims))])
        self.stem = Stem(self.config['ENCODER'])
        self.downsamplings = [ResBlock(self.config['ENCODER']['STEM_DIMS2'][-1], self.en_layer_dims[0], 2)]
        for idx in range(1, len(self.en_layer_dims)):
            self.downsamplings.append(ResBlock(self.en_layer_dims[idx - 1], self.en_layer_dims[idx], 2))
        self.downsamplings = nn.ModuleList(self.downsamplings)
        self.en_layers = nn.ModuleList([self._make_layer(dim, len(self.en_layer_names)) for dim in self.en_layer_dims])

        # 3. decoder
        self.de_layer_names = self.config['DECODER']['LAYER_NAMES']
        self.de_layer_dims = self.config['DECODER']['LAYER_DIMS']
        self.encoder_dims = self.config['DECODER']['ENCODER_LAYER_DIMS_INV']
        self.is_out_convs = self.config['DECODER']['IS_OUT_CONVS']

        if self.positional_mode is not None:
            self.de_positional_encoding = nn.ModuleList(
                [PositionEncodingSine(self.de_layer_dims[idx], positional_mode=self.positional_mode)
                 for idx in range(len(self.de_layer_dims))])
        self.de_layer1 = self._make_layer(self.de_layer_dims[0], len(self.de_layer_names))
        self.de_layers = nn.ModuleList(
            [self._make_layer(dim, len(self.de_layer_names)) for dim in self.de_layer_dims[1:]])

        # out_convs + cat_convs
        if self.is_out_convs:
            tmp = [conv1x1(self.encoder_dims[idx], self.de_layer_dims[idx]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(conv1x1(self.config['ENCODER']['STEM_DIMS2'][-1], self.de_layer_dims[-1]))
            self.out_convs = nn.ModuleList(tmp)
            tmp = [CNNBlock3x3(self.de_layer_dims[idx], self.de_layer_dims[idx + 1]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(CNNBlock3x3(self.de_layer_dims[-1], self.de_layer_dims[-1]))
            self.cat_convs = nn.ModuleList(tmp)
        else:
            # only cat_convs
            tmp = [CNNBlock3x3(self.encoder_dims[idx] + self.de_layer_dims[idx], self.de_layer_dims[idx + 1]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(
                CNNBlock3x3(self.config['ENCODER']['STEM_DIMS2'][-1] + self.de_layer_dims[-1], self.de_layer_dims[-1]))
            self.cat_convs = nn.ModuleList(tmp)

    def _make_layer(self, dim, N):
        layers = nn.ModuleList()
        block = {'default': TransformerBlock, 'semi': SemiCNNTransformerBlock,
                 'pure': PureCNNTransformerBlock, 'multi': MultiCNNTransformerBlock}
        for i in range(N):
            layer = block[self.transformer_type](dim, self.nhead, self.attention, self.FF_type)
            layers.append(layer)
        return layers

    def forward_one_en_layer(self, idx, feat0, feat1, mask0=None, mask1=None):
        # downsampling features
        if feat0.shape == feat1.shape:
            feats = self.downsamplings[idx](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.downsamplings[idx](feat0), self.downsamplings[idx](feat1)

        # downsampling masks
        if mask0 is not None:
            mask0 = F.interpolate(mask0.float().unsqueeze(1), size=(feat0.size(2), feat0.size(3))).bool().squeeze(1)
        if mask1 is not None:
            mask1 = F.interpolate(mask1.float().unsqueeze(1), size=(feat1.size(2), feat1.size(3))).bool().squeeze(1)

        # TransformerBlock process
        if self.positional_mode is not None and self.positional_pos == 'resolution':
            feat0, feat1 = self.en_positional_encoding[idx](feat0), self.en_positional_encoding[idx](feat1)
        for layer, name in zip(self.en_layers[idx], self.en_layer_names):
            if self.positional_mode is not None and self.positional_pos == 'layer':
                feat0, feat1 = self.en_positional_encoding[idx](feat0), self.en_positional_encoding[idx](feat1)
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1, mask0, mask1

    def forward_one_de_layer(self, idx, feat0, feat1, encoder_feat0, encoder_feat1, mask0=None, mask1=None):
        # upsampling features
        feat0, feat1 = F.interpolate(feat0, size=(encoder_feat0.size(2), encoder_feat0.size(3)), mode='bilinear',
                                     align_corners=True), F.interpolate(feat1, size=(
            encoder_feat1.size(2), encoder_feat1.size(3)), mode='bilinear', align_corners=True)

        # cat_convs + out_convs
        if self.is_out_convs:
            feat0 = feat0 + self.out_convs[idx](encoder_feat0)
            feat1 = feat1 + self.out_convs[idx](encoder_feat1)
        else:
            # only cat_convs
            feat0 = torch.cat([feat0, encoder_feat0], dim=1)
            feat1 = torch.cat([feat1, encoder_feat1], dim=1)
        if feat0.shape == feat1.shape:
            feats = self.cat_convs[idx](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.cat_convs[idx](feat0), self.cat_convs[idx](feat1)

        # upsampling masks
        if mask0 is not None:
            mask0 = F.interpolate(mask0.float().unsqueeze(1), size=(encoder_feat0.size(2), encoder_feat0.size(3)),
                                  mode='nearest').bool().squeeze(1)
        if mask1 is not None:
            mask1 = F.interpolate(mask1.float().unsqueeze(1), size=(encoder_feat1.size(2), encoder_feat1.size(3)),
                                  mode='nearest').bool().squeeze(1)

        # TransformerBlock process
        if self.positional_mode is not None and self.positional_pos == 'resolution':
            feat0, feat1 = self.de_positional_encoding[idx + 1](feat0), self.de_positional_encoding[idx + 1](feat1)
        for layer, name in zip(self.de_layers[idx], self.de_layer_names):
            if self.positional_mode is not None and self.positional_pos == 'layer':
                feat0, feat1 = self.de_positional_encoding[idx + 1](feat0), self.de_positional_encoding[idx + 1](feat1)
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1, mask0, mask1

    def forward(self, img0, img1, mask0=None, mask1=None):
        """
        Args:
            img0 (torch.Tensor): [N, C, H, W]
            img1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, H, W] (optional)
            mask1 (torch.Tensor): [N, H, W] (optional)
        Return:
            feat_c0 (torch.Tensor): [N, C, H/8, W/8]
            feat_f0 (torch.Tensor): [N, C, H/2, W/2]
            feat_c1 (torch.Tensor): [N, C, H/8, W/8]
            feat_f1 (torch.Tensor): [N, C, H/2, W/2]
        """
        # 1. encoder
        # 1.1 encoder stem
        if img0.shape == img1.shape:
            feats = self.stem(torch.cat([img0, img1], dim=0))
            feat0, feat1 = feats.split(img0.size(0))
        else:
            feat0, feat1 = self.stem(img0), self.stem(img1)
        stem_feat0, stem_feat1 = feat0.clone(), feat1.clone()
        # 1.2 encoder layer1 -- 1/4
        feat0, feat1, mask0, mask1 = self.forward_one_en_layer(0, feat0, feat1, mask0, mask1)
        en_feat0_0, en_feat1_0 = feat0.clone(), feat1.clone()
        # 1.3 encoder layer2 -- 1/8
        feat0, feat1, mask0, mask1 = self.forward_one_en_layer(1, feat0, feat1, mask0, mask1)
        en_feat0_1, en_feat1_1 = feat0.clone(), feat1.clone()
        # 1.4 encoder layer3 -- 1/16
        feat0, feat1, mask0, mask1 = self.forward_one_en_layer(2, feat0, feat1, mask0, mask1)

        # 2. decoder
        # 2.1 decoder layer1 -- 1/16
        # feat0, feat1 = self.de_positional_encoding[0](feat0), self.de_positional_encoding[0](feat1)
        for layer, name in zip(self.de_layer1, self.de_layer_names):
            # feat0, feat1 = self.de_positional_encoding[0](feat0), self.de_positional_encoding[0](feat1)
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        # feat_c0, feat_c1 = feat0.clone(), feat1.clone()
        # 2.2 decoder layer2 -- 1/8
        feat0, feat1, mask0, mask1 = self.forward_one_de_layer(0, feat0, feat1, en_feat0_1, en_feat1_1, mask0, mask1)
        feat_c0, feat_c1 = feat0.clone(), feat1.clone()
        # 2.3 decoder layer3 -- 1/4
        feat0, feat1, mask0, mask1 = self.forward_one_de_layer(1, feat0, feat1, en_feat0_0, en_feat1_0, mask0, mask1)
        # 2.4 upsample layer and cat stem features -- 1/2
        feat0, feat1 = F.interpolate(feat0, size=(stem_feat0.size(2), stem_feat0.size(3)), mode='bilinear',
                                     align_corners=True), F.interpolate(feat1,
                                                                        size=(stem_feat1.size(2), stem_feat1.size(3)),
                                                                        mode='bilinear', align_corners=True)
        # cat_convs + out_convs
        if self.is_out_convs:
            feat0 = feat0 + self.out_convs[-1](stem_feat0)
            feat1 = feat1 + self.out_convs[-1](stem_feat1)
        else:
            # only cat_convs
            feat0 = torch.cat([feat0, stem_feat0], dim=1)
            feat1 = torch.cat([feat1, stem_feat1], dim=1)
        if feat0.shape == feat1.shape:
            feats = self.cat_convs[-1](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.cat_convs[-1](feat0), self.cat_convs[-1](feat1)
        feat_f0, feat_f1 = feat0.clone(), feat1.clone()

        return feat_c0, feat_f0, feat_c1, feat_f1


class Backbone_vis(nn.Module):
    """
        Backbone including encoder and decoder part
    """

    def __init__(self, config):
        super().__init__()

        # 1. config
        self.config = config
        self.attention = self.config['ATTENTION']
        self.FF_type = self.config['FF_TYPE']
        self.positional_mode = self.config['POSITIONAL_MODE']
        self.positional_pos = self.config['POSITIONAL_POS']
        self.nhead = self.config['NHEAD']

        # 2. encoder
        self.en_layer_names = self.config['ENCODER']['LAYER_NAMES']
        self.en_layer_dims = self.config['ENCODER']['LAYER_DIMS']

        if self.positional_mode is not None:
            self.en_positional_encoding = nn.ModuleList(
                [PositionEncodingSine(self.en_layer_dims[idx], positional_mode=self.positional_mode)
                 for idx in range(len(self.en_layer_dims))])
        self.stem = Stem(self.config['ENCODER'])
        self.downsamplings = [ResBlock(self.config['ENCODER']['STEM_DIMS2'][-1], self.en_layer_dims[0], 2)]
        for idx in range(1, len(self.en_layer_dims)):
            self.downsamplings.append(ResBlock(self.en_layer_dims[idx - 1], self.en_layer_dims[idx], 2))
        self.downsamplings = nn.ModuleList(self.downsamplings)
        self.en_layers = nn.ModuleList([self._make_layer(dim, len(self.en_layer_names)) for dim in self.en_layer_dims])

        # 3. decoder
        self.de_layer_names = self.config['DECODER']['LAYER_NAMES']
        self.de_layer_dims = self.config['DECODER']['LAYER_DIMS']
        self.encoder_dims = self.config['DECODER']['ENCODER_LAYER_DIMS_INV']
        self.is_out_convs = self.config['DECODER']['IS_OUT_CONVS']

        if self.positional_mode is not None:
            self.de_positional_encoding = nn.ModuleList(
                [PositionEncodingSine(self.de_layer_dims[idx], positional_mode=self.positional_mode)
                 for idx in range(len(self.de_layer_dims))])
        self.de_layer1 = self._make_layer(self.de_layer_dims[0], len(self.de_layer_names))
        self.de_layers = nn.ModuleList(
            [self._make_layer(dim, len(self.de_layer_names)) for dim in self.de_layer_dims[1:]])

        # out_convs + cat_convs
        if self.is_out_convs:
            tmp = [conv1x1(self.encoder_dims[idx], self.de_layer_dims[idx]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(conv1x1(self.config['ENCODER']['STEM_DIMS2'][-1], self.de_layer_dims[-1]))
            self.out_convs = nn.ModuleList(tmp)
            tmp = [CNNBlock3x3(self.de_layer_dims[idx], self.de_layer_dims[idx + 1]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(CNNBlock3x3(self.de_layer_dims[-1], self.de_layer_dims[-1]))
            self.cat_convs = nn.ModuleList(tmp)
        else:
            # only cat_convs
            tmp = [CNNBlock3x3(self.encoder_dims[idx] + self.de_layer_dims[idx], self.de_layer_dims[idx + 1]) for idx in
                   range(len(self.de_layer_dims) - 1)]
            tmp.append(
                CNNBlock3x3(self.config['ENCODER']['STEM_DIMS2'][-1] + self.de_layer_dims[-1], self.de_layer_dims[-1]))
            self.cat_convs = nn.ModuleList(tmp)

    def _make_layer(self, dim, N):
        layers = nn.ModuleList()
        for i in range(N):
            layer = MultiCNNTransformerBlock_vis(dim, self.nhead, self.attention, self.FF_type)
            layers.append(layer)
        return layers

    def forward_one_en_layer(self, idx, feat0, feat1, mask0=None, mask1=None):
        # downsampling features
        if feat0.shape == feat1.shape:
            feats = self.downsamplings[idx](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.downsamplings[idx](feat0), self.downsamplings[idx](feat1)

        # downsampling masks
        if mask0 is not None:
            mask0 = F.interpolate(mask0.float().unsqueeze(1), size=(feat0.size(2), feat0.size(3))).bool().squeeze(1)
        if mask1 is not None:
            mask1 = F.interpolate(mask1.float().unsqueeze(1), size=(feat1.size(2), feat1.size(3))).bool().squeeze(1)

        # TransformerBlock process
        if self.positional_mode is not None and self.positional_pos == 'resolution':
            feat0, feat1 = self.en_positional_encoding[idx](feat0), self.en_positional_encoding[idx](feat1)
        attention_vis_list = []
        for layer, name in zip(self.en_layers[idx], self.en_layer_names):
            if self.positional_mode is not None and self.positional_pos == 'layer':
                feat0, feat1 = self.en_positional_encoding[idx](feat0), self.en_positional_encoding[idx](feat1)
            if name == 'self':
                feat0, attention_vis0 = layer(feat0, feat0, mask0, mask0)
                feat1, attention_vis1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, attention_vis0 = layer(feat0, feat1, mask0, mask1)
                feat1, attention_vis1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
            attention_vis_list.append((attention_vis0.detach().cpu().numpy(), attention_vis1.detach().cpu().numpy()))
        return feat0, feat1, mask0, mask1, attention_vis_list

    def forward_one_de_layer(self, idx, feat0, feat1, encoder_feat0, encoder_feat1, mask0=None, mask1=None):
        # upsampling features
        feat0, feat1 = F.interpolate(feat0, size=(encoder_feat0.size(2), encoder_feat0.size(3)), mode='bilinear',
                                     align_corners=True), F.interpolate(feat1, size=(
            encoder_feat1.size(2), encoder_feat1.size(3)), mode='bilinear', align_corners=True)

        # cat_convs + out_convs
        if self.is_out_convs:
            feat0 = feat0 + self.out_convs[idx](encoder_feat0)
            feat1 = feat1 + self.out_convs[idx](encoder_feat1)
        else:
            # only cat_convs
            feat0 = torch.cat([feat0, encoder_feat0], dim=1)
            feat1 = torch.cat([feat1, encoder_feat1], dim=1)
        if feat0.shape == feat1.shape:
            feats = self.cat_convs[idx](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.cat_convs[idx](feat0), self.cat_convs[idx](feat1)

        # upsampling masks
        if mask0 is not None:
            mask0 = F.interpolate(mask0.float().unsqueeze(1), size=(encoder_feat0.size(2), encoder_feat0.size(3)),
                                  mode='nearest').bool().squeeze(1)
        if mask1 is not None:
            mask1 = F.interpolate(mask1.float().unsqueeze(1), size=(encoder_feat1.size(2), encoder_feat1.size(3)),
                                  mode='nearest').bool().squeeze(1)

        # TransformerBlock process
        if self.positional_mode is not None and self.positional_pos == 'resolution':
            feat0, feat1 = self.de_positional_encoding[idx + 1](feat0), self.de_positional_encoding[idx + 1](feat1)
        attention_vis_list = []
        for layer, name in zip(self.de_layers[idx], self.de_layer_names):
            if self.positional_mode is not None and self.positional_pos == 'layer':
                feat0, feat1 = self.de_positional_encoding[idx + 1](feat0), self.de_positional_encoding[idx + 1](feat1)
            if name == 'self':
                feat0, attention_vis0 = layer(feat0, feat0, mask0, mask0)
                feat1, attention_vis1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, attention_vis0 = layer(feat0, feat1, mask0, mask1)
                feat1, attention_vis1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
            attention_vis_list.append((attention_vis0.detach().cpu().numpy(), attention_vis1.detach().cpu().numpy()))
        return feat0, feat1, mask0, mask1, attention_vis_list

    def forward(self, img0, img1, mask0=None, mask1=None):
        """
        Args:
            img0 (torch.Tensor): [N, C, H, W]
            img1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, H, W] (optional)
            mask1 (torch.Tensor): [N, H, W] (optional)
        Return:
            feat_c0 (torch.Tensor): [N, C, H/8, W/8]
            feat_f0 (torch.Tensor): [N, C, H/2, W/2]
            feat_c1 (torch.Tensor): [N, C, H/8, W/8]
            feat_f1 (torch.Tensor): [N, C, H/2, W/2]
        """
        # 1. encoder
        # 1.1 encoder stem
        if img0.shape == img1.shape:
            feats = self.stem(torch.cat([img0, img1], dim=0))
            feat0, feat1 = feats.split(img0.size(0))
        else:
            feat0, feat1 = self.stem(img0), self.stem(img1)
        stem_feat0, stem_feat1 = feat0.clone(), feat1.clone()
        attention_vis_lists = []
        # 1.2 encoder layer1 -- 1/4
        feat0, feat1, mask0, mask1, attention_vis_list = self.forward_one_en_layer(0, feat0, feat1, mask0, mask1)
        en_feat0_0, en_feat1_0 = feat0.clone(), feat1.clone()
        attention_vis_lists.append(attention_vis_list)
        # 1.3 encoder layer2 -- 1/8
        feat0, feat1, mask0, mask1, attention_vis_list = self.forward_one_en_layer(1, feat0, feat1, mask0, mask1)
        en_feat0_1, en_feat1_1 = feat0.clone(), feat1.clone()
        attention_vis_lists.append(attention_vis_list)
        # 1.4 encoder layer3 -- 1/16
        feat0, feat1, mask0, mask1, attention_vis_list = self.forward_one_en_layer(2, feat0, feat1, mask0, mask1)
        attention_vis_lists.append(attention_vis_list)

        # 2. decoder
        # 2.1 decoder layer1 -- 1/16
        # feat0, feat1 = self.de_positional_encoding[0](feat0), self.de_positional_encoding[0](feat1)
        attention_vis_list = []
        for layer, name in zip(self.de_layer1, self.de_layer_names):
            # feat0, feat1 = self.de_positional_encoding[0](feat0), self.de_positional_encoding[0](feat1)
            if name == 'self':
                feat0, attention_vis0 = layer(feat0, feat0, mask0, mask0)
                feat1, attention_vis1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, attention_vis0 = layer(feat0, feat1, mask0, mask1)
                feat1, attention_vis1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
            attention_vis_list.append((attention_vis0.detach().cpu().numpy(), attention_vis1.detach().cpu().numpy()))
        attention_vis_lists.append(attention_vis_list)
        # feat_c0, feat_c1 = feat0.clone(), feat1.clone()
        # 2.2 decoder layer2 -- 1/8
        feat0, feat1, mask0, mask1, attention_vis_list = self.forward_one_de_layer(0, feat0, feat1, en_feat0_1,
                                                                                   en_feat1_1, mask0, mask1)
        feat_c0, feat_c1 = feat0.clone(), feat1.clone()
        attention_vis_lists.append(attention_vis_list)
        # 2.3 decoder layer3 -- 1/4
        feat0, feat1, mask0, mask1, attention_vis_list = self.forward_one_de_layer(1, feat0, feat1, en_feat0_0,
                                                                                   en_feat1_0, mask0, mask1)
        attention_vis_lists.append(attention_vis_list)
        # 2.4 upsample layer and cat stem features -- 1/2
        feat0, feat1 = F.interpolate(feat0, size=(stem_feat0.size(2), stem_feat0.size(3)), mode='bilinear',
                                     align_corners=True), F.interpolate(feat1,
                                                                        size=(stem_feat1.size(2), stem_feat1.size(3)),
                                                                        mode='bilinear', align_corners=True)
        # cat_convs + out_convs
        if self.is_out_convs:
            feat0 = feat0 + self.out_convs[-1](stem_feat0)
            feat1 = feat1 + self.out_convs[-1](stem_feat1)
        else:
            # only cat_convs
            feat0 = torch.cat([feat0, stem_feat0], dim=1)
            feat1 = torch.cat([feat1, stem_feat1], dim=1)
        if feat0.shape == feat1.shape:
            feats = self.cat_convs[-1](torch.cat([feat0, feat1], dim=0))
            feat0, feat1 = feats.split(feat0.size(0))
        else:
            feat0, feat1 = self.cat_convs[-1](feat0), self.cat_convs[-1](feat1)
        feat_f0, feat_f1 = feat0.clone(), feat1.clone()

        return feat_c0, feat_f0, feat_c1, feat_f1, attention_vis_lists
