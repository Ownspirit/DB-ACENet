from collections import OrderedDict
from einops import rearrange

import torch
from torch import nn
from timm.models.layers import DropPath

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN
from .MotionAGFormer import *
from utils.learning import load_model
from .modules.hmr_model import FT_HMR_FEA
from .modules.hmr_block import Block
import copy

class Fusion(nn.Module):
    """
    Implementation of Fusion block.
    """
    def __init__(self, args):
        super().__init__()

        self.mlp_expand = nn.Sequential(
            nn.Linear(args.dim_in, args.fusion_dim ),
            nn.GELU()
        )
        self.mlp_reduce = nn.Sequential(
            nn.Linear(args.fusion_dim*9, args.fusion_dim*6),
            nn.GELU(),
            nn.Linear(args.fusion_dim*6, args.fusion_dim*3),
            nn.GELU(),
            nn.Linear(args.fusion_dim*3, args.fusion_dim),
            nn.GELU()
        )

        self.pose_embed = load_model(args)
        self.AGformer = load_model(args, node_flags=True)
        self.rgb_embed = FT_HMR_FEA(return_embed=True)
        fusion_depth = args.fusion_depth
        fusion_dim = args.fusion_dim
        self.fusion_depth = fusion_depth
        self.out_head = nn.Linear(128, 3)

    def add_hypothetical_coords_dynamic(self, fea_2d, delta_x=0.9, delta_y=0.2):
        # fea_2d: [B, T, J, 3]，3=[x, y, conf]
        B, T, J, C = fea_2d.shape
        device = fea_2d.device
        xy = fea_2d[..., :2]  # [B, T, J, 2]
        conf = fea_2d[..., 2]  # [B, T, J]

        # 置信度区间掩码
        mask_high = (conf > 0.8)
        mask_mid = (conf > 0.5) & (conf <= 0.8)
        mask_low = (conf <= 0.5)

        # 9个偏移模板
        offsets_high = torch.zeros((9, 2), device=device)  # 全部为0
        offsets_mid = torch.tensor([
            [0, 0], [0, delta_y], [0, 0], [-delta_x, 0], [0, 0], [delta_x, 0],
            [0, 0], [0, -delta_y], [0, 0]
        ], device=device)
        offsets_low = torch.tensor([
            [-delta_x, delta_y], [0, delta_y], [delta_x, delta_y], [-delta_x, 0], [0, 0],
            [delta_x, 0], [-delta_x, -delta_y], [0, -delta_y], [delta_x, -delta_y]
        ], device=device)

        # 展开原始点
        xy_exp = xy.unsqueeze(-2).expand(-1, -1, -1, 9, -1)  # [B, T, J, 9, 2]
        conf_exp = conf.unsqueeze(-1).expand(-1, -1, -1, 9)  # [B, T, J, 9]

        # 只扩展到 [B, T, J, 9]
        mask_high_idx = mask_high.unsqueeze(-1).expand(-1, -1, -1, 9)
        mask_mid_idx = mask_mid.unsqueeze(-1).expand(-1, -1, -1, 9)
        mask_low_idx = mask_low.unsqueeze(-1).expand(-1, -1, -1, 9)

        # 生成新坐标
        hypo_xy = torch.zeros_like(xy_exp)  # [B, T, J, 9, 2]
        # 用 ... 取所有后续维度
        a = xy[mask_low].unsqueeze(-2)
        b = (a + offsets_low)
        c = b.detach().cpu().numpy()
        hypo_xy[mask_high_idx, :] = (xy[mask_high].unsqueeze(-2) + offsets_high).reshape(-1, 2)
        hypo_xy[mask_mid_idx, :] = (xy[mask_mid].unsqueeze(-2) + offsets_mid).reshape(-1, 2)
        hypo_xy[mask_low_idx, :] = (xy[mask_low].unsqueeze(-2) + offsets_low).reshape(-1, 2)

        # 拼接置信度
        hypo_conf = conf_exp.unsqueeze(-1)  # [B, T, J, 9, 1]
        hypo = torch.cat([hypo_xy, hypo_conf], dim=-1)  # [B, T, J, 9, 3]

        return hypo

    def forward(self, x, fea):
        bs, T, J, C  = x.shape

        x_win = self.add_hypothetical_coords_dynamic(x)  # [B, T, J, 9, 3]
        x_win = self.mlp_expand(x_win)  # [bs, T, J, 9, fusion_dim]
        # 展开最后两维
        x_win = x_win.reshape(bs, T, J, -1)  # [bs, T, J, fusion_dim*9]
        x_win = self.mlp_reduce(x_win)  # [bs, T, J, 128]

        fea_2d = self.pose_embed(x_win, return_rep=True)
        fea_rgb = self.rgb_embed(fea)
        fea_rgb = fea_rgb.unsqueeze(2)
        
        x = torch.cat((fea_2d, fea_rgb), dim=-2)
        bs, T, J, C  = x.shape
        
        # for i in range(self.fusion_depth):
        #     x = rearrange(x, 'b f n cw -> (b f) n cw')
        #     steblock = self.STEblocks[i]
        #     tteblock = self.TTEblocks[i]
        #     x = steblock(x)
        #     x = self.Spatial_norm(x)
        #     x = rearrange(x, '(b f) n cw -> (b n) f cw', f=T)
        #     x = tteblock(x)
        #     x = self.Temporal_norm(x)
        #     x = rearrange(x, '(b n) f cw -> b f n cw', n=J)
        x = self.AGformer(x)


        out = x[:, :, :17, :]
        #out = self.out_head(x)
        # out_fea = self.fusion_embed(x)
        return out



