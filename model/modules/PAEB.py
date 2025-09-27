from einops import rearrange

import torch
from torch import nn

from model.modules.attention import Attention
from model.modules.mlp import MLP
from model.modules.hmr_block import Block

class PAEB(nn.Module):
    """
    Implementation of Pose-Adaptive Enhanced Block.
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
        self.STE_depth = args.STE_depth
        fusion_dim = args.fusion_dim

        self.STEblock = Block(dim=fusion_dim, num_heads=8, mlp_ratio=2., qkv_bias=False)
        self.Spatial_norm = nn.LayerNorm(fusion_dim)
        self.TTEblock = Block(dim=fusion_dim, num_heads=8, mlp_ratio=2., qkv_bias=False)
        self.Temporal_norm = nn.LayerNorm(fusion_dim)

        # self.STEblocks = nn.ModuleList([
        #     Block(dim=fusion_dim, num_heads=8, mlp_ratio=2., qkv_bias=False) for i in range(args.STE_depth)
        # ])
        # self.Spatial_norm = nn.LayerNorm(fusion_dim)
        # self.TTEblocks = nn.ModuleList([
        #     Block(dim=fusion_dim, num_heads=8, mlp_ratio=2., qkv_bias=False) for i in range(args.STE_depth)
        # ])
        # self.Temporal_norm = nn.LayerNorm(fusion_dim)

        # self.out_head = nn.Linear(128, 3)
        self.delta_x = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.delta_y = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_hypothetical_coords_dynamic(self, fea_2d, delta_x=0.9, delta_y=0.2):
        # fea_2d: [B, T, J, 3]，3=[x, y, conf]
        B, T, J, C = fea_2d.shape
        device = fea_2d.device
        xy = fea_2d[..., :2]  # [B, T, J, 2]
        conf = fea_2d[..., 2]  # [B, T, J]

        
        mask_high = (conf > 0.8)
        mask_mid = (conf > 0.5) & (conf <= 0.8)
        mask_low = (conf <= 0.5)

        
        offsets_high = torch.zeros((9, 2), device=device)  
        offsets_mid = torch.tensor([
            [0, 0], [0, delta_y], [0, 0], [-delta_x, 0], [0, 0], [delta_x, 0],
            [0, 0], [0, -delta_y], [0, 0]
        ], device=device)
        offsets_low = torch.tensor([
            [-delta_x, delta_y], [0, delta_y], [delta_x, delta_y], [-delta_x, 0], [0, 0],
            [delta_x, 0], [-delta_x, -delta_y], [0, -delta_y], [delta_x, -delta_y]
        ], device=device)

        
        xy_exp = xy.unsqueeze(-2).expand(-1, -1, -1, 9, -1)  # [B, T, J, 9, 2]
        conf_exp = conf.unsqueeze(-1).expand(-1, -1, -1, 9)  # [B, T, J, 9]

        # expand [B, T, J, 9]
        mask_high_idx = mask_high.unsqueeze(-1).expand(-1, -1, -1, 9)
        mask_mid_idx = mask_mid.unsqueeze(-1).expand(-1, -1, -1, 9)
        mask_low_idx = mask_low.unsqueeze(-1).expand(-1, -1, -1, 9)

        
        hypo_xy = torch.zeros_like(xy_exp)  # [B, T, J, 9, 2]
        
        a = xy[mask_low].unsqueeze(-2)
        b = (a + offsets_low)
        c = b.detach().cpu().numpy()
        hypo_xy[mask_high_idx, :] = (xy[mask_high].unsqueeze(-2) + offsets_high).reshape(-1, 2)
        hypo_xy[mask_mid_idx, :] = (xy[mask_mid].unsqueeze(-2) + offsets_mid).reshape(-1, 2)
        hypo_xy[mask_low_idx, :] = (xy[mask_low].unsqueeze(-2) + offsets_low).reshape(-1, 2)

        
        hypo_conf = conf_exp.unsqueeze(-1)  # [B, T, J, 9, 1]
        hypo = torch.cat([hypo_xy, hypo_conf], dim=-1)  # [B, T, J, 9, 3]

        return hypo

    def forward(self, x):
        bs, T, J, C  = x.shape
        # x_win = self.add_hypothetical_coords_dynamic(x)  # [B, T, J, 9, 3]

        # x_win = self.add_hypothetical_coords_dynamic(x,self.delta_x,self.delta_y)  # [B, T, J, 9, 3]
        x_win = self.add_hypothetical_coords_dynamic(x, delta_x=self.delta_x, delta_y=self.delta_y)
        x_win = self.mlp_expand(x_win)  # [bs, T, J, 9, fusion_dim]
        
        x_win = x_win.reshape(bs, T, J, -1)  # [bs, T, J, fusion_dim*9]
        x_win = self.mlp_reduce(x_win)  # [bs, T, J, 128]

        x = rearrange(x_win, 'b f n cw -> (b f) n cw')
        x = self.Spatial_norm(self.STEblock(x))
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=T)
        x = self.Temporal_norm(self.TTEblock(x))
        x = rearrange(x, '(b n) f cw -> b f n cw', n=J)

         # STE + TTE blocks
        # for i in range(self.STE_depth):
        #     x = rearrange(x_win, 'b f n cw -> (b f) n cw')
        #     steblock = self.STEblocks[i]
        #     tteblock = self.TTEblocks[i]
        #     x = steblock(x)
        #     x = self.Spatial_norm(x)
        #     x = rearrange(x, '(b f) n cw -> (b n) f cw', f=T)
        #     x = tteblock(x)
        #     x = self.Temporal_norm(x)
        #     x = rearrange(x, '(b n) f cw -> b f n cw', n=J)
        
        return x


