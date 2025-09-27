import torch
import torch.nn as nn
from model.modules.mlp import MLP
from .hmr_block import Mlp, Block


class FT_HMR_FEA(nn.Module):
    def __init__(self, num_frame=243, num_joints=17, embed_dim_ratio=128, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, return_embed=False):
        super(FT_HMR_FEA, self).__init__()

        self.num_frame = num_frame
        
        embed_dim = embed_dim_ratio
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim*8))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        norm_layer = nn.LayerNorm
        # self.RGB_TTEblocks = nn.Sequential(
        #         MLP(in_features=embed_dim*8, hidden_features=embed_dim*4, out_features=embed_dim*2, act_layer=nn.GELU, drop=drop_rate),
        #         Block(dim=embed_dim*2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
        # )

        self.RGB_TTEblocks = nn.Sequential(
                nn.Linear(embed_dim * 8, embed_dim * 4),
                nn.GELU(),
                Block(dim=embed_dim * 4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
                Block(dim=embed_dim*2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
        )

        # self.RGB_TTEblocks = nn.Sequential(
        #         Block(dim=embed_dim*8, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
        #         Block(dim=embed_dim*4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
        #         Block(dim=embed_dim*2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=True, currentdim=0, depth=2),
        # )
        self.out_head = Mlp(in_features=embed_dim, hidden_features=embed_dim*2, out_features=num_joints*3, act_layer=nn.GELU, drop=drop_rate)
        self.return_embed = return_embed

    def forward(self, x):
        bs, f, dim = x.shape
        x = x + self.Temporal_pos_embed
        fea = self.RGB_TTEblocks(x)
        if self.return_embed:
            return fea
        else:
            out = self.out_head(fea)
            return out.reshape(bs, f, 17, 3)
    

