import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FusionCrossAttention(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads= num_heads
        self.joint_norm = nn.LayerNorm(dim_in)
        self.rgb_norm = nn.LayerNorm(dim_in)
        self.query = nn.Linear(dim_in, dim_in)
        self.key = nn.Linear(dim_in, dim_in)
        self.value = nn.Linear(dim_in, dim_in)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(dim_in, dim_out)

    def forward(self, fea_2d, fea_rgb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = fea_2d.shape
        N = fea_rgb.shape[1]
        H = self.num_heads
        query = self.query(self.joint_norm(fea_2d)).unsqueeze(2)
        key = self.key(self.rgb_norm(fea_rgb)).unsqueeze(1)
        key = key.repeat(int(B/key.shape[0]), 1, 1, 1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.rgb_norm(fea_rgb)).unsqueeze(1)
        value = value.repeat(int(B/value.shape[0]), 1, 1, 1)
        value = value.view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        y = fea_2d + self.proj_out(y)
        return y