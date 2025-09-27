import torch
import torch.nn as nn
from model.modules.mlp import MLP
from model.modules.graph import GCN
from model.modules.cross_attention import FusionCrossAttention

class Fusion(nn.Module):
    """
    Implementation of Fusion block.
    """
    def __init__(self, args):
        super().__init__()
        self.fusion_methods = args.fusion_method
        """
            set different fusion methods for different setting
        """
        if self.fusion_methods == 'MLP':
            self.fusion_layer = MLP(
                in_features=args.fusion_dim*18,
                hidden_features=args.fusion_dim *18* 2,
                out_features=args.fusion_dim*18,
                act_layer=nn.GELU,
                drop=0.1,
                channel_first=False
            )
        elif self.fusion_methods == 'GCN':
            self.fusion_layer = GCN(
                args.fusion_dim,args.fusion_dim,num_nodes=args.num_joints+1, mode='spatial')
        elif self.fusion_methods == "CA":
            self.fusion_layer = FusionCrossAttention(
                dim_in=args.fusion_dim,
                dim_out=args.fusion_dim,
                num_heads=8,
                dropout=0.0,
            )
        elif self.fusion_methods == 'CAT' or self.fusion_methods == 'ADD':
            self.fusion_layer = None
        else:
            self.fusion_layer = None
            print(f"Using fusion method: {self.fusion_methods}, no specific implementation details provided.")
            exit()


        """
            todo: add more fusion methods like CA, GCN, etc.
        """

    def forward(self, x, fea):
        """
            many fusion methods
        """
        B, F, J, C = x.shape
        fea = fea.unsqueeze(2)
        if self.fusion_methods == 'CAT':
            out = torch.cat((x, fea), dim=2)
        elif self.fusion_methods == 'ADD':
            fea = fea.repeat(1, 1, J, 1)
            out = x + fea
        elif self.fusion_methods == 'MLP':
            out = torch.cat((x, fea), dim=2)
            out = out.view(B , F, -1)
            out = self.fusion_layer(out)
            out = out.view(B, F, J+1, C)
        elif self.fusion_methods == 'CA':
            x = x.reshape(B * F, J, C)
            fea = fea.unsqueeze(2).reshape(B * F, 1, C)
            out = self.fusion_layer(x, fea)
            out = out.reshape(B, F, J, C)
        elif self.fusion_methods == 'GCN':
            out = torch.cat((x, fea), dim=2)
            out = self.fusion_layer(out)
        """
            todo: add more fusion methods implementation details like CA, GCN, etc.
        """
        return out
