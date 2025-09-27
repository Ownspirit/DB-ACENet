import torch
import torch.nn as nn
from .modules.PAEB import PAEB
from .modules.Fusion import Fusion
from .modules.Lifting_Model import Lifting_Model
from .modules.hmr_model import FT_HMR_FEA

class DBACENet(nn.Module):
    """
        Initialization of DBACENet for monocular human 3D pose estimation
    """
    def __init__(self, args):

        super().__init__()
        self.PAEB = PAEB(args)
        self.Lifting_Model = Lifting_Model(args)

    def forward(self, x, fea):
        bs, T, J, C  = x.shape
        fea_2d = self.PAEB(x)
        fea_rgb = self.rgb_embed(fea)
        fea_fusion = self.Fusion(fea_2d, fea_rgb)
        out = self.Lifting_Model(fea_fusion)
        out = out[:, :, :17, :]  # Ensure output shape matches expected
        return out

