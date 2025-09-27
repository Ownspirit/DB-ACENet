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
        #self.rgb_embed = FT_HMR_FEA(return_embed=True)
        self.rgb_embed = FT_HMR_FEA(return_embed=False)

    def forward(self,fea):
        bs, T, C  = fea.shape
        fea_rgb = self.rgb_embed(fea)
        #out = out[:, :, :17, :]  # Ensure output shape matches expected
        return fea_rgb

