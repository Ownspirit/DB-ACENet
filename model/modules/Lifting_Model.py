import torch
import torch.nn as nn
from utils.learning import load_model
class Lifting_Model(nn.Module):
    """
    Initialization of Lifting Model.
    """
    def __init__(self, args):
        super().__init__()
        self.lifting_type = args.lifting_type
        # if self.lifting_type == 'motionagformer':
        if args.fusion_method in ['CA', 'ADD']:
            self.lifting_model = load_model(args, node_flags=False)
        elif args.fusion_method in ['MLP', 'GCN', 'CAT']:
            self.lifting_model = load_model(args, node_flags=True)
        else:
            raise ValueError(f"Unsupported lifting type: {self.lifting_type}")

    def forward(self, x):
        """
        Implementation of Lifting Model
        """
        out = self.lifting_model(x)
        return out