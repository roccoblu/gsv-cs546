import torch
import torch.nn.functional as F
import torch.nn as nn

class GAPool(nn.Module):
    """Implementation of GAP
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)
