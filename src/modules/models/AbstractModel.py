import torch

from src.modules.layers.AbstarctLayer import AbstractLayer

class AbstractModel(AbstractLayer):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward(self, field:torch.Tensor):
        super(AbstractModel, self).forward(field)