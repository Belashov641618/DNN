import torch

from ..layers.AbstarctLayer import AbstractLayer

class AbstractModel(AbstractLayer):

    def finalize(self):
        return

    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward(self, field:torch.Tensor):
        super(AbstractModel, self).forward(field)