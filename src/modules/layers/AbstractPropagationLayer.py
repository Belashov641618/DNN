import torch

from .AbstarctLayer import AbstractLayer

class AbstractPropagationLayer(AbstractLayer):

    def __init__(self):
        super(AbstractPropagationLayer, self).__init__()

    def forward(self, field:torch.Tensor):
        super(AbstractPropagationLayer, self).forward(field)
        return field