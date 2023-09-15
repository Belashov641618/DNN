import torch

from modules.layers.old.AbstractLayer import AbstractLayer

class AmplificationLayer(AbstractLayer):

    _amplification : torch.nn.Parameter
    def __init__(self):
        super(AmplificationLayer, self).__init__()
        self._amplification = torch.nn.Parameter(torch.tensor([1.0], dtype=self._tensor_float_type))
    def forward(self, field:torch.Tensor):
        super(AmplificationLayer, self).forward(field)
        return field * self._amplification