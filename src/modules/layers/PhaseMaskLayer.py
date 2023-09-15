import torch

from .AbstractMaskLayer import AbstractMaskLayer

class PhaseMaskLayer(AbstractMaskLayer):

    def _mask(self):
        return torch.exp(2.0j*self._scale()*torch.pi)

    def __init__(self, pixels:int=20, up_scaling:int=8):
        super(PhaseMaskLayer, self).__init__(pixels=pixels, up_scaling=up_scaling)