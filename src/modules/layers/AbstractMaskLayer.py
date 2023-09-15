import torch
from typing import List

from .AbstarctLayer import AbstractLayer

sigmoid_normalization:int = 0
sinus_normalization:int = 1

class AbstractMaskLayer(AbstractLayer):

    _parameters : torch.nn.Parameter
    def _mask(self):
        return self._scale()
    def _recalc_parameters(self):
        with torch.no_grad():
            if hasattr(self, '_parameters'):
                device = self._parameters.device
                ParametersCopy = self._parameters.clone()
                if ParametersCopy.size(0) < self._pixels:
                    Parameters = torch.normal(mean=0, std=10, size=(self._pixels, self._pixels), dtype=self._accuracy.tensor_float)
                    nx1, nx2 = int(self._pixels/2 - ParametersCopy.size()[0]/2), int(self._pixels/2 + ParametersCopy.size()[0]/2)
                    ny1, ny2 = int(self._pixels/2 - ParametersCopy.size()[1]/2), int(self._pixels/2 + ParametersCopy.size()[1]/2)
                    if nx2 - nx1 < ParametersCopy.size()[0]: nx2 += 1
                    if ny2 - ny1 < ParametersCopy.size()[1]: ny2 += 1
                    Parameters[nx1:nx2, ny1:ny2] = ParametersCopy
                else:
                    nx1, nx2 = int(ParametersCopy.size()[0]/2 - self._PixelsCount/2), int(ParametersCopy.size()[0]/2 + self._pixels/2) + 1
                    ny1, ny2 = int(ParametersCopy.size()[1]/2 - self._PixelsCount/2), int(ParametersCopy.size()[1]/2 + self._pixels/2) + 1
                    if nx2 - nx1 > self._pixels: nx2 -= 1
                    if ny2 - ny1 > self._pixels: ny2 -= 1
                    Parameters = ParametersCopy[nx1:nx2, ny1:ny2]
                self._parameters = torch.nn.Parameter(Parameters.to(device))
            else:
                self._parameters = torch.nn.Parameter(torch.normal(mean=0, std=10, size=(self._pixels, self._pixels), dtype=self._accuracy.tensor_float))

    _normalization_type : int
    _normalization_parameters : List
    @property
    def normalization(self):
        class Selector:
            _self:AbstractMaskLayer
            def __init__(self, _self:AbstractMaskLayer):
                self._self = _self
            def get(self, as_int:bool=False):
                if as_int: return self._self._normalization_type
                elif self._self._normalization_type == sigmoid_normalization:
                    return 'sigmoid'
                elif self._self._normalization_type == sinus_normalization:
                    return 'sinus'
            def sigmoid(self):
                self._self._normalization_type = sigmoid_normalization
                self._self._normalization_parameters = []
            def sinus(self, period:float=100.0):
                self._self._normalization_type = sigmoid_normalization
                self._self._normalization_parameters = [period]
        return Selector(self)
    def _normalize(self):
        if self._normalization_type == sigmoid_normalization:
            return torch.sigmoid(self._parameters)
        elif self._normalization_type == sinus_normalization:
            return torch.sin(0.5*self._normalization_parameters[0]*self._parameters/torch.pi)
        else: raise Exception('Пока ты не лез, всё работало!')

    def _scale(self):
        return torch.repeat_interleave(torch.repeat_interleave(self._normalize(), self._up_scaling, dim=1), self._up_scaling, dim=0)

    _pixels : int
    @property
    def pixels(self):
        return self._pixels
    @pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._delayed.add(self._recalc_parameters)

    _up_scaling : int
    @property
    def up_scaling(self):
        return self._up_scaling
    @up_scaling.setter
    def up_scaling(self, calculating_pixels_per_pixel:int):
        self._up_scaling = calculating_pixels_per_pixel
        self._delayed.add(self._recalc_parameters)

    def __init__(self, pixels:int=20, up_scaling:int=8):
        super(AbstractMaskLayer, self).__init__()
        self.pixels = pixels
        self.up_scaling = up_scaling
        self._recalc_parameters()
        
    def forward(self, field:torch.Tensor):
        super(AbstractMaskLayer, self).forward(field)
        return field * self._mask()