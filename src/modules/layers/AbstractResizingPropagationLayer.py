import torch
from typing import Union, Iterable
from utilities.DecimalPrefixes import nm, mm
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from src.modules.layers.AbstractPropagationLayer import AbstractPropagationLayer

class AbstractResizingPropagationLayer(AbstractPropagationLayer):

    _in_plane_length : float
    @property
    def in_plane_length(self):
        return self._in_plane_length
    @in_plane_length.setter
    def in_plane_length(self, length:float):
        self._in_plane_length = length
        self._delayed.add(self._recalc_propagation_buffer)

    _in_pixels : int
    @property
    def in_pixels(self):
        return self._pixels
    @in_pixels.setter
    def in_pixels(self, amount:int):
        self._in_pixels = amount
        self._delayed.add(self._recalc_propagation_buffer)

    _in_up_scaling : int
    @property
    def in_up_scaling(self):
        return self._up_scaling
    @in_up_scaling.setter
    def in_up_scaling(self, calculating_pixels_per_pixel:int):
        self._in_up_scaling = calculating_pixels_per_pixel
        self._delayed.add(self._recalc_propagation_buffer)

    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        out_plane_length:float=1.0*mm,
                        out_pixels:int=20,
                        out_up_scaling:int=8,
                        in_plane_length: float = 1.0 * mm,
                        in_pixels: int = 20,
                        in_up_scaling: int = 8,
                        distance:float=20.0*mm):
        super().__init__()
        self.wavelength = wavelength
        self.reflection = reflection
        self.plane_length = out_plane_length
        self.pixels = out_pixels
        self.up_scaling = out_up_scaling
        self.in_plane_length = in_plane_length
        self.in_pixels = in_pixels
        self.in_up_scaling = in_up_scaling
        self.distance = distance
        self.delayed.finalize()

    def forward(self, field:torch.Tensor):
        return super().forward(field)