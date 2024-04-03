import torch
from typing import Union, Iterable
from ...utilities.DecimalPrefixes import nm, mm
from copy import deepcopy

from .AbstractPropagationLayer import AbstractPropagationLayer
from .KirchhoffPropagationLayer import KirchhoffPropagationLayer

class FourierPropagationLayer(AbstractPropagationLayer):

    def _recalc_propagation_buffer(self):
        device = torch.device('cpu')
        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device

        fx = torch.fft.fftshift(torch.fft.fftfreq(self._pixels*self._up_scaling + 2*self._border_pixels, d=self._plane_length / (self._pixels*self._up_scaling), device=device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(self._pixels*self._up_scaling + 2*self._border_pixels, d=self._plane_length / (self._pixels*self._up_scaling), device=device))
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0).to(device)
        space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0).to(device)
        Kz = ((2 * torch.pi) * torch.sqrt(0j + (1.0 / (wave_length * space_reflection)) ** 2 - fxx ** 2 - fyy ** 2)).to(dtype=self._accuracy.tensor_complex)

        self.register_buffer('_propagation_buffer', torch.exp(1.0j * Kz * self._distance))

    _border_pixels : int
    @property
    def border_pixels(self):
        class Selector:
            _self:FourierPropagationLayer
            def __init__(self, _self:FourierPropagationLayer):
                self._self = _self
            def get(self):
                return deepcopy(self._self._border_pixels)
        return Selector(self)
    def _recalc_border_pixels(self):
        self._border_pixels = int(self._border * self._pixels * self._up_scaling / self._plane_length)

    @KirchhoffPropagationLayer.plane_length.setter
    def plane_length(self, length:float):
        self._plane_length = length
        self._delayed.add(self._recalc_propagation_buffer, 1.0)
        self._delayed.add(self._recalc_border_pixels, 0.0)

    @KirchhoffPropagationLayer.pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._delayed.add(self._recalc_propagation_buffer, 1.0)
        self._delayed.add(self._recalc_border_pixels, 0.0)

    @KirchhoffPropagationLayer.up_scaling.setter
    def up_scaling(self, calculating_pixels_per_pixel:int):
        self._up_scaling = calculating_pixels_per_pixel
        self._delayed.add(self._recalc_propagation_buffer, 1.0)
        self._delayed.add(self._recalc_border_pixels, 0.0)

    _border : float
    @property
    def border(self):
        return self._border
    @border.setter
    def border(self, length:float):
        self._border = length
        self._delayed.add(self._recalc_propagation_buffer, 1.0)
        self._delayed.add(self._recalc_border_pixels, 0.0)

    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        plane_length:float=1.0*mm,
                        pixels:int=20,
                        up_scaling:int=8,
                        distance:float=20.0*mm,
                        border:float=0.0*mm):
        super().__init__()
        self.wavelength = wavelength
        self.reflection = reflection
        self.plane_length = plane_length
        self.pixels = pixels
        self.up_scaling = up_scaling
        self.distance = distance
        self.border = border
        self._recalc_border_pixels()
        self._recalc_propagation_buffer()

    def forward(self, field:torch.Tensor):
        super().forward(field)

        field = torch.nn.functional.pad(field, (+self._border_pixels, +self._border_pixels, +self._border_pixels, +self._border_pixels))
        field = torch.fft.fftshift(torch.fft.fft2(field))
        field = torch.fft.ifft2(torch.fft.ifftshift(field * self._propagation_buffer))
        field = torch.nn.functional.pad(field, (-self._border_pixels, -self._border_pixels, -self._border_pixels, -self._border_pixels))

        return field