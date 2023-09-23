import torch
from typing import Union, Iterable

from src.utilities.DecimalPrefixes import mm, nm
from src.modules.layers.AbstractPropagationLayer import AbstractPropagationLayer


class LensLayer(AbstractPropagationLayer):

    def _recalc_propagation_buffer(self):
        wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
        mesh = torch.linspace(-self._plane_length/2, +self._plane_length/2, int(self._pixels*self._up_scaling))
        x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        PhaseBuffer = torch.exp(-1j*torch.pi*(x_grid**2 + y_grid**2)/(wave_length*self._focus)).to(dtype=self._accuracy.tensor_complex)

        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device
            self.register_buffer('_propagation_buffer', PhaseBuffer.to(device))
        else:
            self.register_buffer('_propagation_buffer', PhaseBuffer)


    _focus : float
    @property
    def focus(self):
        return self._focus
    @focus.setter
    def focus(self, length:float):
        self._focus = length
        self._delayed.add(self._recalc_propagation_buffer)


    def __init__(self, focus:float=10*mm, wave_length:Union[float,torch.Tensor,Iterable]=600*nm, plane_length:float=1.0*mm, pixels:int=21, up_scaling:int=20):
        super(LensLayer, self).__init__()

        self._focus = focus
        self.wavelength = wave_length
        self.plane_length = plane_length
        self.pixels = pixels
        self.up_scaling = up_scaling
        self._recalc_propagation_buffer()

    def forward(self, field:torch.Tensor):
        super(LensLayer, self).forward(field)
        return field * self._propagation_buffer
        