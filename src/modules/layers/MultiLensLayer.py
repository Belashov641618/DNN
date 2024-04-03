import torch
import numpy
from typing import Union, Iterable

from ...utilities.DecimalPrefixes import mm, nm
from .LensLayer import LensLayer
from .AbstractPropagationLayer import AbstractPropagationLayer


class MultiLensLayer(LensLayer):

    def _recalc_propagation_buffer(self):
        central_points = []
        for x in numpy.linspace(0.5*self._plane_length/self._lenses, self._plane_length - 0.5*self.plane_length/self._lenses, self._lenses):
            for y in numpy.linspace(0.5*self._plane_length/self._lenses, self._plane_length - 0.5*self.plane_length/self._lenses, self._lenses):
                central_points.append([x,y])

        density = self._pixels * self._up_scaling / self._plane_length
        border_points = [(cx - self._plane_length/self._lenses / 2, cy - self._plane_length/self._lenses / 2, cx + self._plane_length/self._lenses / 2, cy + self._plane_length/self._lenses / 2) for (cx, cy) in central_points]
        border_pixels = [(int(x1 * density), int(y1 * density), int(x2 * density), int(y2 * density)) for (x1, y1, x2, y2) in border_points]

        density = self._plane_length / (self._pixels * self._up_scaling)

        propagation_buffer = torch.ones((self._pixels * self._up_scaling, self._pixels * self._up_scaling), dtype=self._accuracy.tensor_complex)
        for (nx1, ny1, nx2, ny2) in border_pixels:
            nX = nx2 - nx1
            nY = ny2 - ny1
            lengthX = nX * density
            lengthY = nY * density
            rangeX = torch.linspace(-lengthX/2, +lengthX/2, nX)
            rangeY = torch.linspace(-lengthY/2, +lengthY/2, nY)
            meshX, meshY = torch.meshgrid(rangeX, rangeY, indexing='ij')
            propagation_buffer[nx1:nx2, ny1:ny2] = torch.exp(-1j*torch.pi*(meshX**2 + meshY**2)/(self._wavelength*self._focus)).to(dtype=self._accuracy.tensor_complex)

        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device
            self.register_buffer('_propagation_buffer', propagation_buffer.to(device))
        else:
            self.register_buffer('_propagation_buffer', propagation_buffer)

    _lenses : int
    @property
    def lenses(self):
        return self._lenses
    @lenses.setter
    def lenses(self, amount:int):
        self._lenses = amount
        self._delayed.add(self._recalc_propagation_buffer)

    def __init__(self, lenses:int=2, focus:float=10*mm, wave_length:Union[float,torch.Tensor,Iterable]=600*nm, plane_length:float=1.0*mm, pixels:int=21, up_scaling:int=20):
        AbstractPropagationLayer.__init__(self)

        self.focus = focus
        self.wavelength = wave_length
        self.plane_length = plane_length
        self.pixels = pixels
        self.up_scaling = up_scaling
        self.lenses = lenses

        self._recalc_propagation_buffer()

    def forward(self, field:torch.Tensor):
        AbstractPropagationLayer.forward(self, field)
        return field * self._propagation_buffer