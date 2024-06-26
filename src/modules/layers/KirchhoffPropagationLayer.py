import torch
from typing import Union, Iterable
from ...utilities.DecimalPrefixes import nm, mm

from .AbstractPropagationLayer import AbstractPropagationLayer

class KirchhoffPropagationLayer(AbstractPropagationLayer):

    def _recalc_propagation_buffer(self):
        # mesh = torch.linspace(0, self._plane_length, self._pixels*self._up_scaling)
        # x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        # R = torch.sqrt(self._distance**2 + x_grid**2 + y_grid**2)
        # wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
        # space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0)
        # propagator = (self._distance / R**2)+(1.0/(2*torch.pi*R) + 1.0/(1j*wave_length*space_reflection))*torch.exp(2j*torch.pi*R/(wave_length*space_reflection))*(self._plane_length/(self._pixels*self._up_scaling))**2
        # PropagationBuffer = torch.nn.functional.pad(propagator, mode='reflect', pad=(self._pixels*self._up_scaling-1, 0, self._pixels*self._up_scaling-1, 0)).unsqueeze(1).to(self._accuracy.tensor_complex)

        mesh = torch.linspace(-self._plane_length, +self._plane_length, 2*self._pixels*self._up_scaling-1)
        x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        g = torch.sqrt(self._distance**2 + x_grid**2 + y_grid**2)
        wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
        space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0)
        PropagationBuffer = (2*self._plane_length/(2*self._pixels*self._up_scaling-1))**2 * torch.exp(self._distance*2.0j*torch.pi/(space_reflection*wave_length)) * torch.exp(g*2.0j*torch.pi/(space_reflection * wave_length)) / (g*1j*wave_length).unsqueeze(1).to(self._accuracy.tensor_complex)

        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device
            self.register_buffer('_propagation_buffer', PropagationBuffer.to(device))
        else:
            self.register_buffer('_propagation_buffer', PropagationBuffer)


    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        plane_length:float=1.0*mm,
                        pixels:int=20,
                        up_scaling:int=8,
                        distance:float=20.0*mm):
        super(KirchhoffPropagationLayer, self).__init__()
        self.wavelength = wavelength
        self.reflection = reflection
        self.plane_length = plane_length
        self.pixels = pixels
        self.up_scaling = up_scaling
        self.distance = distance
        self._recalc_propagation_buffer()
        
    def forward(self, field:torch.Tensor):
        super(KirchhoffPropagationLayer, self).forward(field)
        field = torch.nn.functional.conv2d(field, self._propagation_buffer, padding=self._pixels*self._up_scaling-1, groups=self._propagation_buffer.size(0))
        return field