import torch
from typing import Union, Iterable

from .AbstarctLayer import AbstractLayer

class AbstractPropagationLayer(AbstractLayer):

    _propagation_buffer : torch.Tensor
    @property
    def propagation_buffer(self):
        class Selector:
            _self:AbstractPropagationLayer
            def __init__(self, _self:AbstractPropagationLayer):
                self._self = _self
            def get(self, to_cpu:bool=True):
                self._self.delayed.finalize()
                if to_cpu:
                    return self._self._propagation_buffer.clone().detach().cpu()
                return self._self._propagation_buffer.clone().detach()
        return Selector(self)
    def _recalc_propagation_buffer(self):
        mesh = torch.linspace(0, self._plane_length, self._pixels*self._up_scaling)
        x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        R = torch.sqrt(self._distance**2 + x_grid**2 + y_grid**2)
        wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
        space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0)
        propagator = (self._distance / R**2)+(1.0/(2*torch.pi*R) + 1.0/(1j*wave_length*space_reflection))*torch.exp(2j*torch.pi*R/(wave_length*space_reflection))*(self._plane_length/(self._pixels*self._up_scaling))**2
        PropagationBuffer = torch.nn.functional.pad(propagator, mode='reflect', pad=(self._pixels*self._up_scaling-1, 0, self._pixels*self._up_scaling-1, 0)).unsqueeze(1).to(self._accuracy.tensor_complex)

        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device
            self.register_buffer('_propagation_buffer', PropagationBuffer.to(device))
        else:
            self.register_buffer('_propagation_buffer', PropagationBuffer)


    _wavelength : torch.Tensor
    @property
    def wavelength(self):
        return self._wavelength
    @wavelength.setter
    def wavelength(self, length:Union[float,Iterable,torch.Tensor]):
        self._wavelength = (length.to(self._accuracy.tensor_float).requires_grad_(False) if torch.is_tensor(length) else torch.tensor([length] if type(length) is float else length, requires_grad=False, dtype=self._accuracy.tensor_float))
        self._delayed.add(self._recalc_propagation_buffer)

    _plane_length : float
    @property
    def plane_length(self):
        return self._plane_length
    @plane_length.setter
    def plane_length(self, length:float):
        self._plane_length = length
        self._delayed.add(self._recalc_propagation_buffer)

    _pixels : int
    @property
    def pixels(self):
        return self._pixels
    @pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._delayed.add(self._recalc_propagation_buffer)

    _up_scaling : int
    @property
    def up_scaling(self):
        return self._up_scaling
    @up_scaling.setter
    def up_scaling(self, calculating_pixels_per_pixel:int):
        self._up_scaling = calculating_pixels_per_pixel
        self._delayed.add(self._recalc_propagation_buffer)

    _distance : float
    @property
    def distance(self):
        return self._distance
    @distance.setter
    def distance(self, diffraction_length:float):
        self._distance = diffraction_length
        self._delayed.add(self._recalc_propagation_buffer)

    _reflection : torch.Tensor
    @property
    def reflection(self):
        return self._reflection
    @reflection.setter
    def reflection(self, space_reflection:Union[float,Iterable,torch.Tensor]):
        self._reflection = (space_reflection.to(self._accuracy.tensor_complex).requires_grad_(False) if torch.is_tensor(space_reflection) else torch.tensor([space_reflection] if type(space_reflection) is float  else space_reflection, requires_grad=False, dtype=self._accuracy.tensor_complex))
        if self._reflection.size() != self._wavelength.size():
            if self._reflection.size(0) == 1:
                self._reflection = self._reflection.repeat(self._wavelength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        self._delayed.add(self._recalc_propagation_buffer)


    def __init__(self):
        super(AbstractPropagationLayer, self).__init__()

    def forward(self, field:torch.Tensor):
        super(AbstractPropagationLayer, self).forward(field)
        return field