import torch

from typing import Union
from copy import deepcopy

from .AbstractModel import AbstractModel

from ..layers.AmplificationLayer import AmplificationLayer
from ..layers.DetectorsLayer import DetectorsLayer
from ..layers.FourierPropagationLayer import FourierPropagationLayer
from ..layers.HeightMaskLayer import HeightMaskLayer

class RealSpaceD2NN(AbstractModel):

    _AmplificationModule    : AmplificationLayer
    _DetectorsModule        : DetectorsLayer
    _HeightMasksModuleList  : torch.nn.ModuleList[HeightMaskLayer]
    _PropagationModule      : FourierPropagationLayer


    _wavelength : torch.Tensor
    @property
    def wavelength(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._wavelength)
            def _synchronize(self, length:Union[torch.Tensor, float]):
                self._self._PropagationModule.wavelength = length
                for module in self._self._HeightMasksModuleList:
                    module.wavelength = length
            def __call__(self, length:float):
                self._synchronize(length)
            def range(self, length0:float, length1:float, N:int):
                wavelengths = torch.linspace(length0, length1, N, dtype=self._self._accuracy.tensor_float)
                self._synchronize(wavelengths)
        return Selector(self)


    _space_reflection : torch.Tensor
    @property
    def space_reflection(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._space_reflection)
            def _synchronize(self, reflection:Union[torch.Tensor, float]):
                self._self._PropagationModule.reflection = reflection
            def __call__(self, reflection:float):
                self._synchronize(reflection)
                for module in self._self._HeightMasksModuleList:
                    module.space_reflection = reflection
            def range(self, reflection0:float, reflection1:float, N:int):
                reflections = torch.linspace(reflection0, reflection1, N, dtype=self._self._accuracy.tensor_float)
                self._synchronize(reflections)
        return Selector(self)

    _mask_reflection : torch.Tensor


    _plane_length : float


    _pixels : int


    _up_scaling : int


    _distance : float


    _border : float


    def __init__(self):
        super(RealSpaceD2NN, self).__init__()

    def forward(self, field:torch.Tensor):
        return