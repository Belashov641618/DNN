import torch
from typing import Union, List, Tuple
from copy import deepcopy

from utilities.DecimalPrefixes import nm, um, mm, cm
from utilities.Formaters import Format

from src.modules.models.AbstractModel import AbstractModel

from modules.layers.AmplificationLayer import AmplificationLayer
from modules.layers.DetectorsLayer import DetectorsLayer
from modules.layers.FourierPropagationLayer import FourierPropagationLayer
from modules.layers.HeightMaskLayer import HeightMaskLayer

from src.modules.layers.AbstractMaskLayer import sigmoid_normalization, sinus_normalization

class RealSpaceD2NN(AbstractModel):

    _AmplificationModule    : AmplificationLayer
    _DetectorsModule        : DetectorsLayer
    _HeightMasksModuleList  : torch.nn.ModuleList
    _PropagationModule      : FourierPropagationLayer

    def finalize(self):
        self._AmplificationModule.delayed.finalize()
        self._DetectorsModule.delayed.finalize()
        self._PropagationModule.delayed.finalize()
        for module in self._HeightMasksModuleList:
            if isinstance(module, HeightMaskLayer):
                module.delayed.finalize()

    @AbstractModel.accuracy.getter
    def accuracy(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def bits16(self):
                self._self._accuracy.set(bits=16)
                self._self._AmplificationModule.accuracy.bits16()
                self._self._DetectorsModule.accuracy.bits16()
                self._self._PropagationModule.accuracy.bits16()
                for module in self._self._HeightMasksModuleList:
                    module.accuracy.bits16()
            def bits32(self):
                self._self._accuracy.set(bits=32)
                self._self._AmplificationModule.accuracy.bits32()
                self._self._DetectorsModule.accuracy.bits32()
                self._self._PropagationModule.accuracy.bits32()
                for module in self._self._HeightMasksModuleList:
                    module.accuracy.bits32()
            def bits64(self):
                self._self._accuracy.set(bits=64)
                self._self._AmplificationModule.accuracy.bits64()
                self._self._DetectorsModule.accuracy.bits64()
                self._self._PropagationModule.accuracy.bits64()
                for module in self._self._HeightMasksModuleList:
                    module.accuracy.bits64()
            def get(self):
                return self._self._accuracy.get()
        return Selector(self)


    # Начало описания параметров
    _wavelength : torch.Tensor
    @property
    def wavelength(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    if isinstance(self._self._wavelength, torch.Tensor):
                        return "range", (torch.min(self._self._wavelength), torch.max(self._self._wavelength).item(), self._self._wavelength.numel())
                    else:
                        return "__call__", (self._self._wavelength,)
                return deepcopy(self._self._wavelength)
            def _synchronize(self, length:Union[torch.Tensor, float]):
                self._self._wavelength = length
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
            def get(self, description:bool=False):
                if description:
                    if isinstance(self._self._space_reflection, torch.Tensor):
                        return "range", (torch.min(self._self._space_reflection), torch.max(self._self._space_reflection).item(), self._self._space_reflection.numel())
                    else:
                        return "__call__", (self._self._space_reflection,)
                return deepcopy(self._self._space_reflection)
            def _synchronize(self, reflection:Union[torch.Tensor, float]):
                self._self._space_reflection = reflection
                self._self._PropagationModule.reflection = reflection
            def __call__(self, reflection:float):
                self._synchronize(reflection)
                for module in self._self._HeightMasksModuleList:
                    module.space_reflection = reflection
            def range(self, reflection0:float, reflection1:float, N:int):
                reflections = torch.linspace(reflection0, reflection1, N, dtype=self._self._accuracy.tensor_complex)
                self._synchronize(reflections)
        return Selector(self)

    _mask_reflection : torch.Tensor
    @property
    def mask_reflection(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    if isinstance(self._self._mask_reflection, torch.Tensor):
                        return "range", (torch.min(self._self._mask_reflection), torch.max(self._self._mask_reflection).item(), self._self._mask_reflection.numel())
                    else:
                        return "__call__", (self._self._mask_reflection,)
                return deepcopy(self._self._mask_reflection)
            def _synchronize(self, reflection:Union[torch.Tensor, float]):
                self._self._mask_reflection = reflection
                for module in self._self._HeightMasksModuleList:
                    module.mask_reflection = reflection
            def __call__(self, reflection:float):
                self._synchronize(reflection)
            def range(self, reflection0:float, reflection1:float, N:int):
                reflections = torch.linspace(reflection0, reflection1, N, dtype=self._self._accuracy.tensor_complex)
                self._synchronize(reflections)
        return Selector(self)

    _plane_length : float
    @property
    def plane_length(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._plane_length,)
                return deepcopy(self._self._plane_length)
            def __call__(self, length:float):
                self._self._plane_length = length
                self._self._PropagationModule.plane_length = length
        return Selector(self)

    _pixels : int
    @property
    def pixels(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._pixels,)
                return deepcopy(self._self._pixels)
            def __call__(self, amount:int):
                self._self._pixels = amount
                self._self._DetectorsModule.pixels = amount
                self._self._PropagationModule.pixels = amount
                for module in self._self._HeightMasksModuleList:
                    module.pixels = amount
        return Selector(self)

    _up_scaling : int
    @property
    def up_scaling(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._up_scaling,)
                return deepcopy(self._self._up_scaling)
            def __call__(self, amount:int):
                self._self._up_scaling = amount
                self._self._DetectorsModule.up_scaling = amount
                self._self._PropagationModule.up_scaling = amount
                for module in self._self._HeightMasksModuleList:
                    module.up_scaling = amount
        return Selector(self)

    _space : float
    @property
    def space(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._space,)
                return deepcopy(self._self._space)
            def __call__(self, length:float):
                self._self._space = length
                self._self._PropagationModule.distance = length
        return Selector(self)

    _border : float
    @property
    def border(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._border,)
                return deepcopy(self._self._border)
            def __call__(self, length:float):
                self._self._border = length
                self._self._PropagationModule.border = length
        return Selector(self)

    _detectors : int
    @property
    def detectors(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._detectors,)
                return deepcopy(self._self._detectors)
            def __call__(self, amount:int):
                self._self._detectors = amount
                self._self._DetectorsModule.detectors = amount
        return Selector(self)

    _detectors_type_variant : Tuple[str, Tuple]
    @property
    def detectors_type(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return self._self._detectors_type_variant
                return self._self._detectors_type_variant[0]
            def polar(self, borders:float=0.05, space:float=0.2, power:float=0.5):
                self._self._DetectorsModule.masks.set.polar(borders=borders, space=space, power=power)
                self._self._detectors_type_variant = ("polar", (borders, space, power))
            def square(self, borders:float=0.05, space:float=0.2):
                self._self._DetectorsModule.masks.set.square(borders=borders, space=space)
                self._self._detectors_type_variant = ("square", (borders, space))
        return Selector(self)

    @property
    def normalization(self):
        return self._DetectorsModule.normalization.set

    _normalization_type : int
    _normalization_parameters : List
    @property
    def parameters_normalization(self):
        class Selector:
            _self:RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False, as_int:bool=False):
                if description:
                    if   self._self._normalization_type == sigmoid_normalization:   return 'sigmoid', tuple(self._self._normalization_parameters)
                    elif self._self._normalization_type == sinus_normalization:     return 'sinus',   tuple(self._self._normalization_parameters)
                if as_int: return self._self._normalization_type
                elif self._self._normalization_type == sigmoid_normalization:   return 'sigmoid(x)'
                elif self._self._normalization_type == sinus_normalization:     return 'sinus(' + str(self._self._normalization_parameters[0]) + '•x)'
            def sigmoid(self):
                self._self._normalization_type = sigmoid_normalization
                self._self._normalization_parameters = []
                for module in self._self._HeightMasksModuleList:
                    if isinstance(module, HeightMaskLayer):
                        module.normalization.sigmoid()
            def sinus(self, period:float=100.0):
                self._self._normalization_type = sigmoid_normalization
                self._self._normalization_parameters = [period]
                for module in self._self._HeightMasksModuleList:
                    if isinstance(module, HeightMaskLayer):
                        module.normalization.sinus(period)
        return Selector(self)

    _layers : int
    @property
    def layers(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    return "__call__", (self._self._layers,)
                return deepcopy(self._self._layers)
            def __call__(self, amount:int):
                self._self._layers = amount
                if hasattr(self._self, "_HeightMasksModuleList"):
                    if   amount > len(self._self._HeightMasksModuleList):
                        for i in range(amount - len(self._self._HeightMasksModuleList)):
                            self._self._HeightMasksModuleList.append(HeightMaskLayer())
                            self._self._HeightMasksModuleList[-1].wavelength        = self._self._wavelength
                            self._self._HeightMasksModuleList[-1].space_reflection  = self._self._space_reflection
                            self._self._HeightMasksModuleList[-1].mask_reflection   = self._self._mask_reflection
                            self._self._HeightMasksModuleList[-1].pixels            = self._self._pixels
                            self._self._HeightMasksModuleList[-1].up_scaling        = self._self._up_scaling
                            self._self._HeightMasksModuleList[-1].accuracy.bits(self._self.accuracy.get())

                    else:
                        for i in range(len(self._self._HeightMasksModuleList) - amount):
                            self._self._HeightMasksModuleList.pop(0)
                else:
                    self._self._HeightMasksModuleList = torch.nn.ModuleList([HeightMaskLayer() for i in range(amount)])
                    for module in self._self._HeightMasksModuleList:
                        module.wavelength        = self._self._wavelength
                        module.space_reflection  = self._self._space_reflection
                        module.mask_reflection   = self._self._mask_reflection
                        module.pixels            = self._self._pixels
                        module.up_scaling        = self._self._up_scaling
                        module.accuracy.bits(self._self.accuracy.get())
        return Selector(self)

    _amplification : bool
    @property
    def amplification(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def get(self, description:bool=False):
                if description:
                    if self._self._amplification:   return "enable", ()
                    else:                           return "disable", ()
                return deepcopy(self._self._amplification)
            def enable(self):
                self._self._amplification = True
            def disable(self):
                self._self._amplification = False
        return Selector(self)
    # Конец описания параметров


    def __init__(self,
                 layers:int=5,
                 wavelength:Union[torch.Tensor,float]=600*nm,
                 space_reflection:Union[torch.Tensor,float]=1.0,
                 mask_reflection:Union[torch.Tensor,float]=1.5,
                 plane_length:float=1.0*mm,
                 pixels:int=21,
                 up_scaling:int=8,
                 space:float=20.0*mm,
                 border:float=0.0*mm,
                 detectors:int=10):
        super(RealSpaceD2NN, self).__init__()
        self._AmplificationModule = AmplificationLayer()
        self._DetectorsModule = DetectorsLayer()
        self._PropagationModule = FourierPropagationLayer()
        self._HeightMasksModuleList = torch.nn.ModuleList([HeightMaskLayer() for i in range(layers)])

        self.amplification.disable()
        self.normalization.softmax()
        self.parameters_normalization.sinus()

        self.wavelength(wavelength)
        self.space_reflection(space_reflection)
        self.mask_reflection(mask_reflection)
        self.plane_length(plane_length)
        self.pixels(pixels)
        self.up_scaling(up_scaling)
        self.space(space)
        self.border(border)
        self.detectors(detectors)
        self.layers(layers)

    def forward(self, field:torch.Tensor, record_history:bool=False, history:List=None, length:float=0):
        super(RealSpaceD2NN, self).forward(field)

        if record_history and history is None:
            history = [("Начальная амплитуда", torch.abs(field.clone().detach().cpu()))]

        if self._amplification:
            field = self._AmplificationModule(field)

        for i, _MaskLayer in enumerate(self._HeightMasksModuleList):
            field = self._PropagationModule(field)
            if record_history:
                length += self._space
                history.append(("Амплитуда перед маской №" + str(i+1) + " (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))
            field = _MaskLayer(field)

        field = self._PropagationModule(field)
        if record_history:
            length += self._space
            history.append(("Амплитуда перед детекторами (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))

        labels = self._DetectorsModule(field)

        if record_history:
            return labels, history
        return labels


if __name__ == "__main__":
    from Test import Test
    Test.emission.pixel(RealSpaceD2NN())
    Test.emission.MNIST(RealSpaceD2NN())
    Test.emission.variate.pixel(RealSpaceD2NN(), param='space', values=(2.*mm, 5.*mm, 10.*mm), unit='m')
    Test.emission.variate.MNIST(RealSpaceD2NN(), param='space', values=(2.*mm, 5.*mm, 10.*mm), unit='m')
