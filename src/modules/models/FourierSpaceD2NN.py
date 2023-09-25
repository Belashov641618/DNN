import torch
from typing import Union, List
from copy import deepcopy

from utilities.DecimalPrefixes import nm, um, mm, cm
from utilities.Formaters import Format

from src.modules.models.RealSpaceD2NN import RealSpaceD2NN
from src.modules.models.AbstractModel import AbstractModel
from src.modules.layers.LensLayer import LensLayer
from src.modules.layers.FourierPropagationLayer import FourierPropagationLayer
from modules.layers.AmplificationLayer import AmplificationLayer
from modules.layers.HeightMaskLayer import HeightMaskLayer
from modules.layers.DetectorsLayer import DetectorsLayer


class FourierSpaceD2NN(RealSpaceD2NN, AbstractModel):

    _LensModule             : LensLayer
    _LensPropagationModule  : FourierPropagationLayer

    def finalize(self):
        super(FourierSpaceD2NN, self).finalize()
        self._LensModule.delayed.finalize()
        self._LensPropagationModule.delayed.finalize()

    # Начало описания параметров
    _focus : float
    @property
    def focus(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._focus)
            def __call__(self, length:float):
                self._self._focus = length
                self._self._LensModule.focus = length
                self._self._LensPropagationModule.distance = length
        return Selector(self)

    _focus_border : float
    @property
    def focus_border(self):
        class Selector:
            _self:FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._focus_border)
            def __call__(self, length:float):
                self._self._focus_border = length
                self._self._LensPropagationModule.border = length
        return Selector(self)

    @RealSpaceD2NN.wavelength.getter
    def wavelength(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._wavelength)
            def _synchronize(self, length:Union[torch.Tensor, float]):
                self._self._wavelength = length
                self._self._PropagationModule.wavelength = length
                self._self._LensModule.wavelength = length
                self._self._LensPropagationModule.wavelength = length
                for module in self._self._HeightMasksModuleList:
                    module.wavelength = length
            def __call__(self, length:float):
                self._synchronize(length)
            def range(self, length0:float, length1:float, N:int):
                wavelengths = torch.linspace(length0, length1, N, dtype=self._self._accuracy.tensor_float)
                self._synchronize(wavelengths)
        return Selector(self)

    @RealSpaceD2NN.space_reflection.getter
    def space_reflection(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._space_reflection)
            def _synchronize(self, reflection:Union[torch.Tensor, float]):
                self._self._space_reflection = reflection
                self._self._PropagationModule.reflection = reflection
                self._self._LensPropagationModule.reflection = reflection
            def __call__(self, reflection:float):
                self._synchronize(reflection)
                for module in self._self._HeightMasksModuleList:
                    module.space_reflection = reflection
            def range(self, reflection0:float, reflection1:float, N:int):
                reflections = torch.linspace(reflection0, reflection1, N, dtype=self._self._accuracy.tensor_complex)
                self._synchronize(reflections)
        return Selector(self)

    @RealSpaceD2NN.plane_length.getter
    def plane_length(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._plane_length)
            def __call__(self, length:float):
                self._self._plane_length = length
                self._self._PropagationModule.plane_length = length
                self._self._LensModule.plane_length = length
                self._self._LensPropagationModule.plane_length = length
        return Selector(self)

    @RealSpaceD2NN.pixels.getter
    def pixels(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._pixels)
            def __call__(self, amount:int):
                self._self._pixels = amount
                self._self._DetectorsModule.pixels = amount
                self._self._PropagationModule.pixels = amount
                self._self._LensModule.pixels = amount
                self._self._LensPropagationModule.pixels = amount
                for module in self._self._HeightMasksModuleList:
                    module.pixels = amount
        return Selector(self)

    @RealSpaceD2NN.up_scaling.getter
    def up_scaling(self):
        class Selector:
            _self : FourierSpaceD2NN
            def __init__(self, _self:FourierSpaceD2NN):
                self._self = _self
            def get(self):
                return deepcopy(self._self._up_scaling)
            def __call__(self, amount:int):
                self._self._up_scaling = amount
                self._self._DetectorsModule.up_scaling = amount
                self._self._PropagationModule.up_scaling = amount
                self._self._LensModule.up_scaling = amount
                self._self._LensPropagationModule.up_scaling = amount
                for module in self._self._HeightMasksModuleList:
                    module.up_scaling = amount
        return Selector(self)
    # Начало описания параметров


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
                 focus:float=10.0*mm,
                 focus_border:float=0.0*mm,
                 detectors:int=10):
        AbstractModel.__init__(self)
        self._AmplificationModule = AmplificationLayer()
        self._DetectorsModule = DetectorsLayer()
        self._PropagationModule = FourierPropagationLayer()
        self._HeightMasksModuleList = torch.nn.ModuleList([HeightMaskLayer() for i in range(layers)])
        self._LensModule = LensLayer()
        self._LensPropagationModule = FourierPropagationLayer()

        self.detectors_type.polar()
        self.amplification.disable()
        self.normalization.maximum()
        self.parameters_normalization.sinus()

        self.wavelength(wavelength)
        self.space_reflection(space_reflection)
        self.mask_reflection(mask_reflection)
        self.plane_length(plane_length)
        self.pixels(pixels)
        self.up_scaling(up_scaling)
        self.space(space)
        self.border(border)
        self.focus(focus)
        self.focus_border(focus_border)
        self.detectors(detectors)
        self.layers(layers)



    def forward(self, field:torch.Tensor, record_history:bool=False, history:List=None, length:float=0):
        AbstractModel.forward(self, field)

        if record_history and history is None:
            history = [("Начальная амплитуда", torch.abs(field.clone().detach().cpu()))]

        field = self._LensPropagationModule(field)
        length += self._focus
        if record_history:
            history.append(("Амплитуда перед входной линзой (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))

        field = self._LensModule(field)

        field = self._LensPropagationModule(field)
        length += self._focus
        if record_history:
            history.append(("Амплитуда перед маской №1 (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))

        field = self._HeightMasksModuleList[0](field)

        for i, _MaskLayer in enumerate(self._HeightMasksModuleList[1:], 1):
            field = self._PropagationModule(field)
            length += self._space
            if record_history:
                history.append(("Амплитуда перед маской №" + str(i+1) + " (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))
            field = _MaskLayer(field)


        field = self._LensPropagationModule(field)
        length += self._focus
        if record_history:
            history.append(("Амплитуда перед выходной линзой (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))

        field = self._LensModule(field)

        field = self._LensPropagationModule(field)
        length += self._focus
        if record_history:
            history.append(("Амплитуда перед детекторами (" + Format.Engineering(length, 'm', 1) + ")", torch.abs(field.clone().detach().cpu())))

        labels = self._DetectorsModule(field)

        if record_history:
            return labels, history
        return labels



if __name__ == '__main__':
    from Test import Test
    Test.emission.pixel(FourierSpaceD2NN())
    Test.emission.MNIST(FourierSpaceD2NN())
    Test.emission.variate.pixel(FourierSpaceD2NN(), param='space', values=(2.*mm, 5.*mm, 10.*mm), unit='m')
    Test.emission.variate.MNIST(FourierSpaceD2NN(), param='space', values=(2.*mm, 5.*mm, 10.*mm), unit='m')