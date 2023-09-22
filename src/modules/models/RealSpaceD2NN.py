import torch

from typing import Union, List
from copy import deepcopy

from utilities.DecimalPrefixes import nm, um, mm, cm
from utilities.Formaters import Format

from AbstractModel import AbstractModel

from src.modules.layers.AmplificationLayer import AmplificationLayer
from src.modules.layers.DetectorsLayer import DetectorsLayer
from src.modules.layers.FourierPropagationLayer import FourierPropagationLayer
from src.modules.layers.HeightMaskLayer import HeightMaskLayer

from src.modules.layers.AbstractMaskLayer import sigmoid_normalization, sinus_normalization

class RealSpaceD2NN(AbstractModel):

    _AmplificationModule    : AmplificationLayer
    _DetectorsModule        : DetectorsLayer
    _HeightMasksModuleList  : torch.nn.ModuleList
    _PropagationModule      : FourierPropagationLayer


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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
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
            def get(self):
                return deepcopy(self._self._detectors)
            def __call__(self, amount:int):
                self._self._detectors = amount
                self._self._DetectorsModule.detectors = amount
        return Selector(self)

    @property
    def detectors_type(self):
        class Selector:
            _self : RealSpaceD2NN
            def __init__(self, _self:RealSpaceD2NN):
                self._self = _self
            def polar(self, borders:float=0.05, space:float=0.2, power:float=0.5):
                self._self._DetectorsModule.masks.set.polar(borders=borders, space=space, power=power)
            def square(self, borders:float=0.05, space:float=0.2):
                self._self._DetectorsModule.masks.set.square(borders=borders, space=space)
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
            def get(self, as_int:bool=False):
                if as_int: return self._self._normalization_type
                elif self._self._normalization_type == sigmoid_normalization:
                    return 'sigmoid(x)'
                elif self._self._normalization_type == sinus_normalization:
                    return 'sinus(' + str(self._self._normalization_parameters[0]) + '•x)'
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
            def get(self):
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
            def get(self):
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

    def forward(self, field:torch.Tensor, record_history:bool=False):
        super(RealSpaceD2NN, self).forward(field)

        if self._amplification:
            field = self._AmplificationModule(field)

        history = []
        length = 0

        if record_history:
            history.append(("Начальная амплитуда", torch.abs(field.clone().detach().cpu())))

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

class Test:
    @staticmethod
    def PixelEmission():
        from belashovplot import TiledPlot
        Model = RealSpaceD2NN()

        pixels = Model.pixels.get()
        up_scaling = Model.up_scaling.get()
        total = pixels * up_scaling
        length = Model.plane_length.get()
        begin   = int(total/2 - up_scaling/2)
        end     = begin + up_scaling

        with torch.no_grad():
            field = torch.zeros(1, 1, total, total)
            field[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
            labels, history = Model.forward(field, record_history=True)

        plot = TiledPlot(20., 10.)
        plot.title("Тест распространения")
        plot.description.top("Распространения излучения от единственного пикселя вдоль нейронной сети")

        for i, (description, field) in enumerate(history):
            axes = plot.axes.add(i, 0)
            unit, mult = Format.Engineering_Separated(length/2, 'm')
            axes.imshow(field.squeeze(), extent=[-mult*length/2, +mult*length/2, -mult*length/2, +mult*length/2])
            plot.graph.label.x("x, " + unit)
            plot.graph.label.y("y, " + unit)
            plot.graph.description(description)

        plot.show()

    @staticmethod
    def PixelEmissionWithParameterVariation(parameter:str="space", values=(10.0*mm, 20.0*mm, 30.0*mm), unit:str='m'):
        from belashovplot import TiledPlot
        Model = RealSpaceD2NN()
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        Model.to(device)

        pixels = Model.pixels.get()
        up_scaling = Model.up_scaling.get()
        total = pixels * up_scaling
        length = Model.plane_length.get()
        begin   = int(total/2 - up_scaling/2)
        end     = begin + up_scaling

        plot = TiledPlot(20., 10.)
        plot.title("Тест распространения")
        plot.description.top("Распространения излучения от единственного пикселя вдоль нейронной сети")
        plot.pad.graph.vertical(0.2)
        plot.pad.graph.horizontal(0.2)

        with torch.no_grad():
            field0 = torch.zeros(1, 1, total, total)
            field0[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
            field0 = field0.to(device)

            for j, value in enumerate(values):
                getattr(Model, parameter)(value)
                labels, history = Model.forward(field0.clone().detach(), record_history=True)
                for i, (description, field) in enumerate(history):
                    axes = plot.axes.add(i, j)
                    unit, mult = Format.Engineering_Separated(length/2, 'm')
                    axes.imshow(field.squeeze(), extent=[-mult*length/2, +mult*length/2, -mult*length/2, +mult*length/2])
                    plot.graph.label.x("x, " + unit)
                    plot.graph.label.y("y, " + unit)
                    plot.graph.description(description)
                plot.description.row.left(parameter + " : " + Format.Engineering(value, unit, 1), j)

        plot.show()



if __name__ == "__main__":
    # Test.PixelEmission()
    Test.PixelEmissionWithParameterVariation()