import torch
import numpy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from belashovplot import TiledPlot
from typing import Union, Callable, Dict, List
import inspect
from functools import partial
import platform
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, FigureManagerBase

from src.modules.layers.AbstarctLayer import AbstractLayer
from src.modules.models.AbstractModel import AbstractModel


import os
_DataRoot = os.path.abspath(__file__) + '../../../../data/'
class DataSets:
    class loader:
        @staticmethod
        def MNIST(sizeX:int=28, sizeY:int=28, dtype:torch.dtype=torch.float32, train:bool=False, batch:int=1):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((sizeX, sizeY)),
                transforms.ConvertImageDtype(dtype)
            ])
            mnist_data = datasets.MNIST(root=_DataRoot, train=train, download=True, transform=transform)
            data_loader = DataLoader(mnist_data, batch_size=batch, shuffle=True)
            return data_loader
    class single:
        @staticmethod
        def MNIST(sizeX:int=28, sizeY:int=28, dtype:torch.dtype=torch.float32, train:bool=False, num:int=None):
            if num is None:
                return next(iter(DataSets.loader.MNIST(sizeX, sizeY, dtype, train, 1)))
            Loader = DataSets.loader.MNIST(sizeX, sizeY, dtype, train, 1)
            for image, label in Loader:
                if label == torch.eye(10,10)[num]:
                    return image, label
            return None

class Inputs:
    _data : Union[torch.Tensor, partial]
    def __init__(self, data:Union[torch.Tensor, partial]):
        self._data = data

    def sizeX(self, module:Union[AbstractModel, AbstractLayer]):
        if isinstance(self._data, partial):
            pixels = ['pixels', 'in_pixels']
            for pixels_attribute in pixels:
                if hasattr(module, pixels_attribute):
                    pixels = getattr(module, pixels_attribute)
                    break
            else: raise Exception('There is no any attribute to attach pixels in module')
            up_scaling = ['up_scaling', 'in_up_scaling']
            for up_scaling_attribute in up_scaling:
                if hasattr(module, up_scaling_attribute):
                    up_scaling = getattr(module, up_scaling_attribute)
                    break
            else: raise Exception('There is no any attribute to attach up_scaling in module')
            pixels = pixels*up_scaling
            self._data = partial(self._data, sizeX=pixels)
    def sizeY(self, module:Union[AbstractModel, AbstractLayer]):
        if isinstance(self._data, partial):
            pixels = ['pixels', 'in_pixels']
            for pixels_attribute in pixels:
                if hasattr(module, pixels_attribute):
                    pixels = getattr(module, pixels_attribute)
                    break
            else: raise Exception('There is no any attribute to attach pixels in module')
            up_scaling = ['up_scaling', 'in_up_scaling']
            for up_scaling_attribute in up_scaling:
                if hasattr(module, up_scaling_attribute):
                    up_scaling = getattr(module, up_scaling_attribute)
                    break
            else: raise Exception('There is no any attribute to attach up_scaling in module')
            pixels = pixels*up_scaling
            self._data = partial(self._data, sizeY=pixels)
    def dtype(self, module:Union[AbstractModel, AbstractLayer]):
        self._data = partial(self._data, dtype=module._accuracy.tensor_complex)

    def free_args(self):
        if isinstance(self._data, partial):
            original_func = self._data.func
            sig = inspect.signature(original_func)
            partial_args = set(self._data.args)
            partial_kwargs = set(self._data.keywords.keys())
            missing_args = [name for name, param in sig.parameters.items()
                            if name not in partial_args and name not in partial_kwargs]
            return missing_args
        return []
    def freeze(self, module:Union[AbstractModel, AbstractLayer]):
        for argument in self.free_args():
            if not hasattr(self, argument):
                raise Exception('No method to freeze argument: ' + argument)
            getattr(self, argument)(module)
    def get(self, module:Union[AbstractModel, AbstractLayer]=None):
        if self.free_args():
            if module is None:
                raise Exception('Provide module to freeze args: ' + ', '.join(self.free_args()))
            self.freeze(module)
        if isinstance(self._data, partial):
            return self._data()
        return self._data

class Results:
    _results : Union[Dict, List]
    _module : Union[AbstractModel, AbstractLayer]
    def __init__(self, results:Union[Dict, List], module:Union[AbstractModel, AbstractLayer]=None):
        self._results = results
    def plot(self, max_width:float=None, max_height:float=None, dpi:int=100):
        if (max_width is None) or (max_height is None):
            figure = plt.figure()
            figure.set_dpi(100)
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            plt.show(block=False)
            max_width_, max_height_ = figure.get_size_inches()
            plt.close(figure)
            if max_width is None: max_width = max_width_
            if max_height is None: max_height = max_height_

        plot = TiledPlot(max_width, max_height, dpi)

        axes = plot.axes.add()

        plot.show()
    @property
    def result(self):
        class Selector:
            _self : Results
            def __init__(self, _self:Results):
                self._self = _self
            def __call__(self):
                return self._self._results
        return Selector(self)

class AbstractResult:
    _name : str
    def name(self):
        return self._name
    def __init__(self, name:str=None):
        self._name = name
class ImageResult(AbstractResult):
    _data : torch.Tensor
    _arguments : Dict
    @staticmethod
    def _prepare_image(image:torch.Tensor):
        return image.squeeze()
    def __init__(self, image:torch.Tensor, name:str=None, length_x:float=1.0, length_y:float=1.0):
        super().__init__(name)
        self._name = name
        self._data = self._prepare_image(image)
        self._arguments = {
            'cmap' : 'viridis',
            'aspect' : 'auto',
            'extent' : [-length_x/2, +length_x/2, -length_y/2, +length_y/2],
        }
    def image(self):
        return self._data
    def arguments(self):
        return self._arguments


class DevicesManager:
    @staticmethod
    def available():
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append((f'cuda:{i}', torch.cuda.get_device_name(i)))
        devices.append(('cpu', platform.processor()))
        return devices

    _devices : List
    _availability : List
    def __init__(self, echo:bool=True):
        self._devices = []
        self._availability = []
        if echo: print('Существующие на системе исполнители:')
        for i, (index, name) in enumerate(self.available(), 1):
            if echo: print(str(i) + ': ' + name)
            self._devices.append(torch.device(index))
            self._availability.append(True)
    def get(self):
        for i, (device, availability) in enumerate(zip(self._devices, self._availability)):
            if availability:
                self._availability[i] = False
                return device
        return None
    def free(self, device:torch.device):
        self._availability[self._devices.index(device)] = True


Devices = DevicesManager(True)

class Test:
    class inputs:
        @staticmethod
        def manual(array_like:Union[torch.Tensor, numpy.ndarray]):
            if not torch.is_tensor(array_like):
                array_like = torch.Tensor(array_like)
            return Inputs(array_like)
        @staticmethod
        def MNIST(num: int = None, dtype: torch.dtype = None, sizeX: int = None, sizeY: int = None):
            kwargs = {}
            if dtype is not None:
                kwargs['dtype'] = dtype
            if sizeX is not None:
                kwargs['sizeX'] = sizeX
            if sizeY is not None:
                kwargs['sizeY'] = sizeY
            return Inputs(partial(DataSets.single.MNIST, num=num, train=False, **kwargs))
    class forward:
        @staticmethod
        def layer(layer:AbstractLayer, initial:Inputs):
            field0 = initial.get(layer)
            result = None
            device = Devices.get()
            layer = layer.to(device)
            field0 = field0.to(device)
            with torch.no_grad():
                result = layer.forward(field0)
            layer = layer.cpu()
            result = result.cpu()
            field0 = field0.cpu()
            Devices.free(device)
            return Results({'Начальные данные' : field0, 'Результат работы' : result}, layer)
        @staticmethod
        def model(model:AbstractModel, initial:Inputs):
            field0 = initial.get(model)
            result = None
            device = Devices.get()
            model = model.to(device)
            field0 = field0.to(device)
            with torch.no_grad():
                result = model.forward(field0)
            model = model.cpu()
            result = result.cpu()
            field0 = field0.cpu()
            Devices.free(device)
            return Results({'':[ImageResult(field0, 'Начальное поле'), ImageResult(result, 'Результат обработки')]}, model)
        @staticmethod
        def __call__(module:Union[AbstractModel, AbstractLayer], initial:Inputs):
            if isinstance(module, AbstractModel):
                return Test.forward.model(module, initial)
            elif isinstance(module, AbstractLayer):
                return Test.forward.layer(module, initial)

if __name__ == '__main__':
    print('Система автоматического тестирования')
