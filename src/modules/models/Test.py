import torch
from torchvision.transforms.functional import resize
from typing import Iterable, List

from utilities.Formaters import Format
from utilities.DecimalPrefixes import mm

from src.modules.layers.DetectorsLayer import DetectorsLayer
from src.utilities.UniversalTestsAndOther import GenerateSingleUnscaledSampleMNIST

from belashovplot import TiledPlot


_basic_width    = 11.69 * 1.0
_basic_height   = 8.21  * 1.0

class Test:
    class emission:
        def __init__(self, model:torch.nn.Module, field0:torch.Tensor):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            field0 = field0.to(device)

            plot = TiledPlot(_basic_width, _basic_height)
            plot.FontLibrary.MultiplyFontSize(0.6)
            plot.FontLibrary.Fonts.ColumnDescriptionTop.FontSize = 11
            plot.title("Тест " + type(model).__name__)
            plot.description.top("Распространения излучения вдоль нейронной сети")

            plane_length = None
            if hasattr(model, 'plane_length') and hasattr(getattr(model, 'plane_length'), 'get'):
                plane_length = model.plane_length.get()

            with torch.no_grad():

                labels, history = model.forward(field0, record_history=True)

                for module in model.modules():
                    if isinstance(module, DetectorsLayer):
                        history.append(('Усреднённые маски детекторов', torch.mean(module.masks.get(), dim=0)))
                        break

                for i, (description, field) in enumerate(history):
                    axes = plot.axes.add(i, 0)
                    if plane_length is not None:
                        unit, mult = Format.Engineering_Separated(plane_length / 2, 'm')
                        axes.imshow(field.squeeze().swapdims(0,1), extent=[-mult*plane_length/2, +mult*plane_length/2, -mult*plane_length/2, +mult*plane_length/2], origin='lower')
                    else:
                        axes.imshow(field.squeeze().swapdims(0,1), origin='lower')
                    plot.graph.label.x("x, " + unit)
                    plot.graph.label.y("y, " + unit)
                    plot.description.column.top(description, i)

            plot.show()

        @staticmethod
        def pixel(model:torch.nn.Module, pixels:int=None, up_scaling:int=None):
            if pixels is None:
                if hasattr(model, 'pixels') and hasattr(getattr(model, 'pixels'), 'get'):
                    pixels = model.pixels.get()
                else: Exception('Автоматическое определение количества пикселей провалено, введите вручную.')
            if up_scaling is None:
                if hasattr(model, 'up_scaling') and hasattr(getattr(model, 'up_scaling'), 'get'):
                    up_scaling = model.up_scaling.get()
                else: Exception('Автоматическое определение множителя разрешения провалено, введите вручную.')

            total = pixels * up_scaling
            begin = int(total / 2 - up_scaling / 2)
            end = begin + up_scaling

            field0 = torch.zeros(1, 1, total, total)
            field0[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
            Test.emission(model, field0=field0)
        @staticmethod
        def MNIST(model:torch.nn.Module, pixels:int=None, up_scaling:int=None):
            if pixels is None:
                if hasattr(model, 'pixels') and hasattr(getattr(model, 'pixels'), 'get'):
                    pixels = model.pixels.get()
                else: Exception('Автоматическое определение количества пикселей провалено, введите вручную.')
            if up_scaling is None:
                if hasattr(model, 'up_scaling') and hasattr(getattr(model, 'up_scaling'), 'get'):
                    up_scaling = model.up_scaling.get()
                else: Exception('Автоматическое определение множителя разрешения провалено, введите вручную.')

            field0 = resize(torch.abs(GenerateSingleUnscaledSampleMNIST(only_image=True)), [pixels*up_scaling, pixels*up_scaling]).to(torch.complex64)
            Test.emission(model, field0=field0)
        class variate:
            def __init__(self, model:torch.nn.Module, param:str, values:Iterable, unit:str, field0:torch.Tensor):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                field0 = field0.to(device)

                plot = TiledPlot(_basic_width, _basic_height)
                plot.FontLibrary.MultiplyFontSize(0.6)
                plot.FontLibrary.Fonts.ColumnDescriptionTop.FontSize = 11
                plot.title("Тест " + type(model).__name__)
                plot.description.top("Распространения излучения вдоль нейронной сети при вариации одного параметра")

                plane_length = None
                if hasattr(model, 'plane_length') and hasattr(getattr(model, 'plane_length'), 'get'):
                    plane_length = model.plane_length.get()

                with torch.no_grad():
                    for j, value in enumerate(values):
                        getattr(model, param)(value)

                        labels, history = model.forward(field0.clone().detach(), record_history=True)

                        for module in model.modules():
                            if isinstance(module, DetectorsLayer):
                                history.append(('Усреднённые маски детекторов', torch.mean(module.masks.get(), dim=0)))
                                break

                        for i, (description, field) in enumerate(history):
                            axes = plot.axes.add(i, j)
                            if plane_length is not None:
                                unit, mult = Format.Engineering_Separated(plane_length / 2, 'm')
                                axes.imshow(field.squeeze().swapdims(0,1), extent=[-mult*plane_length/2, +mult*plane_length/2, -mult*plane_length/2, +mult*plane_length/2], origin='lower')
                            else:
                                axes.imshow(field.squeeze().swapdims(0,1), origin='lower')
                            plot.graph.label.x("x, " + unit)
                            plot.graph.label.y("y, " + unit)
                            if j == 0:
                                plot.description.column.top(description, i)
                        plot.description.row.left(param + ' : ' + Format.Engineering(value, unit), j)

                plot.show()

            @staticmethod
            def pixel(model:torch.nn.Module, param:str, values:Iterable, unit:str='', pixels:int=None, up_scaling:int=None):
                if pixels is None:
                    if hasattr(model, 'pixels') and hasattr(getattr(model, 'pixels'), 'get'):
                        pixels = model.pixels.get()
                    else:
                        Exception('Автоматическое определение количества пикселей провалено, введите вручную.')
                if up_scaling is None:
                    if hasattr(model, 'up_scaling') and hasattr(getattr(model, 'up_scaling'), 'get'):
                        up_scaling = model.up_scaling.get()
                    else:
                        Exception('Автоматическое определение множителя разрешения провалено, введите вручную.')

                total = pixels * up_scaling
                begin = int(total / 2 - up_scaling / 2)
                end = begin + up_scaling

                field0 = torch.zeros(1, 1, total, total)
                field0[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
                Test.emission.variate(model, param=param, values=values, unit=unit, field0=field0)
            @staticmethod
            def MNIST(model:torch.nn.Module, param:str, values:Iterable, unit:str='', pixels:int=None, up_scaling:int=None):
                if pixels is None:
                    if hasattr(model, 'pixels') and hasattr(getattr(model, 'pixels'), 'get'):
                        pixels = model.pixels.get()
                    else: Exception('Автоматическое определение количества пикселей провалено, введите вручную.')
                if up_scaling is None:
                    if hasattr(model, 'up_scaling') and hasattr(getattr(model, 'up_scaling'), 'get'):
                        up_scaling = model.up_scaling.get()
                    else: Exception('Автоматическое определение множителя разрешения провалено, введите вручную.')

                field0 = resize(torch.abs(GenerateSingleUnscaledSampleMNIST(only_image=True)), [pixels*up_scaling, pixels*up_scaling]).to(torch.complex64)
                Test.emission.variate(model, param=param, values=values, unit=unit, field0=field0)