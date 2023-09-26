import torch
from torchvision.transforms.functional import resize
from typing import Iterable, List, Union

from utilities.Formaters import Format
from utilities.DecimalPrefixes import mm

from modules.layers.DetectorsLayer import DetectorsLayer
from src.utilities.UniversalTestsAndOther import GenerateSingleUnscaledSampleMNIST

from belashovplot import TiledPlot


_basic_width    = 11.69 * 1.0
_basic_height   = 8.21  * 1.0

class Test:
    class compare:
        def __init__(self, model1:torch.nn.Module, model2:torch.nn.Module, inputs:Union[List[torch.Tensor], torch.Tensor], description1:str="Модель №1", description2:str="Модель №2"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model1.to(device)
            model2.to(device)
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.concat(inputs, dim=0).to(device)
            else:
                inputs = inputs.to(device)

            plot = TiledPlot(1.5*_basic_width, 1.5*_basic_height)
            plot.FontLibrary.Fonts.ColumnDescriptionTop.FontSize = 6
            plot.FontLibrary.Fonts.ColumnDescriptionBottom.FontSize = 6
            plot.FontLibrary.Fonts.RowDescriptionRight.FontSize = 12
            plot.FontLibrary.Fonts.RowDescriptionLeft.FontSize = 12
            plot.FontLibrary.Fonts.AxisX.FontSize = 6
            plot.FontLibrary.Fonts.AxisY.FontSize = 6

            if type(model1).__name__ == type(model2).__name__:
                plot.title("Сравнение " + type(model1).__name__)
            else:
                plot.title("Сравнение " + type(model1).__name__ + " и " + type(model2).__name__)

            plane_length1 = None
            if hasattr(model1, 'plane_length') and hasattr(getattr(model1, 'plane_length'), 'get'):
                plane_length1 = model1.plane_length.get()
            plane_length2 = None
            if hasattr(model2, 'plane_length') and hasattr(getattr(model2, 'plane_length'), 'get'):
                plane_length2 = model2.plane_length.get()

            with torch.no_grad():
                labels1, history1 = model1(inputs.clone(), record_history=True)
                labels2, history2 = model2(inputs.clone(), record_history=True)

            for module in model1.modules():
                if isinstance(module, DetectorsLayer):
                    history1.append(('Усреднённые маски детекторов', torch.mean(module.masks.get(), dim=0).expand(history1[-1][1].size(0), -1, -1)))
                    break
            for module in model2.modules():
                if isinstance(module, DetectorsLayer):
                    history2.append(('Усреднённые маски детекторов', torch.mean(module.masks.get(), dim=0).expand(history2[-1][1].size(0), -1, -1)))
                    break

            row = 0
            row0 = 0
            for i, (description, images) in enumerate(history1):
                plot.description.column.top(description, i)
                row = 0
                for image in images:
                    axes = plot.axes.add(i, row)
                    axes.xaxis.set_tick_params(labelsize=4)
                    axes.yaxis.set_tick_params(labelsize=4)
                    if plane_length1 is not None:
                        unit, mult = Format.Engineering_Separated(plane_length1/2, 'm')
                        axes.imshow(image.squeeze().swapdims(0,1), extent=[-mult*plane_length1/2, +mult*plane_length1/2, -mult*plane_length1/2, +mult*plane_length1/2], origin='lower')
                        plot.graph.label.x("x, " + unit)
                        plot.graph.label.y("y, " + unit)
                    else:
                        axes.imshow(image.squeeze().swapdims(0,1), origin='lower')
                        plot.graph.label.x("x")
                        plot.graph.label.y("y")
                    row+=1
            plot.description.row.left(description1, row0, row-1)

            row0 = row + 1
            for i, (description, images) in enumerate(history2):
                plot.description.column.bottom(description, i)
                row = row0
                for image in images:
                    axes = plot.axes.add(i, row)
                    axes.xaxis.set_tick_params(labelsize=4)
                    axes.yaxis.set_tick_params(labelsize=4)
                    if plane_length2 is not None:
                        unit, mult = Format.Engineering_Separated(plane_length2/2, 'm')
                        axes.imshow(image.squeeze().swapdims(0,1), extent=[-mult*plane_length2/2, +mult*plane_length2/2, -mult*plane_length2/2, +mult*plane_length2/2], origin='lower')
                        plot.graph.label.x("x, " + unit)
                        plot.graph.label.y("y, " + unit)
                    else:
                        axes.imshow(image.squeeze().swapdims(0,1), origin='lower')
                        plot.graph.label.x("x")
                        plot.graph.label.y("y")
                    row+=1
            plot.description.row.left(description2, row0, row-1)

            plot.show()

        @staticmethod
        def MNIST(model1:torch.nn.Module, model2:torch.nn.Module, samples:int=4, description1:str="Модель №1", description2:str="Модель №2", pixels:int=None, up_scaling:int=None):
            if pixels is None:
                if hasattr(model1, 'pixels') and hasattr(getattr(model1, 'pixels'), 'get'):
                    pixels = model1.pixels.get()
                else: Exception('Автоматическое определение количества пикселей провалено, введите вручную.')
            if up_scaling is None:
                if hasattr(model1, 'up_scaling') and hasattr(getattr(model1, 'up_scaling'), 'get'):
                    up_scaling = model1.up_scaling.get()
                else: Exception('Автоматическое определение множителя разрешения провалено, введите вручную.')

            inputs = [resize(torch.abs(GenerateSingleUnscaledSampleMNIST(only_image=True)), [pixels * up_scaling, pixels * up_scaling]).expand(1,1,-1,-1).to(torch.complex64) for i in range(samples)]

            Test.compare(model1, model2, inputs, description1, description2)


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