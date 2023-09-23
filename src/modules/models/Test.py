import torch
from typing import Iterable

from utilities.Formaters import Format
from utilities.DecimalPrefixes import mm

from belashovplot import TiledPlot

_basic_width    = 11.69 * 1.0
_basic_height   = 8.21  * 1.0

class Test:
    class emission:

        class pixel:
            def __init__(self, model:torch.nn.Module):
                pixels = model.pixels.get()
                up_scaling = model.up_scaling.get()
                total = pixels * up_scaling
                length = model.plane_length.get()
                begin = int(total / 2 - up_scaling / 2)
                end = begin + up_scaling

                plot = TiledPlot(_basic_width, _basic_height)
                plot.FontLibrary.MultiplyFontSize(0.6)
                plot.title("Тест "  + type(model).__name__)
                plot.description.top("Распространения излучения от единственного пикселя вдоль нейронной сети")

                with torch.no_grad():
                    field = torch.zeros(1, 1, total, total)
                    field[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
                    labels, history = model.forward(field, record_history=True)

                    for i, (description, field) in enumerate(history):
                        axes = plot.axes.add(i, 0)
                        unit, mult = Format.Engineering_Separated(length / 2, 'm')
                        axes.imshow(field.squeeze(), extent=[-mult*length/2, +mult*length/2, -mult*length/2, +mult*length/2])
                        plot.graph.label.x("x, " + unit)
                        plot.graph.label.y("y, " + unit)
                        plot.graph.description(description)

                plot.show()

            class variation:
                @staticmethod
                def parameter(model:torch.nn.Module, param:str='space', values:Iterable=(10.0*mm, 20.0*mm, 30.0*mm), unit:str='m'):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)

                    pixels = model.pixels.get()
                    up_scaling = model.up_scaling.get()
                    total = pixels * up_scaling
                    length = model.plane_length.get()
                    begin = int(total / 2 - up_scaling / 2)
                    end = begin + up_scaling

                    plot = TiledPlot(_basic_width, _basic_height)
                    plot.FontLibrary.MultiplyFontSize(0.6)
                    plot.title("Тест " + type(model).__name__)
                    plot.description.top("Распространения излучения от единственного пикселя вдоль нейронной сети при вариации параметра " + param + ".")
                    plot.pad.graph.vertical(0.2)
                    plot.pad.graph.horizontal(0.2)

                    with torch.no_grad():
                        field0 = torch.zeros(1, 1, total, total)
                        field0[0][0][begin:end, begin:end] = torch.ones(up_scaling, up_scaling)
                        field0 = field0.to(device)

                        for j, value in enumerate(values):
                            getattr(model, param)(value)
                            labels, history = model.forward(field0.clone().detach(), record_history=True)
                            for i, (description, field) in enumerate(history):
                                axes = plot.axes.add(i, j)
                                unit, mult = Format.Engineering_Separated(length / 2, 'm')
                                axes.imshow(field.squeeze(), extent=[-mult*length/2, +mult*length/2, -mult*length/2, +mult*length/2])
                                plot.graph.label.x("x, " + unit)
                                plot.graph.label.y("y, " + unit)
                                plot.graph.description(description)
                            plot.description.row.left(param + " : " + Format.Engineering(value, unit, 1), j)

                    plot.show()