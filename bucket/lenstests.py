import torch
import torchvision
from typing import Tuple
from belashovplot import TiledPlot
from src.utilities.Formaters import Format
from src.utilities.CycleTimePredictor import CycleTimePredictor
from src.utilities.UniversalTestsAndOther import GenerateSingleUnscaledSampleMNIST
from utilities.DecimalPrefixes import nm, mm
from src.modules.layers.LensLayer import LensLayer
from src.modules.layers.MultiLensLayer import MultiLensLayer
from src.modules.layers.AbstractPropagationLayer import AbstractPropagationLayer
from src.modules.layers.FourierPropagationLayer import FourierPropagationLayer
from src.modules.layers.KirchhoffPropagationLayer import KirchhoffPropagationLayer

def ComparePropagationLayers(Layer1:AbstractPropagationLayer, Layer2:AbstractPropagationLayer, tests:int=100, batches:int=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if Layer1.pixels != Layer2.pixels:              Layer2.pixels = Layer1.pixels
    if Layer1.up_scaling != Layer2.up_scaling:      Layer2.up_scaling = Layer1.up_scaling
    if Layer1.wavelength != Layer2.wavelength:      Layer2.wavelength = Layer1.wavelength
    if Layer1.plane_length != Layer2.plane_length:  Layer2.plane_length = Layer1.plane_length

    pixels = Layer1.pixels * Layer1.up_scaling
    plane_length = Layer1.plane_length

    Layer1.to(device)
    Layer2.to(device)

    phase_difference =      torch.zeros(pixels, pixels, requires_grad=False)
    amplitude_difference =  torch.zeros(pixels, pixels, requires_grad=False)

    with torch.no_grad():
        for batch in CycleTimePredictor(range(batches)):
            amplitude = torch.rand((tests, 1, pixels, pixels))
            phase = 2.0*torch.pi*torch.rand((tests, 1, pixels, pixels))
            field = amplitude * torch.exp(1j*phase)
            field = field.to(device)

            result1 = Layer1(field).cpu()
            result2 = Layer2(field).cpu()

            phase_difference +=     torch.sum(torch.abs(torch.angle(result1) - torch.angle(result2)), dim=(0, 1)) / tests
            amplitude_difference += torch.sum(torch.abs(torch.abs(result1)   - torch.abs(result2)),   dim=(0, 1)) / tests

        phase_difference        /= batches
        amplitude_difference    /= batches

    name1 = type(Layer1).__name__
    name2 = type(Layer2).__name__

    Plot = TiledPlot(23.4, 33.1)
    Plot.title('Сравнение различных методов распространения света')
    Plot.description.top('Сравниваются ' + name1 + ' и ' + name2 + ' путём усреднения отклонений по фазе и амплитуде за ' + str(int(tests*batches)) + ' тестов')

    axes = Plot.axes.add((0,0))
    axes.imshow(amplitude_difference, aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1,0))
    axes.imshow(phase_difference, aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')
    Plot.graph.title('Фаза')

    Plot.show()

def CompareFourierTransformations(Propagation:AbstractPropagationLayer, Lens:LensLayer, field:torch.Tensor=None, ratios:Tuple[float]=(1.5, 1.0, 0.5)):
    if Propagation.pixels != Lens.pixels:               Lens.pixels = Propagation.pixels
    if Propagation.up_scaling != Lens.up_scaling:       Lens.up_scaling = Propagation.up_scaling
    if Propagation.plane_length != Lens.plane_length:   Lens.plane_length = Propagation.plane_length
    if Propagation.wavelength != Lens.wavelength:       Lens.wavelength = Propagation.wavelength

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixels = Lens.pixels * Lens.up_scaling
    plane_length = Lens.plane_length

    if field is None:
        transform = torchvision.transforms.Resize((pixels, pixels), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True)))
    real = torch.fft.fftshift(torch.fft.fft2(field.clone())).squeeze().cpu()

    field = field.to(device)
    Propagation.to(device)
    Lens.to(device)

    Plot = TiledPlot(23.4, 33.1)
    Plot.title('Сравнение Фурье образа с Фурье образом линзы')

    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(real), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

    axes = Plot.axes.add((0, 1))
    axes.imshow(torch.angle(real), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

    axes = Plot.axes.add((0, 2))
    axes.imshow(torch.real(real), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

    with torch.no_grad():
        for i, ratio in enumerate(CycleTimePredictor(ratios), 1):
            field_ = field.clone().expand(1,1,pixels,pixels)

            Propagation.distance = Lens.focus
            field_ = Propagation(field_)
            field_ = Lens(field_)
            Propagation.distance = Lens.focus * ratio
            field_ = Propagation(field_)
            result = field_.squeeze().cpu()

            axes = Plot.axes.add((i, 0))
            axes.imshow(torch.abs(result), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

            axes = Plot.axes.add((i, 1))
            axes.imshow(torch.angle(result), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

            axes = Plot.axes.add((i, 2))
            axes.imshow(torch.real(result), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

    Plot.show()

def TestMultiLens(N:int=3, f:float=30*mm):
    raw_pixels = 256
    up_scaling = 2
    pixels = raw_pixels*up_scaling
    plane_length = 3.0*mm

    L = 2 * N * f
    b = L * f / (L - f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Lenses = MultiLensLayer(N, f, 600*nm, plane_length=plane_length, pixels=raw_pixels, up_scaling=up_scaling).to(device)
    Propagation1 = FourierPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, L, 3*plane_length).to(device)
    Propagation2 = FourierPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, b, 3*plane_length).to(device)

    transform = torchvision.transforms.Resize((pixels, pixels))
    field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True))).to(device)

    arguments = {
        'aspect':'auto',
        'origin':'lower',
        'extent':[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2]
    }

    Plot = TiledPlot(23.4, 33.1)
    Plot.title('Симуляция массива линз')

    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(field).squeeze().swapdims(0,1).cpu(), **arguments)
    Plot.graph.title('Исходная картинка')

    axes = Plot.axes.add((0, 1))
    axes.imshow(torch.angle(Lenses.propagation_buffer.get()).squeeze().swapdims(0,1), **arguments)
    Plot.graph.title('Фазовая маска массива линз')

    Plot.description.row.right('Вариация фокусного расстояния', 0)
    Plot.description.row.right('Вариация размера плоскости', 1)

    Plot.description.bottom('Длинна волны: ' + Format.Engineering(Lenses.wavelength.item(),'m') + ', Стандартное фокусное расстояние: ' + Format.Engineering(f,'m') + ', Количество пикселей: ' + str(pixels) + ', Падинг 3*размер поля' + ', Базовый размер поля: ' + Format.Engineering(plane_length,'m'))

    f_ = f

    for i, f in enumerate(CycleTimePredictor([5*mm, 10*mm, 20*mm, 30*mm, 50*mm]), 1):
        L = 2 * N * f
        b = L * f / (L - f)
        Lenses.focus = f
        Propagation1.distance = L
        Propagation2.distance = b
        axes = Plot.axes.add((i, 0))
        field_ = Propagation2(Lenses(Propagation1(field.clone())))
        axes.imshow(torch.abs(field_).squeeze().swapdims(0,1).cpu(), **arguments)
        Plot.graph.title('Фокусное расстояние: ' + Format.Engineering(f, 'm'))

    f = f_
    L = 2 * N * f
    b = L * f / (L - f)
    Lenses.focus = f
    Propagation1.distance = L
    Propagation2.distance = b

    for i, pl in enumerate(CycleTimePredictor([2*mm, 3*mm, 4*mm, 5*mm, 6*mm]), 1):
        Lenses.plane_length = pl
        Propagation1.plane_length = pl
        Propagation1.border = 3*pl
        Propagation2.plane_length = pl
        Propagation2.border = 3*pl
        axes = Plot.axes.add((i, 1))
        field_ = Propagation2(Lenses(Propagation1(field.clone())))
        axes.imshow(torch.abs(field_).squeeze().swapdims(0,1).cpu(), **arguments)
        Plot.graph.title('Размер поля: ' + Format.Engineering(pl, 'm'))


    Plot.show()

if __name__ == '__main__':
    print("Запуск теста линз")
    # ComparePropagationLayers(FourierPropagationLayer(), KirchhoffPropagationLayer())
    # CompareFourierTransformations(FourierPropagationLayer(border=12*1.0E-3, up_scaling=24), LensLayer())
    # TestMultiLens()
