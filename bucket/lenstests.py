import torch
import numpy
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
from src.modules.layers.AbstarctLayer import AbstractLayer
from src.examiner.manager.parameters.generator import get

FormatWidth = 23.4/4
FormatHeight = 33.1/4/2

def FormatProperties(Layer:AbstractLayer):
    properties_to_describe = {}
    description = []
    for prop in get.properties(Layer):
        if hasattr(getattr(Layer, prop), 'get'):
            value = getattr(Layer, prop).get()
        else:
            value = getattr(Layer, prop)
        if torch.is_tensor(value):
            if torch.numel(value) == 1:
                properties_to_describe[prop] = value.item()
        else:
            properties_to_describe[prop] = value
    for (prop, value) in properties_to_describe.items():
        if prop in ['wavelength', 'plane_length', 'distance']:
            description.append(prop + ":" + Format.Engineering(value, 'm'))
        else:
            description.append(prop + ":" + str(value))
    description = ', '.join(description)
    return description

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

    axes = Plot.axes.add((0, 0))
    axes.imshow(amplitude_difference, aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1, 0))
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

    axes = Plot.axes.add((1, 0))
    axes.imshow(torch.real(real), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

    axes = Plot.axes.add((1, 1))
    axes.imshow(torch.imag(real), aspect='auto', extent=[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2], origin='lower')

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

def TestMultiLens(N:int=2, f:float=100*mm):
    raw_pixels = 1000
    up_scaling = 1
    pixels = raw_pixels*up_scaling
    plane_length = 10.0*mm

    L = 2 * N * f
    b = L * f / (L - f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Lenses = MultiLensLayer(N, f, 600*nm, plane_length=plane_length, pixels=raw_pixels, up_scaling=up_scaling).to(device)
    Propagation1 = FourierPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, L, 3*plane_length).to(device)
    Propagation2 = FourierPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, b, 3*plane_length).to(device)
    # Propagation1 = KirchhoffPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, L).to(device)
    # Propagation2 = KirchhoffPropagationLayer(600*nm, 1.0, plane_length, raw_pixels, up_scaling, b).to(device)

    transform = torchvision.transforms.Resize((pixels, pixels))
    field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True))).to(device).to(Propagation1._accuracy.tensor_complex)

    arguments = {
        'aspect':'auto',
        'origin':'lower',
        'extent':[-plane_length/2, +plane_length/2, -plane_length/2, +plane_length/2]
    }

    Plot = TiledPlot(FormatWidth, FormatHeight)
    Plot.title('Симуляция массива линз')

    description_propagation1    = 'Свойства слоя распростронения до линзы: ('       + FormatProperties(Propagation1)    + ')'
    description_propagation2    = 'Свойства слоя распростронения после линзы: ('    + FormatProperties(Propagation2)    + ')'
    description_lenses          = 'Свойства слоя массива линз: ('                   + FormatProperties(Lenses)          + ')'
    description = '\n'.join([description_propagation1, description_propagation2, description_lenses])
    Plot.description.bottom(description)

    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(field).squeeze().swapdims(0,1).cpu(), **arguments)
    Plot.graph.title('Исходная картинка')

    axes = Plot.axes.add((0, 1))
    axes.imshow(torch.angle(Lenses.propagation_buffer.get()).squeeze().swapdims(0,1), **arguments)
    Plot.graph.title('Фазовая маска массива линз')

    with torch.no_grad(): result = Propagation2(Lenses(Propagation1(field))).squeeze().cpu()

    axes = Plot.axes.add((1, 0))
    axes.imshow(torch.abs(result), **arguments)
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1, 1))
    axes.imshow(torch.angle(result), **arguments)
    Plot.graph.title('Фаза')

    axes = Plot.axes.add((2, 0))
    axes.imshow(torch.real(result), **arguments)
    Plot.graph.title('Реальная часть')

    axes = Plot.axes.add((2, 1))
    axes.imshow(torch.imag(result), **arguments)
    Plot.graph.title('Мнимая часть')

    Plot.description.column.top('Исходные данные', 0)
    Plot.description.column.top('Результаты', 1,2)

    Plot.show()

def ShowPropagationBuffer(Layer:AbstractPropagationLayer):
    buffer = Layer.propagation_buffer.get().squeeze()
    amplitude = torch.abs(buffer)
    phase = torch.angle(buffer)
    real = torch.real(buffer)
    imag = torch.imag(buffer)

    plane_length = (Layer.plane_length if hasattr(Layer, 'plane_length') else 1.0)

    arguments = {
        'aspect': 'auto',
        'origin': 'lower',
        'extent': [-plane_length / 2, +plane_length / 2, -plane_length / 2, +plane_length / 2]
    }

    Plot = TiledPlot(FormatWidth, FormatHeight)
    Plot.title("Буфер распространения метода: " + type(Layer).__name__)
    properties_to_describe = {}
    description = []
    for prop in get.properties(Layer):
        if hasattr(getattr(Layer, prop), 'get'):
            value = getattr(Layer, prop).get()
        else:
            value = getattr(Layer, prop)
        if torch.is_tensor(value):
            if torch.numel(value) == 1:
                properties_to_describe[prop] = value.item()
        else:
            properties_to_describe[prop] = value
    for (prop, value) in properties_to_describe.items():
        if prop in ['wavelength', 'plane_length', 'distance']:
            description.append(prop + ":" + Format.Engineering(value, 'm'))
        else:
            description.append(prop + ":" + str(value))
    description = ', '.join(description)
    Plot.description.bottom(description)

    axes = Plot.axes.add((0, 0))
    axes.imshow(amplitude, **arguments)
    Plot.graph.title('Амплитуда буфера')

    axes = Plot.axes.add((1, 0))
    axes.imshow(phase, **arguments, vmin=-numpy.pi, vmax=+numpy.pi)
    Plot.graph.title('Фаза буфера')

    axes = Plot.axes.add((0, 1))
    axes.imshow(real, **arguments)
    Plot.graph.title('Реальная часть буфера')

    axes = Plot.axes.add((1, 1))
    axes.imshow(imag, **arguments)
    Plot.graph.title('Мнимая часть буфера')

    Plot.show()

def TestMultiplication(N:int=3):
    field = torch.abs(GenerateSingleUnscaledSampleMNIST(True))
    print(field.size())
    size = field.size(2)
    # field[0][0][0][:] = torch.ones(size)
    # field[0][0][size-1][:] = torch.ones(size)
    # field[0][0][0][:] = torch.ones(size)
    # field[0][0][size - 1][:] = torch.ones(size)
    # field[0][0][:][0] = torch.ones(size)
    # field[0][0][:][size - 1] = torch.ones(size)
    result_ = torch.tile(field, (N, N))

    field = torch.nn.functional.pad(field, ((int(N/2)+1)*size + int(N/2)*int(size/2), (int(N/2)+1)*size + int(N/2*size/2), (int(N/2)+1)*size + int(N/2)*int(size/2), (int(N/2)+1)*size + int(N/2)*int(size/2)), mode='constant')
    # field = torch.nn.functional.pad(field, ((N+2)*size, (N+2)*size, (N+2)*size, (N+2)*size), mode='constant')
    kernel = torch.zeros((size,size))
    kernel[int(size/2)][int(size/2)] = 1.0
    kernel = torch.tile(kernel, (N, N))
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    result = torch.nn.functional.conv2d(field, kernel, dilation=(1,1))

    arguments = {
        'aspect': 'auto',
        'origin': 'lower'
    }

    Plot = TiledPlot(FormatWidth, FormatHeight)
    Plot.title('Сравнение')

    axes = Plot.axes.add((0, 0))
    axes.imshow(field.squeeze(), **arguments)
    Plot.graph.title('Исходная картинка')

    axes = Plot.axes.add((1, 0))
    axes.imshow(result_.squeeze(), **arguments)
    Plot.graph.title('Необходимый результат')

    axes = Plot.axes.add((0, 1))
    axes.imshow(kernel.squeeze(), **arguments)
    Plot.graph.title('Ядро свёртки')

    axes = Plot.axes.add((1, 1))
    axes.imshow(result.squeeze(), **arguments)
    Plot.graph.title('Результат')

    Plot.show()


def TestConvMultiplication(N:int=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    focus = 300*mm
    plane_length = 10.0*mm
    pixels = 51

    # Propagation = KirchhoffPropagationLayer(plane_length=plane_length, pixels=N*pixels, up_scaling=1, distance=focus).to(device)
    Propagation = FourierPropagationLayer(plane_length=plane_length, pixels=N*pixels, up_scaling=1, distance=focus, border=25*plane_length).to(device)
    Lens = LensLayer(plane_length=plane_length, pixels=N*pixels, up_scaling=1, focus=focus).to(device)

    s = 1
    n0 = int(pixels/2 - s)
    n1 = n0 + s
    kernel = torch.zeros((1, 1, pixels, pixels))
    kernel[0][0][n0:n1, n0:n1] = torch.ones((s, s))
    kernel = torch.tile(kernel, (1,1,N,N))
    kernel_ = kernel.clone()
    kernel = torch.fft.fftshift(torch.fft.fft2(kernel)).to(device)

    kernel = (torch.ones((1,1,N*pixels,N*pixels)) / (16.0 * plane_length*N * plane_length*N)).to(device)

    kernel = torch.zeros((1,1,N*pixels,N*pixels)).to(device)
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(pixels*N, d=plane_length / (pixels*N), device=device))
    min_freq, max_freq = torch.min(frequencies), torch.max(frequencies)
    # print(frequencies)
    frequencies = (frequencies * plane_length / (2.0 * torch.pi)).to(torch.int32)
    # print(frequencies)
    frequencies = torch.arange(min(frequencies), max(frequencies)+1, device=device)*2*torch.pi / plane_length
    print(frequencies)
    numbers = (N*pixels * (frequencies - min_freq) / (max_freq - min_freq)).to(torch.long)
    mesh = torch.meshgrid(numbers, numbers, indexing='ij')
    print(numbers)
    kernel[0][0][mesh] = 1.0 / (16.0 * plane_length*N * plane_length*N)
    # for n in numbers:
    #     for m in numbers:
    #         kernel[0][0][n][m] = 1.0 / (16.0 * plane_length*N * plane_length*N)

    transform = torchvision.transforms.Resize((pixels, pixels))
    field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True))).to(device)
    result = torch.tile(field, (N, N))
    field = torch.nn.functional.pad(field, (pixels*int(N/2), pixels*int(N/2), pixels*int(N/2), pixels*int(N/2))).to(torch.complex64).to(device)

    field_ = field.clone()

    field_ = Propagation(Lens(Propagation(field_)))
    field_ = field_*kernel
    field_ = Propagation(Lens(Propagation(field_))).cpu()

    arguments = {
        'aspect': 'auto',
        'origin': 'lower'
    }

    Plot = TiledPlot(FormatWidth, FormatHeight)
    Plot.title('Сравнение')

    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(field.squeeze()).cpu(), **arguments)
    Plot.graph.title('Исходная картинка')

    axes = Plot.axes.add((1, 0))
    axes.imshow(torch.abs(kernel_.squeeze()).cpu(), **arguments)
    Plot.graph.title('Ядро свёртки')

    axes = Plot.axes.add((0, 1))
    axes.imshow(torch.abs(kernel.squeeze()).cpu(), **arguments)
    Plot.graph.title('Фурбе образ ядра свёртки')

    axes = Plot.axes.add((1, 1))
    axes.imshow(torch.abs(field_.squeeze()).cpu(), **arguments)
    Plot.graph.title('Результат')

    Plot.show()


def CompareConvolutionMultiplication(copies=3, plane_length=30*mm, wavelength=600*nm, focus=300*mm, pixels=(15, 31, 63, 127, 255, 511, 1023), choose_input=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Propagation = KirchhoffPropagationLayer(plane_length=plane_length, pixels=copies * max(pixels), up_scaling=1, distance=focus, wavelength=wavelength).to(device)
    Lens = LensLayer(plane_length=plane_length, pixels=copies * max(pixels), up_scaling=1, focus=focus, wave_length=wavelength).to(device)

    arguments = {
        'aspect': 'auto',
        'origin': 'lower',
        'extent': [0, +plane_length, 0, plane_length]
    }

    transform = torchvision.transforms.Resize((max(pixels), max(pixels)))
    field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True))).to(device).swapdims(2,3)
    if choose_input:
        answer = False
        while not answer:
            field = transform(torch.abs(GenerateSingleUnscaledSampleMNIST(True))).to(device).swapdims(2,3)
            Plot = TiledPlot(FormatWidth, FormatHeight)
            Plot.title('Вас устраивает картинка?\n1-Да, 0-Нет')
            axes = Plot.axes.add((0, 0))
            axes.imshow(torch.abs(field).cpu().squeeze(), **arguments)
            Plot.show()
            answer = True
    field0 = field.clone()
    result = torch.abs(torch.tile(torch.flip(field, (2,3)), (copies, copies)))
    field = torch.nn.functional.pad(field, (max(pixels) * int(copies / 2), max(pixels) * int(copies / 2), max(pixels) * int(copies / 2), max(pixels) * int(copies / 2))).to(torch.complex64).to(device)
    fft = torch.fft.fftshift(torch.fft.fft2(field))

    Plot = TiledPlot(MaxWidth=FormatWidth*2, MaxHeight=FormatHeight*2)
    Plot.FontLibrary.MultiplyFontSize(0.7)
    Plot.title('Сравнение результата размножения изображения c помошью свёртки')
    Plot.description.top('Вычисляется картинка, создаваемая 4f системой при различной дискретизации поля расчёта')
    Plot.description.bottom('Параметры эксперимента:\n'
                            'Длинна волны:' + Format.Engineering(wavelength, 'm') + ', '
                            'Размер плоскости:' + Format.Engineering(plane_length, 'm') + ', '
                            'Фокусное расстояние линз:' + Format.Engineering(focus, 'm') + ', '
                            'Количество копий:' + str(copies) + ' шт.')

    axes = Plot.axes.add((0, 0),(1, 1))
    axes.imshow(torch.abs(field).squeeze().cpu(), **arguments)
    axes = Plot.axes.add((0, 2),(1, 3))
    axes.imshow(torch.abs(fft).squeeze().cpu(), **arguments)
    axes = Plot.axes.add((0, 4), (1, 5))
    axes.imshow(torch.angle(fft).squeeze().cpu(), **arguments)
    axes = Plot.axes.add((0, 6), (1, 7))
    axes.imshow(torch.abs(result).squeeze().cpu(), **arguments)
    Plot.description.column.top('Референсные картинки', 0, 1)

    Plot.description.row.left('Изображение', 0, 1)
    Plot.description.row.left('Амплитуда Фурье образа', 2, 3)
    Plot.description.row.left('Фаза Фурье образа', 4, 5)
    Plot.description.row.left('Результат', 6, 7)

    result_amplitude_difference_history = []
    result_phase_difference_history = []
    fft_amplitude_difference_history = []
    fft_phase_difference_history = []

    with torch.no_grad():
        for i, pixels_ in enumerate(CycleTimePredictor(pixels), 1):
            Plot.description.column.top('Кол-во пикселей: ' + str(pixels_), 2*i, 2*i+1)

            Propagation.pixels = pixels_*copies
            Lens.pixels = pixels_*copies

            transform_ = torchvision.transforms.Resize((pixels_, pixels_))
            field_ = transform_(torch.abs(field0.clone()))
            field_ = torch.nn.functional.pad(field_, (pixels_ * int(copies / 2), pixels_ * int(copies / 2), pixels_ * int(copies / 2), pixels_ * int(copies / 2))).to(torch.complex64).to(device)
            axes = Plot.axes.add((0+2*i, 0), (1+2*i, 1))
            axes.imshow(torch.abs(field_).squeeze().cpu(), **arguments)

            fft_ = Propagation(Lens(Propagation(field_)))
            axes = Plot.axes.add((0+2*i, 2), (1+2*i, 3))
            axes.imshow(torch.abs(fft_).squeeze().cpu(), **arguments)
            axes = Plot.axes.add((0+2*i, 4), (1+2*i, 5))
            axes.imshow(torch.angle(fft_).squeeze().cpu(), **arguments)

            kernel = torch.zeros((1,1,copies*pixels_,copies*pixels_)).to(device)
            frequencies = torch.fft.fftshift(torch.fft.fftfreq(pixels_*copies, d=plane_length / (pixels_*copies), device=device))
            min_freq, max_freq = torch.min(frequencies), torch.max(frequencies)
            frequencies = (frequencies * plane_length / (2.0 * torch.pi)).to(torch.int32)
            frequencies = torch.arange(min(frequencies), max(frequencies)+1, device=device)*2*torch.pi / plane_length
            numbers = (copies*pixels_ * (frequencies - min_freq) / (max_freq - min_freq)).to(torch.long)
            mesh = torch.meshgrid(numbers, numbers, indexing='ij')
            kernel[0][0][mesh] = 1.0 / (16.0 * plane_length*copies * plane_length*copies)

            fft__ = fft_*kernel

            result_ = Propagation(Lens(Propagation(fft__)))
            axes = Plot.axes.add((0+2*i, 6), (1+2*i, 7))
            axes.imshow(torch.abs(result_).squeeze().cpu(), **arguments)

            def unwrap(x):
                return torch.min(x, torch.abs(x-2.0*torch.pi))

            transform__ = torchvision.transforms.Resize((copies*pixels_, copies*pixels_))
            result_amplitude_difference_history.append(torch.mean(torch.abs(transform__(torch.abs(result)) - torch.abs(result_))).squeeze().item())
            fft_amplitude_difference_history.append(torch.mean(torch.abs(transform__(torch.abs(fft)) - torch.abs(fft_))).squeeze().item())
            result_phase_difference_history.append(torch.mean(unwrap(torch.abs(transform__(torch.angle(result)) - torch.angle(result_)))).squeeze().item())
            fft_phase_difference_history.append(torch.mean(unwrap(torch.abs(transform__(torch.angle(fft)) - torch.angle(fft_)))).squeeze().item())

    axes = Plot.axes.add((2 + 2 * len(pixels), 0), (3 + 2 * len(pixels), 1))
    axes.grid(True)
    if max(result_amplitude_difference_history) / min(result_amplitude_difference_history) >= 100:
        axes.semilogy()
    axes.plot(pixels, result_amplitude_difference_history)
    axes.scatter(pixels, result_amplitude_difference_history)
    Plot.description.row.right('Среднее отлонение амплитуды результата', 0, 1)

    axes = Plot.axes.add((2 + 2 * len(pixels), 2), (3 + 2 * len(pixels), 3))
    axes.grid(True)
    if max(result_phase_difference_history) / min(result_phase_difference_history) >= 100:
        axes.semilogy()
    axes.plot(pixels, result_phase_difference_history)
    axes.scatter(pixels, result_phase_difference_history)
    Plot.description.row.right('Среднее отлонение фазы результата', 2, 3)

    axes = Plot.axes.add((2 + 2 * len(pixels), 4), (3 + 2 * len(pixels), 5))
    axes.grid(True)
    if max(fft_amplitude_difference_history) / min(fft_amplitude_difference_history) >= 100:
        axes.semilogy()
    axes.plot(pixels, fft_amplitude_difference_history)
    axes.scatter(pixels, fft_amplitude_difference_history)
    Plot.description.row.right('Среднее отлонение амплитуды Фурье образа', 4, 5)

    axes = Plot.axes.add((2 + 2 * len(pixels), 6), (3 + 2 * len(pixels), 7))
    axes.grid(True)
    if max(fft_phase_difference_history) / min(fft_phase_difference_history) >= 100:
        axes.semilogy()
    axes.plot(pixels, fft_phase_difference_history)
    axes.scatter(pixels, fft_phase_difference_history)
    Plot.description.row.right('Среднее отлонение фазы Фурье образа', 6, 7)

    Plot.description.column.top('Средние отклонения', 2+2*len(pixels), 3+2*len(pixels))

    Plot.show()

if __name__ == '__main__':
    print("Запуск теста линз")
    # CompareFourierTransformations(FourierPropagationLayer(border=12*1.0E-3, up_scaling=24), LensLayer())
    # TestMultiLens()
    # ShowPropagationBuffer(KirchhoffPropagationLayer())
    # ComparePropagationLayers(FourierPropagationLayer(), KirchhoffPropagationLayer())
    # TestMultiplication(5)
    # TestConvMultiplication()
    CompareConvolutionMultiplication(plane_length=10*mm, pixels=(7,15,31), choose_input=True)
