import torch
import numpy
from PIL import Image
from itertools import product
import matplotlib.pyplot as plt

from belashovplot import TiledPlot

from src.utilities.CycleTimePredictor import CycleTimePredictor
from src.utilities.Formaters import Format
from src.utilities.DecimalPrefixes import nm,um,mm,cm

image_arguments = {
    'aspect' : 'auto',
    'cmap' : 'gray'
}
plot_arguments = {
    'MaxWidth'  : 14.0,
    'MaxHeight' : 14.0*9/16,
}

def plot_image(image):
    Plot = TiledPlot(**plot_arguments)
    Plot.title('Исходное поле')

    axes = Plot.axes.add((0,0))
    axes.imshow(numpy.abs(image), **image_arguments)
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1, 0))
    axes.imshow(numpy.angle(image), **image_arguments)
    Plot.graph.title('Фаза')

    Plot.show()

def plot_fft(image):
    image = numpy.fft.fftshift(numpy.fft.fft2(image))
    Plot = TiledPlot(**plot_arguments)
    Plot.title('Фурье образ поля')

    axes = Plot.axes.add((0,0))
    axes.imshow(numpy.abs(image), **image_arguments)
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1, 0))
    axes.imshow(numpy.angle(image), **image_arguments)
    Plot.graph.title('Фаза')

    Plot.show()

def plot_magnitude(image, kx=5.0, ky=5.0):
    N = image.shape[0]
    x_mesh, y_mesh = numpy.meshgrid(numpy.arange(N), numpy.arange(N))
    magnitude = 1.0*numpy.exp(1j*(kx*x_mesh + ky*y_mesh))

    Plot = TiledPlot(**plot_arguments)
    Plot.title('Мода Фурье обрза')

    axes = Plot.axes.add((0, 0))
    axes.imshow(numpy.abs(magnitude), **image_arguments, vmin=0, vmax=1)
    Plot.graph.title('Амплитуда')

    axes = Plot.axes.add((1, 0))
    axes.imshow(numpy.angle(magnitude), **image_arguments)
    Plot.graph.title('Фаза')

    axes = Plot.axes.add((0, 1))
    axes.imshow(numpy.real(magnitude), **image_arguments)
    Plot.graph.title('Вещественная часть')

    axes = Plot.axes.add((1, 1))
    axes.imshow(numpy.imag(magnitude), **image_arguments)
    Plot.graph.title('Мнимая часть')

    Plot.show()

def plot_image_from_magnitudes(image, steps=4):
    fft = numpy.fft.fftshift(numpy.fft.fft2(image))

    Plot = TiledPlot(**plot_arguments)
    Plot.FontLibrary.Fonts.GraphTitle.FontSize *= 0.7
    Plot.title('Восстановление изображения из Фурье образа')

    rows = int(numpy.sqrt(steps))
    cols = int(steps/rows) + 1 - (steps%rows == 0)

    N = image.shape[0]
    n = N / steps
    deltas = numpy.geomspace(10, N, steps+1)
    for i, (row, col) in zip(CycleTimePredictor(range(1, steps+1)), product(range(rows), range(cols))):
        left = int(N/2-deltas[i]/2)
        right = int(N/2+deltas[i]/2)
        print(left, right, N, i, steps)
        fft_ = numpy.zeros_like(fft, dtype=complex)
        fft_[left:right, left:right] = fft[left:right, left:right]
        image_ = numpy.fft.ifft2(numpy.fft.ifftshift(fft_))

        axes = Plot.axes.add((col, row))
        axes.imshow(numpy.abs(image_), **image_arguments)
        Plot.graph.title('Процент мод: ' + str(int(100*deltas[i]/N)) + '%')

    Plot.show()
def check_fft(dl=1.0, N=300):
    frequencies = numpy.fft.fftfreq(N, dl)
    df = 1.0/(N*dl)
    k_min = min(frequencies)
    k_max = max(frequencies)
    print('N =', N, ', dl =', dl)
    print('dk =', df, ', real =', frequencies[1]-frequencies[0])
    print('k_min =', k_min, ', n_min =', k_min/df)
    print('k_max =', k_max, ', n_max =', k_max/df)

def main():
    image = Image.open('C:/Users/uclap/Downloads/Putin.jpg').convert('RGB')
    image = numpy.moveaxis(numpy.array(image.getdata()).reshape(list(image.size) + [3]), 2, 0)
    image = numpy.sum(image, axis=0) * numpy.exp(2j * numpy.pi * image[0])
    #plot_image(image)

    #check_fft(1.0, 10)
    #check_fft(1.0, 11)

    #plot_fft(image)
    #plot_magnitude(image, 1/20, 1/10)
    plot_image_from_magnitudes(image, 9)

if __name__ == '__main__':
    main()