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
    'MaxWidth'  : 12.0,
    'MaxHeight' : 12.0*9/16,
}

def plot_discrete_convolution():
    N = 1000
    M = int(N/3)
    S = 5
    frequencies = numpy.linspace(-10000,+10000,N)
    smooth_kernel = 1.0 / ((numpy.linspace(-1, +1, M))**2 + 1.0)**10
    input_array = numpy.convolve(numpy.random.rand(N), smooth_kernel, mode='same')
    input_array = (input_array - numpy.min(input_array)) / (numpy.max(input_array) - numpy.min(input_array))
    kernel_array = numpy.convolve(numpy.random.rand(N), smooth_kernel, mode='same')
    kernel_array = (kernel_array - numpy.min(kernel_array)) / (numpy.max(kernel_array) - numpy.min(kernel_array))
    multiplication_array = kernel_array * input_array

    frequencies_step = (numpy.max(frequencies) - numpy.min(frequencies)) / S
    frequencies_S = numpy.linspace(numpy.min(frequencies) + frequencies_step/2, numpy.max(frequencies)-frequencies_step/2, S)
    index_S = numpy.array(N*(frequencies_S - numpy.min(frequencies))/(numpy.max(frequencies) - numpy.min(frequencies)) ,dtype=int)
    kernel_array_S = kernel_array[index_S]

    Plot = TiledPlot(**plot_arguments)
    Plot.FontLibrary.MultiplyFontSize(0.7)
    Plot.graph.width_to_height(1.7)
    Plot.title('Отличие дискретной и непрерывной свёртки')

    axes = Plot.axes.add((0,0))
    Plot.graph.title('Непрерывные Фурье образы ядра и входной функции')
    axes.plot(frequencies, kernel_array,
                linestyle='--', color='green', label='Фурье образ ядра')
    axes.plot(frequencies, input_array,
                linestyle='-', color='orange', label='Фурье образ входной функции')
    axes.scatter(frequencies_S, kernel_array_S,
                color='green', label='Точки дискретизации')
    axes.grid(True)
    axes.legend()

    kernel_array_D = numpy.zeros(N)
    start = 0
    for i, value in enumerate(kernel_array_S):
        stop = frequencies_step*(i+1)
        stop = int(N*stop/(numpy.max(frequencies)-numpy.min(frequencies)))
        kernel_array_D[start:stop] = value
        start = stop
    multiplication_array_D = kernel_array_D * input_array

    axes = Plot.axes.add((1,0))
    Plot.graph.title('Непрерывный Фурье образ входной функции и дискретизованный образ ядра')
    axes.plot(frequencies, kernel_array_D,
              linestyle='--', color='green', label='Дискретизованный Фурье образ ядра')
    axes.plot(frequencies, kernel_array,
              linestyle=':', color='lightgreen', label='Фурье образ ядра')
    axes.scatter(frequencies_S, kernel_array_S,
                 color='green')
    axes.plot(frequencies, input_array,
              linestyle='-', color='orange')
    axes.grid(True)
    axes.legend()


    axes = Plot.axes.add((0, 1))
    Plot.graph.title('Результат перемножения образов без/c дискретизации ядра')
    axes.plot(frequencies, multiplication_array,
              linestyle='-', color='lightblue', label='Без дискретизации')
    axes.plot(frequencies, multiplication_array_D,
              linestyle='-', color='darkblue', label='С дискретизацией')
    axes.grid(True)
    axes.legend()

    axes = Plot.axes.add((1, 1))
    Plot.graph.title('Относительная разница результатов')
    axes.plot(frequencies, numpy.abs(multiplication_array - multiplication_array_D),
              linestyle='-', color='gray')
    axes.grid(True)

    Plot.show()

if __name__ == '__main__':
    print('Convolution illustrations:')
    plot_discrete_convolution()