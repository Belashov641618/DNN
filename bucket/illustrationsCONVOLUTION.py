import numpy
import torch
from torchvision.transforms.functional import pad, resize
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

def GenerateGaborFilter(N:int=10, lam:float=1.0, theta:float=0, psi:float=0, sigma:float=1.0, gamma:float=1.0):
    mesh = torch.linspace(-1,+1,N)
    x_grid_, y_grid_ = torch.meshgrid(mesh, mesh, indexing='ij')
    x_grid = x_grid_*numpy.cos(theta) + y_grid_*numpy.sin(theta)
    y_grid = y_grid_*numpy.cos(theta) - x_grid_*numpy.sin(theta)
    return numpy.exp(-(x_grid**2 + (gamma*y_grid)**2)/(2*sigma**2))*torch.cos(2*torch.pi*x_grid/lam + psi)

def multy_kernel_convolution(filters:bool=False, convolutions:bool=True, summ:bool=True, core:bool=False, compare:bool=False):
    with torch.no_grad():
        n = 9
        kernels_amount = n*n
        kernel_pixels = 15 #15
        image_address = 'C:/Users/uclap/Downloads/ConvolutionPhoto1.jpg'

        lam = 2.0 / 5.0
        psi = 0.0
        sigma = 0.25
        gamma = 0.7

        kernels = [GenerateGaborFilter(kernel_pixels, lam, theta, psi, sigma, gamma) for theta in numpy.linspace(0, 2*numpy.pi, kernels_amount+1)[:-1]]
        image = Image.open(image_address).convert('RGB')
        size = image.size
        image = torch.tensor(image.getdata())
        image = torch.reshape(image, (size[0], size[1], 3))
        image = torch.sum(image, dim=2).to(torch.float32)
        image = resize(image.unsqueeze(0).unsqueeze(0), [int(size[0]/3), int(size[1]/3)], antialias=True).squeeze()
        size = list(image.size())
        if size[0] % 2 == 0:
            image = image[1:, :]
            size = (size[0]-1, size[1])
        if size[1] % 2 == 0:
            image = image[:, 1:]
            size = (size[0], size[1]-1)

        padding = int((kernel_pixels-1)/2)
        image_ = pad(image, [padding, padding, padding, padding])
        multy_results = [torch.conv2d(image_.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze() for kernel in CycleTimePredictor(kernels)]

        if filters:
            Plot = TiledPlot(**plot_arguments)
            Plot.title('Фильтры габора')
            for kernel, (col, row) in zip(CycleTimePredictor(kernels), product(range(n), range(n))):
                axes = Plot.axes.add((col, row))
                axes.imshow(kernel, **image_arguments)
            Plot.show()

        if convolutions:
            Plot = TiledPlot(**plot_arguments)
            Plot.title('Свёртки с фильтрами')
            for result, (col, row) in zip(CycleTimePredictor(multy_results), product(range(n), range(n))):
                axes = Plot.axes.add((col, row))
                axes.imshow(result, **image_arguments)
            Plot.show()

        result_ = torch.zeros(size)
        for result in CycleTimePredictor(multy_results):
            result_ += result
        if summ:
            Plot = TiledPlot(**plot_arguments)
            Plot.title('Сумма свёрток')
            axes = Plot.axes.add((0, 0))
            axes.imshow(result, **image_arguments)
            Plot.show()

        multy_kernel_size = n*kernel_pixels + (n - 1)*(size[0]-1)
        multi_kernel = torch.zeros((multy_kernel_size, multy_kernel_size))
        for (i,j), kernel in zip(product(range(n), range(n)), kernels):
            x1 = i * (kernel_pixels + size[0] - 1)
            x2 = x1 + kernel_pixels
            y1 = j * (kernel_pixels + size[0] - 1)
            y2 = y1 + kernel_pixels
            print(x1,x2,y1,y2)
            multi_kernel[x1:x2,y1:y2] = kernel

        padding = int((multy_kernel_size-1)/2) + n*(kernel_pixels+size[0]-1) - size[0]
        padding = multy_kernel_size - int((kernel_pixels - 1)/2)
        image_ = pad(image, [padding, padding, padding, padding])
        print(image_.size())
        multi_result = torch.conv2d(image_.unsqueeze(0).unsqueeze(0), multi_kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        #multi_result = torch.zeros_like(image_)

        if compare:
            Plot = TiledPlot(**plot_arguments)
            Plot.title('Сравнение')
            axes = Plot.axes.add((0, 0), (n-1, n-1))
            axes.imshow(image, **image_arguments)
            #Plot.graph.title('Исходное изображение')

            axes = Plot.axes.add((n, 0), (2*n-1, n-1))
            axes.imshow(multi_kernel, **image_arguments)
            #Plot.graph.title('Составное ядро')

            for kernel, (x, y) in zip(kernels, product(range(n), range(n))):
                axes = Plot.axes.add((2*n+x, 0+y))
                axes.imshow(kernel, **image_arguments)

            axes = Plot.axes.add((0, n), (n-1, 2*n-1))
            axes.imshow(result_, **image_arguments)
            #Plot.graph.title('Сумма свёрток')

            axes = Plot.axes.add((n, n), (2*n-1, 2*n-1))
            axes.imshow(multi_result, **image_arguments)
            #Plot.graph.title('Составная свёртка')

            for result, (x, y) in zip(multy_results, product(range(n), range(n))):
                axes = Plot.axes.add((2*n+x, n+y))
                axes.imshow(result, **image_arguments)

            Plot.show()

def image_multiplication():
    image_path = 'C:/Users/uclap/Downloads/Laboratory.jpg'
    Copies = 3
    L = 1.0
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        size = image.size
        image = torch.tensor(image.getdata())
        image = torch.reshape(image, (size[0], size[1], 3))
        image = torch.sum(image, dim=2).to(torch.float32)
        pixels = size[0]
        padding = int((Copies-1)/2)*pixels
        image = pad(image, [padding, padding, padding, padding])

        d = L / (pixels * Copies)

        fft = torch.fft.fftshift(torch.fft.fft2(image))
        frequencies = torch.fft.fftfreq(Copies*pixels, d)
        fft_ = torch.zeros_like(fft, dtype=torch.complex64)
        for i in range(0, pixels*Copies, 3):
            fft_[::Copies, i] = fft[::Copies, i]
            fft_[i, ::Copies] = fft[i, ::Copies]

        ifft = torch.fft.ifft2(torch.fft.ifftshift(fft_))

        Plot = TiledPlot(**plot_arguments)

        axes = Plot.axes.add((0, 0))
        axes.imshow(torch.abs(image), **image_arguments)
        Plot.graph.title('Изображение')

        axes = Plot.axes.add((0, 1))
        axes.imshow(torch.log10(torch.abs(fft) + 1.0), **image_arguments)
        Plot.graph.title('Фурье образ')

        axes = Plot.axes.add((1, 1))
        axes.imshow(torch.log10(torch.abs(fft_) + 1.0), **image_arguments)
        Plot.graph.title('Изменённый Фурье образ')

        axes = Plot.axes.add((1, 0))
        axes.imshow(torch.abs(ifft), **image_arguments)
        Plot.graph.title('Размноженное изображение')

        Plot.show()




if __name__ == '__main__':
    print('Convolution illustrations:')
    # plot_discrete_convolution()
    multy_kernel_convolution()
    #image_multiplication()