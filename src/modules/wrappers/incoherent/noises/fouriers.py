import math

import matplotlib.pyplot
import torch
import numpy
from tqdm import tqdm

from typing import Iterable, Union, Tuple
from itertools import product
from belashovplot import TiledPlot
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation
# animation.writers['ffmpeg'].executable = 'C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe'

if __name__ == '__main__':
    from filters import Gaussian as _gaussian_generator
else:
    from . import gaussian as _gaussian_generator

class FourierMask:
    _mask:torch.Tensor
    @property
    def dims(self):
        return len(self._mask.size())
    @property
    def size(self):
        return self._mask.size()
    @property
    def device(self):
        return self._mask.device
    @property
    def dtype(self):
        return self._mask.dtype

    def __init__(self, mask:torch.Tensor):
        self._mask = mask

    def sample(self) -> torch.Tensor:
        spectrum = torch.rand(self.size, device=self.device, dtype=self.dtype)*torch.exp(2j*torch.pi*torch.rand(self.size, device=self.device, dtype=self.dtype)) * self._mask
        return torch.abs(torch.fft.ifftn(spectrum)).to(self.dtype)

def gaussian(sigmas:Union[Iterable[float],float], counts:Union[Iterable[int],int], limits:Union[Iterable[Tuple[float,float]],Tuple[float, float]]=None, device:torch.device=None, generator:bool=False) -> Union[torch.Tensor, FourierMask]:
    sigmas_:Tuple[float, ...]
    if isinstance(sigmas, float):   sigmas_ = (sigmas, )
    else:                           sigmas_ = tuple(sigmas)

    dims = len(sigmas_)

    counts_:Tuple[int, ...]
    if isinstance(counts, int):     counts_ = (counts, )
    else:                           counts_ = tuple(counts)

    limits_:Tuple[Tuple[float, float], ...]
    if limits is None:              limits_ = tuple([(-1., +1.) for _ in range(dims)])
    elif isinstance(limits, tuple) and len(limits) == 2 and all(isinstance(item, float) for item in limits):
        limits:tuple[float, float]
        limits_ = (limits, )
    else:                           limits_ = tuple(limits)

    if len(sigmas_) != dims or len(counts_) != dims or len(limits_) != dims: raise AssertionError('Lengths of sigmas and counts and limits are not equal')

    sigmas_temp = []
    for sigma, count in zip(sigmas_, counts_):
        sigmas_temp.append(sigma*100/count)
    sigmas_ = tuple(sigmas_temp)

    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coordinates:list[torch.Tensor] = [torch.linspace(limit0, limit1, count, device=device) for (limit0, limit1), count in zip(limits_, counts_)]
    mask = _gaussian_generator(sigmas_)(*coordinates)
    if generator:
        return FourierMask(mask)
    else:
        return FourierMask(mask).sample()




test_figure_width:float  = 16.
test_figure_height:float = 9.
def test_gaussian_3d(samples_t:int, samples_xy:int, Nxy=1000, fps:float=30, seconds:float=5.0):
    Nt = int(seconds*fps)
    limits = ((-1., +1.), (-1., +1.), (0., 1.))
    sigmas_xy = numpy.logspace(-2, 0, samples_xy)
    sigmas_t = numpy.logspace(-2, 0, samples_t)

    col_generator = zip(range(samples_t),  sigmas_t)
    row_generator = zip(range(samples_xy), sigmas_xy)

    data:list[tuple[int, int, torch.Tensor], ...] = []
    for (col, sigma_t), (row, sigma_xy) in tqdm(product(col_generator, row_generator)):
        sigmas = (sigma_t, sigma_xy, sigma_xy)
        counts = (Nt, Nxy, Nxy)
        data.append((col, row, gaussian(sigmas, counts, limits).cpu()))

    plot = TiledPlot(test_figure_width, test_figure_height)
    plot.FontLibrary.MultiplyFontSize(0.7)
    plot.FontLibrary.SynchronizeFont('DejaVu Sans')
    plot.title('Генерация шума через модуляцию случайного Фурье образа Гауссом')

    for col, sigma_t in col_generator:
        power = int(math.log10(sigma_t))
        number = round(sigma_t / (10**power), 2)
        plot.description.column.top(f'{number}*10^{power}', col)
    for row, sigma_xy in row_generator:
        power = int(math.log10(sigma_xy))
        number = round(sigma_xy / (10**power), 2)
        plot.description.row.left(f'{number}*10^{power}', row)

    kwargs = {'aspect':'auto', 'cmap':'gray', 'extent':[-1., +1., -1., +1.]}
    images_list = []
    for col, row, video in data:
        axes = plot.axes.add(col, row)
        image = axes.imshow(video[0], **kwargs)
        images_list.append(image)
    plot.description.bottom(f'$t={round(0, 2)}s$')
    plot.finalize()

    iterator = iter(tqdm(range(Nt)))
    def update(frame:int):
        for (col_, row_, video_), image_ in zip(data, images_list):
            plot.description.bottom(f'$t={round(frame/(Nt-1), 2)}s$')
            image_.set_data(video_[frame])
        try:
            next(iterator)
        except StopIteration:
            pass
        return images_list
    animation = FuncAnimation(plot._Figure, update, frames=Nt, interval=seconds/Nt, blit=True)

    animation.save('test.gif')
    plot.show()
def test_gaussian_2d(samples:int, N=1000):
    limits = ((-1., +1.), (-1., +1.))
    sigmas = numpy.logspace(-2, 0, samples)

    plot = TiledPlot(test_figure_width, test_figure_height)
    plot.FontLibrary.MultiplyFontSize(0.7)
    plot.FontLibrary.SynchronizeFont('DejaVu Sans')
    plot.title('Генерация шума через модуляцию случайного Фурье образа Гауссом')

    plot.description.row.left('Результат', 0)
    plot.description.row.left('Ядро модуляции', 1)

    kwargs = {'aspect':'auto', 'cmap':'viridis', 'extent':[-1., +1., -1., +1.]}
    for (col, sigma) in zip(range(samples), sigmas):
        power = int(math.log10(sigma))
        number = round(sigma / (10 ** power), 2)
        # plot.description.column.top(f'$\\sigma={number}{{\bullet}}10^{{{power}}}$', col)
        plot.description.column.top(f'{number}*10^{power}', col)


        sigmas_ = (sigma, sigma)
        counts = (N, N)
        generator = gaussian(sigmas_, counts, limits, generator=True)
        core = generator._mask.cpu()
        image = generator.sample().cpu()
        spectrum = torch.fft.fftshift(torch.fft.fft2(image))

        axes = plot.axes.add(col, 0)
        axes.imshow(image, **kwargs)

        axes = plot.axes.add(col, 1)
        axes.imshow(core, **kwargs)
    plot.show()
def test():
    # test_gaussian_2d(7, 1000)
    test_gaussian_3d(5, 4, 512, 60, 3.0)
if __name__ == '__main__':
    test()