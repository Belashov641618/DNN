import belashovplot
import torch
import torchvision.transforms
from torchvision.transforms.functional import resize
from typing import Union, Iterable

from src.modules.layers.AbstractPropagationLayer import AbstractPropagationLayer
from src.utilities.DecimalPrefixes import nm, mm, um, cm

EQLMode = 1
PPRMode = 2
RPPMode = 3

class FourierResizingPropagationLayer(AbstractPropagationLayer):

    _antialiasing : bool
    @property
    def antialiasing(self):
        class Selector:
            _self : FourierResizingPropagationLayer
            def __init__(self, _self:FourierResizingPropagationLayer):
                self._self = _self
            def get(self):
                return self._self._antialiasing
            def enable(self):
                self._self._antialiasing = True
            def disable(self):
                self._self._antialiasing = False
        return Selector(self)

    _interpolation : torchvision.transforms.InterpolationMode
    @property
    def interpolation(self):
        class Selector:
            _self : FourierResizingPropagationLayer
            def __init__(self, _self):
                self._self = _self
            def nearest(self):
                self._self._interpolation = torchvision.transforms.InterpolationMode.NEAREST
            def bilinear(self):
                self._self._interpolation = torchvision.transforms.InterpolationMode.BILINEAR
            def bicubic(self):
                self._self._interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        return Selector(self)

    def _reset_options(self):
        self.antialiasing.enable()
        self.interpolation.bicubic()


    _padding: int
    _resize: int
    _mode : int
    def _recalc_mode(self):
        if self.in_plane_length < self.out_plane_length:
            self._mode = PPRMode
            in_pixel_size = self.in_plane_length / (self.in_pixels * self.up_scaling)
            padding = (self.out_plane_length / in_pixel_size - (self.in_pixels * self.in_up_scaling))/2
            padding = int(padding) + (padding % 1.0 > 0)
            self._padding = padding
            self._resize = self.out_pixels * self.out_up_scaling
        elif self.in_plane_length > self.out_plane_length:
            self._mode = RPPMode
            in_ar_pixel_size = self.out_plane_length / (self.out_pixels * self.up_scaling)
            in_ar_pixels = int(self.in_plane_length / in_ar_pixel_size)
            in_ar_pixels = in_ar_pixels + (not in_ar_pixels%2)
            self._resize = in_ar_pixels
            padding = int((in_ar_pixels - self.out_pixels*self.out_up_scaling)/2)
            self._padding = padding
        else:
            self._mode = EQLMode
            self._resize = 0
            self._padding = 0

        echo = True
        if echo:
            print('\n')
            print('Размер входного поля:  ', self.in_plane_length)
            print('Пиксели входного поля: ', self.in_pixels*self.in_up_scaling)
            print('Шаг входного поля:     ', self.in_plane_length / (self.in_pixels*self.in_up_scaling))
            print('Размер выходного поля: ', self.out_plane_length)
            print('Пиксели выходного поля:', self.out_pixels * self.out_up_scaling)
            print('Шаг выходного поля:    ', self.out_plane_length / (self.out_pixels * self.out_up_scaling))
            print('В текущей конфигурации ', end='')
            if self.in_plane_length < self.out_plane_length:
                print('размер выходного поля больше размера входного, поэтому будет использован следующий порядок операций:')
                print('Padding -> Propagation -> Resizing')
                print('Количество пикселей входного поля после падинга:', self.in_pixels*self.in_up_scaling + 2*self._padding)
                print('Количество пикселей после ресайзинга:', self._resize)
            elif self.in_plane_length > self.out_plane_length:
                print('размер выходного поля меньше размера входного, поэтому будет использован следующий порядок операций:')
                print('Resizing -> Propagation -> Padding')
                print('Количество пикселей входного поля после ресайзинга:', self._resize)
                print('Количество пикселей выходного поля:', self.out_pixels*self.out_up_scaling)
                print('Паддинг:', self._padding)
                print('Количество пикселей после падинга:', self._resize - 2*self._padding)
            else:
                print('размер выходного и входного поля равны')

    _border_pixels : int
    @property
    def border_pixels(self):
        return self._border_pixels
    def _recalc_border_pixels(self):
        self._border_pixels = int(self._border * self._pixels * self._up_scaling / self._plane_length)

    def _recalc_propagation_buffer(self):
        plane_length = None
        pixels = None
        if self._mode == PPRMode:
            pixels = self.in_pixels*self.in_up_scaling + 2*self._padding
            plane_length = self.in_plane_length*(1.0 + 2*self._padding / (self.in_pixels*self.in_up_scaling))
        elif self._mode == RPPMode:
            pixels = self._resize
            plane_length = self.in_plane_length
        else:
            pixels = self.in_pixels*self.in_up_scaling
            plane_length = self.in_plane_length

        device = torch.device('cpu')
        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device

        fx = torch.fft.fftshift(torch.fft.fftfreq(pixels + 2*self._border_pixels, d=plane_length / pixels, device=device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(pixels + 2*self._border_pixels, d=plane_length / pixels, device=device))
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0).to(device)
        space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0).to(device)
        Kz = ((2 * torch.pi) * torch.sqrt(0j + (1.0 / (wave_length * space_reflection)) ** 2 - fxx ** 2 - fyy ** 2)).to(dtype=self._accuracy.tensor_complex)

        self.register_buffer('_propagation_buffer', torch.exp(1.0j * Kz * self._distance))

    @AbstractPropagationLayer.plane_length.setter
    def plane_length(self, length:float):
        self._plane_length = length
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)
    @AbstractPropagationLayer.pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)
    @AbstractPropagationLayer.up_scaling.setter
    def up_scaling(self, amount:int):
        self._up_scaling = amount
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)

    @property
    def in_plane_length(self):
        return self.plane_length
    @in_plane_length.setter
    def in_plane_length(self, length:float):
        self.plane_length = length
    @property
    def in_pixels(self):
        return self.pixels
    @in_pixels.setter
    def in_pixels(self, amount:int):
        self.pixels = amount
    @property
    def in_up_scaling(self):
        return self.up_scaling
    @in_up_scaling.setter
    def in_up_scaling(self, amount:int):
        self.up_scaling = amount

    _out_plane_length : float
    @property
    def out_plane_length(self):
        return self._out_plane_length
    @out_plane_length.setter
    def out_plane_length(self, length:float):
        self._out_plane_length = length
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)

    _out_pixels : int
    @property
    def out_pixels(self):
        return self._out_pixels
    @out_pixels.setter
    def out_pixels(self, amount:int):
        print('\tУстановка out_pixels в:', amount)
        self._out_pixels = amount
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)

    _out_up_scaling : int
    @property
    def out_up_scaling(self):
        return self._out_up_scaling
    @out_up_scaling.setter
    def out_up_scaling(self, amount:int):
        self._out_up_scaling = amount
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_mode, -1)

    _border: float
    @property
    def border(self):
        return self._border
    @border.setter
    def border(self, length: float):
        self._border = length
        self._delayed.add(self._recalc_propagation_buffer)
        self._delayed.add(self._recalc_border_pixels, -1)



    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        distance:float=20.0*mm,
                        in_plane_length:float=1.0*mm,
                        in_pixels:int=20,
                        in_up_scaling:int=8,
                        out_plane_length:float=None,
                        out_pixels:int=None,
                        out_up_scaling:int=None,
                        border:float=1.0*mm):
        super().__init__()

        if out_plane_length is None:    out_plane_length = in_plane_length
        if out_pixels is None:          out_pixels = in_pixels
        if out_up_scaling is None:      out_up_scaling = in_up_scaling

        self.wavelength = wavelength
        self.reflection = reflection
        self.distance = distance

        self.in_plane_length = in_plane_length
        self.in_pixels = in_pixels
        self.in_up_scaling = in_up_scaling

        self.out_plane_length = out_plane_length
        self.out_pixels = out_pixels
        self.out_up_scaling = out_up_scaling

        self.border = border

        self._reset_options()

        self.delayed.finalize()

    def forward(self, field:torch.Tensor):
        super().forward(field)

        if self._mode == EQLMode:
            field = torch.nn.functional.pad(field, (+self._border_pixels, +self._border_pixels, +self._border_pixels, +self._border_pixels))
            field = torch.fft.fftshift(torch.fft.fft2(field))
            field = torch.fft.ifft2(torch.fft.ifftshift(field * self._propagation_buffer))
            field = torch.nn.functional.pad(field, (-self._border_pixels, -self._border_pixels, -self._border_pixels, -self._border_pixels))
        elif self._mode == PPRMode:
            field = torch.nn.functional.pad(field, (+self._border_pixels+self._padding, +self._border_pixels+self._padding, +self._border_pixels+self._padding, +self._border_pixels+self._padding))
            field = torch.fft.fftshift(torch.fft.fft2(field))
            field = torch.fft.ifft2(torch.fft.ifftshift(field * self._propagation_buffer))
            field = torch.nn.functional.pad(field, (-self._border_pixels, -self._border_pixels, -self._border_pixels, -self._border_pixels))
            field =          resize(torch.real(field), [self._resize, self._resize], interpolation=self._interpolation, antialias=self._antialiasing) \
                    + 1.0j * resize(torch.imag(field), [self._resize, self._resize], interpolation=self._interpolation, antialias=self._antialiasing)
        elif self._mode == RPPMode:
            field =          resize(torch.real(field), [self._resize, self._resize], interpolation=self._interpolation, antialias=self._antialiasing) \
                    + 1.0j * resize(torch.imag(field), [self._resize, self._resize], interpolation=self._interpolation, antialias=self._antialiasing)
            field = torch.nn.functional.pad(field, (+self._border_pixels, +self._border_pixels, +self._border_pixels, +self._border_pixels))
            field = torch.fft.fftshift(torch.fft.fft2(field))
            field = torch.fft.ifft2(torch.fft.ifftshift(field * self._propagation_buffer))
            field = torch.nn.functional.pad(field, (-self._border_pixels-self._padding, -self._border_pixels-self._padding, -self._border_pixels-self._padding, -self._border_pixels-self._padding))

        return field


class Test:
    class gap:
        @staticmethod
        def _get_distance():
            return
        @staticmethod
        def _two(propagation:FourierResizingPropagationLayer, d:float, b:float, h:float, a:float, show:int=3):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            propagation.to(device)

            wave_length = propagation.wavelength.item()
            in_pixels = propagation.in_pixels * propagation.in_up_scaling
            out_pixels = propagation.out_pixels * propagation.out_up_scaling
            in_plane_length = 3 * d
            L = a * in_plane_length * d / (wave_length * (2 * show + 1))

            propagation.in_plane_length = in_plane_length
            propagation.distance = L

            field = torch.zeros((1, 1, in_pixels, in_pixels), dtype=torch.complex64).to(device)
            fill = torch.tensor([[-d / 2 - b, -h / 2, -d / 2, +h / 2], [+d / 2, -h / 2, +d / 2 + b, +h / 2]])
            fill = (in_pixels - 1) * (fill + in_plane_length / 2) / in_plane_length
            fill = fill.to(torch.int32)
            fill = fill.tolist()
            for (nx1, ny1, nx2, ny2) in fill:
                field[0, 0, nx1:nx2, ny1:ny2] = torch.ones((nx2 - nx1, ny2 - ny1), dtype=torch.complex64, device=device)
            field = field.swapdims(2, 3)

            out_plane_length = (2 * show + 1) * L * wave_length / d
            propagation.out_plane_length = out_plane_length

            result = propagation.forward(field)

            return field, result, L, in_plane_length, in_plane_length*a
        @staticmethod
        def two(propagation:FourierResizingPropagationLayer, d:float, b:float, h:float, a:float, show:int=3, plot:belashovplot.TiledPlot=None):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            propagation.to(device)

            wave_length = propagation.wavelength.item()
            in_pixels = propagation.in_pixels * propagation.in_up_scaling
            out_pixels = propagation.out_pixels * propagation.out_up_scaling
            in_plane_length = 3*d
            L = a * in_plane_length * d / (wave_length * (2*show+1))

            propagation.in_plane_length = in_plane_length
            propagation.distance = L

            field = torch.zeros((1,1,in_pixels,in_pixels), dtype=torch.complex64).to(device)
            fill = torch.tensor([[-d/2-b, -h/2, -d/2, +h/2], [+d/2, -h/2, +d/2+b, +h/2]])
            fill = (in_pixels - 1) * (fill + in_plane_length/2) / in_plane_length
            fill = fill.to(torch.int32)
            fill = fill.tolist()
            for (nx1,ny1,nx2,ny2) in fill:
                field[0, 0, nx1:nx2, ny1:ny2] = torch.ones((nx2-nx1, ny2-ny1), dtype=torch.complex64, device=device)
            field = field.swapdims(2,3)

            out_plane_length = (2*show + 1)*L*wave_length/d
            propagation.out_plane_length = out_plane_length

            result = propagation.forward(field)

            field = field.cpu().squeeze()
            result = result.cpu().squeeze()

            from belashovplot import TiledPlot
            if plot is None: plot = TiledPlot(8, 8)
            plot.title('Дифракция на двух щелях при разных размерах входа и выхода')

            arguments = {
                'cmap' : 'gray',
                'aspect' : 'auto',
                'extent' : [-in_plane_length/2, +in_plane_length/2, -in_plane_length/2, +in_plane_length/2]
            }
            axes = plot.axes.add(0,0)
            axes.imshow(torch.abs(field), **arguments)
            plot.graph.title('Амплитуда исходного поля')

            arguments['extent'] = [-out_plane_length/2, +out_plane_length/2, -out_plane_length/2, +out_plane_length/2]
            axes = plot.axes.add(1,0)
            axes.imshow(torch.abs(result), **arguments)
            plot.graph.title('Амплитуда распространённого излучения')
            for i in range(-show, show+1):
                x = i * L * wave_length / d
                axes.axvline(x, color='maroon', linestyle='--')

            print('Расстояние:', L)

            plot.show()
        @staticmethod
        def two_multy(propagation:FourierResizingPropagationLayer, d:float, b:float, h:float, a_max:float, show:int=3, plot:belashovplot.TiledPlot=None, points:int=11):
            import numpy
            field = None
            results = []
            distances = []
            in_plane_lengths = []
            out_plane_lengths = []
            from src.utilities.CycleTimePredictor import CycleTimePredictor
            points1 = int(points/2)
            points2 = points - points1
            a_range_1 = numpy.linspace(1.0, a_max, points1+1)
            a_range_2 = 1.0 / numpy.linspace(1.0, a_max, points2)[1:]
            a_range = numpy.concatenate([a_range_1, a_range_2])
            for a in CycleTimePredictor(a_range):
                field, result, distance, in_plane_length, out_plane_length = Test.gap._two(propagation, d, b, h, a, show)
                results.append(result)
                distances.append(distance)
                in_plane_lengths.append(in_plane_length)
                out_plane_lengths.append(out_plane_length)

            from belashovplot import TiledPlot
            if plot is None: plot = TiledPlot(8,8)
            plot.title('Распространение с разными параметрами')
            plot.FontLibrary.MultiplyFontSize(0.7)

            amount = points + 1
            rows = int(numpy.sqrt(amount))
            if rows == 0: rows = 1
            cols = amount / rows
            cols = int(cols) + (cols % 1.0 > 0)

            from itertools import product
            col_row = list(product(range(cols), range(rows)))

            from src.utilities.Formaters import Format
            for result, ratio, distance, out_plane_length, (col, row) in zip(results, a_range, distances, out_plane_lengths, col_row[1:]):
                unit, mult = Format.Engineering_Separated(out_plane_length, 'm')
                arguments = {
                    'cmap': 'gray',
                    'aspect': 'auto',
                    'extent': [-out_plane_length*mult / 2, +out_plane_length*mult / 2, -out_plane_length*mult / 2, +out_plane_length*mult / 2]
                }
                axes = plot.axes.add(col, row)
                axes.imshow(torch.abs(result).squeeze().cpu(), **arguments)
                plot.graph.description('Отношение к размеру внешнего поля: ' + str(round(ratio, 2)) + '\nРасстояние дифракции: ' + Format.Engineering(distance, 'm'))
                plot.graph.label.x(unit)
                plot.graph.label.y(unit)

            arguments = {
                'cmap': 'gray',
                'aspect': 'auto',
                'extent': [-in_plane_lengths[0] / 2, +in_plane_lengths[0] / 2, -in_plane_lengths[0] / 2, +in_plane_lengths[0] / 2]
            }
            axes = plot.axes.add(0,0)
            axes.imshow(torch.abs(field).squeeze().cpu(), **arguments)
            plot.graph.description("Исходное изображение")

            plot.show()
        @staticmethod
        def two_cut(propagation:FourierResizingPropagationLayer, d:float, b:float, h:float, a_max:float, a_min:float, show:int=3, plot:belashovplot.TiledPlot=None, points:int=50):
            return

if __name__ == '__main__':
    PropagationLayer = FourierResizingPropagationLayer(600*nm, 1.0, 20*mm, 1*mm, 202, 3, 1*mm, 202, 3, 1*mm*3)
    PropagationLayer.antialiasing.disable()
    PropagationLayer.interpolation.bilinear()
    Test.gap.two(PropagationLayer, 1*mm, 30*um, 30*um, 2.0, 3)
    #Test.gap.two_cut(PropagationLayer, 1*mm, 30*um, 30*um, 1.05)