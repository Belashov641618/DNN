import torch
from typing import Union, Iterable

from src.utilities.DecimalPrefixes import nm,mm,um

from src.modules.layers.AbstarctLayer import AbstractLayer

from src.modules.layers.LensLayer import LensLayer
from src.modules.layers.FraunhoferPropagationLayer import FraunhoferPropagationLayer

class FFOpticalLayer(AbstractLayer):

    _PropagationLayer1 : FraunhoferPropagationLayer
    _PropagationLayer2 : FraunhoferPropagationLayer
    _LensLayer : LensLayer

    @property
    def focus(self):
        return self._LensLayer.focus
    @focus.setter
    def focus(self, length:float):
        self._PropagationLayer1.distance = length
        self._PropagationLayer2.distance = length
        self._LensLayer.focus = length

    @property
    def wavelength(self):
        return self._LensLayer.wavelength
    @wavelength.setter
    def wavelength(self, length:Union[Iterable, torch.Tensor, float]):
        self._LensLayer.wavelength = length
        self._PropagationLayer1.wavelength = length
        self._PropagationLayer2.wavelength = length

    @property
    def in_plane_length(self):
        return self._PropagationLayer1.in_plane_length
    @in_plane_length.setter
    def in_plane_length(self, length:float):
        self._PropagationLayer1.in_plane_length = length
    @property
    def in_pixels(self):
        return self._PropagationLayer1.in_pixels
    @in_pixels.setter
    def in_pixels(self, amount:int):
        self._PropagationLayer1.pixels = amount
    @property
    def in_up_scaling(self):
        return self._PropagationLayer1.in_up_scaling
    @in_up_scaling.setter
    def in_up_scaling(self, amount:int):
        self._PropagationLayer1.in_up_scaling = amount


    @property
    def lens_plane_length(self):
        return self._LensLayer.plane_length
    @lens_plane_length.setter
    def lens_plane_length(self, length:float):
        self._LensLayer.plane_length = length
        self._PropagationLayer1.plane_length = length
        self._PropagationLayer2.in_plane_length = length
    @property
    def lens_pixels(self):
        return self._LensLayer.pixels
    @lens_pixels.setter
    def lens_pixels(self, amount:int):
        self._LensLayer.pixels = amount
        self._PropagationLayer1.pixels = amount
        self._PropagationLayer2.in_pixels = amount
    @property
    def lens_up_scaling(self):
        return self._LensLayer.up_scaling
    @lens_up_scaling.setter
    def lens_up_scaling(self, amount:int):
        self._LensLayer.up_scaling = amount
        self._PropagationLayer1.up_scaling = amount
        self._PropagationLayer2.in_up_scaling = amount

    @property
    def out_plane_length(self):
        return self._PropagationLayer2.plane_length
    @out_plane_length.setter
    def out_plane_length(self, length:float):
        self._PropagationLayer2.plane_length = length
    @property
    def out_pixels(self):
        return self._PropagationLayer2.pixels
    @out_pixels.setter
    def out_pixels(self, amount:int):
        self._PropagationLayer2.pixels = amount
    @property
    def out_up_scaling(self):
        return self._PropagationLayer2.up_scaling
    @out_up_scaling.setter
    def out_up_scaling(self, amount:int):
        self._PropagationLayer2.up_scaling = amount

    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        out_plane_length:float=1.0*mm,
                        out_pixels:int=20,
                        out_up_scaling:int=8,
                        lens_plane_length:float=1.0*mm,
                        lens_pixels:int=20,
                        lens_up_scaling:int=8,
                        in_plane_length: float = 1.0 * mm,
                        in_pixels: int = 20,
                        in_up_scaling: int = 8,
                        focus:float=20.0*mm):
        super().__init__()

        self._LensLayer = LensLayer()
        self._PropagationLayer1 = FraunhoferPropagationLayer()
        self._PropagationLayer2 = FraunhoferPropagationLayer()

        self.wavelength = wavelength
        self.focus = focus
        self.in_plane_length = in_plane_length
        self.in_pixels = in_pixels
        self.in_up_scaling = in_up_scaling
        self.lens_plane_length = lens_plane_length
        self.lens_pixels = lens_pixels
        self.lens_up_scaling = lens_up_scaling
        self.out_plane_length = out_plane_length
        self.out_pixels = out_pixels
        self.out_up_scaling = out_up_scaling

        self._LensLayer.delayed.finalize()
        self._PropagationLayer1.delayed.finalize()
        self._PropagationLayer2.delayed.finalize()

    def forward(self, field:torch.Tensor):
        super().forward(field)
        return self._PropagationLayer1(self._LensLayer(self._PropagationLayer2(field)))




def Test():
    wavelength = 600*nm
    focus = 100*mm

    plane_length =  [30*mm, 40*mm,  80*mm]
    pixels =        [50,    50,     50]
    up_scaling =    [1,     1,      1]

    Layer = FFOpticalLayer(wavelength,
                           plane_length[2], pixels[2], up_scaling[2],
                           plane_length[1], pixels[1], up_scaling[1],
                           plane_length[0], pixels[0], up_scaling[0], focus)

    from torchvision.transforms.functional import resize
    from torchvision.transforms import InterpolationMode
    from src.utilities.UniversalTestsAndOther import GenerateSingleUnscaledSampleMNIST
    input_field = resize(torch.abs(GenerateSingleUnscaledSampleMNIST(True)), [pixels[0]*up_scaling[0], pixels[0]*up_scaling[0]], interpolation=InterpolationMode.BICUBIC).to(torch.complex64)
    output_field = Layer(input_field)

    input_field = input_field.squeeze()
    output_field = output_field.squeeze()

    from belashovplot import TiledPlot
    from itertools import product
    Plot = TiledPlot(7,7,100)
    Plot.title('Тест f-f системы')

    titles = [
        'Амплитуда входа',
        'Фаза входа',
        'Амплитуда выхода',
        'Фаза выхода'
    ]
    images = [
        torch.abs(input_field),
        torch.angle(input_field),
        torch.abs(output_field),
        torch.angle(output_field)
    ]

    for title, (col, row), image in zip(titles, product((0,1),(0,1)), images):
        axes = Plot.axes.add((row, col))
        axes.imshow(image, aspect='auto')
        Plot.graph.title(title)

    print('Starting to plot')
    Plot.show()

if __name__ == '__main__':
    Test()
