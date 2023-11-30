import torch
from typing import Union, Iterable
from utilities.DecimalPrefixes import nm, mm, um
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from src.modules.layers.AbstractPropagationLayer import AbstractPropagationLayer
from src.modules.layers.AbstractResizingPropagationLayer import AbstractResizingPropagationLayer

class FraunhoferPropagationLayer(AbstractResizingPropagationLayer):

    _target_pixel_size : float
    def _recalc_target_pixel_size(self):
        self._target_pixel_size = min(self._in_plane_length / (self._in_up_scaling * self._in_pixels), self._plane_length / (self._up_scaling * self._pixels))
        print('Расчётные пиксели входного поля:', int(self._in_plane_length/self._target_pixel_size))
        print('Расчётные пиксели выходного поля:', int(self._plane_length / self._target_pixel_size))
    @AbstractPropagationLayer.pixels.setter
    def pixels(self, amount:int):
        AbstractPropagationLayer.pixels.fset(self, amount)
        self._delayed.add(self._recalc_target_pixel_size, -1)
    @AbstractPropagationLayer.up_scaling.setter
    def up_scaling(self, calculating_pixels_per_pixel:int):
        AbstractPropagationLayer.up_scaling.fset(self, calculating_pixels_per_pixel)
        self._delayed.add(self._recalc_target_pixel_size, -1)
    @AbstractPropagationLayer.plane_length.setter
    def plane_length(self, length:float):
        AbstractPropagationLayer.plane_length.fset(self, length)
        self._delayed.add(self._recalc_target_pixel_size, -1)

    def _recalc_propagation_buffer(self):
        if self._plane_length / (self._up_scaling * self._pixels) == self._target_pixel_size:
            # Дополнительное разбиение входной плоскости
            new_size = int(self._in_plane_length / self._target_pixel_size) + (self._in_plane_length % self._target_pixel_size != 0)
            new_plane_length = new_size * self._target_pixel_size
            mesh = torch.linspace(-new_plane_length, +new_plane_length, 2*new_size-1)
            x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
            g = torch.sqrt(self._distance ** 2 + x_grid ** 2 + y_grid ** 2)
            wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
            space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0)
            PropagationBuffer = self._target_pixel_size**2 * torch.exp(self._distance * 2.0j * torch.pi / (space_reflection * wave_length)) * torch.exp(g * 2.0j * torch.pi / (space_reflection * wave_length)) / (g * 1j * wave_length).unsqueeze(1).to(self._accuracy.tensor_complex)
        else:
            # Дополнительное разбиение выходной плоскости
            mesh = torch.linspace(-self._plane_length, +self._plane_length, 2*self._pixels*self._up_scaling-1)
            x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
            g = torch.sqrt(self._distance**2 + x_grid**2 + y_grid**2)
            wave_length = self._wavelength.expand(1, 1, -1).movedim(2, 0)
            space_reflection = self._reflection.expand(1, 1, -1).movedim(2, 0)
            PropagationBuffer = self._target_pixel_size**2 * torch.exp(self._distance*2.0j*torch.pi/(space_reflection*wave_length)) * torch.exp(g*2.0j*torch.pi/(space_reflection * wave_length)) / (g*1j*wave_length).unsqueeze(1).to(self._accuracy.tensor_complex)
        if hasattr(self, '_propagation_buffer'):
            device = self._propagation_buffer.device
            self.register_buffer('_propagation_buffer', PropagationBuffer.to(device))
        else:
            self.register_buffer('_propagation_buffer', PropagationBuffer)


    @AbstractResizingPropagationLayer.in_plane_length.setter
    def in_plane_length(self, length:float):
        AbstractResizingPropagationLayer.in_plane_length.fset(self, length)
        self._delayed.add(self._recalc_target_pixel_size, -1)

    @AbstractResizingPropagationLayer.in_pixels.setter
    def in_pixels(self, amount:int):
        AbstractResizingPropagationLayer.in_pixels.fset(self, amount)
        self._delayed.add(self._recalc_target_pixel_size, -1)

    @AbstractResizingPropagationLayer.in_up_scaling.setter
    def in_up_scaling(self, calculating_pixels_per_pixel:int):
        AbstractResizingPropagationLayer.in_up_scaling.fset(self, calculating_pixels_per_pixel)
        self._delayed.add(self._recalc_target_pixel_size, -1)

    def __init__(self,  wavelength:Union[float,Iterable,torch.Tensor]=600*nm,
                        reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        out_plane_length:float=1.0*mm,
                        out_pixels:int=20,
                        out_up_scaling:int=8,
                        in_plane_length: float = 1.0 * mm,
                        in_pixels: int = 20,
                        in_up_scaling: int = 8,
                        distance:float=20.0*mm):
        AbstractPropagationLayer.__init__(self)
        self.wavelength = wavelength
        self.reflection = reflection
        self.plane_length = out_plane_length
        self.pixels = out_pixels
        self.up_scaling = out_up_scaling
        self.in_plane_length = in_plane_length
        self.in_pixels = in_pixels
        self.in_up_scaling = in_up_scaling
        self.distance = distance
        self.delayed.finalize()

    def forward(self, field:torch.Tensor):
        super().forward(field)
        if self._plane_length / (self._up_scaling * self._pixels) == self._target_pixel_size:
            # Дополнительное разбиение входной плоскости
            new_size = int(self._in_plane_length / self._target_pixel_size) + (self._in_plane_length % self._target_pixel_size != 0)
            field = resize(torch.real(field), size=[new_size, new_size], interpolation=InterpolationMode.BICUBIC, antialias=True) + 1j*resize(torch.imag(field), size=[new_size, new_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
            field = torch.nn.functional.conv2d(field, self._propagation_buffer, padding=new_size-1, groups=self._propagation_buffer.size(0))
            return field
        else:
            # Дополнительное разбиение выходной плоскости
            new_size = int(self._plane_length / self._target_pixel_size) + (self._plane_length % self._target_pixel_size != 0)
            field = torch.nn.functional.conv2d(field, self._propagation_buffer, padding=self._in_pixels*self._in_up_scaling-1, groups=self._propagation_buffer.size(0))
            field = resize(torch.real(field), size=[self._pixels*self._up_scaling, self._pixels*self._up_scaling], interpolation=InterpolationMode.BICUBIC, antialias=True) + 1j*resize(torch.imag(field), size=[self._pixels*self._up_scaling, self._pixels*self._up_scaling], interpolation=InterpolationMode.BICUBIC, antialias=True)
            return field



from belashovplot import TiledPlot
def ShowPropagationBuffer(Layer:FraunhoferPropagationLayer):
    Plot = TiledPlot(8,8)
    Plot.title('Ядро слоя')
    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(Layer.propagation_buffer.get().squeeze()))
    Plot.graph.title('Амплитуда')
    axes = Plot.axes.add((1, 0))
    axes.imshow(torch.angle(Layer.propagation_buffer.get().squeeze()))
    Plot.graph.title('Фаза')
    Plot.show()

def TwoSlits():
    wavelength = 600*nm
    d = 3*mm
    b = 10*um
    h = 10*um
    distance = 1000*mm
    step = distance*wavelength/d
    N = 2

    in_plane_length = 2*(d+b)
    in_pixels = 100
    in_up_scaling = 1

    out_plane_length = step*(2*N+1)
    out_pixels = 100
    out_up_scaling = 1

    input_field = torch.zeros((in_pixels*in_up_scaling, in_pixels*in_up_scaling), dtype=torch.complex64)
    x1_ = in_plane_length/2 - d/2 - b/2
    x2_ = in_plane_length/2 - d/2 + b/2
    x3_ = in_plane_length/2 + d/2 - b/2
    x4_ = in_plane_length/2 + d/2 + b/2
    y1_ = in_plane_length/2 - h/2
    y2_ = in_plane_length/2 + h/2
    import numpy
    coordinates = numpy.array([(x1_,y1_, x2_,y2_), (x3_,y1_, x4_,y2_)])
    coordinates = (in_pixels * in_up_scaling * coordinates / in_plane_length)
    for (x1, y1, x2, y2) in coordinates:
        nx1, ny1, nx2, ny2 = int(x1), int(y1), int(x2), int(y2)
        input_field[nx1:nx2, ny1:ny2] = torch.ones((nx2-nx1, ny2-ny1))
    input_field = input_field.unsqueeze(0).unsqueeze(0)

    Layer = FraunhoferPropagationLayer(wavelength, 1.0, out_plane_length, out_pixels, out_up_scaling, in_plane_length, in_pixels, in_up_scaling, distance)

    output_field = Layer(input_field)

    input_field = input_field.squeeze().cpu()
    output_field = output_field.squeeze().cpu()

    Plot = TiledPlot(8,8)
    Plot.title('Дифракция на двух щелях')

    extent = [-in_plane_length/2, +in_plane_length/2, -in_plane_length/2, +in_plane_length/2]
    axes = Plot.axes.add((0, 0))
    axes.imshow(torch.abs(input_field).swapdims(0,1), aspect='auto', extent=extent)
    Plot.graph.title('Входное изображение')

    extent = [-out_plane_length/2, +out_plane_length/2, -out_plane_length/2, +out_plane_length/2]
    axes = Plot.axes.add((1, 0))
    axes.imshow(torch.abs(output_field).swapdims(0,1), aspect='auto', extent=extent)
    Plot.graph.title('Входное изображение')
    for i in range(-N, N+1):
        axes.axvline(i*step, color='maroon', linestyle='--')

    Plot.show()

def Test():
    #Layer = FraunhoferPropagationLayer(out_plane_length=1*mm, out_pixels=100, out_up_scaling=1, in_plane_length=2*mm, in_pixels=50, in_up_scaling=1, distance=100*mm)
    #ShowPropagationBuffer(Layer)
    TwoSlits()

if __name__ == '__main__':
    Test()