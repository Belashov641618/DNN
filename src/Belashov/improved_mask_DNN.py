import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision.transforms import functional
from torch.utils.data import DataLoader
import numpy
import pickle
from itertools import product
from time import time as seconds
from datetime import datetime
import sys

from src.AdditionalUtilities.DecimalPrefixes import nm, um, mm
from src.AdditionalUtilities.Formaters import Format
from src.AdditionalUtilities.TitledFigure import Titles
from src.utils import set_det_pos
from src.AdditionalUtilities.Colorize import Colorizer
from src.diffraction import Lens as LensLayer


class HeightMaskLayer(torch.nn.Module):
    """
    Описание класса
    """

    _PixelsCount : int
    _MaximumHeight : float

    _UpScaling : int
    _UpScaledPixelsCount : int

    _SmoothMatrix : None
    _SmoothMatrixSum = None

    _PropagationBuffer : None
    _PropagationShiftBuffer : None

    _Parameters : torch.nn.Parameter

    def __init__(self, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, pixels_count=50, up_scaling=20, smoothing_matrix=None, parameters=None):
        super(HeightMaskLayer, self).__init__()

        #Setting all class fields:
        if torch.is_tensor(wave_length):
            _Lambda = wave_length.to(torch.float32)
        else:
            _Lambda = torch.tensor(requires_grad=False, dtype=torch.float32, data=[wave_length])

        _SpaceComplexReflection = torch.tensor(requires_grad=False, dtype=torch.complex64,  data=space_reflection)
        _MaskComplexReflection  = torch.tensor(requires_grad=False, dtype=torch.complex64,  data=mask_reflection)

        self._MaximumHeight = torch.max(_Lambda / torch.real(_MaskComplexReflection - _SpaceComplexReflection)).item()

        self._PixelsCount           = int(pixels_count)
        self._UpScaling             = int(up_scaling)
        self._UpScaledPixelsCount   = self._PixelsCount * self._UpScaling

        if smoothing_matrix is not None:
            if isinstance(smoothing_matrix, type(torch.tensor)):
                self.register_buffer('_SmoothMatrix', smoothing_matrix.to(torch.float32))
            else:
                self.register_buffer('_SmoothMatrix', torch.tensor(smoothing_matrix).to(torch.float32))
            self._SmoothMatrixSum = torch.sum(self._SmoothMatrix).item()


        #Registring parameters
        self._Parameters = torch.nn.Parameter(torch.rand(pixels_count, pixels_count))
        if parameters is not None: self._Parameters = torch.nn.Parameter(parameters.to(torch.float32))


        #Checking if everything ok and fixing some issues
        if True:
            if _Lambda.size() == 1:
                if _SpaceComplexReflection.size() != torch.Size([]):
                    raise Exception('Space reflection should be single if wave_length is single')
                if _MaskComplexReflection.size() != torch.Size([]):
                    raise Exception('Mask reflection should be single if wave_length is single')
            else:
                if _SpaceComplexReflection.size() != _Lambda.size():
                    if _SpaceComplexReflection.size() == torch.Size([]):
                        _SpaceComplexReflection = _SpaceComplexReflection.repeat(_Lambda.size())
                    else:
                        raise Exception('Space reflection should be same size as wave length or single value')
                if _MaskComplexReflection.size() != _Lambda.size():
                    if _MaskComplexReflection.size() == torch.Size([]):
                        _MaskComplexReflection = _MaskComplexReflection.repeat(_Lambda.size())
                    else:
                        raise Exception('Mask reflection should be same size as wave length or single value')
            if self._Parameters.size() != torch.Size((pixels_count, pixels_count)):
                raise Exception('parameters must have size (pixels_count, pixels_count) or equals None for auto creating')


        #Registering buffers
        self.register_buffer('_PropagationBuffer',      (2j*torch.pi*(_MaskComplexReflection + _SpaceComplexReflection)/_Lambda).to(torch.complex64))
        self.register_buffer('_PropagationShiftBuffer', torch.exp(2j*torch.pi*self._MaximumHeight*_SpaceComplexReflection/_Lambda).to(torch.complex64))

    def _heights(self):
        return self._MaximumHeight * torch.sigmoid(self._Parameters)
    def _prepared_heights(self):
        if self._SmoothMatrixSum is None:
            return torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0)
        else:
            return torch.nn.functional.conv2d(
                torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0).expand(1, 1, -1, -1),
                self._SmoothMatrix.expand(1, 1, -1, -1),
                padding='same').squeeze() / self._SmoothMatrixSum
    def _propagator(self):
        return self._PropagationShiftBuffer.expand(self._UpScaledPixelsCount, self._UpScaledPixelsCount, -1).movedim(2,0) \
               * \
               torch.exp(self._PropagationBuffer.expand(self._UpScaledPixelsCount, self._UpScaledPixelsCount, -1).movedim(2,0) * self._prepared_heights())

    def GetOriginalHeights(self):
        return self._heights().clone().detach()
    def GetPreparedHeights(self):
        return self._prepared_heights().clone().detach()
    def GetMaximumHeight(self):
        return self._MaximumHeight
    def GetSmoothMatrix(self):
        return self._SmoothMatrix.clone().detach() / self._SmoothMatrixSum

    def forward(self, field):
        return field * self._propagator()

class PaddedDiffractionLayer(torch.nn.Module):

    _BorderPixelsCount : int
    _PropagationBuffer : None
    def _init_PropagationArguments(self, wave_length, up_scaled_pixels_count, up_scaled_pixel_length, diffraction_length, space_reflection, border_pixels_count):
        # Этот код скопирован из DiffractiveLayer, но затухающие моды учитываются и добавлен учёт границы
        wave_length         = wave_length.expand(1, 1, -1).movedim(2,0)
        space_reflection    = space_reflection.expand(1, 1, -1).movedim(2,0)

        fx = torch.fft.fftshift(torch.fft.fftfreq(up_scaled_pixels_count + 2*border_pixels_count, d=up_scaled_pixel_length))
        fy = torch.fft.fftshift(torch.fft.fftfreq(up_scaled_pixels_count + 2*border_pixels_count, d=up_scaled_pixel_length))
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

        Kz = ((2 * torch.pi) * torch.sqrt(0j + (1.0/(wave_length*space_reflection))**2 - fxx**2 - fyy**2)).to(dtype=torch.complex64)
        self.register_buffer('_PropagationBuffer', torch.exp(1j * Kz * diffraction_length))

    def __init__(self, wave_length=600*nm, space_reflection=1.0, plane_length=1.5*mm, pixels_count=50, diffraction_length=1.0*mm, up_scaling=20, border_length=0.0):
        super(PaddedDiffractionLayer, self).__init__()

        if not torch.is_tensor(wave_length):
            wave_length = torch.tensor([wave_length])
        if not torch.is_tensor(space_reflection):
            space_reflection = torch.tensor([space_reflection])
        if space_reflection.size() != wave_length.size():
            space_reflection = space_reflection.expand_as(wave_length)

        self._BorderPixelsCount = int(border_length * pixels_count * up_scaling / plane_length)

        self._init_PropagationArguments(wave_length, pixels_count*up_scaling, plane_length/(pixels_count*up_scaling), diffraction_length, space_reflection, self._BorderPixelsCount)

    def forward(self, field):
        field = torch.nn.functional.pad(field, (+self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount))
        field = torch.fft.fftshift(torch.fft.fft2(field))
        field = torch.fft.ifft2(torch.fft.ifftshift(field * self._PropagationBuffer))
        field = torch.nn.functional.pad(field, (-self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount))
        return field

class SquareDetectorsLayer(torch.nn.Module):
    _DetectorsRectangles : type(torch.tensor)
    _DetectorsRectanglesPixels : type(torch.tensor)
    _DetectorsMasksBuffer : None
    _DetectorsRectanglesPixelsBuffer : None
    def GetDetectorsRectanglesCoordinates(self):
        return self._DetectorsRectangles.clone().detach()
    def GetDetectorRectanglesPixels(self):
        if self._DetectorsRectanglesPixels is None:
            return self._DetectorsRectanglesPixelsBuffer.clone().detach()
        else:
            return self._DetectorsRectanglesPixels.clone().detach()
    def __init__(self, plane_length=1.5*mm, detector_length=None, pixels_count=50, up_scaling=20, detectors_border_ratio=0.1, use_masks=False):
        super(SquareDetectorsLayer, self).__init__()

        if detector_length is None:
            detector_length = plane_length * (1.0-detectors_border_ratio) / 5

        if detector_length*4 > plane_length*(1.0-detectors_border_ratio):
            raise Exception('Detector length is to large')

        initial_detector_centers = torch.tensor([
            [-detector_length,      +detector_length],
            [0,                     +detector_length],
            [+detector_length,      +detector_length],

            [-1.5*detector_length,  0],
            [-0.5*detector_length,  0],
            [+0.5*detector_length,  0],
            [+1.5*detector_length,  0],

            [-detector_length,      -detector_length],
            [0,                     -detector_length],
            [+detector_length,      -detector_length]
        ], dtype=torch.float64)
        initial_detector_centers = initial_detector_centers * (plane_length * (1.0-detectors_border_ratio) / (3.0*detector_length) - 1.0/3.0)

        detectors_rectangles = initial_detector_centers.unsqueeze(dim=2).repeat(1,1,2).swapdims(1,2)
        detectors_rectangles[:, 0, :] = detectors_rectangles[:, 0, :] - 0.5*detector_length
        detectors_rectangles[:, 1, :] = detectors_rectangles[:, 1, :] + 0.5*detector_length
        self._DetectorsRectangles = detectors_rectangles
        detectors_rectangles = detectors_rectangles + plane_length / 2
        detectors_rectangles_pixels = (detectors_rectangles * pixels_count * up_scaling / plane_length).to(torch.int32)
        detectors_rectangles_pixels = torch.where(detectors_rectangles_pixels >= pixels_count*up_scaling, pixels_count*up_scaling-1, detectors_rectangles_pixels)
        detectors_rectangles_pixels = torch.where(detectors_rectangles_pixels < 0, 0, detectors_rectangles_pixels)
        self._DetectorsRectanglesPixels = detectors_rectangles_pixels

        if use_masks:
            self._DetectorsRectanglesPixelsBuffer = None
            DetectorsMasks = torch.zeros((len(initial_detector_centers), pixels_count * up_scaling, pixels_count * up_scaling), dtype=torch.float32)
            for num, rectangle in enumerate(detectors_rectangles_pixels):
                DetectorsMasks[num, rectangle[0][0]:rectangle[1][0]+1, rectangle[0][1]:rectangle[1][1]+1] = torch.ones(rectangle[1][0]-rectangle[0][0]+1, rectangle[1][1]-rectangle[0][1]+1)
            self.register_buffer('_DetectorsMasksBuffer', DetectorsMasks)
        else:
            self._DetectorsMasksBuffer = None
            self._DetectorsRectanglesPixels = None
            self.register_buffer('_DetectorsRectanglesPixelsBuffer', detectors_rectangles_pixels)
    def forward(self, field):
        if self._DetectorsMasksBuffer is None:
            results_list = []
            for n, rectangle in enumerate(self._DetectorsRectanglesPixelsBuffer):
                results_list.append(torch.sum(torch.abs(field[:, :, rectangle[0][0]:rectangle[1][0]+1, rectangle[0][1]:rectangle[1][1]+1])**2, dim=(1,2,3)))
            field_integral = torch.sum(torch.abs(field)**2, dim=(1,2,3))
            results = torch.stack(results_list, dim=1)
            results = results / field_integral.expand(10, -1).swapdims(0,1)
            # results = results / (results.max(dim=1).values[:, None])
            return results
        else:
            results = torch.sum(field.expand(10,-1,-1,-1,-1).movedim(0,4) * self._DetectorsMasksBuffer, dim=(0,1))
            field_integral = torch.sum(torch.abs(field) ** 2, dim=(1, 2, 3))
            results = results / field_integral.expand(10, -1).swapdims(0, 1)
            # results = results / (results.max(dim=1).values[:, None])
            return results

class ConcentratorsDetectorsLayer(torch.nn.Module):
    _DetectorsRectanglesPixels : type(torch.tensor)
    _DetectorsRectanglesCoordinates : type(torch.tensor)
    _DetectorsMasksBuffer : None
    def GetDetectorsRectanglesCoordinates(self):
        return self._DetectorsRectanglesCoordinates.clone().detach()
    def GetDetectorRectanglesPixels(self):
        return self._DetectorsRectanglesPixels.clone().detach()

    def __init__(self, plane_length=1.5*mm, pixels_count=50, up_scaling=20, detectors_masks=None):
        super(ConcentratorsDetectorsLayer, self).__init__()
        if detectors_masks is None:
            initial_detector_centers = torch.tensor([
                [-plane_length/5,       +plane_length/5],
                [0,                     +plane_length/5],
                [+plane_length/5,       +plane_length/5],

                [-1.5*plane_length/5,   0],
                [-0.5*plane_length/5,   0],
                [+0.5*plane_length/5,   0],
                [+1.5*plane_length/5,   0],

                [-plane_length/5,       -plane_length/5],
                [0,                     -plane_length/5],
                [+plane_length/5,       -plane_length/5]
            ], dtype=torch.float64)
            initial_detector_centers = initial_detector_centers * (plane_length / (3.0*plane_length/5.0) - 1.0/3.0)
            initial_detector_centers = (initial_detector_centers + plane_length/2)*pixels_count*up_scaling/plane_length
            initial_detector_centers.to(torch.int32)

            DetectorsMasks = torch.zeros((len(initial_detector_centers), pixels_count * up_scaling, pixels_count * up_scaling), dtype=torch.float32)
            for n, center in enumerate(initial_detector_centers):
                x_array = torch.arange(pixels_count*up_scaling) - center[0]
                y_array = torch.arange(pixels_count*up_scaling) - center[1]
                x_grid, y_grid = torch.meshgrid(x_array, y_array, indexing='ij')
                length_grid = torch.sqrt(x_grid**2 + y_grid**2)
                DetectorsMasks[n] = 1.0 / (1.0 + length_grid)**1.0

            self.register_buffer('_DetectorsMasksBuffer', DetectorsMasks)
        else:
            self.register_buffer('_DetectorsMasksBuffer', detectors_masks)

        self._DetectorsRectanglesPixels = torch.zeros(10, 2, 2, dtype=torch.int32)
        for n, mask in enumerate(self._DetectorsMasksBuffer):
            Cx = torch.argmax(torch.max(mask, dim=1)[0])
            Cy = torch.argmax(torch.max(mask, dim=0)[0])
            # self._DetectorsRectanglesPixels[n][0][0] = int(Cx - pixels_count*up_scaling/24)
            # self._DetectorsRectanglesPixels[n][0][1] = int(Cy - pixels_count*up_scaling/24)
            # self._DetectorsRectanglesPixels[n][1][0] = int(Cx + pixels_count*up_scaling/24)
            # self._DetectorsRectanglesPixels[n][1][1] = int(Cy + pixels_count*up_scaling/24)
            self._DetectorsRectanglesPixels[n][0][0] = Cx
            self._DetectorsRectanglesPixels[n][0][1] = Cy
            self._DetectorsRectanglesPixels[n][1][0] = Cx
            self._DetectorsRectanglesPixels[n][1][1] = Cy
        self._DetectorsRectanglesCoordinates = (self._DetectorsRectanglesPixels.to(torch.float32) - pixels_count*up_scaling/2) * plane_length / (pixels_count*up_scaling)

    def forward(self, field):
        field = torch.abs(field)**2
        results = torch.sum(field*self._DetectorsMasksBuffer.expand(field.size(dim=0), field.size(dim=1), -1, -1, -1).movedim(2,0), dim=(2,3,4))
        field_sum = torch.sum(field, dim=(1,2,3)).expand(10, -1)
        results = (results / field_sum).swapdims(0,1) * 10.0
        # results = results.swapdims(0,1) / torch.max(results, dim=1)[0]
        return results

class ImprovedD2NN(torch.nn.Module):

    _MaskLayers         : type(torch.nn.ModuleList)
    _PropagationLayer   : PaddedDiffractionLayer

    _PlaneLength    : float
    _PixelsCount    : int
    _UpScaling      : int
    _Lambdas        : type(torch.tensor)
    def __init__(self, layers_count=4, pixels_count=20, pixel_length=50*um, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, layer_spacing_length=5*mm, up_scaling=None, border_length=None, smoothing_matrix=None):
        super(ImprovedD2NN, self).__init__()

        if up_scaling is None:
            up_scaling = 32
        if border_length is None:
            border_length = pixel_length*pixels_count/2

        self._PlaneLength   = pixel_length * pixels_count
        self._PixelsCount   = pixels_count
        self._UpScaling     = up_scaling
        if torch.is_tensor(wave_length):
            self._Lambdas = wave_length
        else:
            self._Lambdas = torch.tensor([wave_length])

        self._PropagationLayer = PaddedDiffractionLayer(wave_length, space_reflection, pixel_length*pixels_count, pixels_count, layer_spacing_length, up_scaling, border_length)
        self._MaskLayers = torch.nn.ModuleList([HeightMaskLayer(wave_length, space_reflection, mask_reflection, pixels_count, up_scaling, smoothing_matrix) for _ in range(layers_count)])
    def _restore_attributes(self):
        if (not hasattr(self, '_PlaneLength')) or (self._PlaneLength is None):
            print('Restoring plane length is failed because it`s no resource')
        if (not hasattr(self, '_PixelsCount')) or (self._PixelsCount is None):
            print('Restoring pixels count')
            self._PixelsCount = self._MaskLayers[0].__getattribute__('_PixelsCount')
        if (not hasattr(self, '_UpScaling')) or (self._UpScaling is None):
            print('Restoring pixels count')
            self._UpScaling = self._MaskLayers[0].__getattribute__('_UpScaling')

    def forward(self, field, variable_that_has_to_be_deleted):
        field = self._PropagationLayer(field)
        for _MaskLayer in self._MaskLayers:
            field = _MaskLayer(field)
            field = self._PropagationLayer(field)
        field = torch.abs(field)**2
        return field.sum(dim=1), variable_that_has_to_be_deleted

    def save(self, file_name='FileImprovedD2NN.data'):
        try:
            file = open(file_name, 'wb')
            pickle.dump(self, file)
            file.close()
        except Exception as e:
            print(e)
    @staticmethod
    def load(file_name='FileImprovedD2NN.data', class_was_updated=False):
        try:
            file = open(file_name, 'rb')
            self = pickle.load(file)
            if class_was_updated:
                self._restore_attributes()
            file.close()
            return self
        except Exception as e:
            print(e)

    def VisualizeHeights(self, show=True):
        N = len(self._MaskLayers)
        Ny = int(numpy.sqrt(N))
        Nx = int(N/Ny) + ((N/Ny - int(N/Ny)) != 0)

        fig = plt.figure(figsize=(12*Nx/Ny+0.16*Nx/4, 12))
        fig.suptitle('Матрицы высот сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05)

        for (n, MaskLayer), (ny, nx)  in zip(enumerate(self._MaskLayers), product(range(Ny), range(Nx))):
            heights = MaskLayer.GetPreparedHeights().cpu()

            unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')
            unitZ,  multZ  = Format.Engineering_Separated(torch.max(heights), 'm')

            heights *= multZ

            axes = Fig.add_axes((nx+1, ny+1))
            axes.set_title('Маска №' + str(n+1), **Format.Text('Header'))
            axes.xaxis.set_tick_params(labelsize=8)
            axes.yaxis.set_tick_params(labelsize=8)
            axes.set_xlabel('X, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold'}))
            axes.set_ylabel('Y, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
            image = axes.imshow(heights, cmap='viridis', origin='lower', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])

            cbar = fig.colorbar(image, ax=axes, location='right', shrink=0.8, pad=0.12*Nx/4)
            cbar.ax.set_ylabel('Высота Z, ' + unitZ, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90, 'verticalalignment':'top'}))
            cbar.ax.tick_params(labelsize=8, labelleft=True, left=True, labelright=False, right=False)
        if show:
            plt.show()
    def VisualizeFieldCuts(self, show=True, samples=6, dataset=None, detector_pos=None, device='cpu'):
        #Dataset
        if dataset is None:
            transformation = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
                torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
            ])
            dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = torch.utils.data.DataLoader(dataset, batch_size=samples, shuffle=True, num_workers=0)
        images, labels = next(iter(loader))
        images = images.to(device)

        #Detectors
        if detector_pos is None:
            edge_x = self._PixelsCount * self._UpScaling // 20
            edge_y = edge_x * 2
            det_size = (self._PixelsCount * self._UpScaling - 5 * edge_x) // 4
            detector_pos = set_det_pos(det_size, edge_x, edge_y, self._PixelsCount*self._UpScaling)
        detector_masks = torch.zeros((len(detector_pos), int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling)), dtype=torch.float32)
        for i, (x1, x2, y1, y2) in enumerate(detector_pos):
            detector_masks[i, x1:x2+1, y1:y2+1] = torch.ones((x2-x1+1, y2-y1+1), dtype=torch.float32)

        #Processing
        with torch.no_grad():
            History = torch.zeros([2 + len(self._MaskLayers)] + list(images.size()))
            field = images
            History[0] = (torch.abs(field.clone().detach())**2).cpu()
            field = self._PropagationLayer(images)
            History[1] = (torch.abs(field.clone().detach())**2).cpu()
            for n, _MaskLayer in enumerate(self._MaskLayers):
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
                History[n+2] = (torch.abs(field.clone().detach())**2).cpu()
        if images.size(dim=1) != 1:
            colorizer = Colorizer()
            History = colorizer.Colorize(History, self._Lambdas)
        else:
            History = History.squeeze(dim=2)
        History = History.swapaxes(0,1)

        #Drawing
        Ny = samples
        Nx = 2 + len(self._MaskLayers)

        fig = plt.figure(figsize=(12*Nx/Ny, 12))
        fig.suptitle('Распространение поля по сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05, hspace=0.03, wspace=-0.08*Nx/Ny, leftspace=-0.001*Nx*Nx/Ny, rightspace=-0.001*Nx*Nx/Ny)

        for m, history in enumerate(History):
            for n, image in enumerate(history):
                name = 'Поле перед маской #' + str(n)
                if n == 0: name = 'Начальное изображение №' + str(m+1)
                if n == History.size(dim=1)-1: name = 'Слой Детекторов'

                unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')

                axes = Fig.add_axes((n+1, m+1))
                axes.set_title(name, **Format.Text('Caption', {'fontweight':'bold'}))
                axes.imshow(image, cmap='viridis', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])
                if m == History.size(dim=0)-1:
                    axes.xaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_xlabel('X, ' + unitXY, **Format.Text('Caption'))
                else:
                    axes.xaxis.set_tick_params(labelbottom=False, labelsize=8, direction='inout')
                if n == 0:
                    axes.yaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_ylabel('Y, ' + unitXY, **Format.Text('Caption', {'rotation': 90}))
                else:
                    axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')

                if n == History.size(dim=1)-1:
                    det_size = (detector_pos[0][1] - detector_pos[0][0]) * self._PlaneLength * multXY / (self._PixelsCount*self._UpScaling)
                    labels_data = torch.sum(detector_masks * image, dim=(1,2))
                    labels_data = labels_data / torch.sum(labels_data)
                    for i, det in enumerate(detector_pos):
                        det = (numpy.array(det) - int(self._PixelsCount*self._UpScaling/2)) * self._PlaneLength * multXY / (self._PixelsCount*self._UpScaling)
                        color = 'maroon'
                        if i == labels[m]: color = 'yellow'
                        if i == int(torch.argmax(labels_data)): color = 'dodgerblue'
                        if (i == labels[m]) and (i == int(torch.argmax(labels_data))): color = 'green'
                        rect = patches.Rectangle((det[2], det[0]), det_size, det_size, linewidth=1, edgecolor=color, facecolor='none')
                        axes.add_patch(rect)
                        text_pos = (det[2] + det_size/2, det[0] + det_size/2)
                        axes.text(text_pos[0], text_pos[1], Format.Scientific(labels_data[i].item(),precision=1), **Format.Text('Header', {'fontsize':9 ,'c':color}))

        if show:
            plt.show()

class ImprovedWithDetectorsD2NN(torch.nn.Module):

    _MaskLayers         : type(torch.nn.ModuleList)
    _PropagationLayer   : PaddedDiffractionLayer
    _DetectorsLayer     : SquareDetectorsLayer

    _PlaneLength    : float
    _PixelsCount    : int
    _UpScaling      : int
    _Lambdas        : type(torch.tensor)
    def __init__(self, layers_count=4, pixels_count=20, pixel_length=50*um, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, layer_spacing_length=5*mm, up_scaling=None, border_length=None, smoothing_matrix=None):
        super(ImprovedWithDetectorsD2NN, self).__init__()

        if up_scaling is None:
            up_scaling = 32
        if border_length is None:
            border_length = pixel_length*pixels_count/2

        self._PlaneLength   = pixel_length * pixels_count
        self._PixelsCount   = pixels_count
        self._UpScaling     = up_scaling
        if torch.is_tensor(wave_length):
            self._Lambdas = wave_length
        else:
            self._Lambdas = torch.tensor([wave_length])

        self._PropagationLayer = PaddedDiffractionLayer(wave_length, space_reflection, pixel_length*pixels_count, pixels_count, layer_spacing_length, up_scaling, border_length)
        self._MaskLayers = torch.nn.ModuleList([HeightMaskLayer(wave_length, space_reflection, mask_reflection, pixels_count, up_scaling, smoothing_matrix) for _ in range(layers_count)])
        self._DetectorsLayer = SquareDetectorsLayer(pixel_length*pixels_count, pixels_count=pixels_count, up_scaling=up_scaling)
    def _restore_attributes(self):
        if (not hasattr(self, '_PlaneLength')) or (self._PlaneLength is None):
            print('Restoring plane length is failed because it`s no resource')
        if (not hasattr(self, '_PixelsCount')) or (self._PixelsCount is None):
            print('Restoring pixels count')
            self._PixelsCount = self._MaskLayers[0].__getattribute__('_PixelsCount')
        if (not hasattr(self, '_UpScaling')) or (self._UpScaling is None):
            print('Restoring pixels count')
            self._UpScaling = self._MaskLayers[0].__getattribute__('_UpScaling')

    def forward(self, field, record_history=False):
        if record_history:
            history = [field]
            field = self._PropagationLayer(field)
            history.append(field)
            for _MaskLayer in self._MaskLayers:
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
                history.append(field)
            field = self._DetectorsLayer(field)
            return field, history
        else:
            field = self._PropagationLayer(field)
            for _MaskLayer in self._MaskLayers:
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
            field = self._DetectorsLayer(field)
            return field

    def save(self, file_name='FileImprovedD2NN.data'):
        try:
            file = open(file_name, 'wb')
            pickle.dump(self, file)
            file.close()
        except Exception as e:
            print(e)
    @staticmethod
    def load(file_name='FileImprovedD2NN.data', class_was_updated=False):
        try:
            file = open(file_name, 'rb')
            self = pickle.load(file)
            if class_was_updated:
                self._restore_attributes()
            file.close()
            return self
        except Exception as e:
            print(e)

    def load_parameters(self, other_model):
        with torch.no_grad():
            for (name, param), (name_, param_) in zip(self.named_parameters(), other_model.named_parameters()):
                param.copy_(param_)

        # for n, module in enumerate(self._MaskLayers):
        #     other_module = other_model.__getattr__('_MaskLayers')[n]
        #     module = HeightMaskLayer()
        #     # print(other_module.__getattr__('_Parameters'))
        #     module.__setattr__('_Parameters', torch.nn.Parameter(other_module.__getattr__('_Parameters')))

    def AccuracyTest(self, batch_size=16, test_loader=None, device=None):
        if test_loader is None:
            transformation = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomRotation((-90, -90)),
                torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
                torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
            ])
            testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
            test_loader = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.eval()
        with torch.no_grad():
            TimeStart = seconds()
            TotalIterations = len(test_loader)
            Iteration = 0

            Correct = 0
            Total = 0

            for batch_num, (images, labels) in enumerate(test_loader):
                Iteration += 1

                images = images.to(device)
                labels = labels.to(device)

                output = self(images)
                Correct += torch.sum(torch.eq(torch.argmax(output, dim=1), labels)).item()
                Total += len(labels)

                string = 'Тестирование, | Осталось: ' + Format.Time((seconds() - TimeStart) * (TotalIterations - Iteration) / Iteration) \
                         + ' | Прошло: ' + Format.Time(seconds() - TimeStart) \
                         + ' | Итерация: ' + str(Iteration) + ' из ' + str(TotalIterations) \
                         + ' | Правильно: ' + str(Correct) + ' из ' + str(Total) \
                         + ' | Точность %: ' + str(round(Correct / Total, 2) * 100)
                sys.stdout.write(f"\r{string + '                                                                                     '}")
        print('')
        return 100 * Correct / Total

    def Train(self, show=True, batch_size=8, epochs_count=1, criterion=None, learning_rate=0.003):

        # Loading Data
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
        train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
        testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)

        # Setting device
        torch.cuda.init()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('Using device:', torch.cuda.get_device_name(), '')
            available, total = torch.cuda.mem_get_info()
            print("Memory | Available: %.2f GB | Total:     %.2f GB" % (available / 1e9, total / 1e9))
        else:
            print('Using device:', 'cpu')
        self.to(device)

        #Initializing criterion and optimizer
        if criterion is None:
            def MSELoss(outputs, corrects):
                return torch.mean((outputs - torch.eye(outputs.size(dim=1),outputs.size(dim=1), device=device, dtype=torch.float32)[corrects])**2)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            criterion = MSELoss
        else:
            criterion = criterion.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        #Initializing loss graph
        colors_list = ['maroon', 'chocolate', 'darkorange', 'goldenrod', 'olivedrab', 'darkgreen', 'darkcyan', 'steelblue', 'navy', 'slateblue', 'darkviolet', 'palevioletred','maroon', 'chocolate', 'darkorange', 'goldenrod', 'olivedrab', 'darkgreen', 'darkcyan', 'steelblue', 'navy', 'slateblue', 'darkviolet', 'palevioletred']
        colors_list_size = len(colors_list)
        loss_buffer = []
        average_loss_buffer = []
        max_loss = -1.0E-50
        min_loss = +1.0E+50
        iter_buffer = numpy.arange(1, len(train_loader)+1)
        x_shift = 1
        fig = plt.figure(figsize=(12,12))
        fig.suptitle('Эволюция лосса при обучении дифрационной сети', **Format.Text('BigHeader', {'fontsize':24}))
        axes = fig.add_subplot(111)
        axes.set_title('Лосс в зависимости от итерации', **Format.Text('BigHeader'))
        axes.xaxis.set_tick_params(labelsize=11)
        axes.yaxis.set_tick_params(labelsize=11)
        axes.set_xlabel('Итерация', **Format.Text('Header'))
        axes.set_ylabel('Лосс', **Format.Text('Header', {'rotation':90}))
        axes.grid(True)
        axes.set_yscale('log')
        fig.show()

        accuracy = 0
        #Training
        for epoch_num in range(epochs_count):
            TimeStart = seconds()
            TotalIterations = len(train_loader)
            Iteration = 0

            LossBufferSize = 20
            LossBuffer = torch.zeros(LossBufferSize, requires_grad=False)
            LossBufferIndex = 0

            loss_buffer.clear()
            average_loss_buffer.clear()
            if epoch_num != 0: iter_buffer += len(train_loader)
            # line, = axes.plot(iter_buffer[0], 10, color=colors_list[epoch_num], linestyle='--', linewidth=0.7)
            line_average, = axes.plot(iter_buffer[0], 10, color=colors_list[epoch_num], linewidth=3.0)

            self.train()
            for batch_num, (images, labels) in enumerate(train_loader):
                Iteration += 1

                images = images.to(device)
                labels = labels.to(device)

                output = self(images)

                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                LossBuffer[LossBufferIndex % LossBufferSize] = loss.item()
                LossBufferIndex += 1

                loss_buffer.append(loss.item())
                if len(average_loss_buffer) == 0:
                    average_loss_buffer.append(loss.item())
                else:
                    average_loss_buffer.append((average_loss_buffer[-1]*LossBufferSize + loss.item())/(LossBufferSize + 1))
                if loss.item() < min_loss: min_loss = loss.item()
                if loss.item() > max_loss: max_loss = loss.item()
                if min_loss == max_loss: max_loss += 1.0E-20
                # line.set_xdata(iter_buffer[:len(loss_buffer)])
                # line.set_ydata(loss_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()
                axes.set_xlim(0, iter_buffer[len(loss_buffer)-1])
                if min_loss != max_loss:
                    axes.set_ylim(min_loss, max_loss)
                line_average.set_xdata(iter_buffer[:len(loss_buffer)])
                line_average.set_ydata(average_loss_buffer)

                string  = 'Обучение, Эпоха ' + str(epoch_num+1) + ' из ' + str(epochs_count) \
                    + ' | Осталось: ' + Format.Time((seconds() - TimeStart)*(TotalIterations - Iteration)/Iteration) \
                    + ' | Прошло: ' + Format.Time(seconds() - TimeStart) \
                    + ' | Итерация: ' + str(Iteration) + ' из ' + str(TotalIterations) \
                    + ' | Средний по ' + str(LossBufferSize) + ' тестам лосс: ' + Format.Scientific(torch.mean(LossBuffer).item())
                sys.stdout.write(f"\r{string + '                                                                                     '}")
            self.eval()
            print('')

            accuracy = self.AccuracyTest(test_loader=test_loader, device=device)
            print('')

        path = 'TrainedModels/AutoSave/'
        if __name__ == '__main__':
            path = '../' + path
        file_name = path + datetime.now().strftime('%m.%d.%Y_%H.%M') + '_' + str(int(accuracy)) + '%' + '.data'
        self.save(file_name)
    def VisualizeHeights(self, show=True):
        N = len(self._MaskLayers)
        Ny = int(numpy.sqrt(N))
        Nx = int(N/Ny) + ((N/Ny - int(N/Ny)) != 0)

        fig = plt.figure(figsize=(12*Nx/Ny+0.16*Nx/4, 12))
        fig.suptitle('Матрицы высот сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05)

        for (n, MaskLayer), (ny, nx)  in zip(enumerate(self._MaskLayers), product(range(Ny), range(Nx))):
            heights = MaskLayer.GetPreparedHeights().cpu()

            unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')
            unitZ,  multZ  = Format.Engineering_Separated(torch.max(heights), 'm')

            heights *= multZ

            axes = Fig.add_axes((nx+1, ny+1))
            axes.set_title('Маска №' + str(n+1), **Format.Text('Header'))
            axes.xaxis.set_tick_params(labelsize=8)
            axes.yaxis.set_tick_params(labelsize=8)
            axes.set_xlabel('X, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold'}))
            axes.set_ylabel('Y, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
            image = axes.imshow(heights, cmap='viridis', origin='lower', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])

            cbar = fig.colorbar(image, ax=axes, location='right', shrink=0.8, pad=0.12*Nx/4)
            cbar.ax.set_ylabel('Высота Z, ' + unitZ, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90, 'verticalalignment':'top'}))
            cbar.ax.tick_params(labelsize=8, labelleft=True, left=True, labelright=False, right=False)
        if show:
            plt.show()
    def VisualizeFieldCuts(self, show=True, samples=3, dataset=None, device='cpu'):
        #Dataset
        if dataset is None:
            transformation = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomRotation((-90,-90)),
                torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
                torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
            ])
            dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = DataLoader(dataset, batch_size=samples, shuffle=True, num_workers=0)
        images, labels = next(iter(loader))
        images = images.to(device)

        #Detectors
        detectors_data = torch.zeros((samples, 10), dtype=torch.float32)
        detectors_pos = self._DetectorsLayer.GetDetectorsRectanglesCoordinates().cpu()

        #Processing
        History = torch.zeros([2 + len(self._MaskLayers)] + list(images.size()))
        with torch.no_grad():
            field = images
            History[0] = (torch.abs(field.clone().detach())**2).cpu()
            field = self._PropagationLayer(field)
            History[1] = (torch.abs(field.clone().detach())**2).cpu()
            for n, _MaskLayer in enumerate(self._MaskLayers):
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
                History[n+2] = (torch.abs(field.clone().detach())**2).cpu()
            detectors_data = self._DetectorsLayer(field).clone().detach().cpu()
        if images.size(dim=1) != 1:
            colorizer = Colorizer()
            History = colorizer.Colorize(History, self._Lambdas)
        else:
            History = History.squeeze(dim=2)
        History = History.swapaxes(0,1)

        #Drawing
        Ny = samples
        Nx = 2 + len(self._MaskLayers)

        fig = plt.figure(figsize=(12*Nx/Ny, 12))
        fig.suptitle('Распространение поля по сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05, hspace=0.03, wspace=-0.08*Nx/Ny, leftspace=-0.001*Nx*Nx/Ny, rightspace=-0.001*Nx*Nx/Ny)

        for m, history in enumerate(History):
            for n, image in enumerate(history):
                name = 'Поле перед маской #' + str(n)
                if n == 0: name = 'Начальное изображение №' + str(m+1)
                if n == History.size(dim=1)-1: name = 'Слой Детекторов'

                unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')

                axes = Fig.add_axes((n+1, m+1))
                axes.set_title(name, **Format.Text('Caption', {'fontweight':'bold'}))
                axes.imshow(image.swapaxes(0,1), origin='lower', cmap='viridis', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])
                if m == History.size(dim=0)-1:
                    axes.xaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_xlabel('X, ' + unitXY, **Format.Text('Caption'))
                else:
                    axes.xaxis.set_tick_params(labelbottom=False, labelsize=8, direction='inout')
                if n == 0:
                    axes.yaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_ylabel('Y, ' + unitXY, **Format.Text('Caption', {'rotation': 90}))
                else:
                    axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')

                if n == History.size(dim=1)-1:
                    for i, det in enumerate(detectors_pos):
                        labels_data = detectors_data[m]
                        det_scaled = det * multXY
                        width = det_scaled[1][0] - det_scaled[0][0]
                        height = det_scaled[1][1] - det_scaled[0][1]
                        color = 'maroon'
                        if i == labels[m]: color = 'yellow'
                        if i == int(torch.argmax(labels_data)): color = 'dodgerblue'
                        if (i == labels[m]) and (i == int(torch.argmax(labels_data))): color = 'green'
                        rect = patches.Rectangle((det_scaled[0][0], det_scaled[0][1]), width, height, linewidth=1, edgecolor=color, facecolor='none')
                        axes.add_patch(rect)
                        text_pos = (det_scaled[0][0] + width/2, det_scaled[0][1] + height/2)
                        axes.text(text_pos[0], text_pos[1], Format.Scientific(labels_data[i].item(),precision=1), **Format.Text('Header', {'fontsize':9 ,'c':color}))

        if show:
            plt.show()

class ImprovedWithDetectorsWithLensD2NN(torch.nn.Module):

    _MaskLayers             : type(torch.nn.ModuleList)
    _LensLayer              : LensLayer
    _LensPropagationLayer   : PaddedDiffractionLayer
    _PropagationLayer       : PaddedDiffractionLayer
    _DetectorsLayer         : SquareDetectorsLayer

    _PlaneLength    : float
    _PixelsCount    : int
    _UpScaling      : int
    _Lambdas        : type(torch.tensor)
    def __init__(self, layers_count=4, pixels_count=20, pixel_length=50*um, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, layer_spacing_length=5*mm, focus_length=30*mm, up_scaling=None, border_length=None, smoothing_matrix=None):
        super(ImprovedWithDetectorsWithLensD2NN, self).__init__()

        if up_scaling is None:
            up_scaling = 32
        if border_length is None:
            border_length = pixel_length*pixels_count/2

        self._PlaneLength   = pixel_length * pixels_count
        self._PixelsCount   = pixels_count
        self._UpScaling     = up_scaling
        if torch.is_tensor(wave_length):
            self._Lambdas = wave_length
        else:
            self._Lambdas = torch.tensor([wave_length])

        self._PropagationLayer = PaddedDiffractionLayer(wave_length, space_reflection, pixel_length*pixels_count, pixels_count, layer_spacing_length, up_scaling, border_length)
        self._MaskLayers = torch.nn.ModuleList([HeightMaskLayer(wave_length, space_reflection, mask_reflection, pixels_count, up_scaling, smoothing_matrix) for _ in range(layers_count)])
        self._LensLayer = LensLayer(focus_length, wave_length, pixels_count*up_scaling, pixel_length/up_scaling)
        self._LensPropagationLayer = PaddedDiffractionLayer(wave_length, space_reflection, pixel_length*pixels_count, pixels_count, 1.0*focus_length, up_scaling, border_length)
        self._DetectorsLayer = SquareDetectorsLayer(pixel_length*pixels_count, pixels_count=pixels_count, up_scaling=up_scaling)
    def _restore_attributes(self):
        if (not hasattr(self, '_PlaneLength')) or (self._PlaneLength is None):
            print('Restoring plane length is failed because it`s no resource')
        if (not hasattr(self, '_PixelsCount')) or (self._PixelsCount is None):
            print('Restoring pixels count')
            self._PixelsCount = self._MaskLayers[0].__getattribute__('_PixelsCount')
        if (not hasattr(self, '_UpScaling')) or (self._UpScaling is None):
            print('Restoring pixels count')
            self._UpScaling = self._MaskLayers[0].__getattribute__('_UpScaling')

    def forward(self, field, record_history=False):
        if record_history:
            history = [field]
            field = self._LensPropagationLayer(field)
            field = self._LensLayer(field)
            field = self._LensPropagationLayer(field)
            history.append(field)
            for _MaskLayer in self._MaskLayers[:-1]:
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
                history.append(field)
            field = self._MaskLayers[-1](field)
            field = self._LensPropagationLayer(field)
            field = self._LensLayer(field)
            field = self._LensPropagationLayer(field)
            history.append(field)
            field = self._DetectorsLayer(field)
            return field, history
        else:
            field = self._LensPropagationLayer(field)
            field = self._LensLayer(field)
            field = self._LensPropagationLayer(field)
            for _MaskLayer in self._MaskLayers[:-1]:
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
            field = self._MaskLayers[-1](field)
            field = self._LensPropagationLayer(field)
            field = self._LensLayer(field)
            field = self._LensPropagationLayer(field)
            field = self._DetectorsLayer(field)
            return field

    def save(self, file_name='FileImprovedD2NN.data'):
        try:
            file = open(file_name, 'wb')
            pickle.dump(self, file)
            file.close()
        except Exception as e:
            print(e)
    @staticmethod
    def load(file_name='FileImprovedD2NN.data', class_was_updated=False):
        try:
            file = open(file_name, 'rb')
            self = pickle.load(file)
            if class_was_updated:
                self._restore_attributes()
            file.close()
            return self
        except Exception as e:
            print(e)

    def load_parameters(self, other_model):
        with torch.no_grad():
            for (name, param), (name_, param_) in zip(self.named_parameters(), other_model.named_parameters()):
                param.copy_(param_)

        # for n, module in enumerate(self._MaskLayers):
        #     other_module = other_model.__getattr__('_MaskLayers')[n]
        #     module = HeightMaskLayer()
        #     # print(other_module.__getattr__('_Parameters'))
        #     module.__setattr__('_Parameters', torch.nn.Parameter(other_module.__getattr__('_Parameters')))

    def AccuracyTest(self, batch_size=16, test_loader=None, device=None):
        if test_loader is None:
            transformation = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomRotation((-90, -90)),
                torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
                torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
            ])
            testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
            test_loader = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.eval()
        with torch.no_grad():
            TimeStart = seconds()
            TotalIterations = len(test_loader)
            Iteration = 0

            Correct = 0
            Total = 0

            for batch_num, (images, labels) in enumerate(test_loader):
                Iteration += 1

                images = images.to(device)
                labels = labels.to(device)

                output = self(images)
                Correct += torch.sum(torch.eq(torch.argmax(output, dim=1), labels)).item()
                Total += len(labels)

                string = 'Тестирование, | Осталось: ' + Format.Time((seconds() - TimeStart) * (TotalIterations - Iteration) / Iteration) \
                         + ' | Прошло: ' + Format.Time(seconds() - TimeStart) \
                         + ' | Итерация: ' + str(Iteration) + ' из ' + str(TotalIterations) \
                         + ' | Правильно: ' + str(Correct) + ' из ' + str(Total) \
                         + ' | Точность %: ' + str(round(Correct / Total, 2) * 100)
                sys.stdout.write(f"\r{string + '                                                                                     '}")
        print('')
        return 100 * Correct / Total

    def Train(self, show=True, batch_size=8, epochs_count=1, criterion=None, learning_rate=0.003):

        # Loading Data
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
        train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
        testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)

        # Setting device
        torch.cuda.init()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('Using device:', torch.cuda.get_device_name(), '')
            available, total = torch.cuda.mem_get_info()
            print("Memory | Available: %.2f GB | Total:     %.2f GB" % (available / 1e9, total / 1e9))
        else:
            print('Using device:', 'cpu')
        self.to(device)

        #Initializing criterion and optimizer
        if criterion is None:
            def MSELoss(outputs, corrects):
                return torch.mean((outputs - torch.eye(outputs.size(dim=1),outputs.size(dim=1), device=device, dtype=torch.float32)[corrects])**2)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            criterion = MSELoss
        else:
            criterion = criterion.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        #Initializing loss graph
        colors_list = ['maroon', 'chocolate', 'darkorange', 'goldenrod', 'olivedrab', 'darkgreen', 'darkcyan', 'steelblue', 'navy', 'slateblue', 'darkviolet', 'palevioletred','maroon', 'chocolate', 'darkorange', 'goldenrod', 'olivedrab', 'darkgreen', 'darkcyan', 'steelblue', 'navy', 'slateblue', 'darkviolet', 'palevioletred']
        colors_list_size = len(colors_list)
        loss_buffer = []
        average_loss_buffer = []
        max_loss = -1.0E-50
        min_loss = +1.0E+50
        iter_buffer = numpy.arange(1, len(train_loader)+1)
        x_shift = 1
        fig = plt.figure(figsize=(12,12))
        fig.suptitle('Эволюция лосса при обучении дифрационной сети', **Format.Text('BigHeader', {'fontsize':24}))
        axes = fig.add_subplot(111)
        axes.set_title('Лосс в зависимости от итерации', **Format.Text('BigHeader'))
        axes.xaxis.set_tick_params(labelsize=11)
        axes.yaxis.set_tick_params(labelsize=11)
        axes.set_xlabel('Итерация', **Format.Text('Header'))
        axes.set_ylabel('Лосс', **Format.Text('Header', {'rotation':90}))
        axes.grid(True)
        axes.set_yscale('log')
        fig.show()

        accuracy = 0
        #Training
        for epoch_num in range(epochs_count):
            TimeStart = seconds()
            TotalIterations = len(train_loader)
            Iteration = 0

            LossBufferSize = 20
            LossBuffer = torch.zeros(LossBufferSize, requires_grad=False)
            LossBufferIndex = 0

            loss_buffer.clear()
            average_loss_buffer.clear()
            if epoch_num != 0: iter_buffer += len(train_loader)
            # line, = axes.plot(iter_buffer[0], 10, color=colors_list[epoch_num], linestyle='--', linewidth=0.7)
            line_average, = axes.plot(iter_buffer[0], 10, color=colors_list[epoch_num], linewidth=3.0)

            self.train()
            for batch_num, (images, labels) in enumerate(train_loader):
                Iteration += 1

                images = images.to(device)
                labels = labels.to(device)

                output = self(images)

                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                LossBuffer[LossBufferIndex % LossBufferSize] = loss.item()
                LossBufferIndex += 1

                loss_buffer.append(loss.item())
                if len(average_loss_buffer) == 0:
                    average_loss_buffer.append(loss.item())
                else:
                    average_loss_buffer.append((average_loss_buffer[-1]*LossBufferSize + loss.item())/(LossBufferSize + 1))
                if loss.item() < min_loss: min_loss = loss.item()
                if loss.item() > max_loss: max_loss = loss.item()
                if min_loss == max_loss: max_loss += 1.0E-20
                # line.set_xdata(iter_buffer[:len(loss_buffer)])
                # line.set_ydata(loss_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()
                axes.set_xlim(0, iter_buffer[len(loss_buffer)-1])
                if min_loss != max_loss:
                    axes.set_ylim(min_loss, max_loss)
                line_average.set_xdata(iter_buffer[:len(loss_buffer)])
                line_average.set_ydata(average_loss_buffer)

                string  = 'Обучение, Эпоха ' + str(epoch_num+1) + ' из ' + str(epochs_count) \
                    + ' | Осталось: ' + Format.Time((seconds() - TimeStart)*(TotalIterations - Iteration)/Iteration) \
                    + ' | Прошло: ' + Format.Time(seconds() - TimeStart) \
                    + ' | Итерация: ' + str(Iteration) + ' из ' + str(TotalIterations) \
                    + ' | Средний по ' + str(LossBufferSize) + ' тестам лосс: ' + Format.Scientific(torch.mean(LossBuffer).item())
                sys.stdout.write(f"\r{string + '                                                                                     '}")
            self.eval()
            print('')

            accuracy = self.AccuracyTest(test_loader=test_loader, device=device)
            print('')

        path = 'TrainedModels/AutoSave/'
        if __name__ == '__main__':
            path = '../' + path
        file_name = path + datetime.now().strftime('%m.%d.%Y_%H.%M') + '_' + str(int(accuracy)) + '%' + '.data'
        self.save(file_name)
    def VisualizeHeights(self, show=True):
        N = len(self._MaskLayers)
        Ny = int(numpy.sqrt(N))
        Nx = int(N/Ny) + ((N/Ny - int(N/Ny)) != 0)

        fig = plt.figure(figsize=(12*Nx/Ny+0.16*Nx/4, 12))
        fig.suptitle('Матрицы высот сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05)

        for (n, MaskLayer), (ny, nx)  in zip(enumerate(self._MaskLayers), product(range(Ny), range(Nx))):
            heights = MaskLayer.GetPreparedHeights().cpu()

            unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')
            unitZ,  multZ  = Format.Engineering_Separated(torch.max(heights), 'm')

            heights *= multZ

            axes = Fig.add_axes((nx+1, ny+1))
            axes.set_title('Маска №' + str(n+1), **Format.Text('Header'))
            axes.xaxis.set_tick_params(labelsize=8)
            axes.yaxis.set_tick_params(labelsize=8)
            axes.set_xlabel('X, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold'}))
            axes.set_ylabel('Y, ' + unitXY, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
            image = axes.imshow(heights, cmap='viridis', origin='lower', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])

            cbar = fig.colorbar(image, ax=axes, location='right', shrink=0.8, pad=0.12*Nx/4)
            cbar.ax.set_ylabel('Высота Z, ' + unitZ, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90, 'verticalalignment':'top'}))
            cbar.ax.tick_params(labelsize=8, labelleft=True, left=True, labelright=False, right=False)
        if show:
            plt.show()
    def VisualizeFieldCuts(self, show=True, samples=3, dataset=None, device='cpu'):
        #Dataset
        if dataset is None:
            transformation = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomRotation((-90,-90)),
                torchvision.transforms.Resize(size=(int(self._PixelsCount * self._UpScaling), int(self._PixelsCount * self._UpScaling))),
                torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
            ])
            dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = DataLoader(dataset, batch_size=samples, shuffle=True, num_workers=0)
        images, labels = next(iter(loader))
        images = images.to(device)

        #Detectors
        detectors_data = torch.zeros((samples, 10), dtype=torch.float32)
        detectors_pos = self._DetectorsLayer.GetDetectorsRectanglesCoordinates().cpu()

        #Processing
        History = torch.zeros([2 + len(self._MaskLayers)] + list(images.size()))
        with torch.no_grad():
            field = images
            History[0] = (torch.abs(field.clone().detach())**2).cpu()
            field = self._PropagationLayer(field)
            History[1] = (torch.abs(field.clone().detach())**2).cpu()
            for n, _MaskLayer in enumerate(self._MaskLayers):
                field = _MaskLayer(field)
                field = self._PropagationLayer(field)
                History[n+2] = (torch.abs(field.clone().detach())**2).cpu()
            detectors_data = self._DetectorsLayer(field).clone().detach().cpu()
        if images.size(dim=1) != 1:
            colorizer = Colorizer()
            History = colorizer.Colorize(History, self._Lambdas)
        else:
            History = History.squeeze(dim=2)
        History = History.swapaxes(0,1)

        #Drawing
        Ny = samples
        Nx = 2 + len(self._MaskLayers)

        fig = plt.figure(figsize=(12*Nx/Ny, 12))
        fig.suptitle('Распространение поля по сети', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Nx, Ny), topspace=0.05, hspace=0.03, wspace=-0.08*Nx/Ny, leftspace=-0.001*Nx*Nx/Ny, rightspace=-0.001*Nx*Nx/Ny)

        for m, history in enumerate(History):
            for n, image in enumerate(history):
                name = 'Поле перед маской #' + str(n)
                if n == 0: name = 'Начальное изображение №' + str(m+1)
                if n == History.size(dim=1)-1: name = 'Слой Детекторов'

                unitXY, multXY = Format.Engineering_Separated(self._PlaneLength, 'm')

                axes = Fig.add_axes((n+1, m+1))
                axes.set_title(name, **Format.Text('Caption', {'fontweight':'bold'}))
                axes.imshow(image.swapaxes(0,1), origin='lower', cmap='viridis', extent=[-self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2, -self._PlaneLength * multXY / 2, +self._PlaneLength * multXY / 2])
                if m == History.size(dim=0)-1:
                    axes.xaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_xlabel('X, ' + unitXY, **Format.Text('Caption'))
                else:
                    axes.xaxis.set_tick_params(labelbottom=False, labelsize=8, direction='inout')
                if n == 0:
                    axes.yaxis.set_tick_params(labelsize=8, direction='inout')
                    axes.set_ylabel('Y, ' + unitXY, **Format.Text('Caption', {'rotation': 90}))
                else:
                    axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')

                if n == History.size(dim=1)-1:
                    for i, det in enumerate(detectors_pos):
                        labels_data = detectors_data[m]
                        det_scaled = det * multXY
                        width = det_scaled[1][0] - det_scaled[0][0]
                        height = det_scaled[1][1] - det_scaled[0][1]
                        color = 'maroon'
                        if i == labels[m]: color = 'yellow'
                        if i == int(torch.argmax(labels_data)): color = 'dodgerblue'
                        if (i == labels[m]) and (i == int(torch.argmax(labels_data))): color = 'green'
                        rect = patches.Rectangle((det_scaled[0][0], det_scaled[0][1]), width, height, linewidth=1, edgecolor=color, facecolor='none')
                        axes.add_patch(rect)
                        text_pos = (det_scaled[0][0] + width/2, det_scaled[0][1] + height/2)
                        axes.text(text_pos[0], text_pos[1], Format.Scientific(labels_data[i].item(),precision=1), **Format.Text('Header', {'fontsize':9 ,'c':color}))

        if show:
            plt.show()



def TestModelUpScaling(show=False, dataset=None, up_scaling_list=(1,2,3,4,5,6), layers_count=4, pixels_count=28, pixel_length=10*um, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, layer_spacing_length=1*mm, border_length=None, parameters_file_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MaximumUpScaling = max(up_scaling_list)
    if dataset is None:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    images = images.to(device)

    if border_length is None: border_length = pixels_count*pixel_length*2

    reference_model = ImprovedWithDetectorsD2NN(layers_count=layers_count, pixels_count=pixels_count, pixel_length=pixel_length, wave_length=wave_length, space_reflection=space_reflection, mask_reflection=mask_reflection, layer_spacing_length=layer_spacing_length, up_scaling=MaximumUpScaling, border_length=border_length)
    if parameters_file_name is not None:
        reference_model = ImprovedWithDetectorsD2NN.load(parameters_file_name)

    unitXY, multXY = Format.Engineering_Separated(pixels_count*pixel_length, 'm')
    extentXY = [-pixels_count*pixel_length*multXY/2, +pixels_count*pixel_length*multXY/2, -pixels_count*pixel_length*multXY/2, +pixels_count*pixel_length*multXY/2]

    Cols = layers_count + 2
    Rows = len(up_scaling_list)
    fig = plt.figure(figsize=(12*Cols/Rows, 12))
    Fig = Titles(fig, (Cols, Rows), hspace=0.06/Rows, wspace=0.01/Cols)
    fig.suptitle('Распространение поля в модели в зависимости от размера вычислительного пикселя.\nДлинна волны: ' + Format.Engineering(wave_length,'m') + ', Расстояние между слоями: ' + Format.Engineering(layer_spacing_length, 'm'), **Format.Text('BigHeader', {'verticalalignment':'top'}))
    for row, up_scaling in enumerate(up_scaling_list):
        model = ImprovedWithDetectorsD2NN(layers_count=layers_count, pixels_count=pixels_count, pixel_length=pixel_length, wave_length=wave_length, space_reflection=space_reflection, mask_reflection=mask_reflection, layer_spacing_length=layer_spacing_length, up_scaling=up_scaling, border_length=border_length)
        model.load_parameters(reference_model)
        model.to(device)
        with torch.no_grad():
            image = functional.resize(torch.abs(images), [int(up_scaling*pixels_count), int(up_scaling*pixels_count)], interpolation=torchvision.transforms.InterpolationMode.BILINEAR).to(torch.complex64)
            _, history = model.forward(image, True)
            history = (torch.abs(torch.squeeze(torch.stack(history, 0)))**2).cpu()
        for col, image in enumerate(history):
            axes = Fig.add_axes((col+1, row+1))
            axes.imshow(image.swapaxes(0,1), origin='lower', cmap='viridis', extent=extentXY)
            if row == 0:
                if col == 0:
                    axes.set_title('Входная интенсивность', **Format.Text('Default', {'fontweight':'bold'}))
                elif col == layers_count + 1:
                    axes.set_title('Плоскость детекторов', **Format.Text('Default', {'fontweight':'bold'}))
                else:
                    axes.set_title('Перед маской №' + str(col), **Format.Text('Default', {'fontweight':'bold'}))
            if row == len(up_scaling_list) - 1:
                axes.xaxis.set_tick_params(labelsize=8, direction='inout')
                axes.set_xlabel('X, ' + unitXY, **Format.Text('Caption'))
            else:
                axes.xaxis.set_tick_params(labelbottom=False, labelsize=8, direction='inout')
            axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')
            if col == 0:
                axes.set_ylabel('Размер\nвычислительного\nпикселя: ' + Format.Engineering(pixel_length/up_scaling, 'm'), **Format.Text('Default', {'rotation':90, 'verticalalignment':'bottom'}))

    if show:
        plt.show()

def TestLens(show=False, focus_length=10*mm, plane_length=None, pixels_count=None, up_scaling=None, wave_length=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if wave_length is None:
        wave_length = torch.linspace(350*nm, 700*nm, 12)
    if plane_length is None:
        plane_length = 5*mm
    if pixels_count is None:
        pixels_count = 200 #1000
    if up_scaling is None:
        up_scaling = 1

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((pixels_count*up_scaling, pixels_count*up_scaling), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomRotation((-90, -90)),
        torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    images = images.to(device)

    colorizer = None
    if torch.is_tensor(wave_length):
        images = images.expand(-1, wave_length.size(dim=0), -1, -1)
        colorizer = Colorizer()

    Lens = LensLayer(focus_length, wave_length, pixels_count*up_scaling, plane_length/(pixels_count*up_scaling)).to(device)

    configurations = [
        (2.0*focus_length, 2.0*focus_length),
        (1.0*focus_length, 1.0*focus_length),
        (3.0*focus_length, 1.5*focus_length)
    ]

    Cols = len(configurations)
    Rows = 2

    unitXY, multXY = Format.Engineering_Separated(plane_length, 'm')
    extentXY = [-plane_length * multXY / 2, +plane_length * multXY / 2,
                -plane_length * multXY / 2, +plane_length * multXY / 2]

    fig = plt.figure(figsize=(12*Cols/Rows,12))
    Fig = Titles(fig, (Cols,Rows), topspace=0.15, hspace=0.06/Rows/2, wspace=0.01/Cols/2)
    fig.suptitle('Проверка работы линзы', **Format.Text('BigHeader', {'verticalalignment':'top', 'fontsize':36}))
    for col, (a_length, b_length) in enumerate(configurations):
        Propagation_a = PaddedDiffractionLayer(wave_length = wave_length, plane_length = plane_length, pixels_count = pixels_count, diffraction_length = a_length, up_scaling = up_scaling, border_length = plane_length*2).to(device)
        Propagation_b = PaddedDiffractionLayer(wave_length = wave_length, plane_length = plane_length, pixels_count = pixels_count, diffraction_length = b_length, up_scaling = up_scaling, border_length = plane_length*2).to(device)

        with torch.no_grad():
            image_a = images
            image_b = Propagation_b(Lens(Propagation_a(images)))
            # image_b = Propagation_b(Propagation_a(images))
            if colorizer is None:
                image_a = (torch.abs(image_a[0][0])**2).cpu()
                image_b = (torch.abs(image_b[0][0])**2).cpu()
            else:
                image_a = colorizer.Colorize((torch.abs(image_a)**2).cpu(), wave_length)[0]
                image_b = colorizer.Colorize((torch.abs(image_b)**2).cpu(), wave_length)[0]

        axes = Fig.add_axes((col+1, 1))
        axes.imshow(image_a.swapaxes(0,1), origin='lower', cmap='viridis', extent=extentXY)
        axes.set_title('Конфигурация: ' + Format.Scientific(a_length/focus_length,'',0)[:-1] + 'f - ' + Format.Scientific(b_length/focus_length,'',1)[:-1] + 'f\n' +
                       'Размер вычислительного пикселя: ' + Format.Engineering(plane_length/(pixels_count*up_scaling),'m') + '\n' +
                       'Размер изображения: ' + Format.Engineering(plane_length,'m') + '\n' +
                       'Длинна волны: ' + (Format.Engineering(wave_length,'m') if colorizer is None else Format.Engineering(wave_length[0].item(),'m') + ' - ' + Format.Engineering(wave_length[-1].item(),'m') + ' <' + str(len(wave_length)) + '>')
                       , **Format.Text('Header', {'verticalalignment':'top'}))
        axes.xaxis.set_tick_params(labelbottom=False, labelsize=8, direction='inout')
        axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')
        if col == 0:
            axes.yaxis.set_tick_params(labelleft=True)
            axes.set_ylabel('Y, ' + unitXY)

        axes = Fig.add_axes((col+1, 2))
        axes.imshow(image_b.swapaxes(0, 1), origin='lower', cmap='viridis', extent=extentXY)
        axes.xaxis.set_tick_params(labelbottom=True, labelsize=8, direction='inout')
        axes.yaxis.set_tick_params(labelleft=False, labelsize=8, direction='inout')
        if col == 0:
            axes.yaxis.set_tick_params(labelleft=True)
            axes.set_ylabel('Y, ' + unitXY)
        axes.set_xlabel('X, ' + unitXY)

    if show:
        plt.show()

def test_TrainingImprovedD2NN(show=True, batch_size=8, epochs_count=1, file_name=None, layers_count=5, pixels_count=20, pixel_length=50*um, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, layer_spacing_length=5*mm, up_scaling=None, border_length=None, smoothing_matrix=None):
    from src.DNN import Trainer
    from src.DNN import DETECTOR_POS
    from src.utils import set_det_pos, get_detector_imgs, visualize_n_samples, mask_visualization

    if up_scaling is None:
        up_scaling = 32
    if border_length is None:
        border_length = pixel_length * pixels_count / 2



    #Loading Data
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(int(pixels_count * up_scaling), int(pixels_count * up_scaling))),
        torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
    ])
    training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
    testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
    test_loader = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)



    #Setting device
    torch.cuda.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using device:', torch.cuda.get_device_name(), '')
        available, total = torch.cuda.mem_get_info()
        print("Memory | Available: %.2f GB | Total:     %.2f GB" % (available / 1e9, total / 1e9))
    else:
        print('Using device:', 'cpu')



    #Setting Detector Positions
    edge_x = pixels_count * up_scaling // 20
    edge_y = edge_x * 2
    det_size = (pixels_count * up_scaling - 5 * edge_x) // 4
    labels_image_tensors, detector_pos = get_detector_imgs(det_size=det_size, edge_x=edge_x, edge_y=edge_y, N_pixels=pixels_count * up_scaling)
    labels_image_tensors = labels_image_tensors.to(device)



    #Creating Custom Losses
    def custom_loss_MSE(images, labels):
        full_int_img = images.sum(dim=(1, 2))[:, None, None]
        full_int_label = labels_image_tensors[labels].sum(dim=(1, 2))[:, None, None]
        loss = ((images / full_int_img - labels_image_tensors[labels] / full_int_label) ** 2).sum(dim=(1, 2))
        return loss.mean()
    crossEntropy = torch.nn.CrossEntropyLoss().to(device)
    # def custom_loss(images, labels):
    #     full_int = images.sum(dim=(1, 2))
    #     loss = 1 - (images * labels_image_tensors[labels]).sum(dim=(1, 2)) / full_int
    #     return loss.mean()
    def custom_loss_CrossEntropy(images, labels, multiplier=10):
        full_int_img = images.sum(dim=(1, 2))[:, None, None]
        detector_parts = (images[:, None, :, :] * (labels_image_tensors[None, :, :, :])).sum(dim=(2, 3))
        detector_parts = detector_parts / (detector_parts.max(dim=1).values[:, None]) * multiplier
        return crossEntropy(detector_parts, labels)
    # def custom_loss_CrossEntropy_MSE(images, labels, multiplier=10):
    #     full_int_img = images.sum(dim=(1, 2))[:, None, None]
    #     detector_parts = (images[:, None, :, :] * (labels_image_tensors[None, :, :, :])).sum(dim=(2, 3))
    #     detector_parts = detector_parts / (detector_parts.max(dim=1).values[:, None]) * multiplier
    #     detector_intensity = (1 - detector_parts.sum(dim=1) / full_int_img).mean()
    #     return 0.2 * crossEntropy(detector_parts, labels) + 0.8 * detector_intensity
    def custom_loss_sum(images, labels, multiplier=10):
        return 500 * custom_loss_MSE(images, labels) + 1 * custom_loss_CrossEntropy(images, labels, multiplier=10)
    def custom_loss_concentrate(images, labels, multiplier=10):
        detector_parts = (images[:, None, :, :] * (labels_image_tensors[None, :, :, :])).sum(dim=(2, 3))
        # detector_parts = detector_parts / (detector_parts.max(dim=1).values[:, None]) * multiplier
        # images_integrals = torch.sum(images, dim=(1,2))
        # detectors_integrals = torch.sum(detector_parts, dim=1)
        # detector_parts = (detector_parts.swapdims(0,1)*detectors_integrals/images_integrals).swapdims(0,1)
        # return crossEntropy(detector_parts, labels)
        images_integrals = torch.sum(images, dim=(1,2))
        detectors_integrals = torch.sum(detector_parts, dim=1)
        return torch.sum((images_integrals - detectors_integrals)/images_integrals)
    def custom_loss_combination(images, labels):
        return custom_loss_CrossEntropy(images, labels, multiplier=10) + custom_loss_concentrate(images, labels)

    #Creating Model Optimizer and LossFunction
    if file_name is None:
        model = ImprovedD2NN(layers_count, pixels_count, pixel_length, wave_length, space_reflection, mask_reflection, layer_spacing_length, up_scaling, border_length, smoothing_matrix).to(device)
    else:
        try:
            model = ImprovedD2NN.load(file_name)
        except Exception as e:
            return
    criterion = custom_loss_combination
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)



    #Creating Trainer
    trainer = Trainer(model, detector_pos, 0, device)



    #Visualizing Masks
    model.VisualizeHeights(False)
    # mask_visualization(model)
    # visualize_n_samples(model, testing, n=5, padding=0, detector_pos=detector_pos)



    #Training
    histograms, best_model = trainer.train(criterion, optimizer, train_loader, test_loader, epochs=epochs_count)
    best_model.save()
    best_model.VisualizeHeights(show)
def test_HeightMaskLayer(show=False):
    from itertools import product
    import matplotlib.pyplot as plt

    print('<HeightMaskLayer> - Height smoothing test: ', end='')
    try:
        PixelsCount = 5
        UpScaling = 50
        Samples = 4

        SmoothingMatrix = numpy.zeros((Samples, int(UpScaling / 2), int(UpScaling / 2)))

        SmoothingParameters = numpy.linspace(3.0, 0.5, Samples)
        MaxPixelsShift = UpScaling / 4.0
        CentralPixel = UpScaling / 4.0
        for s, i, j in product(range(Samples), range(int(UpScaling / 2)), range(int(UpScaling / 2))):
            weight = MaxPixelsShift - numpy.sqrt((CentralPixel - i) ** 2 + (CentralPixel - j) ** 2)
            if weight >= 0:
                SmoothingMatrix[s][i][j] = weight ** SmoothingParameters[s]

        fig = plt.figure(figsize=(3.0 * Samples, 3.0 * 4))
        fig.suptitle('Сравнение сглаженных и не сглаженных масок при различных фильтрах', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Samples, 4), topspace=0.05, wspace=0.05)

        for s in range(Samples):
            Mask = HeightMaskLayer(pixels_count=PixelsCount, up_scaling=UpScaling, smoothing_matrix=SmoothingMatrix[s])

            Heights = Mask.GetOriginalHeights().numpy()
            SmoothedHeights = Mask.GetPreparedHeights().numpy()
            SmoothingFilter = Mask.GetSmoothMatrix().numpy()

            hmin = numpy.min(Heights)
            hmax = numpy.max(Heights)

            axes1 = Fig.add_axes((s+1, 1))
            axes1.set_title('Сглаживающая маска', **Format.Text('Default', {'fontweight': 'bold'}))
            axes1.xaxis.set_tick_params(labelsize=6)
            axes1.yaxis.set_tick_params(labelsize=6)
            axes1.set_xlabel('Up Scaled Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes1.set_ylabel('Up Scaled Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes1.imshow(SmoothingFilter, cmap='gray', origin='lower', extent=[-UpScaling / 4, +UpScaling / 4, -UpScaling / 4, +UpScaling / 4])
            cbar = fig.colorbar(image, ax=axes1, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes2 = Fig.add_axes((s + 1, 2))
            axes2.set_title('Оригинальные высоты', **Format.Text('Default', {'fontweight': 'bold'}))
            axes2.xaxis.set_tick_params(labelsize=6)
            axes2.yaxis.set_tick_params(labelsize=6)
            axes2.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes2.set_ylabel('Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes2.imshow(Heights, cmap='viridis', origin='lower', extent=[-PixelsCount / 2, +PixelsCount / 2, -PixelsCount / 2, +PixelsCount / 2], vmin=hmin, vmax=hmax)
            cbar = fig.colorbar(image, ax=axes2, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes3 = Fig.add_axes((s + 1, 3))
            axes3.set_title('Сглаженные высоты', **Format.Text('Default', {'fontweight': 'bold'}))
            axes3.xaxis.set_tick_params(labelsize=6)
            axes3.yaxis.set_tick_params(labelsize=6)
            axes3.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes3.set_ylabel('Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes3.imshow(SmoothedHeights, cmap='viridis', origin='lower', extent=[-PixelsCount / 2, +PixelsCount / 2, -PixelsCount / 2, +PixelsCount / 2], vmin=hmin, vmax=hmax)
            cbar = fig.colorbar(image, ax=axes3, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes4 = Fig.add_axes((s + 1, 4))
            axes4.set_title('Сравнение профилей', **Format.Text('Default', {'fontweight': 'bold'}))
            axes4.xaxis.set_tick_params(labelsize=6)
            axes4.yaxis.set_tick_params(labelsize=6)
            axes4.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes4.set_ylabel('Height', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes4.plot(numpy.repeat(numpy.repeat(Heights, UpScaling, axis=1), UpScaling, axis=0)[int(UpScaling*PixelsCount/2)], color='maroon', linestyle='--')
            axes4.plot(SmoothedHeights[int(UpScaling * PixelsCount / 2)], color='indigo', linestyle='--')

        print('Pass! Check result in plot')
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    print('<HeightMaskLayer> - Transmitting test: ', end='')
    try:
        Toller = 1.0E-5
        Channels = 5
        Tests = 5

        WaveLength = numpy.linspace(100*nm, 1000*nm, Channels)
        SpaceReflection = numpy.linspace(2.0, 1.0, Channels) + 1j*numpy.linspace(0.5, 1.0, Channels)
        MaskReflection  = numpy.linspace(2.5, 1.5, Channels) + 1j*numpy.linspace(0.2, 1.0, Channels)
        PixelsCount = 20
        UpScaling = 10

        Mask = HeightMaskLayer(WaveLength, SpaceReflection, MaskReflection, PixelsCount, UpScaling)

        MaskInputField = torch.rand(Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling) * torch.exp(1j * torch.pi * torch.rand(Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling))
        MaskOutputField = Mask.forward(MaskInputField).clone().detach().numpy()

        Heights = Mask.GetPreparedHeights().numpy()
        MaximumHeight = Mask.GetMaximumHeight()

        RealInputField = MaskInputField.clone().detach().numpy()
        RealOutputField = numpy.zeros((Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling), dtype=complex)
        for s, k, i, j in product(range(Tests), range(Channels), range(PixelsCount*UpScaling), range(PixelsCount*UpScaling)):
            RealOutputField[s][k][i][j] = RealInputField[s][k][i][j] * numpy.exp(2j*numpy.pi*(Heights[i][j]*MaskReflection[k] + (MaximumHeight-Heights[i][j])*SpaceReflection[k])/WaveLength[k])

        Deviations          = numpy.abs(MaskOutputField - RealOutputField)
        MaximumDeviation    = numpy.max(Deviations)
        AverageDeviation    = numpy.mean(Deviations)

        if MaximumDeviation >= Toller:
            raise Exception('Maximum absolute value deviation greater then Toller. Maximum Deviation = ' + str(MaximumDeviation) + ' |  Average Deviation = ' + str(AverageDeviation) + ' |  Toller = ' + Format.Scientific(Toller))

        print('Pass! Maximum Deviation = ' + Format.Scientific(MaximumDeviation) + ' |  Average Deviation = ' + Format.Scientific(AverageDeviation) + ' |  Toller = ' + Format.Scientific(Toller))
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    if show:
        plt.show()
def test_PaddedDiffractionLayer(show=False, field_propagation_test=True, old_scaling_propagation_test=False, scaling_propagation_test=True):
    import matplotlib.pyplot as plt
    from torchvision.transforms import functional
    from itertools import product

    if field_propagation_test:
        print('<PaddedDiffractionLayer> - Field propagation test: ', end='')
        try:
            WaveLength = 600*nm
            PixelsCount = 9
            PlaneLength = PixelsCount * 50*um
            MaxLength = 10.0*mm

            UpScaling = numpy.array([1, 3, 15])
            BorderLength = numpy.linspace(0, 2*PlaneLength, len(UpScaling))

            N = len(UpScaling)

            fig = plt.figure(figsize=(N*4.0, N*4.0))
            fig.suptitle('Распрастронение поля при различных параметрах вычисления', **Format.Text('BigHeader'))
            Fig = Titles(fig, (N, N), hspace=0.1)
            for (nx, US), (ny, BL) in product(enumerate(UpScaling), enumerate(BorderLength)):
                ls = numpy.linspace(0, MaxLength, 100)
                cut = numpy.zeros((100, PixelsCount*US))
                field = torch.nn.functional.pad(torch.ones((US, US), dtype=torch.complex64), (int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2)))
                for i, l in enumerate(ls):
                    Propagator = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=l, up_scaling=US, border_length=BL)
                    cut[i] = (numpy.abs(Propagator.forward(field).clone().detach().numpy())**1)[0][int(PixelsCount*US/2)]

                unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
                unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

                axes = Fig.add_axes((nx+1, ny+1))
                axes.set_title('Множитель разрешения: ' + str(US) + '\nРасширение на: ' + Format.Engineering(BL, 'm'), **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
                axes.xaxis.set_tick_params(labelsize=6)
                axes.yaxis.set_tick_params(labelsize=6)
                axes.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
                axes.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
                axes.imshow(cut, cmap='viridis', origin='lower', extent=[-PlaneLength*multX/2, +PlaneLength*multX/2, 0, MaxLength*multZ], aspect='auto')

                axes.autoscale(enable=False)

                axes.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
                for k in range(1, 8+1):
                    tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                    xy0 = (0, 0)
                    y1 = MaxLength * multZ
                    x1 = y1 * tan * multX / multZ
                    axes.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                    axes.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if old_scaling_propagation_test:
        print('<PaddedDiffractionLayer> - Field propagation scaling test: ', end='')
        try:
            MaxLength = 10.0*mm
            UpScaling = 15

            Ratios = [1.0, 2.0, 0.5, 4.0, 0.25]

            WaveLength_  = 600*nm
            PixelSize_   = 50*um

            N = len(Ratios)

            normed_cuts = []

            fig = plt.figure(figsize=(N * 4.0, 3 * 4.0))
            fig.suptitle('Распрастронение поля при пропорционально изменяющихся длинах волн и размерах пикселя', **Format.Text('BigHeader'))
            Fig = Titles(fig, (N, 3), hspace=0.1)
            for (c, ratio) in enumerate(Ratios):
                WaveLength = WaveLength_*ratio
                PixelSize = PixelSize_*ratio

                PixelsCount = int(8 / ratio)

                PlaneLength = PixelsCount * PixelSize
                BorderLength = PlaneLength

                ls = numpy.linspace(0, MaxLength, 100)
                cut = numpy.zeros((100, PixelsCount * UpScaling))

                field = torch.zeros(PixelsCount*UpScaling, PixelsCount*UpScaling)
                min_index = int(PixelsCount*UpScaling/2 - UpScaling/2)
                max_index = int(min_index + UpScaling)
                field[min_index:max_index, min_index:max_index] = torch.ones(UpScaling, UpScaling)
                for i, l in enumerate(ls):
                    Propagator = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=l, up_scaling=UpScaling, border_length=BorderLength)
                    cut[i] = (numpy.abs(Propagator.forward(field).clone().detach().numpy()) ** 1)[0][int(PixelsCount * UpScaling / 2)]

                unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
                unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

                axes1 = Fig.add_axes((c+1, 1))
                axes1.set_title('Длинна волны: ' + Format.Engineering(WaveLength) + '\nРазмер пикселя: ' + Format.Engineering(PixelSize) + '\nМножитель: ' + str(ratio), **Format.Text('Default', {'fontweight': 'bold'}), pad=21)
                axes1.xaxis.set_tick_params(labelsize=6)
                axes1.yaxis.set_tick_params(labelsize=6)
                axes1.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
                axes1.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
                axes1.imshow(cut, cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
                axes1.autoscale(enable=False)
                axes1.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
                for k in range(1, 8 + 1):
                    tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                    xy0 = (0, 0)
                    y1 = MaxLength * multZ
                    x1 = y1 * tan * multX / multZ
                    axes1.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                    axes1.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

                axes2 = Fig.add_axes((c+1, 2))
                axes2.set_title('Нормированное на множитель', **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
                axes2.xaxis.set_tick_params(labelsize=6)
                axes2.yaxis.set_tick_params(labelsize=6)
                axes2.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
                axes2.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
                axes2.imshow(cut/ratio, cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
                axes2.autoscale(enable=False)
                axes2.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
                for k in range(1, 8 + 1):
                    tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                    xy0 = (0, 0)
                    y1 = MaxLength * multZ
                    x1 = y1 * tan * multX / multZ
                    axes2.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                    axes2.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

                normed_cut = torch.tensor(cut/ratio)
                if c != 0:
                    normed_cut = functional.resize(normed_cut.expand(1,1,-1,-1), normed_cuts[0].shape).squeeze()
                normed_cuts.append(normed_cut.numpy())

            for c, cut in enumerate(normed_cuts):
                PixelSize = PixelSize_ * Ratios[c]
                PixelsCount = int(8 / Ratios[c])
                PlaneLength = PixelsCount * PixelSize

                unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
                unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

                axes = Fig.add_axes((c+1, 3))
                axes.set_title('Разница нормированных срезов', **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
                axes.xaxis.set_tick_params(labelsize=6)
                axes.yaxis.set_tick_params(labelsize=6)
                axes.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
                axes.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
                axes.imshow(numpy.abs(cut-normed_cuts[0]), cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
                if c == 0:
                    axes.text(0, MaxLength * multZ/2, 'Reference', **Format.Text('BigHeader'))

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if scaling_propagation_test:
        print('<PaddedDiffractionLayer> - Field propagation scaling test: ', end='')
        try:
            use_fast_method = False

            PixelsCount = 9
            UpScaling = 20

            InitialDiffractionLength = 10*mm
            InitialPixelLength = 50*um
            InitialWaveLength = 600*nm

            Ratios = [0.125, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00]
            Samples = len(Ratios)

            Steps = 150

            Cols = Samples
            Rows = 2

            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                InitialField = torch.zeros((PixelsCount*UpScaling, PixelsCount*UpScaling), dtype=torch.complex64).to(device)
                Center = int(PixelsCount*UpScaling/2)
                Nx1 = int((PixelsCount*UpScaling - UpScaling)/2)
                Ny1 = int((PixelsCount*UpScaling - UpScaling)/2)
                Nx2 = int((PixelsCount*UpScaling + UpScaling)/2) + 1
                Ny2 = int((PixelsCount*UpScaling + UpScaling)/2) + 1
                InitialField[Nx1:Nx2, Ny1:Ny2] = torch.ones((Nx2-Nx1, Ny2-Ny1), dtype=torch.complex64)

                Pictures = torch.zeros((Samples, Steps, PixelsCount*UpScaling), dtype=torch.float32)

                fig = plt.figure(figsize=(1.4*3.0*Cols, 1.4*3.5*Rows))
                Fig = Titles(fig, (Cols, Rows), topspace=0.22)
                fig.suptitle('Распространение поля от пикселя\nпри сохранение отношения длинны волны к длинне пикселя', **Format.Text('BigHeader', {'verticalalignment':'top'}))
                for col, ratio in enumerate(Ratios):
                    PixelLength = InitialPixelLength * ratio
                    PlaneLength = PixelLength * PixelsCount
                    WaveLength = InitialWaveLength * ratio
                    DiffractionLength = InitialDiffractionLength * ratio
                    DiffractionLengthStep = DiffractionLength / (Steps - 1)

                    if use_fast_method:
                        PropagatorLayer = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=DiffractionLengthStep, up_scaling=UpScaling, border_length=PlaneLength).to(device)
                        field = InitialField.clone().expand(1,1,-1,-1)
                        Pictures[col][0] = (torch.abs(field[0][0][Center])**1).cpu()
                        for step in range(1, Steps):
                            field = PropagatorLayer.forward(field)
                            Pictures[col][step] = (torch.abs(field[0][0][Center])**1).cpu()
                    else:
                        Lengths = torch.linspace(0, DiffractionLength, Steps)
                        initial_field = InitialField.clone().expand(1, 1, -1, -1)
                        for step, length in enumerate(Lengths):
                            PropagatorLayer = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=length, up_scaling=UpScaling, border_length=PlaneLength).to(device)
                            field = PropagatorLayer.forward(initial_field)
                            Pictures[col][step] = (torch.abs(field[0][0][Center]) ** 1).cpu()

                    unitX, multX = Format.Engineering_Separated(PlaneLength/2, 'm')
                    unitY, multY = Format.Engineering_Separated(DiffractionLength, 'm')

                    axes = Fig.add_axes((col+1, 1))
                    axes.set_title(
                        'Размер пикселя: ' + Format.Engineering(PixelLength, 'm') + '\n' +
                        'Размер маски: ' + Format.Engineering(PlaneLength, 'm') + '\n' +
                        'Длинна волны: ' + Format.Engineering(WaveLength, 'm') + '\n' +
                        'Множитель: ' + str(ratio),
                        **Format.Text('Default', {'fontweight':'bold'})
                    )
                    axes.xaxis.set_tick_params(labelsize=8)
                    axes.yaxis.set_tick_params(labelsize=8)
                    axes.set_xlabel('Вдоль маски, ' + unitX, **Format.Text('Caption'))
                    axes.set_ylabel('Поперёк маски, ' + unitY, **Format.Text('Caption', {'rotation':90}))
                    axes.imshow(Pictures[col].numpy(), origin='lower', cmap='viridis', extent=[-PlaneLength*multX/2, +PlaneLength*multX/2, 0, DiffractionLength*multY], aspect='auto')

                ReferenceIndex = Ratios.index(1.0)
                for col in range(Samples):
                    axes = Fig.add_axes((col+1, 2))
                    axes.set_title('Сравнение: ', **Format.Text('Default', {'fontweight':'bold'}))
                    axes.xaxis.set_tick_params(labelsize=8)
                    axes.yaxis.set_tick_params(labelsize=8)
                    axes.set_xlabel('Номер вычислительного пикселя', **Format.Text('Caption'))
                    axes.set_ylabel('Номер шага дифракции', **Format.Text('Caption', {'rotation':90}))
                    image = torch.abs(Pictures[col] - Pictures[ReferenceIndex])
                    axes.imshow(image.numpy(), origin='lower', cmap='viridis', aspect='auto')
                    if col == ReferenceIndex:
                        axes.text(PixelsCount*UpScaling/2, Steps/2, 'Reference', **Format.Text('BigHeader', {'c':'white'}))
                    elif torch.sum(image) == 0:
                        axes.text(PixelsCount*UpScaling/2, Steps/2, 'Zero\neverywhere', **Format.Text('BigHeader', {'c':'white'}))

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if show:
        plt.show()
def test_ImprovedD2NN(show=False):
    import matplotlib.pyplot as plt

    print('<ImprovedD2NN> - Save and load test: ', end='')
    try:
        PixelsCount = 20
        UpScaling = 16
        LambdasCount = 5
        Lambdas = torch.linspace(400*nm, 700*nm, LambdasCount)
        Tests = 10

        Network0 = ImprovedD2NN(pixels_count=PixelsCount, up_scaling=UpScaling, wave_length=Lambdas)
        Network0.save()

        Network1 = ImprovedD2NN.load()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Network0 = Network0.to(device)
        Network1 = Network1.to(device)

        Inputs = torch.rand((Tests, LambdasCount, PixelsCount*UpScaling, PixelsCount*UpScaling), device=device) * torch.exp(2j*torch.pi*torch.rand((Tests, LambdasCount, PixelsCount*UpScaling, PixelsCount*UpScaling), device=device))

        Outputs0 = Network0.forward(Inputs)
        Outputs1 = Network1.forward(Inputs)

        MaxError = torch.max(torch.abs(Outputs1 - Outputs0)).cpu()

        if MaxError != 0:
            raise Exception('Maximum Error greater than zero ' + Format.Scientific(MaxError))
        print('Pass!')
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    if show:
        plt.show()
def test_show():
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == '__main__':
    # TestModelUpScaling(up_scaling_list=(1, 11, 21, 31), pixel_length=50*um, layer_spacing_length=20*mm)
    # TestModelUpScaling(up_scaling_list=(1, 2, 3, 4, 5, 6), pixel_length=50*um, layer_spacing_length=20*mm)
    # TestLens(wave_length=500*nm)
    #TestLens()
    # test_show()

    # test_HeightMaskLayer()
    test_PaddedDiffractionLayer(field_propagation_test=False)
    # test_ImprovedD2NN()
    test_show()

    # Model = ImprovedWithDetectorsD2NN(layers_count=5, pixels_count=30, up_scaling=20, layer_spacing_length=100*mm).cuda()
    # Model.load_parameters(ImprovedWithDetectorsD2NN.load())
    # Model.save('Model_85%_accuracy.data')
    # # Model = ImprovedWithDetectorsD2NN.load()
    # # Model.VisualizeHeights(show=False)
    # # Model.Train(batch_size=50, epochs_count=4, learning_rate=0.0005)
    # Model.VisualizeHeights(show=False)
    # Model.VisualizeFieldCuts(show=False, samples=4, device='cuda')
    # # Model.AccuracyTest(50)
    # import winsound
    # for freq in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
    #     winsound.Beep(freq, 100)
    # test_show()

    # Model = ImprovedWithDetectorsD2NN(layers_count=2, pixels_count=10, up_scaling=8, layer_spacing_length=100*mm).cuda()
    # Model.Train(batch_size=500, epochs_count=1, learning_rate=0.003)


