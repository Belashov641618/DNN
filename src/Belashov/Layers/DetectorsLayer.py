import torch
from torchvision.transforms.functional import resize as f_resize
from torchvision.transforms import InterpolationMode
from typing import Union, Iterable
from copy import deepcopy
from src.AdditionalUtilities.DecimalPrefixes import mm, um, nm

from src.Belashov.Layers.AbstractLayer import AbstractLayer


class DetectorsLayer(AbstractLayer):
    @staticmethod
    def GenerateSquareDetectorsMasks(plane_length:float, total_pixels_count:int, detector_length:float=None, detectors_layout_border_ratio:float=0.1):
        if detector_length is None:
            detector_length = plane_length * (1.0-detectors_layout_border_ratio) / 5
        if detector_length*4 > plane_length*(1.0-detectors_layout_border_ratio):
            raise ValueError('Detector length is to large')

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
        initial_detector_centers = initial_detector_centers * (plane_length * (1.0-detectors_layout_border_ratio) / (3.0*detector_length) - 1.0/3.0)

        detectors_rectangles = initial_detector_centers.unsqueeze(dim=2).repeat(1,1,2).swapdims(1,2)
        detectors_rectangles[:, 0, :] = detectors_rectangles[:, 0, :] - 0.5*detector_length
        detectors_rectangles[:, 1, :] = detectors_rectangles[:, 1, :] + 0.5*detector_length
        detectors_rectangles = detectors_rectangles + plane_length / 2

        detectors_rectangles_pixels = (detectors_rectangles * total_pixels_count / plane_length).to(torch.int32)
        detectors_rectangles_pixels = torch.where(detectors_rectangles_pixels >= total_pixels_count, total_pixels_count-1, detectors_rectangles_pixels)
        detectors_rectangles_pixels = torch.where(detectors_rectangles_pixels < 0, 0, detectors_rectangles_pixels)

        DetectorsMasks = torch.zeros((len(initial_detector_centers), total_pixels_count, total_pixels_count), dtype=torch.float64)
        for num, rectangle in enumerate(detectors_rectangles_pixels):
            DetectorsMasks[num, rectangle[0][0]:rectangle[1][0]+1, rectangle[0][1]:rectangle[1][1]+1] = torch.ones(rectangle[1][0]-rectangle[0][0]+1, rectangle[1][1]-rectangle[0][1]+1)

        return DetectorsMasks
    @staticmethod
    def GeneratePolarDetectorsMasks(plane_length:float, total_pixels_count:int, detector_length:float=None, detectors_layout_border_ratio:float=0.1):
        if detector_length is None:
            detector_length = plane_length * (1.0-detectors_layout_border_ratio) / 5
        if detector_length*4 > plane_length*(1.0-detectors_layout_border_ratio):
            raise ValueError('Detector length is to large')

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
        initial_detector_centers = initial_detector_centers * (plane_length * (1.0-detectors_layout_border_ratio) / (3.0*detector_length) - 1.0/3.0)

        DetectorsMasks = torch.zeros(len(initial_detector_centers), total_pixels_count, total_pixels_count)
        xx, yy = torch.meshgrid(torch.linspace(-plane_length/2, +plane_length/2, total_pixels_count), torch.linspace(-plane_length/2, +plane_length/2, total_pixels_count), indexing='ij')

        for i, center in enumerate(initial_detector_centers):
            x0 = center[0]
            y0 = center[1]
            R = torch.sqrt((xx-x0)**2 + (yy-y0)**2)
            DetectorsMasks[i] = 1.0 / (R**0.5 + 1)

        return DetectorsMasks

    @staticmethod
    def GenerateDetectorsMasksStringRedirector(generating_type:str, plane_length:float, total_pixels_count:int):
        if generating_type == 'Square':
            return DetectorsLayer.GenerateSquareDetectorsMasks(plane_length, total_pixels_count)
        elif generating_type == 'Polar':
            return DetectorsLayer.GeneratePolarDetectorsMasks(plane_length, total_pixels_count)
        else:
            raise ValueError("\033[31m\033[1m{}".format('GenerateDetectorsMasksStringRedirector: there is no generating_type with name ' + generating_type + '!'))


    _DetectorsMasksBuffer : torch.Tensor
    def _recalc_DetectorsMasksBuffer(self):
        if self._DetectorsMasksBuffer.size() != torch.Size((10, self._PixelsCount*self._UpScaling, self._PixelsCount*self._UpScaling)):
            DetectorsMaskCopy = self._DetectorsMasksBuffer.clone().detach()
            self.register_buffer('_DetectorsMasksBuffer', f_resize(DetectorsMaskCopy.expand(1,-1,-1,-1), [self._PixelsCount*self._UpScaling, self._PixelsCount*self._UpScaling], interpolation=InterpolationMode.NEAREST).squeeze())
    def GetDetectorsMasksBuffer(self, to_cpu:bool=True):
        self._launch_DelayedFunctions()
        if to_cpu:
            return deepcopy(self._DetectorsMasksBuffer).cpu()
        return deepcopy(self._DetectorsMasksBuffer)
    def GetDetectorsMasksBufferSum(self, to_cpu:bool=True):
        self._launch_DelayedFunctions()
        if to_cpu:
            return torch.sum(deepcopy(self._DetectorsMasksBuffer)).cpu()
        return torch.sum(deepcopy(self._DetectorsMasksBuffer))


    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        self._add_DelayedFunction(self._recalc_DetectorsMasksBuffer, 1.0)

    _UpScaling : int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling
        self._add_DelayedFunction(self._recalc_DetectorsMasksBuffer, 1.0)


    def __init__(self, pixels_count:int=50, up_scaling:int=20, detectors_masks:Union[str,Iterable,torch.Tensor]='Square'):
        super(DetectorsLayer, self).__init__()

        self._PixelsCount   = pixels_count
        self._UpScaling     = up_scaling

        if isinstance(detectors_masks, str):
            detectors_masks = DetectorsLayer.GenerateDetectorsMasksStringRedirector(detectors_masks, 1.0, pixels_count*up_scaling)
        elif not torch.is_tensor(detectors_masks):
            detectors_masks = torch.tensor(detectors_masks)
        detectors_masks.to(self._tensor_float_type)

        self.register_buffer('_DetectorsMasksBuffer', detectors_masks)

    def forward(self, field:torch.Tensor):
        super(DetectorsLayer, self).forward(field)
        results = torch.sum((torch.abs(field)**2).expand(10,-1,-1,-1,-1).movedim(0,2) * self._DetectorsMasksBuffer, dim=(1,3,4))

        # field_integral = torch.sum(torch.abs(field)**2, dim=(1, 2, 3))
        # results = results / field_integral.expand(10, -1).swapdims(0,1)

        results = results / (results.max(dim=1).values[:, None])

        # results = torch.softmax(results, dim=1)

        return results