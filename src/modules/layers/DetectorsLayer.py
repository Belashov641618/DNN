import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy
from copy import deepcopy
from typing import Union, Iterable, Tuple, List, Dict

from .AbstarctLayer import AbstractLayer

manual_type:int = 0
square_type:int = 1
polar_type:int  = 2

integral_normalization:int = 0
maximum_normalization:int  = 1
softmax_normalization:int  = 2

class DetectorsLayer(AbstractLayer):

    _masks : torch.Tensor
    _masks_type : Tuple[int, List, Dict]
    @property
    def masks(self):
        class Selector:
            _self : DetectorsLayer
            def __init__(self, _self:DetectorsLayer):
                self._self = _self
            @property
            def get(self):
                class Getter:
                    _self : DetectorsLayer
                    def __init__(self, _self:DetectorsLayer):
                        self._self = _self
                    def __call__(self, to_cpu:bool=True):
                        self._self.delayed.finalize()
                        if to_cpu:
                            return self._self._masks.clone().detach().cpu()
                        return self._self._masks.clone().detach()
                    def sum(self, to_cpu:bool=True):
                        self._self.delayed.finalize()
                        if to_cpu:
                            return torch.sum(self._self._masks.clone().detach()).cpu()
                        return torch.sum(self._self._masks.clone().detach())
                return Getter(self._self)
            @property
            def set(self):
                class Setter:
                    _self : DetectorsLayer
                    def __init__(self, _self:DetectorsLayer):
                        self._self = _self
                    def __call__(self, masks:Union[Iterable[torch.Tensor],torch.Tensor]):
                        self._self._masks_type = (manual_type, [], {})
                        new_masks = torch.zeros(self._self._detectors,
                                                self._self.pixels * self._self.up_scaling,
                                                self._self.pixels * self._self.up_scaling,
                                                dtype=self._self._accuracy.tensor_float)
                        if isinstance(masks, torch.Tensor):
                            if len(masks.size()) != 3 or masks.size(0) != self._self._detectors or masks.size(1) != self._self.pixels*self._self.up_scaling or masks.size(2) != self._self.pixels*self._self.up_scaling:
                                raise Exception('masks заданные тензором должны иметь 3 измерения: [кол-во детекторов]x[вычислительные пиксели]x[вычислительные пиксели] !')
                            new_masks = masks
                        else:
                            n = 0
                            for i, mask in enumerate(masks):
                                if mask.size(0) != self._self.pixels*self._self.up_scaling or mask.size(1) != self._self.pixels*self._self.up_scaling:
                                    raise Exception('masks заданные списком должны иметь 2 измерения: [вычислительные пиксели]x[вычислительные пиксели] !')
                                new_masks[i] = mask
                                n += 1
                            if n != self._self._detectors:
                                raise Exception('Недостаточно элементов в списке masks!')
                        self._self.register_buffer('_masks', new_masks)
                    def square(self, borders:float=0.05, space:float=0.2):
                        self._self._masks_type = (square_type, [], {
                            'borders':borders,
                            'space':space
                        })
                        detectors = self._self._detectors
                        pixels = self._self.pixels * self._self.up_scaling
                        new_masks = torch.zeros(detectors, pixels, pixels, dtype=self._self._accuracy.tensor_float)

                        rows = int(numpy.sqrt(detectors))
                        if rows == 0: rows = 1
                        cols = int(detectors/rows)
                        rest = detectors - cols*rows
                        amounts = [cols for i in range(rows)]
                        for i in range(rest):
                            amounts[int(int(rows/2)+(1-2*(i%2))*i)] += 1
                        max_cols = cols
                        if rest: max_cols += 1

                        length = 1.0 - 2.0*borders
                        width = length / (max_cols * (1.0 + space))
                        height = length / (rows * (1.0 + space))
                        y_pad = height*space
                        x_pad = width*space

                        central_points = []
                        for ny, amount in enumerate(amounts):
                            y = borders + height/2 + (y_pad+height)*ny
                            pad = (length - amount*width + (amount-1)*x_pad) / 2
                            for nx in range(amount):
                                x = borders + pad + width/2 + (x_pad+width)*nx
                                central_points.append((x, y))
                        border_points = [(cx-width/2, cy-height/2, cx+width/2, cy+height/2) for (cx, cy) in central_points]
                        border_pixels = [(int(x1*pixels), int(y1*pixels), int(x2*pixels), int(y2*pixels)) for (x1, y1, x2, y2) in border_points]

                        for i, (nx1, ny1, nx2, ny2) in enumerate(border_pixels):
                            new_masks[i][nx1:nx2][ny1:ny2] = torch.ones(nx2-nx1, ny2-ny1)

                        self._self.register_buffer('_masks', new_masks)
                    def polar(self, borders:float=0.05, space:float=0.2, power:float=0.5):
                        self._self._masks_type = (polar_type, [], {
                            'borders':borders,
                            'space':space,
                            'power':power
                        })
                        detectors = self._self._detectors
                        pixels = self._self.pixels * self._self.up_scaling
                        new_masks = torch.zeros(detectors, pixels, pixels, dtype=self._self._accuracy.tensor_float)

                        rows = int(numpy.sqrt(detectors))
                        if rows == 0: rows = 1
                        cols = int(detectors/rows)
                        rest = detectors - cols*rows
                        amounts = [cols for i in range(rows)]
                        for i in range(rest):
                            amounts[int(int(rows/2)+(1-2*(i%2))*i)] += 1
                        max_cols = cols
                        if rest: max_cols += 1

                        length = 1.0 - 2.0*borders
                        width = length / (max_cols * (1.0 + space))
                        height = length / (rows * (1.0 + space))
                        y_pad = height*space
                        x_pad = width*space

                        central_points = []
                        for ny, amount in enumerate(amounts):
                            y = borders + height/2 + (y_pad+height)*ny
                            pad = (length - amount*width + (amount-1)*x_pad) / 2
                            for nx in range(amount):
                                x = borders + pad + width/2 + (x_pad+width)*nx
                                central_points.append((x, y))

                        x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1, +1, pixels), torch.linspace(-1, +1, pixels))
                        for i, (cx, cy) in enumerate(central_points):
                            radius = torch.sqrt(torch.square(x_mesh - cx) + torch.square(y_mesh - cy))
                            new_masks[i] = 1.0 / (1.0 + torch.pow(radius, power))

                        self._self.register_buffer('_masks', new_masks)
                return Setter(self._self)
        return Selector(self)
    def _recalc_masks(self):
        if hasattr(self, '_masks_type'):
            variant, args, kwargs = self._masks_type
            if variant == manual_type:
                old_masks = self._masks.clone().detach()
                pixels = self._pixels*self._up_scaling
                self.masks.set(resize(old_masks.expand(1,-1,-1,-1), [pixels, pixels], interpolation=InterpolationMode.NEAREST).squeeze())
            elif variant == square_type:
                self.masks.set.square(*args, **kwargs)
            elif variant == polar_type:
                self.masks.set.polar(*args, **kwargs)
        else:
            self.masks.set.square()

    _detectors : int
    @property
    def detectors(self):
        return self._detectors
    @detectors.setter
    def detectors(self, amount:int):
        self._detectors = amount
        self._delayed.add(self._recalc_masks)

    _pixels : int
    @property
    def pixels(self):
        return self._pixels
    @pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._delayed.add(self._recalc_masks)

    _up_scaling : int
    @property
    def up_scaling(self):
        return self._up_scaling
    @up_scaling.setter
    def up_scaling(self, calculating_pixels_per_pixel:int):
        self._up_scaling = calculating_pixels_per_pixel
        self._delayed.add(self._recalc_masks)

    _normalization : int
    @property
    def normalization(self):
        class Selector:
            _self:DetectorsLayer
            def __init__(self, _self:DetectorsLayer):
                self._self = _self
            def get(self):
                return deepcopy(self._self._normalization)
            @property
            def set(self):
                class Variants:
                    _self:DetectorsLayer
                    def __init__(self, _self:DetectorsLayer):
                        self._self = _self
                    def integral(self):
                        self._self._normalization = integral_normalization
                    def maximum(self):
                        self._self._normalization = maximum_normalization
                    def softmax(self):
                        self._self._normalization = softmax_normalization
                return Variants(self._self)
        return Selector(self)

    def __init__(self,  detectors:int=10,
                        pixels:int=20,
                        up_scaling:int=8):
        super(DetectorsLayer, self).__init__()

        self._detectors = detectors
        self._pixels = pixels
        self._up_scaling = up_scaling
        self._normalization = integral_normalization
        self._recalc_masks()

    def forward(self, field:torch.Tensor):
        super(DetectorsLayer, self).forward(field)

        results = torch.sum((torch.abs(field)**2).expand(10,-1,-1,-1,-1).movedim(0,2) * self._masks, dim=(1,3,4))

        if self._normalization == integral_normalization:
            field_integral = torch.sum(torch.abs(field)**2, dim=(1, 2, 3))
            results = results / field_integral.expand(10, -1).swapdims(0,1)
        elif self._normalization == maximum_normalization:
            results = results / (results.max(dim=1).values[:, None])
        elif self._normalization == softmax_normalization:
            results = torch.softmax(results, dim=1)

        return results