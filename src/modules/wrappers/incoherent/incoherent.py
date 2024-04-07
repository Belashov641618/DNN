import torch

from noises.normalizations import GaussianNormalizer
from noises.fouriers import gaussian

if __name__ == '__main__':
    from src.modules.models.AbstractModel import AbstractModel
else:
    from ...models.AbstractModel import AbstractModel

class Incoherent(AbstractModel):
    _model:torch.nn.Module

    _spatial_coherence:float
    @property
    def spatial_coherence(self):
        return self._spatial_coherence
    @spatial_coherence.setter
    def spatial_coherence(self, meters:float):
        self._spatial_coherence = meters
        self._delayed.add(self._reset_generator)

    _time_coherence:float
    @property
    def time_coherence(self):
        return self._time_coherence
    @time_coherence.setter
    def time_coherence(self, seconds:float):
        self._time_coherence = seconds
        self._delayed.add(self._reset_generator)
    _pixels:int
    @property
    def pixels(self):
        return self._pixels
    @pixels.setter
    def pixels(self, pixels:int):
        self._pixels = pixels
        self._delayed.add(self._reset_generator)

    _generator:GaussianNormalizer
    def _reset_generator(self):
        pass

    def __init__(self, model:torch.nn.Module, Nt:int=100):
        super(Incoherent, self).__init__()
        self._model = model

    def forward(self, field:torch.Tensor):
        if not hasattr(self, '_pixels'):
            self.pixels = field.shape[2]
        super(Incoherent, self).forward(field)


