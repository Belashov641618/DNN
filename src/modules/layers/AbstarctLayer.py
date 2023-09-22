import torch

from utilities.DelayedFunctions import DelayedFunctions
from utilities.Accuracy import Accuracy

class AbstractLayer(torch.nn.Module):

    _delayed : DelayedFunctions
    _auto_delayed : bool
    @property
    def delayed(self):
        class Container:
            _self : AbstractLayer
            def __init__(self, _self:AbstractLayer):
                self._self = _self
            def finalize(self):
                self._self._delayed.launch()
            def auto(self, state:bool=True):
                self._self._auto_delayed = state
        return Container(self)

    _accuracy : Accuracy
    @property
    def accuracy(self):
        class Container:
            _self : AbstractLayer
            def __init__(self, _self:AbstractLayer):
                self._self = _self
            def bits(self, count:int):
                self._self._accuracy.set(bits=count)
            def bits16(self):
                self._self._accuracy.set(bits=16)
            def bits32(self):
                self._self._accuracy.set(bits=32)
            def bits64(self):
                self._self._accuracy.set(bits=64)
            def get(self):
                return self._self._accuracy.get()
        return Container(self)

    def __init__(self):
        super(AbstractLayer, self).__init__()

        self._delayed = DelayedFunctions()
        self._auto_delayed = True

        self._accuracy = Accuracy()

    def forward(self, field:torch.Tensor):
        if self._auto_delayed:
            self._delayed.launch()