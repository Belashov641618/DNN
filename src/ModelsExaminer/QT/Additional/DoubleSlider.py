from PyQt6.QtWidgets import (
    QSlider,
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot

class DoubleSlider(QSlider):
    _Ratio  : float

    def __init__(self, parent=None):
        super(DoubleSlider, self).__init__(parent)

        super(DoubleSlider, self).setSingleStep(1)

        self._Ratio = 1.0

    def setSingleStep(self, step:float):
        value = self.value()
        min_value = super(DoubleSlider, self).minimum()*self._Ratio
        max_value = super(DoubleSlider, self).maximum()*self._Ratio
        self._Ratio = step
        self.setValue(value)
        self.setMinimum(min_value)
        self.setMaximum(max_value)
    def setMinimum(self, min_value:float):
        super(DoubleSlider, self).setMinimum(int(min_value/self._Ratio))
    def setMaximum(self, max_value:float):
        super(DoubleSlider, self).setMaximum(int(max_value/self._Ratio))
    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value/self._Ratio))

    def value(self):
        return super(DoubleSlider, self).value()*self._Ratio