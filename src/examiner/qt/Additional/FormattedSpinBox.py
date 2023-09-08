from PyQt6.QtWidgets import (
    QDoubleSpinBox,
)

class FormattedSpinBox(QDoubleSpinBox):

    def __init__(self, parent=None, unit:str=None, multiplier:float=None):
        super(FormattedSpinBox, self).__init__(parent)

    _Multiplier : float
    _Unit : str
    def SetMultiplier(self, multiplier:float):
        self._Multiplier = multiplier
    def SetUnit(self, unit:str):
        self._Unit = unit
        super(FormattedSpinBox, self).setSuffix(self._Unit)

    def setSingleStep(self, step: float):
        step = step * self._Multiplier
        super(FormattedSpinBox, self).setSingleStep(step)
    def setMinimum(self, min_value: float):
        min_value = min_value * self._Multiplier
        super(FormattedSpinBox, self).setMinimum(min_value)
    def setMaximum(self, max_value: float):
        max_value = max_value * self._Multiplier
        super(FormattedSpinBox, self).setMaximum(max_value)
    def setValue(self, value):
        value = value * self._Multiplier
        super(FormattedSpinBox, self).setValue(value)

    def value(self):
        return super(FormattedSpinBox, self).value() / self._Multiplier