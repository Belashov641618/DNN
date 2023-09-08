from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
)

from src.examiner.qt.Additional.DoubleSlider import DoubleSlider
from src.examiner.qt.Additional.FormattedSpinBox import FormattedSpinBox


class ContinuousSettingWidget(QWidget):
    Label   : QLabel
    SpinBox : FormattedSpinBox
    Slider  : DoubleSlider

    valueChanged = pyqtSignal()
    def _slider_to_spinbox(self):
        self.SpinBox.blockSignals(True)
        self.SpinBox.setValue(self.Slider.value())
        self.SpinBox.blockSignals(False)
    def _spinbox_to_slider(self):
        self.Slider.blockSignals(True)
        self.Slider.setValue(self.SpinBox.value())
        self.Slider.blockSignals(False)
    def _connect(self):
        # noinspection PyUnresolvedReferences
        self.Slider.valueChanged.connect(self._slider_to_spinbox)
        # noinspection PyUnresolvedReferences
        self.SpinBox.valueChanged.connect(self._spinbox_to_slider)
        # noinspection PyUnresolvedReferences
        self.Slider.valueChanged.connect(self.valueChanged.emit)
        # noinspection PyUnresolvedReferences
        self.SpinBox.valueChanged.connect(self.valueChanged.emit)


    def __init__(self, parent:QWidget=None, label:str=None, unit:str=None, multiplier:float=None, value:float=None, min_value:float=None, max_value:float=None, step:float=None):
        super(ContinuousSettingWidget, self).__init__(parent)

        self.Label      = QLabel(parent)
        self.SpinBox    = FormattedSpinBox(parent)
        self.Slider     = DoubleSlider(parent)

        Layout = QHBoxLayout(parent)

        Layout.addWidget(self.Label)
        Layout.addWidget(self.SpinBox)
        Layout.addWidget(self.Slider)

        self.setLayout(Layout)

        self.Slider.setOrientation(Qt.Orientation.Horizontal)
        self.Label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.Label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.SetLabel(label)
        self.SetUnit(unit)
        self.SetMultiplier(multiplier)
        self.SetStep(step)
        self.SetValueMinimum(min_value)
        self.SetValueMaximum(max_value)
        self.SetValue(value)


        self._connect()


    def value(self):
        return self.SpinBox.value()

    def SetLabel(self, label:str):
        if label is None: label = 'Переменная'
        self.Label.setText(label)
    def SetUnit(self, unit:str):
        if unit is None: unit = 'УЕ'
        self.SpinBox.SetUnit(unit)
    def SetMultiplier(self, multiplier:float):
        if multiplier is None: multiplier = 1.0
        self.SpinBox.SetMultiplier(multiplier)
    def SetValue(self, value:float):
        if value is None: value = 0.5
        self.Slider.setValue(value)
        self.SpinBox.setValue(value)
    def SetValueMinimum(self, min_value:float):
        if min_value is None: min_value = 0.0
        self.Slider.setMinimum(min_value)
        self.SpinBox.setMinimum(min_value)
    def SetValueMaximum(self, max_value:float):
        if max_value is None: max_value = 1.0
        self.Slider.setMaximum(max_value)
        self.SpinBox.setMaximum(max_value)
    def SetStep(self, step:float):
        if step is None: step = 0.01
        self.Slider.setSingleStep(step)
        self.SpinBox.setSingleStep(step)
