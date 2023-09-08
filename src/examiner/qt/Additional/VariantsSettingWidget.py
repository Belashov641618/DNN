from typing import List, Tuple, Union
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QComboBox,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
)

class VariantsSettingWidget(QWidget):

    Label : QLabel
    ComboBox : QComboBox

    valueChanged = pyqtSignal()
    def _connect(self):
        # noinspection PyUnresolvedReferences
        self.ComboBox.currentTextChanged.connect(self.valueChanged.emit)

    def __init__(self, parent:QWidget=None, label:str=None, variants:Union[List[str], Tuple[str]]=None):
        super(VariantsSettingWidget, self).__init__(parent)

        self.Label      = QLabel(parent)
        self.ComboBox   = QComboBox(parent)

        Layout = QHBoxLayout(parent)

        Layout.addWidget(self.Label)
        Layout.addWidget(self.ComboBox)

        self.setLayout(Layout)

        self.ComboBox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.Label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.Label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.SetLabel(label)
        self.SetVariants(variants)

        self._connect()

    def SetLabel(self, label:str):
        if label is None: label = 'Переменная'
        self.Label.setText(label)
    def SetVariants(self, variants:Union[List[str], Tuple[str]]):
        if variants is None: variants = ['Вариант №1', 'Вариант №2', 'Вариант №3']
        self.ComboBox.addItems(variants)

    def value(self):
        return self.ComboBox.currentText()