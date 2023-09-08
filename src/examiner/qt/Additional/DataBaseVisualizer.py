from typing import Union, List
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QBoxLayout,
    QWidget,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QProgressBar,
)
from src.examiner.qt.Additional.ContinuousSettingWidget import ContinuousSettingWidget
from src.examiner.qt.Additional.VariantsSettingWidget import VariantsSettingWidget

class DataBaseVisualizer(QWidget):
    AxesSettingX : VariantsSettingWidget
    AxesSettingY : VariantsSettingWidget
    Graph : QLabel
    ProgressBar: QProgressBar
    def _init_Left(self, layout:QBoxLayout, parent:QWidget):
        LeftLayout = QVBoxLayout(parent)
        layout.addLayout(LeftLayout)

        AxesSettingLayout = QHBoxLayout(parent)
        LeftLayout.addLayout(AxesSettingLayout)

        self.AxesSettingX = VariantsSettingWidget(parent)
        self.AxesSettingY = VariantsSettingWidget(parent)
        AxesSettingLayout.addWidget(self.AxesSettingX)
        AxesSettingLayout.addWidget(self.AxesSettingY)

        self.Graph = QLabel(parent)
        self.Graph.setText('Graph Spacer')
        self.Graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.Graph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Graph.setFrameStyle(QFrame.Shape.StyledPanel)
        LeftLayout.addWidget(self.Graph)

    ParametersLayout : QVBoxLayout
    ParametersList : List[Union[ContinuousSettingWidget, VariantsSettingWidget]]
    def _init_Right(self, layout:QBoxLayout, parent:QWidget):
        RightLayout = QVBoxLayout(parent)
        layout.addLayout(RightLayout)

        Label = QLabel(parent)
        Label.setText('Настройки выборки')
        Label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        RightLayout.addWidget(Label)

        ScrollWidget = QScrollArea(parent)
        ScrollWidget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        ScrollWidget.setWidgetResizable(True)
        ScrollWidget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)
        RightLayout.addWidget(ScrollWidget)

        ContentWidget = QWidget(parent)
        ScrollWidget.setWidget(ContentWidget)

        self.ParametersLayout = QVBoxLayout(parent)
        ContentWidget.setLayout(self.ParametersLayout)

        self.ParametersList = [
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            VariantsSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
            ContinuousSettingWidget(parent),
        ]
        for widget in self.ParametersList:
            self.ParametersLayout.addWidget(widget)


    def __init__(self, parent:QWidget=None):
        super(DataBaseVisualizer, self).__init__(parent)

        Layout = QHBoxLayout(parent)
        self.setLayout(Layout)

        self._init_Left(Layout, parent)
        self._init_Right(Layout, parent)


