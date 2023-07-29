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
from src.ModelsExaminer.QT.Additional.ContinuousSettingWidget import ContinuousSettingWidget
from src.ModelsExaminer.QT.Additional.VariantsSettingWidget import VariantsSettingWidget

class ModelVisualizer(QWidget):

    ParameterSetting1 : ContinuousSettingWidget
    ParameterSetting2 : ContinuousSettingWidget
    GraphChoice : VariantsSettingWidget
    Graph: QLabel
    ProgressBar: QProgressBar
    def _init_Left(self, layout:QBoxLayout, parent:QWidget):
        LeftLayout = QVBoxLayout(parent)
        layout.addLayout(LeftLayout)

        ParametersSettingLayout = QHBoxLayout(parent)
        LeftLayout.addLayout(ParametersSettingLayout)

        self.ParameterSetting1 = ContinuousSettingWidget(parent)
        self.ParameterSetting2 = ContinuousSettingWidget(parent)
        ParametersSettingLayout.addWidget(self.ParameterSetting1)
        ParametersSettingLayout.addWidget(self.ParameterSetting2)

        self.GraphChoice = VariantsSettingWidget(parent)
        LeftLayout.addWidget(self.GraphChoice)

        self.Graph = QLabel(parent)
        self.Graph.setText('Graph Spacer')
        self.Graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.Graph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Graph.setFrameStyle(QFrame.Shape.StyledPanel)
        LeftLayout.addWidget(self.Graph)


    ParametersLayout : QBoxLayout
    def _init_Right(self, layout:QBoxLayout, parent:QWidget):
        RightLayout = QVBoxLayout(parent)
        layout.addLayout(RightLayout)

        Label = QLabel(parent)
        Label.setText('Параметры модели')
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

        self.ParametersList = [QLabel(parent) for i in range(100)]
        for i, widget in enumerate(self.ParametersList):
            widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
            widget.setFrameStyle(QFrame.Shape.StyledPanel)
            widget.setText('Параметр №' + str(i+1))
            self.ParametersLayout.addWidget(widget)



    def __init__(self, parent:QWidget=None):
        super(ModelVisualizer, self).__init__(parent)

        Layout = QHBoxLayout(parent)
        self.setLayout(Layout)

        self._init_Left(Layout, parent)
        self._init_Right(Layout, parent)