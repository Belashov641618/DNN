from typing import Union, List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QMainWindow,
    QLabel,
    QLayout,
    QHBoxLayout,
    QVBoxLayout,
    QBoxLayout,
    QWidget,
    QComboBox,
    QSplitter,
    QProgressBar,
    QTabWidget,
    QSizePolicy,
    QFrame,
    QScrollArea,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from src.ModelsExaminer.QT.Additional.ContinuousSettingWidget import ContinuousSettingWidget
from src.ModelsExaminer.QT.Additional.VariantsSettingWidget import VariantsSettingWidget
from src.ModelsExaminer.QT.Additional.DataBaseVisualizer import DataBaseVisualizer
from src.ModelsExaminer.QT.Additional.ModelVisualizer import ModelVisualizer

class DataBaseWidget(QWidget):

    DataBaseTable : QTableWidget

    def _init_Table(self, parent:QWidget, layout:QBoxLayout):
        Label = QLabel(parent)
        Label.setText('База данных')
        Label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(Label)

        self.DataBaseTable = QTableWidget(parent)
        self.DataBaseTable.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self.DataBaseTable)

        self.DataBaseTable.setRowCount(30)
        self.DataBaseTable.setColumnCount(3)
        self.DataBaseTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.DataBaseTable.setHorizontalHeaderLabels(['Column 1', 'Column 2'])
        for row in range(3):
            for column in range(2):
                item = QTableWidgetItem(f'Row {row}, Column {column}')
                self.DataBaseTable.setItem(row, column, item)

    def __init__(self, parent:QWidget=None):
        super(DataBaseWidget, self).__init__(parent)

        Layout = QVBoxLayout(parent)
        self.setLayout(Layout)

        self._init_Table(parent, Layout)