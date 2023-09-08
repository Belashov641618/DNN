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
)

from src.examiner.qt.Additional.ContinuousSettingWidget import ContinuousSettingWidget
from src.examiner.qt.Additional.VariantsSettingWidget import VariantsSettingWidget
from src.examiner.qt.Additional.DataBaseVisualizer import DataBaseVisualizer
from src.examiner.qt.Additional.ModelVisualizer import ModelVisualizer
from src.examiner.qt.Additional.DataBaseWidget import DataBaseWidget

class MainWindow(QMainWindow):
    _MainWidget : QWidget


    _ModelTabsWidget : QTabWidget
    _ModelInfoTab : ModelVisualizer
    def _init_model_tabs(self, Splitter:QSplitter):
        self._ModelTabsWidget = QTabWidget(self._MainWidget)
        Splitter.addWidget(self._ModelTabsWidget)

        self._ModelTabsWidget.setCornerWidget(QComboBox())

        self._ModelInfoTab = ModelVisualizer(self._MainWidget)
        self._ModelTabsWidget.addTab(self._ModelInfoTab, 'Выбранная модель')


    _DataBaseVisualizer : DataBaseVisualizer
    def _init_graph(self, Splitter:QSplitter):
        self._DataBaseVisualizer = DataBaseVisualizer(self._MainWidget)
        Splitter.addWidget(self._DataBaseVisualizer)


    _DataBaseInfo : DataBaseWidget
    def _init_data_base(self, Splitter:QSplitter):
        self._DataBaseInfo = DataBaseWidget(self._MainWidget)
        Splitter.addWidget(self._DataBaseInfo)


    def _init_GUI(self):
        self._MainWidget = QWidget(self)
        self.setCentralWidget(self._MainWidget)

        MainLayout = QVBoxLayout(self._MainWidget)

        HorizontalSplitter = QSplitter(self._MainWidget)
        HorizontalSplitter.setOrientation(Qt.Orientation.Vertical)
        MainLayout.addWidget(HorizontalSplitter)

        TopSplitter = QSplitter(self._MainWidget)
        TopSplitter.setOrientation(Qt.Orientation.Horizontal)
        HorizontalSplitter.addWidget(TopSplitter)

        BottomSplitter = QSplitter(self._MainWidget)
        BottomSplitter.setOrientation(Qt.Orientation.Horizontal)
        HorizontalSplitter.addWidget(BottomSplitter)

        self._init_graph(TopSplitter)
        self._init_model_tabs(TopSplitter)
        self._init_data_base(BottomSplitter)


        TrainSpacer = QLabel(self._MainWidget)
        TrainSpacer.setText('Train Spacer')
        TrainSpacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        TrainSpacer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        TrainSpacer.setFrameStyle(QFrame.Shape.StyledPanel)
        BottomSplitter.addWidget(TrainSpacer)




        # central_widget = QWidget(self)
        # self.setCentralWidget(central_widget)

        # # Создаем разделитель
        # splitter = QSplitter(central_widget)
        # splitter.setHandleWidth(1)
        # #splitter.setStyleSheet("QSplitter::handle { background-color: black; }")
        #
        # # Создаем левую часть
        # left_widget = QWidget(splitter)
        #
        # # Создаем ComboBox1 и ComboBox2
        # combo_box1 = QComboBox(left_widget)
        # combo_box2 = QComboBox(left_widget)
        # combo_box_model = QStandardItemModel(combo_box1)
        # combo_box_model.appendRow(QStandardItem("Item 1"))
        # combo_box_model.appendRow(QStandardItem("Item 2"))
        # combo_box_model.appendRow(QStandardItem("Item 3"))
        # combo_box1.setModel(combo_box_model)
        # combo_box2.setModel(combo_box_model)
        #
        # # Создаем график
        # graph_label = QLabel(left_widget)
        # graph_label.setText("Graph")
        # graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # graph_label.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        #
        # # Создаем строку загрузки
        # progress_bar = QProgressBar(left_widget)
        #
        # # Создаем менеджер компоновки и добавляем виджеты на левую часть
        # left_layout = QVBoxLayout(left_widget)
        # left_layout.addWidget(combo_box1)
        # left_layout.addWidget(combo_box2)
        # left_layout.addWidget(graph_label)
        # left_layout.addWidget(progress_bar)
        #
        # # Создаем правую часть
        # right_widget = QWidget(splitter)
        #
        # # Создаем вкладки
        # tab_widget = QTabWidget(right_widget)
        # tab_widget.addTab(QWidget(), "Tab 1")
        # tab_widget.addTab(QWidget(), "Tab 2")
        #
        # # Создаем менеджер компоновки и добавляем виджеты на правую часть
        # right_layout = QVBoxLayout(right_widget)
        # right_layout.addWidget(tab_widget)
        #
        # # Создаем менеджер компоновки и добавляем левую и правую части на разделитель
        # splitter_layout = QHBoxLayout()
        # splitter_layout.addWidget(left_widget)
        # splitter_layout.addWidget(right_widget)
        # splitter.setLayout(splitter_layout)
        #
        # # Устанавливаем разделитель в главный виджет
        # central_layout = QHBoxLayout(central_widget)
        # central_layout.addWidget(splitter)


    def __init__(self):
        super(MainWindow, self).__init__()

        screen_size = self.screen().availableGeometry()

        width       = screen_size.width() * 0.75
        height      = screen_size.height() * 0.75
        width_pos   = screen_size.width() / 2 - width / 2
        height_pos  = screen_size.height() / 2 - height / 2

        self.setGeometry(width_pos, height_pos, width, height)

        self._init_GUI()

        # self._MainWidget = QWidget(self)
        # self.setCentralWidget(self._MainWidget)
        # MainLayout = QVBoxLayout(self._MainWidget)
        #
        # widget = ContinuousSettingWidget(self._MainWidget)
        # widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding))
        # MainLayout.addWidget(widget)
        # widget.valueChanged.connect(lambda: print('Текущее значение:' ,widget.value()))
        #
        # widget = VariantsSettingWidget(self._MainWidget)
        # widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding))
        # MainLayout.addWidget(widget)
        # widget.valueChanged.connect(lambda: print('Текущее значение:', widget.value()))





