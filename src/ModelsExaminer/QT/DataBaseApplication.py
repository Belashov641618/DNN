import sys
from typing import Union
import torch

from src.ModelsExaminer.ModelsData.DataBaseManager import DataBase

from PyQt6.QtWidgets import QApplication
from src.ModelsExaminer.QT.Additional.MainWindow import MainWindow

class DataBaseApplication(QApplication):

    _DataBaseManager : DataBase

    def __init__(self, argv, model:Union[str,torch.nn.Module,type(torch.nn.Module)]):
        super(DataBaseApplication, self).__init__(argv)

        if isinstance(model, type(torch.nn.Module)):
            model = model()
        self._DataBaseManager = DataBase(model)

        self.main_window = MainWindow()
        self.main_window.show()


if __name__ == '__main__':
    from src.Belashov.Models.FourierSpaceD2NN import FourierSpaceD2NN
    app = DataBaseApplication(sys.argv, FourierSpaceD2NN)
    app.setStyle('fusion')
    sys.exit(app.exec())