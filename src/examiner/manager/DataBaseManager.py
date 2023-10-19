from typing import Union, List, Tuple, Dict, Any
import pickle
import json
import sqlite3
import os
import importlib.util
from copy import deepcopy

import torch.nn

from src.utilities.Formaters import Format

from src.examiner.manager.parameters.generator import GenerateParametersDict
from modules.Trainer import Trainer

from src.utilities.UniversalTestsAndOther import PlotTroughModelPropagationSamples
from src.utilities.SimplePlot import TiledPlot

_ABSOLUTE_PATH = __file__ + '/../'

_ADDITIONAL_COLUMNS = [
    ['Id', 'int NOT NULL PRIMARY KEY AUTOINCREMENT'],
    ['Variant', 'int'],
    ['FinalAccuracy', 'real']
]
_TABLE_NAME = 'MAIN_TABLE'

class DataBase:
    # Название типа хранимых моделей и сам тип
    _ModelType : Any
    _ModelName : str
    @property
    def ModelName(self):
        return self._ModelName
    @ModelName.setter
    def ModelName(self, model:Union[str, torch.nn.Module]):
        if isinstance(model, torch.nn.Module):
            self._ModelName = type(model).__name__
            self._ModelType = type(model)
        else:
            self._ModelName = model
            self._ModelType = None
        self._changed_ModelName()
    def _changed_ModelName(self):
        self._reset_ParametersDict()
        self._reset_DataBaseConnection()

    # Параметры которые ожидаются у модели и способ их хранения
    _ModelParameters : Dict
    def _reset_ParametersDict(self):
        file_name = _ABSOLUTE_PATH + 'parameters/' + self._ModelName + '_PD.py'
        if not os.path.exists(file_name):
            if self._ModelType is not None:
                answer = input('\033[93m{}\033[0m'.format('Словаря параметров для модели ' + self._ModelName + ' не существует, хотите автоматически создать его? (Y/N): '))
                if answer == 'Y':
                    try: GenerateParametersDict(self._ModelType())
                    except Exception as e: raise Exception("\033[31m\033[1m{}".format('Что-то пошло не так при автоматической генерации словаря параметров! (' + str(e) + ').'))
                else: Exception("\033[31m\033[1m{}".format('Словаря параметров для модели ' + self._ModelName + ' не существует, автоматическое создание было отменено!'))
                if not os.path.exists(file_name): raise Exception("\033[31m\033[1m{}".format('Автоматически созданный словарь находится в другой директории! Пытаемся найти словарь в (' + file_name + ').'))
            else: raise Exception("\033[31m\033[1m{}".format('Словаря параметров для модели ' + self._ModelName + ' не существует, название модели было дано ввиде строки, поэтому невозможно автоматически сгенерировать словарь! Вы можете в качестве модели предоставить экземпляр.'))

        spec = importlib.util.spec_from_file_location("module.name", file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._ModelParameters = deepcopy(module.ParametersDict)

    # Соединение базы данных и прочие методы
    _DataBaseConnection : sqlite3.Connection
    def _reset_DataBaseConnection(self):
        file_name = _ABSOLUTE_PATH + 'databases/' + self._ModelName + '_ModelsBase.db'
        if not os.path.exists(file_name):
            self._DataBaseConnection = sqlite3.connect(file_name)
            self._create_DataBase()
        else:
            self._DataBaseConnection = sqlite3.connect(file_name)
        parameters_list = [param.split(' ')[0] for param in self._generate_ParametersString().split(', ')]
        database_parameters = self._get_DataBaseColumns()
        if set(parameters_list) != set(database_parameters): raise Exception("\033[31m\033[1m{}".format('База данных моделей ' + self._ModelName + ' содержит другой отличный набор столбцов!\nПараметры словаря: ' + str(parameters_list) + '\nСтолбцы базы данных' + str(database_parameters)))
    @staticmethod
    def _get_parameters_dict_minimum(parameters_dict:Dict):
        parameters_minimum = {}
        for parameter, variants in parameters_dict.items():
            attributes_count = {}
            for variant, attributes in variants.items():
                _attributes_count = {}
                for attribute, info in attributes.items():
                    _attributes_count[info['type']] += 1
                for key, value in _attributes_count.items():
                    if key not in attributes_count.keys():
                        attributes_count[key] = 0
                    if attributes_count[key] < value:
                        attributes_count[key] = value
            parameters_minimum[parameter] = attributes_count
        return parameters_minimum
    @staticmethod
    def _get_parameter_dict_minimum_columns(parameters_dict:Dict):
        parameters_dict_minimum = DataBase._get_parameters_dict_minimum(parameters_dict)
        columns = []
        for parameter, attributes_count in parameters_dict_minimum.items():
            columns.append(parameter + '_variant text')
            for table_type, amount in attributes_count.items():
                for i in range(amount):
                    columns.append(parameter + '_' + table_type + '_' + str(i+1) + ' ' + table_type)
        return columns
    def _generate_ParametersString(self):
        columns = []
        for (name, table_type) in _ADDITIONAL_COLUMNS:
            columns.append(name + ' ' + table_type)

        model_columns = DataBase._get_parameter_dict_minimum_columns(self._ModelParameters)
        columns += model_columns

        return ', '.join(columns)
    def _get_DataBaseColumns(self):
        cursor = self._DataBaseConnection.cursor()
        cursor.execute('PRAGMA table_info(databases)')
        return [col[1] for col in cursor.fetchall()]
    def _create_DataBase(self):
        self._DataBaseConnection.execute('CREATE TABLE ' + _TABLE_NAME + ' (' + self._generate_ParametersString() + ')')
    def _clear_DataBase(self):
        self._DataBaseConnection.execute('DROP TABLE ' + _TABLE_NAME)
        self._create_DataBase()
    def _add_training(self):
        return



    # Внутренние методы
    def _pull_parameters_from_model(self, model:torch.nn.Module):
        return


    # Конструктор, диструктор и операторы
    def __init__(self, model:Union[str, torch.nn.Module]):
        self.ModelName = model
        self._initialize_selection()
        self._initialize_Trainer()
    def __del__(self):
        if hasattr(self, '_DataBaseConnection') and self._DataBaseConnection is not None:
            self._DataBaseConnection.commit()
            self._DataBaseConnection.close()


    # Селектор базы данных и действия с выделением
    _SelectedIds : List[int]
    @property
    def selection(self):
        class Selector:
            _self : DataBase
            def __init__(self, original):
                self._self = original
            def __call__(self, ModelParameters:Dict=None, Id:Union[int,List[int],Tuple[int]]=None, FinalAccuracy:Union[float,List[float],Tuple[float,float]]=None, **kwargs):
                if ModelParameters is None and Id is None and FinalAccuracy is None and kwargs == {}:
                    query = 'SELECT Id FROM databases'
                    cursor = self._self._DataBaseConnection.cursor()
                    cursor.execute(query)
                    self._self._SelectedIds = [IdT_[0] for IdT_ in cursor.fetchall()]
                    return self
                if ModelParameters is None:
                    ModelParameters = {}
                for key, value in kwargs.items():
                    ModelParameters[key] = value
                if 'Id' in ModelParameters.keys():
                    Id = ModelParameters.pop('Id')
                if FinalAccuracy is not None:
                    ModelParameters['FinalAccuracy'] = FinalAccuracy
                if isinstance(Id, int):
                    Id = [Id]

                queries = []
                if Id is not None:
                    queries.append('Id IN (' + ', '.join([str(Id_) for Id_ in Id]) + ')')
                for key, value in ModelParameters.items():
                    if isinstance(value, (List, Tuple)):
                        queries.append(key + ' BETWEEN ' + str(value[0]) + ' AND ' + str(value[1]))
                    else:
                        queries.append(key + ' = ' + str(value))
                query = 'SELECT Id FROM databases WHERE ' + ' AND '.join(queries)
                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute(query)
                self._self._SelectedIds = [IdT_[0] for IdT_ in cursor.fetchall()]
                return self
            def delete(self):
                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute('DELETE FROM databases WHERE Id IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')')
                self._self._SelectedIds = []
            def get(self, num:int=None):
                query = 'SELECT File FROM databases WHERE Id '
                if num is None:
                    query += 'IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')'
                else:
                    if num >= len(self._self._SelectedIds): raise ValueError("\033[31m\033[1m{}".format('Слишком большой num!'))
                    query += '= ' + str(self._self._SelectedIds[num])
                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute(query)
                files = cursor.fetchall()
                networks = []
                for file in files:
                    networks.append(pickle.loads(file[0]))
                if num is None:
                    return networks
                else:
                    return networks[0]
            def print(self, **kwargs):
                print(self.table.string(**kwargs))
            @property
            def table(self):
                class Table:
                    _self : DataBase
                    _rows : List
                    _cols : List[str]
                    def __init__(self, original):
                        self._self = original
                    def __call__(self, **kwargs):
                        default_list = {'Id': True, 'FinalAccuracy': True}
                        for parameter in self._self._ModelParameters.keys():
                            default_list[parameter] = True
                        for key, trigger in kwargs.items():
                            if not isinstance(trigger, bool): raise AttributeError("\033[31m\033[1m{}".format('Аттрибуты должны иметь булевые значения True или False'))
                            default_list[key] = trigger
                        final_list = []
                        for key, trigger in default_list.items():
                            if trigger: final_list.append(key)

                        query = 'SELECT ' + ', '.join(final_list) + ' FROM databases WHERE Id IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')'
                        cursor = self._self._DataBaseConnection.cursor()
                        cursor.execute(query)
                        self._rows = cursor.fetchall()
                        self._cols = final_list
                        return self
                    def _check_call(self, **kwargs):
                        if kwargs != {} or not hasattr(self, '_rows'): self.__call__(**kwargs)
                    def string(self, **kwargs):
                        self._check_call(**kwargs)
                        labels_string = '| ' + ' | '.join(self._cols) + ' |'
                        top_bottom_string = ''.ljust(len(labels_string), '-')
                        strings = [top_bottom_string, labels_string]
                        for row in self._rows:
                            formatted_values = []
                            for value, label in zip(row, self._cols):
                                value_string = ''
                                if label in self._self._ModelParameters.keys():
                                    value_string = self._self._ModelParameters[label]['Format'](value)
                                elif label == 'FinalAccuracy':
                                    value_string = str(round(value, 1)) + '%'
                                else:
                                    value_string = str(value)
                                value_string = value_string.center(len(label))
                                formatted_values.append(value_string)
                            strings.append('| ' + ' | '.join(formatted_values) + ' |')
                        strings.append(top_bottom_string)
                        return '\n'.join(strings)
                    def excel(self, column_separator='\t', row_separator='\n', **kwargs):
                        self._check_call(**kwargs)
                        strings = [column_separator.join(self._cols)]
                        for row in self._rows:
                            values = []
                            for value in row:
                                values.append(str(value))
                            strings.append(column_separator.join(values))
                        return row_separator.join(strings)
                return Table(self._self)
            def train(self, echo:bool=True):
                if echo: print('Pulling model files.')
                query = 'SELECT File FROM databases WHERE Id IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')'
                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute(query)
                files = cursor.fetchall()
                for i, file in enumerate(files):
                    if echo: print('Training model #' + str(i+1))
                    model = pickle.loads(file[0])
                    self._self._Trainer.model = model
                    if echo: print('Trainer properties: ' + self._self.trainer.info)

                    model.FinalizeChanges()
                    Plot = TiledPlot(25, 12)
                    Plot.title('Сравнение дифракционных масок до и после обучения')

                    heights1 = model.Heights()
                    for n, matrix in enumerate(heights1):
                        axes = Plot.axes.add(n, 0)
                        Plot.graph.title('Маска №' + str(n+1))
                        axes.imshow(matrix)

                    PlotTroughModelPropagationSamples(model, show=False)
                    self._self._Trainer.train(echo=echo)
                    PlotTroughModelPropagationSamples(model, show=False)

                    heights2 = model.Heights()
                    for n, matrix in enumerate(heights2):
                        axes = Plot.axes.add(n, 1)
                        Plot.graph.title('Маска №' + str(n + 1))
                        axes.imshow(matrix)

                    for n, (matrix1, matrix2) in enumerate(zip(heights1, heights2)):
                        difference = 100.0*(matrix2 - matrix1)/matrix2
                        axes = Plot.axes.add(n, 2)
                        Plot.graph.title('Маска №' + str(n + 1))
                        axes.imshow(difference)
                        Plot.graph.description('Максимальное изменение: ' + Format.Scientific(torch.max(difference).item(), '%'))

                    Plot.description.row.left('До обучения',    0)
                    Plot.description.row.left('После обучения', 1)
                    Plot.description.row.left('Разница',        2)
                    Plot.show(block=True)

                if echo: print('Done!')

        return Selector(self)
    def _initialize_selection(self):
        self._SelectedIds = []


    # Тренировочный модуль
    _Trainer : Trainer
    _TrainerParameters : Dict
    def _reset_TrainerParametersDict(self):
        file_name = _ABSOLUTE_PATH + 'parameters/' + 'Trainer' + '_PD.py'
        if not os.path.exists(file_name):
            answer = input('\033[93m{}\033[0m'.format('Словаря параметров для тренировщика не существует, хотите автоматически создать его? (Y/N): '))
            if answer == 'Y':
                try: GenerateParametersDict(self._Trainer)
                except Exception as e: raise Exception("\033[31m\033[1m{}".format('Что-то пошло не так при автоматической генерации словаря параметров! (' + str(e) + ').'))
            else: Exception("\033[31m\033[1m{}".format('Словаря параметров для тренировщика не существует, автоматическое создание было отменено!'))
            if not os.path.exists(file_name): raise Exception("\033[31m\033[1m{}".format('Автоматически созданный словарь находится в другой директории! Пытаемся найти словарь в (' + file_name + ').'))

        spec = importlib.util.spec_from_file_location("module.name", file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._TrainerParameters = deepcopy(module.ParametersDict)
    @property
    def trainer(self):
        return self._Trainer
    def _initialize_Trainer(self):
        self._Trainer = Trainer()
        self._reset_TrainerParametersDict()

    # Методы пользователя
    def check_existence(self, parameters_dict_or_model:Union[Dict,torch.nn.Module]):
        parameters = parameters_dict_or_model
        if isinstance(parameters_dict_or_model, torch.nn.Module):
            model = parameters_dict_or_model
            parameters = self._pull_parameters_from_model(model)

        PreviousSelectedIds = self._SelectedIds
        self.selection(ModelParameters=parameters)
        SelectedIds = self._SelectedIds
        self._SelectedIds = PreviousSelectedIds

        if not SelectedIds:
            return False
        return True
    def add(self, model:torch.nn.Module):
        parameters = self._pull_parameters_from_model(model)

        if self.check_existence(parameters):
            answer = input('\033[93m{}\033[0m'.format('Модель с теми же самыми параметрами уже существует, хотите заменить её? (Y/N): '))
            if answer != 'Y':
                return
            PreviousSelectedIds = self._SelectedIds
            self.selection(parameters)
            self.selection.delete()
            self._SelectedIds = PreviousSelectedIds

        final_accuracy:float = 0.0
        train_info = {}
        file = pickle.dumps(model)
        parameters['FinalAccuracy'] = final_accuracy
        parameters['TrainInfo'] = json.dumps(train_info)
        parameters['File'] = file

        cursor = self._DataBaseConnection.cursor()
        query = 'INSERT INTO databases (' + ', '.join(parameters.keys()) + ') Values (' + ', '.join(['?']*len(parameters.keys())) + ')'
        cursor.execute(query, list(parameters.values()))
        # self._DataBaseConnection.commit()



class Test:
    @staticmethod
    def AddingModels():
        from modules.models.old.FourierSpaceD2NN import FourierSpaceD2NN
        Base = DataBase(FourierSpaceD2NN())

        import random
        n = 20
        limits = {
            'masks_count': (2, 8, int),
            'wave_length': (100.0E-9, 1000.0E-9, float),
            'space_reflection': (1.0, 1.5, float),
            'mask_reflection': (1.0, 1.5, float),
            'plane_length': (1.0E-3, 10.0E-3, float),
            'pixels_count': (10, 100, int),
            'up_scaling': (1, 50, int),
            'mask_interspacing': (5.0E-3, 100.0E-3, float),
            'mask_propagation_border_length': (1.0E-3, 10.0E-3, float),
            'lens_propagation_border_length': (0.0, 10.0E-3, float),
            'focus_length': (0.0, 10.0E-3, float)
        }
        for i in range(n):
            kwargs = {}
            for key, (min_value, max_value, value_type) in limits.items():
                kwargs[key] = value_type(random.random() * (max_value - min_value) + min_value)
            Base.add(FourierSpaceD2NN(**kwargs))
        Base.selection().print()

        Base._clear_DataBase()
    @staticmethod
    def TrainingSingleModel():
        from modules.models.old.FourierSpaceD2NN import FourierSpaceD2NN
        Base = DataBase(FourierSpaceD2NN())

        WaveLength = 532.0E-9
        PixelsCount = 20
        UpScaling = 8
        PlaneLength = PixelsCount * 30.0E-6
        FocusLength = 100.0E-3
        BorderLength = PlaneLength
        DiffractionLength = 10.0E-3
        Model = FourierSpaceD2NN(
            masks_count=4,
            wave_length=WaveLength,
            plane_length=PlaneLength,
            pixels_count=PixelsCount,
            up_scaling=UpScaling,
            mask_interspacing=DiffractionLength,
            focus_length=FocusLength,
            mask_propagation_border_length=1 * BorderLength,
            lens_propagation_border_length=3 * BorderLength,
            detectors_masks='Polar'
        )
        Model.DiffractionLengthAsParameter(False)
        Model.DisableAmplification()
        Base.add(Model)

        import random
        n = 7
        limits = {
            'wave_length': (100.0E-9, 1000.0E-9, float),
            'mask_reflection': (1.2, 1.5, float),
            'plane_length': (0.7E-3, 0.2E-3, float),
            'mask_interspacing': (5.0E-3, 10.0E-3, float),
            'focus_length': (1.0E-3, 10.0E-3, float)
        }
        for i in range(n):
            kwargs = {}
            for key, (min_value, max_value, value_type) in limits.items():
                kwargs[key] = value_type(random.random() * (max_value - min_value) + min_value)
            Base.add(FourierSpaceD2NN(**kwargs))
        Base.selection().print()

        Base.trainer.device = 'cuda'
        Base.trainer.dataset = 'MNIST'
        Base.trainer.epochs = 1
        Base.trainer.batches = 64

        Base.selection(Id=1)
        Base.selection.print()
        Base.selection.train()

        Base._clear_DataBase()

if __name__ == '__main__':
    # Test.AddingModels()
    Test.TrainingSingleModel()