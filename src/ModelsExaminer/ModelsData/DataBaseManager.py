from typing import Union, List, Tuple, Dict, Any
import pickle
import json
import sqlite3
import os
import importlib.util
from copy import deepcopy

import torch.nn

from src.ModelsExaminer.ModelsData.ModelParametersDicts.Utilities import GenerateModelParametersDict

_ABSOLUTE_PATH = __file__ + '/../'

class DataBase:
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

    _ParametersDict : Dict
    def _reset_ParametersDict(self):
        file_name = _ABSOLUTE_PATH + 'ModelParametersDicts/' + self._ModelName + '_ParametersDict.py'
        if not os.path.exists(file_name):
            if self._ModelType is not None:
                answer = input('\033[93m{}\033[0m'.format('Словаря параметров для модели ' + self._ModelName + ' не существует, хотите автоматически создать его? (Y/N): '))
                if answer == 'Y':
                    try: GenerateModelParametersDict(self._ModelType())
                    except Exception as e: raise Exception("\033[31m\033[1m{}".format('Что-то пошло не так при автоматической генерации словаря параметров! (' + str(e) + ').'))
                else: Exception("\033[31m\033[1m{}".format('Словаря параметров для модели ' + self._ModelName + ' не существует, автоматическое создание было отменено!'))
                if not os.path.exists(file_name): raise Exception("\033[31m\033[1m{}".format('Автоматически созданный словарь находится в другой директории! Пытаемся найти словарь в (' + file_name + ').'))
            else: raise Exception("\033[31m\033[1m{}".format('Словаря параметров для модели ' + self._ModelName + ' не существует, название модели было дано ввиде строки, поэтому невозможно автоматически сгенерировать словарь! Вы можете в качестве модели предоставить экземпляр.'))

        spec = importlib.util.spec_from_file_location("module.name", file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._ParametersDict = deepcopy(module.ParametersDict)

        for key in self._ParametersDict.keys():
            if isinstance(self._ParametersDict[key]['Format'], str):
                self._ParametersDict[key]['Format'] = pickle.loads(bytes.fromhex(self._ParametersDict[key]['Format']))

    _DataBaseConnection : sqlite3.Connection
    def _reset_DataBaseConnection(self):
        file_name = _ABSOLUTE_PATH + 'TrainedModels/' + self._ModelName + '_ModelsBase.db'
        if not os.path.exists(file_name):
            self._DataBaseConnection = sqlite3.connect(file_name)
            self._create_DataBase()
        else:
            self._DataBaseConnection = sqlite3.connect(file_name)
        parameters_list = [param.split(' ')[0] for param in self._generate_ParametersString().split(', ')]
        database_parameters = self._get_DataBaseColumns()
        if set(parameters_list) != set(database_parameters): raise Exception("\033[31m\033[1m{}".format('База данных моделей ' + self._ModelName + ' содержит другой отличный набор столбцов!\nПараметры словаря: ' + str(parameters_list) + '\nСтолбцы базы данных' + str(database_parameters)))
    def _generate_ParametersString(self):
        parameters_string = 'Id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
        for (parameter_name, data) in self._ParametersDict.items():
            data_type = data['DataType']
            parameters_string += parameter_name + ' ' + data_type + ', '
        parameters_string += 'FinalAccuracy real, TrainInfo text, File blob'
        return parameters_string
    def _get_DataBaseColumns(self):
        cursor = self._DataBaseConnection.cursor()
        cursor.execute('PRAGMA table_info(TrainedModels)')
        return [col[1] for col in cursor.fetchall()]
    def _create_DataBase(self):
        self._DataBaseConnection.execute('CREATE TABLE TrainedModels (' + self._generate_ParametersString() + ')')
    def _clear_DataBase(self):
        self._DataBaseConnection.execute('DROP TABLE TrainedModels')
        self._create_DataBase()

    def __init__(self, model:Union[str, torch.nn.Module]):
        self.ModelName = model
    def __del__(self):
        if hasattr(self, '_DataBaseConnection') and self._DataBaseConnection is not None:
            self._DataBaseConnection.commit()
            self._DataBaseConnection.close()

    def CheckExistence(self, parameters_dict:Dict):
        parameters_value_list = []
        parameters_names_list = []
        for database_column in self._get_DataBaseColumns():
            if database_column in self._ParametersDict:
                parameters_value_list.append(parameters_dict[database_column])
                parameters_names_list.append(database_column)
        cursor = self._DataBaseConnection.cursor()
        cursor.execute('SELECT * FROM TrainedModels WHERE ' + ' AND '.join([key + '=?' for key in parameters_names_list]), parameters_value_list)
        return cursor.fetchone()
    def AddModel(self, model:torch.nn.Module, final_accuracy:float=0.0, train_info:Dict=None):
        if train_info is None: train_info = json.dumps({})

        parameters_values_dict = {}
        for parameter_name in self._ParametersDict.keys():
            if not hasattr(model, parameter_name): raise AttributeError("\033[31m\033[1m{}".format('У предоставленной модели отсутсвует аттрибут ' + parameter_name + '!'))
            parameter_value = getattr(model, parameter_name)
            if torch.is_tensor(parameter_value):
                if parameter_value.numel() == 1:
                    parameter_value = parameter_value.item()
                else: raise ValueError("\033[31m\033[1m{}".format('Параметр ' + parameter_name + ' явняется не еденичным тензором!' + str(parameter_value)))
            parameters_values_dict[parameter_name] = parameter_value
        parameters_values_dict['FinalAccuracy'] = final_accuracy
        parameters_values_dict['TrainInfo'] = train_info
        parameters_values_dict['File'] = pickle.dumps(model)

        if self.CheckExistence(parameters_values_dict):
            answer = input('\033[93m{}\033[0m'.format('Модель с теми же самыми параметрами уже существует, хотите заменить её? (Y/N): '))
            if answer != 'Y':
                return
            self.DeleteModel(parameters_values_dict)

        parameters_value_list = [parameters_values_dict[key] for key in self._get_DataBaseColumns() if key not in ['Id']]
        cursor = self._DataBaseConnection.cursor()
        cursor.execute('INSERT INTO TrainedModels (' + ', '.join(parameters_values_dict.keys()) + ') Values (' + ', '.join(['?']*len(parameters_value_list)) + ')', parameters_value_list)
        self._DataBaseConnection.commit()

    _SelectedIds : List[int]
    @property
    def selection(self):
        class Selector:
            _self : DataBase
            def __init__(self, original):
                self._self = original
            def __call__(self, ModelParameters:Dict=None, Id:Union[int,List[int],Tuple[int]]=None, FinalAccuracy:Union[float,List[float],Tuple[float,float]]=None, **kwargs):
                if ModelParameters is None and Id is None and FinalAccuracy is None and kwargs == {}:
                    query = 'SELECT Id FROM TrainedModels'
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
                query = 'SELECT Id FROM TrainedModels WHERE ' + ' AND '.join(queries)

                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute(query)
                self._self._SelectedIds = [IdT_[0] for IdT_ in cursor.fetchall()]
                return self
            def delete(self):
                cursor = self._self._DataBaseConnection.cursor()
                cursor.execute('DELETE FROM TrainedModels WHERE Id IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')')
                self._self._SelectedIds = []
            def get(self, num:int=None):
                query = 'SELECT File FROM TrainedModels WHERE Id '
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
                        for parameter in self._self._ParametersDict.keys():
                            default_list[parameter] = True
                        for key, trigger in kwargs.items():
                            if not isinstance(trigger, bool): raise AttributeError("\033[31m\033[1m{}".format('Аттрибуты должны иметь булевые значения True или False'))
                            default_list[key] = trigger
                        final_list = []
                        for key, trigger in default_list.items():
                            if trigger: final_list.append(key)

                        query = 'SELECT ' + ', '.join(final_list) + ' FROM TrainedModels WHERE Id IN (' + ', '.join([str(Id) for Id in self._self._SelectedIds]) + ')'
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
                                if label in self._self._ParametersDict.keys():
                                    value_string = self._self._ParametersDict[label]['Format'](value)
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

        return Selector(self)

def Test():
    from src.Belashov.Models.FourierSpaceD2NN import FourierSpaceD2NN
    Base = DataBase(FourierSpaceD2NN())

    import random
    n = 20
    limits = {
        'masks_count'                       : (2, 8, int),
        'wave_length'                       : (100.0E-9, 1000.0E-9, float),
        'space_reflection'                  : (1.0, 1.5, float),
        'mask_reflection'                   : (1.0, 1.5, float),
        'plane_length'                      : (1.0E-3, 10.0E-3, float),
        'pixels_count'                      : (10, 100, int),
        'up_scaling'                        : (1, 50, int),
        'mask_interspacing'                 : (5.0E-3, 100.0E-3, float),
        'mask_propagation_border_length'    : (1.0E-3, 10.0E-3, float),
        'lens_propagation_border_length'    : (0.0, 10.0E-3, float),
        'focus_length'                      : (0.0, 10.0E-3, float)
    }
    for i in range(n):
        kwargs = {}
        for key, (min_value, max_value, value_type) in limits.items():
            kwargs[key] = value_type(random.random()*(max_value - min_value) + min_value)
        Base.AddModel(FourierSpaceD2NN(**kwargs))
    Base.selection().print()

    Base._clear_DataBase()

if __name__ == '__main__':
    Test()
