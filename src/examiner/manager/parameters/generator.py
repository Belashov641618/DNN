import torch.nn
import torch
import json
import pickle
import os.path
import inspect
from src.utilities.Formaters import Format
from functools import partial
from typing import Union

from modules.Trainer import Trainer

class get:
    @staticmethod
    def default_value(variable_type):
        if   variable_type == int:
            return 1
        elif variable_type == float:
            return 1.0
        else:
            print("Нету подходящей реализации для класса", variable_type)
    @staticmethod
    def type_name(variable_type):
        if   variable_type == int:
            return 'int'
        elif variable_type == float:
            return 'real'

    @staticmethod
    def properties(obj):
        return [parameter for parameter in dir(obj) if parameter not in ['delayed'] and hasattr(type(obj), parameter) and isinstance(getattr(type(obj), parameter), property)]
    @staticmethod
    def methods(obj):
        return [parameter for parameter in dir(obj) if parameter in ["__call__"] or (parameter not in ["get"] and inspect.ismethod(getattr(obj, parameter)) and not parameter.startswith('_'))]
    @staticmethod
    def attributes(function):
        result = {}
        for name, info in inspect.signature(function).parameters.items():
            typename = get.type_name(info.annotation)
            default_value = (info.default if not info.default is info.empty else get.default_value(info.annotation))
            max_value = info.annotation(default_value * 2)
            min_value = info.annotation(default_value / 2)
            result[name] = {
                "type":typename,
                "default":default_value,
                "max":max_value,
                "min":min_value,
                "format":None
            }
        return result

def parse(model:torch.nn.Module):
    result = {}
    for parameter in get.properties(model):
        variants = {}
        for variant in get.methods(getattr(model, parameter)):
            variants[variant] = get.attributes(getattr(getattr(model, parameter), variant))
        result[parameter] = variants
    return result

def GenerateParametersDict(model:Union[torch.nn.Module, Trainer]):
    data = parse(model)

    print(type(model).__name__ + " parameters:")
    for parameter, variants in data.items():
        print("\t" + parameter + " variants:")
        for variant, attributes in variants.items():
            print("\t\t" + variant + " attributes:")
            for name, info in attributes.items():
                print("\t\t\t" + name + " params:")
                for param, value in info.items():
                    print("\t\t\t\t" + param + ":", value)

    # Создаём путь к папке со словарями:
    directory_path = __file__ + '/../'

    # Получаем название модели:
    rewrite_warning = True
    model_name = type(model).__name__
    if rewrite_warning and os.path.exists(directory_path + model_name + '_PD.py'):
        answer = input('\033[93m{}\033[0m'.format('Словарь для этой модели уже существует, вы хотите переписать его? (Y/N): '))
        if answer != 'Y':
            return None

    # Создаём файл .py и сохраняем в него словарь:
    max_symbols = 0
    for key in data.keys():
        if len(key) > max_symbols:
            max_symbols = len(key)
    with open(directory_path + model_name + '_PD.py', 'w+') as file:
        file.write('from src.utilities.Formaters import Format\n'
                   '\n'
                   'ParametersDict = ')
        file.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    from src.modules.models.FourierSpaceD2NN import FourierSpaceD2NN
    GenerateParametersDict(FourierSpaceD2NN())