import torch.nn
import torch
import json
import pickle
import os.path
import inspect
from src.utilities.Formaters import Format
from functools import partial

class get:
    @staticmethod
    def properties(obj):
        return [parameter for parameter in dir(obj) if hasattr(type(obj), parameter) and isinstance(getattr(type(obj), parameter), property)]
    @staticmethod
    def methods(obj):
        return [parameter for parameter in dir(obj) if parameter in ["__call__"] or (parameter not in ["get"] and inspect.ismethod(getattr(obj, parameter)) and not parameter.startswith('_'))]
    @staticmethod
    def attributes(function):
        result = {}
        for name, info in inspect.signature(function).parameters.items():
            result[name] = {"type":info.annotation, "default":info.default}
        return result
def parse(model:torch.nn.Module):
    result = {}
    for parameter in get.properties(model):
        variants = {}
        for variant in get.methods(getattr(model, parameter)):
            variants[variant] = get.attributes(getattr(getattr(model, parameter), variant))
        result[parameter] = variants
    return result

def GenerateModelParametersDict(Model:torch.nn.Module):
    '''
    В данной функции реализованно автоматическое создание словарей для моделей, в качестве аргкмента передавайте экземпляр модели.
    В качестве параметров в словарь будут добавляться только аттрибуты-сетеры
    '''


    # Создаём путь к папке со словарями:
    directory_path = __file__ + '/../'

    # Получаем название модели:
    RewriteWarning = True
    ModelName = type(Model).__name__
    if RewriteWarning and os.path.exists(directory_path + ModelName + '_ParametersDict.py'):
        answer = input('\033[93m{}\033[0m'.format('Словарь для этой модели уже существует, вы хотите переписать его? (Y/N): '))
        if answer != 'Y':
            return None

    # Получаем атрибуты с сетерами:
    ModelParametersNames = []
    for attribute_name in dir(Model):
        if hasattr(type(Model), attribute_name) and isinstance(getattr(type(Model), attribute_name), property):
            ModelParametersNames.append(attribute_name)

    # Создаём словарь и заполняем его в зависимости от установленных в сети на данный момент параметров:
    ModelParametersDict = {}
    for parameter_name in ModelParametersNames:
        parameter_value = getattr(Model, parameter_name)
        parameter_type = type(parameter_value)
        if parameter_type in [int]:
            ModelParametersDict[parameter_name] = {
                'DataType'  : 'int',
                'MinValue'  : 0,
                'MaxValue'  : parameter_value * 5,
                'ValueStep' : 1,
                'Format'    : pickle.dumps(partial(Format.Scientific, unit='', precision=1)).hex()
            }
        elif parameter_type in [float, complex]:
            ModelParametersDict[parameter_name] = {
                'DataType'  : 'int',
                'MinValue'  : 0,
                'MaxValue'  : parameter_value * 5,
                'ValueStep' : parameter_value/200,
                'Format'    : pickle.dumps(partial(Format.Scientific, unit='', precision=3)).hex()
            }
        elif parameter_type in [torch.Tensor] and parameter_value.numel() == 1:
            ModelParametersDict[parameter_name] = {
                'DataType': 'int',
                'MinValue': 0,
                'MaxValue': parameter_value.item() * 5,
                'ValueStep': parameter_value.item() / 200,
                'Format': pickle.dumps(partial(Format.Scientific, unit='', precision=3)).hex()
            }

    # Создаём файл .py и сохраняем в него словарь:
    max_symbols = 0
    for key in ModelParametersDict.keys():
        if len(key) > max_symbols:
            max_symbols = len(key)
    with open(directory_path + ModelName + '_ParametersDict.py', 'w+') as file:
        file.write('from src.utilities.Formaters import Format\n'
                   '\n'
                   'ParametersDict = ')
        file.write(json.dumps(ModelParametersDict, indent=4))

    return ModelParametersDict

def GenerateParametersDict(model:torch.nn.Module):
    data = parse(model)
    print(type(model).__name__ + " parameters:")
    for parameter, variants in data.items():
        print("\t" + parameter + " variants:")
        for variant, attributes in variants.items():
            print("\t\t" + variant + " attributes:")
            for name, info in attributes.items():
                print("\t\t\t" + name, info["type"], info["default"])


if __name__ == '__main__':
    from src.modules.models.FourierSpaceD2NN import FourierSpaceD2NN
    GenerateParametersDict(FourierSpaceD2NN())