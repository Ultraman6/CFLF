# 将JSON字符串转换为Python字典
import json
from argparse import ArgumentParser

import yaml
from nicegui import ui

# 构造前端组件的映射
# ui_mapping = {
#     'int': ui.number,
#     'float': ui.number,
#     'str': ui.select,
#     'bool': ui.switch,
#     'dict': ui.row,
#     'list': ui.row,
#     'root': ui.upload,  # 目前先用按钮代替，点击按钮后进入文件夹选择界面
#     'choice': ui.select,
#     'mod': ui.tab_panel,
#     'area': ui.row,
# }
dataset_model_mapping = {
    'mnist': ['cnn', 'logistic'],
    'cifar10': ['cnn', 'logistic'],
    'femnist': ['cnn', 'logistic'],
    'shakespeare': ['lstm'],
    'fmnist': ['lstm'],
}


def unit_mapping(type, name=None, default=None, help=None, option=None, on_click=None):  # 此方法用于根据参数类型返回对应的组件
    if type == 'mod':
        return ui.tab_panel(name)
    elif type == 'area':
        return ui.row()
    elif type == 'number':
        return ui.number(label=name, value=default, placeholder=help)
    elif type == 'choice':
        return ui.select(label=name, value=default, options=option)
    elif type == 'bool':
        return ui.switch(text=name, value=default)
    elif type == 'dict':  # 如果是dict，默认为一个容器，且是 字符:数组 的键值对(后续需要完善格式与数量的限制)
        for key, value in value.items():
            ui.number(label=key, value=value)
    elif type == 'root':  # 如果是root，需要调用webview选择文件夹
        return ui.button(text=name, on_click=on_click)


# 参数模板设置规则：
# 1. 最外层最为参数的大类，如federated_learning_settings，下面于该大类的作为参数
# 2. 每个参数的值可能还需要参数，位于下面的param(无论大小参数，都有基本的格式，因为都要转为args类型)
# 3. 某些参数之间存在映射，如dataset和model，写程序时需要注意
# 4. option字段可能是列表或者字典（有说明）
# 5. type字段为dict的参数，在转为args时，需要将其转为json字符串
# 6. .json格式保留参数最原始的设计(模板、前端、保存) .yaml格式便于开发者快速设置 .py格式(args)便于后端读取
# 作为衔接前后端与实际配置文件的任务参数类（Json作为1.参数原始模板(全部) 2.参数保存格式(部分)）
class TaskArgs(ArgumentParser):  # 负责读取参数配置模板、接受前端的参数输入构造args、保存参数配置
    def __init__(self, path, description=None):  # 初始化方法仅读取参数模板，并不涉及args的构造
        super().__init__(description=description)
        self.config = self.load_json_config(path)  # 参数配置模板
        self.tab_dict = {}
        self.unit_dict = {}  # 用于”接住“所有的传值组件对象（键为参数名称: 值为组件对象）
        self.complete_btn = ui.button('完成', on_click=self.complete)  # 完成按钮

    @staticmethod
    def load_json_config(path):  # 静态方法，用于加载JSON配置文件
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def load_yaml_config(self, yaml_str):  # 加载YAML字符串并将其转换为Python字典
        config = yaml.safe_load(yaml_str)
        for key, value in self.config.items():  # 遍历模块
            for key_i, value_i in value['param'].items():  # 遍历模块下的参数
                value_i['default'] = config[key][key_i]
                if value_i['param']:  # 如果存在嵌套子参数，需要继续遍历(鉴于json的结果，不会出现遍历到键标)
                    for key_ii, value_ii in value_i['param'].items():
                        value_ii['default'] = config[key][key_ii]  # yaml不允许重名嵌套

    # 前端方法，根据json初始化组件
    def init_unit_from_json(self):  # 此方法构造参数配置交互界面
        for key, value in self.config.items():
            self.unit_dict[key] = self.unit_searching(value)

    # 1. 无子参数：键:{’unit‘:组件} 2. 有子参数 键:{'con':容器,'unit':组件, 'child':{...}} 需要明确显示的参数(默认常显，需要传入bool参数或者表达式)
    # 待完成对象属性与组件的数据参数绑定
    def unit_searching(self, value, bind_obj=None,
                       bind_value=None):  # 此方法用于根据参数类型返回对应的组件（专门处理组件初始化与参数嵌套导致的块整理）嵌套参数取决于参数的值
        # 先获得顶层参数对应的组件
        unit_dict = {}
        if value['type']:  # 不管有没有子参数，都需要创建一个组件
            unit_dict['unit'] = unit_mapping(value['type'])
        if value['param']:  # 如果该组件有子参数，需要创建一个容器
            unit_con = unit_mapping(value['con'])
            if unit_dict['unit']:  # 如果顶层参数有组件，需要将其装入容器
                unit_dict['unit'].move(unit_con)  # 顶层参数的组件
            unit_dict['child'] = {}
            for key, value in value['param'].items():
                unit_child_dict = self.unit_searching(value)
                if unit_child_dict['con']:  # 如果孩子有容器，那么直接处理容易
                    unit_child_dict['con'].move(unit_con)  # 对于子组件，先装入容器，再考虑可视化的值绑定
                    if unit_dict['unit']:  # 如果父亲有组件，需要指定孩子的可见性
                        unit_child_dict['con'].bind_visibility_from(bind_obj, bind_value)
                    unit_child_dict['con'].bind_visibility_from(bind_obj, bind_value)
                elif unit_child_dict['unit']:  # 如果孩子无容易有组件，直接处理组件
                    unit_child_dict['unit'].move(unit_con)
                    if unit_dict['unit']:  # 如果父亲有组件，需要指定孩子的可见性
                        unit_child_dict['unit'].bind_visibility_from(bind_obj, bind_value)
                unit_dict['child'][key] = unit_child_dict  # 需要保存子组件的字典—尊重孩子
        return unit_dict

    def complete(self):  # 将前端ui读入的参数转为运行args
        for key, value in self.unit_dict.items():
            self.add_argument(key, default=value)

    def save_args_to_json(self, path):  # 将运行args保存为json文件(如果有才保存，说明是修改过的)
        args_dict = vars(self)
        for key, value in self.config.items():
            for key_i, value_i in value['param'].items():
                if args_dict[key_i]:
                    self.config[key][key_i]['default'] = args_dict[key_i]  # 都是py的对象，且都是dict，无需转换
                if value_i['param']:
                    for key_ii, value_ii in value_i['param'].items():
                        if args_dict[key_ii]:
                            self.config[key][key_i][key_ii]['default'] = args_dict[key_ii]

    # def __getitem__(self, item):
    #     return self.config[item]
    #
    # def __setitem__(self, key, value):
    #     self.config[key] = value
    #
    # def __delitem__(self, key):
    #     del self.config[key]
