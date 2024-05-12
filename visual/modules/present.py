# 配置介绍模块 包括: 算法配置、数据集配置、模型配置
from nicegui import ui


def get_present():
    with ui.tabs().classes('w-full items-center') as tabs:
        ui.tab('算法配置')
        ui.tab('数据集配置')
        ui.tab('模型配置')

    with ui.tab_panels(tabs):
        with ui.tab_panel('算法配置'):
            get_algo()
        with ui.tab_panel('数据集配置'):
            get_dataset()
        with ui.tab_panel('模型配置'):
            get_model()


def get_algo():
    ui.label('算法配置介绍')

def get_dataset():
    ui.label('数据集配置介绍')

def get_model():
    ui.label('模型配置介绍')