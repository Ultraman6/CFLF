import base64
import colorsys
import os
import random
from tkinter import filedialog
import tkinter as tk
import requests
from ex4nicegui.reactive import local_file_picker
from ex4nicegui.utils.signals import to_ref_wrapper, to_ref
from nicegui import app, ui
from visual.parts.local_file_picker import local_file_picker

def get_image_data(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    return image_data

def to_base64(input_data):
    """
    将给定的在线图片URL或本地图片的二进制内容转换为Base64编码。
    如果输入是URL，则从该URL下载图片并转换为Base64编码。
    如果输入是二进制内容，则直接转换为Base64编码。

    :param input_data: 图片的URL字符串或图片的二进制内容。
    :return: 图片的Base64编码字符串。
    """
    # 检查是否是URL
    if isinstance(input_data, str) and input_data.startswith('http'):
        # 从URL下载图片
        response = requests.get(input_data)
        if response.status_code == 200:
            # 将图片内容转换为Base64编码
            image_data = response.content
        else:
            raise ValueError("无法从URL下载图片")
    elif isinstance(input_data, bytes):
        # 如果输入是二进制内容，直接使用
        image_data = input_data
    else:
        raise ValueError("输入类型不支持，必须是图片URL或二进制内容")
    # 将图片内容转换为Base64编码
    return 'data:image/png;base64,' + base64.b64encode(image_data).decode('utf-8')

async def han_fold_choice(ref):
    result = await local_file_picker(ref.value, upper_limit=None, directories_only=True)
    if result is not None:
        ref.value = result[0]

async def han_file_choice(ref, limited):
    result = await local_file_picker(ref.value, upper_limit=None, multiple=True, allowed_file_types=limited)
    # 假设result返回的是选中文件的路径列表
    if result:
        for item in result:
            if item not in ref.value:
                ref.value.append(item)  # 若不存在，则添加到ref.value中

def my_vmodel(data, key):
    def setter(new):
        data[key] = new
    return to_ref_wrapper(lambda: data[key], setter)


def convert_dict_to_list(mapping, mapping_dict):
    mapping_list = []
    for key, value in mapping.items():
        in_dict = {'id': key}
        if type(value) is not list:
            value = (value,)
        for i, k in enumerate(mapping_dict):
            in_dict[k] = value[i]
        mapping_list.append(in_dict)
    return mapping_list


def convert_list_to_dict(mapping_list, mapping_dict):
    mapping = {}
    for item in mapping_list:
        if len(mapping_dict) == 1:
            for k in mapping_dict:
                mapping[item['id']] = item[k]
        else:
            in_list = []
            for k in mapping_dict:
                in_list.append(item[k])
            mapping[item['id']] = tuple(in_list)
    return mapping


def convert_tuple_to_dict(mapping, mapping_dict):
    new_dict = {}
    for i, k in enumerate(mapping_dict):
        new_dict[k] = mapping[i]
    return new_dict


def convert_dict_to_tuple(mapping):
    new_list = []
    for k, v in mapping.items():
        new_list.append(v)
    return tuple(new_list)


def get_dicts(colors, infos_each, infos_all, ):
    legend_dict = [
        {
            'name': '总数',
            'icon': 'circle',
        },
        {
            'name': '噪声',
            'icon': 'circle',
        }
    ]
    for label in infos_each:
        legend_dict.append(
            {
                'name': '类别' + str(label),
                'icon': 'circle',
            }
        )
    series_dict = [
        {
            "data": infos_all['total'],
            "type": "bar",
            "name": '总数',
            "barWidth": '10%',  # 设置柱子的宽度
            "barCategoryGap": '0%',  # 设置类目间柱形距离（类目宽的百分比）
            'itemStyle': {
                'color': 'black',
            },
            'emphasis': {
                'focus': 'self',
                'itemStyle': {
                    'borderColor': 'black',
                    'color': 'black',
                    'borderWidth': 20,
                    'shadowBlur': 10,
                    'shadowOffsetX': 0,
                    'shadowColor': 'rgba(0, 0, 0, 0)',
                    'scale': True  # 实际放大柱状图的部分
                },
                'label': {
                    'show': True,
                    'formatter': '{b}\n数量{c}',
                    'position': 'inside',
                }
            },
            # 如果您想要放大当前柱子
            'label': {
                'show': False
            },
            'labelLine': {
                'show': False
            }
        },
        {
            "data": infos_all['noise'],
            "type": "bar",
            "name": '噪声',
            "barWidth": '10%',  # 设置柱子的宽度
            "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
            'itemStyle': {
                'color': 'grey',
            },
            'emphasis': {
                'focus': 'self',
                'itemStyle': {
                    'borderColor': 'grey',
                    'color': 'grey',
                    'borderWidth': 20,
                    'shadowBlur': 10,
                    'shadowOffsetX': 0,
                    'shadowColor': 'rgba(0, 0, 0, 0)',
                    'scale': True  # 实际放大柱状图的部分
                },
                'label': {
                    'show': True,
                    'formatter': '{b}\n噪声{c}',
                    'position': 'inside',
                }
            },
            # 如果您想要放大当前柱子
            'label': {
                'show': False
            },
            'labelLine': {
                'show': False
            }
        }
    ]
    for label, (distribution, noises) in infos_each.items():
        series_dict.append(
            {
                "data": [
                    {
                        "value": dist,  # 总数据量
                        "itemStyle": {
                            "color": {
                                "type": "linear",
                                "x": 0,
                                "y": 0,
                                "x2": 0,
                                "y2": 1,
                                "colorStops": [
                                    {"offset": 0, "color": colors[label]},  # 原始颜色
                                    {"offset": (1 - noise / dist) if dist != 0 else 0, "color": colors[label]},  # 与原始颜色相同，此处为噪声数据位置
                                    {"offset": (1 - noise / dist) if dist != 0 else 1, "color": 'grey'},  # 从噪声数据位置开始渐变
                                    {"offset": 1, "color": 'grey'}  # 底部透明
                                ]
                            }
                        }
                    }
                    for dist, noise in zip(distribution, noises)
                ],
                "type": "bar",
                "stack": 'each',
                "name": '类别' + str(label),
                "barWidth": '10%',  # 设置柱子的宽度
                "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
                'itemStyle': {
                    'color': colors[label],
                },
                'emphasis': {
                    'focus': 'self',
                    'itemStyle': {
                        'borderColor': colors[label],
                        'color': colors[label],
                        'borderWidth': 20,
                        'shadowBlur': 10,
                        'shadowOffsetX': 0,
                        'shadowColor': 'rgba(0, 0, 0, 0)',
                        'scale': True  # 实际放大柱状图的部分
                    },
                    'label': {
                        'show': True,
                        'formatter': '{b}\n数量{c}',
                        'position': 'inside',
                    }
                },
                # 如果您想要放大当前柱子
                'label': {
                    'show': False
                },
                'labelLine': {
                    'show': False
                }
            })
    return legend_dict, series_dict

def build_task_loading(message: str, is_done=False):
    with ui.row().classes("flex-center"):
        if not is_done:
            ui.spinner(color="negative")
        else:
            ui.icon("done", color="positive")

        with ui.row():
            ui.label(message)


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)
