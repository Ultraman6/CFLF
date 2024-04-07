import base64
import os
from tkinter import filedialog
import tkinter as tk
import requests
from ex4nicegui.reactive import local_file_picker
from ex4nicegui.utils.signals import to_ref_wrapper, to_ref
from nicegui import app, ui
from visual.models import User
from visual.parts.local_file_picker import local_file_picker


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



async def detect_use():
    user = await User.get_or_none(username=app.storage.user["user"]['username'])
    if user is None:
        app.storage.user.clear()
        ui.navigate.to('/hall')
    else:
        return user

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