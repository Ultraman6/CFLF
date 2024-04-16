# 此类用于创建冗余参数配置模板 传入tab名称和绑定的value容器
import copy
from typing import Dict, List
from ex4nicegui import to_raw, deep_ref, Ref, to_ref
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper
from nicegui import ui

from visual.parts.func import my_vmodel

algo_param_mapping = {
    'number': rxui.number,
    'choice': rxui.select,
    'check': rxui.checkbox
}


class params_tab:
    def __init__(self, name: str, nums, type, format=None, options=None, default=None, info_dict=None):
        self.nums = nums
        with rxui.card().classes('w-full'):
            rxui.label(name).classes('w-full')
            with ui.row():
                @rxui.vfor(nums)
                def _(s):
                    self.clickable_num(rxui.vmodel(s.get()), type, format, options, info_dict)
                ui.button("追加", on_click=lambda: self.nums.value.append(copy.deepcopy(default)))


    # 名称、双向绑定值、类型、格式、选项
    def clickable_num(self, num, type: str, format: str = None, options: Dict[str, str] or List = None,
                      info_dict: Dict = None):

        def delete(dialog):
            if len(self.nums.value) == 1:
                ui.notify('至少保留一个参数')
                dialog.submit('No')
            else:
                self.nums.value.remove(num.value)
                dialog.submit('Yes')

        with rxui.grid():
            with ui.dialog() as dialog, ui.card():
                if type == "number":
                    rxui.number(value=num, format=format)
                elif type == "choice":
                    rxui.select(value=num, options=options)
                elif type == "check":
                    rxui.checkbox(value=num)
                elif type == 'dict':
                    with rxui.grid(columns=2):
                        for k, v in info_dict.items():
                            rxui.number(label=v['name'], value=my_vmodel(num.value, k), format=v['format'])

                elif type == 'mapping':
                    print(num.value)
                    with rxui.grid(columns=4):
                        @rxui.vfor(num, key='id')
                        def _(store: rxui.VforStore[Dict]):
                            item = store.get()
                            with rxui.column():
                                with rxui.row():
                                    for k2, v2 in info_dict['mapping'].items():
                                        if 'discard' not in v2:
                                            rxui.number(label='客户' + item.value['id'] + v2['name'],
                                                        value=my_vmodel(item.value, k2), format=v2['format']).classes('w-full')
                                if 'watch' not in info_dict:
                                    rxui.button('删除', on_click=lambda: num.value.remove(item.value))
                    if 'watch' not in info_dict:
                        ui.button("追加", on_click=lambda: num.value.append({'id': str(len(num.value)), **{k2: v2['default'] for k2, v2 in info_dict['mapping'].items()}}))

                ui.button("删除", on_click=lambda: delete(dialog)).classes('center')

            if type == 'dict':
                with(
                    rxui.grid(columns=len(info_dict))
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                ):
                    for k, v in info_dict.items():
                        rxui.label(text=lambda k=k: v['name'] + ': ' + str(num.value[k]) + ' ')

            elif type == 'mapping':
                with(
                    rxui.grid(columns=3)
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                ):
                    @rxui.vfor(num, key='id')
                    def _(store: rxui.VforStore[Dict]):
                        item = store.get()
                        with rxui.column():
                            rxui.label(lambda: '客户' + item.value['id'])
                            with rxui.row():
                                for k, v in info_dict['mapping'].items():
                                    if 'discard' not in v:
                                        rxui.label(lambda k=k: v['name'] + ': ' + str(item.value[k]) + ' ')
            else:
                (
                    rxui.label(text=num)
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                )
