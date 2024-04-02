# 此类用于创建冗余参数配置模板 传入tab名称和绑定的value容器
from typing import Dict, List
from ex4nicegui import to_raw, deep_ref, Ref, to_ref
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper
from nicegui import ui

algo_param_mapping = {
    'number': rxui.number,
    'choice': rxui.select,
    'check': rxui.checkbox
}


def my_vmodel(data, key):
    def setter(new):
        data[key] = new

    return to_ref_wrapper(lambda: data[key], setter)


class params_tab:
    def __init__(self, name: str, nums, type, format=None, options=None, default=None, name_dict=None):
        self.nums = nums
        with rxui.card().classes('w-full'):
            rxui.label(name).classes('w-full')
            with ui.row():
                @rxui.vfor(nums)
                def _(s):
                    self.clickable_num(rxui.vmodel(s.get()), type, format, options, name_dict)
                ui.button("追加", on_click=lambda: self.nums.value.append(default))


    # 名称、双向绑定值、类型、格式、选项
    def clickable_num(self, num, type: str, format: str = None, options: Dict[str, str] or List = None,
                      name_dict: Dict = None):

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
                elif type == "list":
                    with rxui.grid(columns=5):
                        @rxui.vfor(num, key='id')
                        def _(store: rxui.VforStore[Dict]):
                            item = store.get()
                            rxui.number(label='客户' + item.value['id'], value=my_vmodel(item.value, 'value'),
                                        format=format)
                elif type == 'label':
                    with rxui.grid(columns=2):
                        for k in name_dict:
                            rxui.number(label=name_dict[k], value=my_vmodel(num.value, k), format=format)
                elif type == 'dict':
                    with rxui.grid(columns=5):
                        @rxui.vfor(num, key='id')
                        def _(store: rxui.VforStore[Dict]):
                            item = store.get()
                            with rxui.column():
                                rxui.label('客户' + item.value['id'])
                                with rxui.row():
                                    for k in name_dict:
                                        rxui.number(label=name_dict[k], value=my_vmodel(item.value['value'], k),
                                                    format=format)
                                ui.button("清除", on_click=lambda: num.value.remove(item.value))
                    ui.button("追加", on_click=lambda: num.value.append(
                        {'id': str(len(num.value)), 'value': {'mean': 0.2, 'std': 0.2}}))
                ui.button("删除", on_click=lambda: delete(dialog)).classes('center')

            if type == 'list':
                with(
                    rxui.grid(columns=5)
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                ):
                    @rxui.vfor(num, key='id')
                    def _(store: rxui.VforStore[Dict]):
                        item = store.get()
                        rxui.label(text=lambda: '客户' + item.value['id'] + ': ' + str(item.value['value']))

            elif type == 'label':
                with(
                    rxui.grid(columns=len(name_dict))
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                ):
                    for k in name_dict:
                        rxui.label(text=lambda k=k: name_dict[k] + ': ' + str(num.value[k]) + ' ')

            elif type == 'dict':
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
                                for k in name_dict:
                                    rxui.label(lambda k=k: name_dict[k] + ': ' + str(item.value['value'][k]) + ' ')
            else:
                (
                    rxui.label(text=num)
                    .classes('w-full outline-double outline-blue-100 outline hover:outline-red-500 p-1')
                    .on('click', dialog.open)
                )
