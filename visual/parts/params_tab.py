# 此类用于创建冗余参数配置模板 传入tab名称和绑定的value容器
from typing import Dict, List
from ex4nicegui import to_raw, deep_ref, Ref, to_ref
from ex4nicegui.reactive import rxui
from nicegui import ui

algo_param_mapping = {
    'number': rxui.number,
    'choice': rxui.select
}


class params_tab:
    def __init__(self, name: str, nums, type, format=None, options=None, default=None):
        self.nums = nums
        with rxui.card():
            rxui.label(name)
            with ui.row():
                @rxui.vfor(nums)
                def _(s):
                    self.clickable_num(rxui.vmodel(s.get()), type, format, options)

                ui.button("追加", on_click=lambda: self.nums.value.append(default))

    # 名称、双向绑定值、类型、格式、选项
    def clickable_num(self, num: Ref[int], type: str, format: str = None, options: Dict[str, str] or List = None):
        # def onclick():
        #     is_show.value = not is_show.value
        # def edit():
        #     is_show.value = False
        # def close():
        #     is_show.value = True
        def delete(dialog):
            self.nums.value.remove(num.value)
            dialog.submit('Yes')

        is_show = deep_ref(True)
        with rxui.grid():
            with ui.dialog() as dialog, ui.card():
                if type == "number":
                    rxui.number(value=num, format=format)
                elif type == "choice":
                    rxui.select(value=num, options=options)
                ui.button("删除", on_click=lambda: delete(dialog))
            (
                rxui.label(text=num)
                .on('click', dialog.open)
                .tailwind.cursor("pointer")
                .user_select("none")
                .outline_color("blue-100")
                .outline_width("4")
                .outline_style("double")
                .padding("p-1")
            )

# n_list = [{'key': [1, 2, 3], 'a': 0}]
# n_ref = deep_ref(n_list[0]['key'])
# nums = deep_ref(n_list[0]['key'])
# n1 = deep_ref(n_list[0]['a'])
# params_tab(nums, 'number', "%.0f")
# rxui.select(value=n_ref, options={0:'device', 1:'cpu'}, on_change=lambda: print(n_list))








# data = deep_ref({"id": "xx", "key1": "cpu", "key2": "xx"})
#
# rxui.label(data)
#
#
# def my_vmodel(data, key):
#     def setter(new):
#         data.value[key] = new
#
#     return to_ref_wrapper(lambda: data.value[key], setter)
#
#
# for key in ["key1", "key2"]:
#     rxui.input(value=my_vmodel(data, key))
# ui.run()
