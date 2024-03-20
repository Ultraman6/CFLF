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

        def delete(dialog):
            self.nums.value.remove(num.value)
            dialog.submit('Yes')

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

