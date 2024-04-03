from weakref import WeakValueDictionary

from nicegui import ui, run
from typing import Any, Callable, Optional, Union, cast
from nicegui.elements.stepper import Step
from nicegui.events import handle_event


def build_task_loading(message: str, is_done=False):
    with ui.row().classes("flex-center"):
        if not is_done:
            ui.spinner(color="negative")
            with ui.row():
                ui.label(message)



class lazy_stepper(ui.stepper):
    def __init__(self, *,
                 value: Union[str, Step, None] = None,
                 on_value_change: Optional[Callable[..., Any]] = None,
                 keep_alive: bool = True
                 ) -> None:

        self.__steps: WeakValueDictionary[str, lazy_step] = WeakValueDictionary()

        def inject_onchange(e):
            step = self.__steps.get(self.value)
            if step:
                step.try_run_build_fn()
            if on_value_change:
                handle_event(on_value_change, e)

        super().__init__(
            value=value,
            on_value_change=inject_onchange,
            keep_alive=keep_alive
        )

    def step(self, name: Union[ui.tab, str]):
        step = lazy_step(name)
        str_name = step._props["name"]
        self.__steps[str_name] = step
        return step


class lazy_step(ui.step):
    def __init__(self, name: str, title: Optional[str] = None, icon: Optional[str] = None) -> None:
        super().__init__(name, title, icon)
        self._build_fn = None

    def try_run_build_fn(self):
        if self._build_fn:
            with self:
                self._build_fn(self._props["name"])
            self._build_fn = None

    def build_fn(self, fn: Callable[[str], None]):
        self._build_fn = fn
        return fn

# def onchange():
#     stepper.set_value(r.value)
#
#
# r = ui.radio(list("abcd"), on_change=onchange)
#
#
# # 每次 panel 切换，都会执行
# def on_panels_change(e):
#     tab_name = e.value
#     ui.notify(f"页面切换了:{tab_name=}")
#
#
# with lazy_stepper(on_value_change=on_panels_change) as stepper:
#     for name in "abcd":
#         # 使用 panels 创建 tab panel
#         step = stepper.step(name)
#         # 被装饰的函数，只在该页首次显示时执行，并且只执行一次
#         @step.build_fn
#         def _(name: str):
#             ui.notify(f"创建页面:{name}")
#             ui.label(f"page:{name}")
#
# r.set_value("a")
#
# ui.run()
