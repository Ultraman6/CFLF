from nicegui import ui, app

from visual.modules.subscribers import self_info, self_record, self_experiment
from visual.pages.AI import ai_interface, ai_config
from visual.pages.epxeriment import experiment_page
from visual.parts.func import locked_page_height
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_tabs import lazy_tabs


class FramWindow:
    def __init__(self):
        self.tab_mapping = {
            '实验平台': ['实验模拟', '算法配置', '模型配置', '数据集配置', '机器配置'],
            '个人设置': ['个人信息', '历史记录', '历史模型'],
            'AI分析': ['AI配置', 'AI结果']
        }
        self.unit_mapping = {
            '实验模拟': experiment_page, '算法配置': ui.label, '模型配置': ui.label, '数据集配置': ui.label, '机器配置': ui.label,
            '个人信息': self_info, '历史记录': self_record, '历史实验': self_experiment,
            'AI配置': ai_config, 'AI结果': ai_interface
        }
        self.tabs = lazy_tabs(on_change=lambda: self.on_main_tab_change())
        self.sub_tabs = lazy_tabs().props('vertical').classes('w-full')

    def get_ui(self, name: str):
        self.unit_mapping[name]()

    def on_main_tab_change(self):
        tab_list = []
        v = self.tabs.value
        for key in self.tab_mapping[v]:
            tab_list.append(ui.tab(key))
        self.sub_tabs.swap(tab_list)

    def create_main_window(self):
        with ui.header().classes(replace='row items-center') as header:
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')

        for key in self.tab_mapping:
            self.tabs.add(ui.tab(key))
        self.tabs.move(header)
        left_drawer = ui.left_drawer(top_corner=False, bottom_corner=True).classes('bg-neutral-100 p-4 overflow-y-auto')
        ui.space().move(header)
        with ui.avatar() as avatar:
            ui.image(source=app.storage.user["user"]['avatar']).classes('w-full h-full')
        avatar.move(header)
        ui.label(f'Hello {app.storage.user["user"]["username"]}  !').move(header)
        ui.space().move(header)
        with ui.row().classes('h-full') as opt:
            ui.button(on_click=lambda: ui.navigate.to('/self'), icon='settings').props('round')
            ui.button(on_click=lambda: (app.storage.user.clear(), ui.navigate.to('/hall')), icon='logout').props('round')
        opt.move(header)
        self.sub_tabs.move(left_drawer)
        with lazy_tab_panels(self.sub_tabs).classes('w-full h-full') as panels:
            for key, value in self.tab_mapping.items():
                for v in value:
                    panel = panels.tab_panel(v)
                    @panel.build_fn
                    def _(name: str):
                        # 显示 loading 效果
                        # build_ui_loading(f"正在加载{name}页面...")
                        # ui.label(f"正在加载{name}页面...")
                        self.get_ui(name)
                        ui.notify(f"创建页面:{name}")
                        # self.unit_mapping[name]()


