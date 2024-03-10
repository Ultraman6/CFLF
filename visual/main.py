from nicegui import ui
from visual.modules.configuration import config_ui
from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.lazy_tabs import lazy_tabs


class MainWindow:
    def __init__(self):
        self.tab_mapping = {
            '实验平台': ['实验模拟', '算法配置', '模型配置', '数据集配置', '机器配置'],
            '个人设置': ['个人信息', '历史配置', '历史结果', '历史模型'],
            'AI分析': ['AI配置', 'AI结果']
        }
        self.unit_mapping = {
            '实验模拟': config_ui, '算法配置': ui.label, '模型配置': ui.label, '数据集配置': ui.label, '机器配置': ui.label,
            '个人信息': ui.label, '历史配置': ui.label, '历史结果': ui.label, '历史模型': ui.label,
            'AI配置': ui.label, 'AI结果': ui.label
        }
        self.tabs = lazy_tabs(on_change=lambda: self.on_main_tab_change())
        self.sub_tabs = lazy_tabs().props('vertical').classes('w-full')

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
        left_drawer = ui.left_drawer().classes('bg-blue-100')
        self.sub_tabs.move(left_drawer)
        with lazy_tab_panels(self.sub_tabs).classes('w-full') as panels:
            for key, value in self.tab_mapping.items():
                for v in value:
                    panel = panels.tab_panel(v)
                    @panel.build_fn
                    def _(name: str):
                        ui.notify(f"创建页面:{name}")
                        self.unit_mapping[name](f"page:{name}")


# Initialize and run the main window
main_window = MainWindow()
main_window.create_main_window()
ui.run(native=True, fullscreen=True)
