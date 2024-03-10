from nicegui import ui

from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.lazy_tabs import lazy_tabs


class MainWindow:
    def __init__(self):
        self.tab_names = {'a': ['1', '2', '3'], 'b': ['4', '5', '6'], 'c': ['7', '8', '9']}
        self.tabs = lazy_tabs(on_change=lambda: self.on_main_tab_change())
        self.sub_tabs = lazy_tabs().props('vertical')

    def on_main_tab_change(self):
        tab_list = []
        v = self.tabs.value
        for key in self.tab_names[v]:
            tab_list.append(ui.tab(key))
        self.sub_tabs.swap(tab_list)


    def create_main_window(self):
        for key in self.tab_names:
            self.tabs.add(ui.tab(key))
        with lazy_tab_panels(self.sub_tabs) as panels:
            for key, value in self.tab_names.items():
                for v in value:
                    panel = panels.tab_panel(v)
                    @panel.build_fn
                    def _(name: str):
                        ui.notify(f"创建页面:{name}")
                        ui.label(f"page:{name}")


main = MainWindow()
main.create_main_window()
ui.run(native=True)
