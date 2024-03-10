from nicegui import ui


# 封装这个tabs，使得有其他组件move入时，tabs的value会自动更新
class lazy_tabs(ui.tabs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__tabs = []

    def add(self, tab: ui.tab):
        tab.move(self)
        if self.value is None:
            self.set_value(tab._props['name'])

    def swap(self, tab_list: list[ui.tab]):
        self.clear()
        self.set_value(tab_list[0]._props['name'])
        for tab in tab_list:
            tab.move(self)
