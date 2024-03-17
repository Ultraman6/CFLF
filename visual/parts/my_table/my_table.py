from typing import Callable, Optional
from nicegui import ui
from nicegui.element import Element


class My_Table(ui.table, component='my_table.js'):

    def __init__(self, title: str, *, on_change: Optional[Callable] = None) -> None:
        super().__init__()
        self._props['title'] = title
        self.on('change', on_change)

    def reset(self) -> None:
        self.run_method('reset')




# ui.markdown()
# with ui.card():
#     counter = Counter('Clicks', on_change=lambda e: ui.notify(f'The value changed to {e.args}.'))
# ui.button('Reset', on_click=counter.reset).props('small outline')

ui.run()