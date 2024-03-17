from contextlib import contextmanager

from nicegui import ui


@contextmanager
def frame(navtitle: str):
    """Custom page frame to share the same styling and behavior across all pages"""
    ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')
    with ui.header().classes('justify-between text-white'):
        ui.label('联邦学习可视化实验平台').classes('font-bold')
        ui.label(navtitle)
        with ui.row():
            menu()
    with ui.column().classes('absolute-center items-center'):
        yield

        with ui.header().classes(replace='row items-center') as header:
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
            with ui.tabs() as tabs:
                ui.tab('实验平台')
                ui.tab('个人设置')
                ui.tab('AI分析')

        with ui.footer(value=False) as footer:
            ui.label('Footer')
        left_drawer = sub_menu()

        with ui.page_sticky(position='bottom-right', x_offset=20, y_offset=20):
            ui.button(on_click=footer.toggle, icon='contact_support').props('fab')

        with ui.tab_panels(tabs, value='实验平台').classes('w-full'):
            with ui.tab_panel('实验平台'):
                ui.label('Content of A')
            with ui.tab_panel('个人设置'):
                ui.label('Content of B')
            with ui.tab_panel('AI分析'):
                ui.label('Content of C')
