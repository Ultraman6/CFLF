from nicegui import ui

# 包括上菜单和左菜单
def menu_full():
    with ui.header().classes(replace='row items-center') as header:
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
        with ui.tabs() as tabs:
            ui.tab('实验平台')
            ui.tab('个人设置')
            ui.tab('AI分析')
    with ui.left_drawer().classes('bg-blue-100') as left_drawer:
        with ui.tab_panels(tabs, value='实验平台').classes('w-full'):
            with ui.tab_panel('实验平台'):
                ui.link('A', '/a').classes(replace='text-white')
            with ui.tab_panel('个人设置'):
                ui.link('A', '/a').classes(replace='text-white')
            with ui.tab_panel('AI分析'):
                ui.link('B', '/b').classes(replace='text-white')



def sub_menu():
    with ui.left_drawer().classes('bg-blue-100') as left_drawer:
        with ui.tab_panels(tabs, value='实验平台').classes('w-full'):
            with ui.tab_panel('实验平台'):
                ui.label('Content of A')
            with ui.tab_panel('个人设置'):
                ui.label('Content of B')
            with ui.tab_panel('AI分析'):
                ui.label('Content of C')
    return left_drawer
