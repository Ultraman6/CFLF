from nicegui import ui
from constants import *
with ui.stepper().props('vertical').classes('w-full') as stepper:
    with ui.step('配置'):
        ui.label('在这里进行运行配置的设置')
        select1 = ui.select(federated_learning_algorithm, value=1, label='联邦算法')
        select2 = ui.select({1: 'One', 2: 'Two', 3: 'Three'}).bind_value(select1, 'value')
        with ui.stepper_navigation():
            ui.button('Next', on_click=stepper.next)
    with ui.step('运行'):
        ui.label('Mix the ingredients')

        with ui.stepper_navigation():
            ui.button('Next', on_click=stepper.next)
            ui.button('Back', on_click=stepper.previous).props('flat')
    with ui.step('结果'):
        ui.label('Bake for 20 minutes')
        with ui.stepper_navigation():
            ui.button('Done', on_click=lambda: ui.notify('Yay!', type='positive'))
            ui.button('Back', on_click=stepper.previous).props('flat')

ui.run()