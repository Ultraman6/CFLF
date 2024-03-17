from nicegui import ui

# def replace():
#     dialog.clear()
#     with dialog, ui.card().classes('w-64 h-64'):
#         ui.label('New Content')
#     dialog.open()

dialog = ui.dialog()
with ui.card() as card:
    ui.label('冗余参数设置')
card.move(dialog)


ui.button('Open', on_click=dialog.open)
# ui.button('Replace', on_click=replace)

ui.run()