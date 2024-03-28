from ex4nicegui.reactive import rxui
from nicegui import ui

def add_row():
    with ui.item(on_click=lambda: ui.notify('Selected contact 1')) as row:
        with ui.item_section().props('avatar'):
            ui.icon('person')
        with ui.item_section():
            ui.item_label('Nice Guy')
            ui.item_label('name').props('caption')
        with ui.item_section().props('side').on('click', lambda row=row: row.delete()):
            ui.icon('delete')
    row.move(lister)

def delete_row(row):
    row.delete()

with ui.list().props('bordered separator') as lister:
    ui.item_label('Contacts').props('header').classes('text-bold')
    ui.separator()
    for i in range(5):
        with ui.item(on_click=lambda: ui.notify('Selected contact 1')) as row:
            with ui.item_section().props('avatar'):
                ui.icon('person')
            with ui.item_section():
                ui.item_label('Nice Guy')
                ui.item_label('name').props('caption')
            with ui.item_section().props('side').on('click', lambda row=row: row.delete()):
                ui.icon('delete')

ui.button('Add Contact', on_click=lambda: add_row())

ui.run()
