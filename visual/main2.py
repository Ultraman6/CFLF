from nicegui import ui
for i in range(100):
    ui.label(f'Hello, world! {i}')
# ui.footer(fixed=False).style('height: 250px')
with ui.page_sticky(position='bottom-right', x_offset=12, y_offset=15):
    ui.button(icon='phone', color='primary')
with ui.page_sticky(position='bottom-left', x_offset=12, y_offset=15):
    ui.button(icon='message', color='primary')

ui.run(reload=True)