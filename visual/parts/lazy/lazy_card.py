from nicegui import ui

from visual.parts.constant import idx_dict


def build_card(icon: str, title: str, description: str, button_text: str, color: str):
    with ui.card().classes("min-w-[300px]"):
        with ui.column():
            ui.avatar(icon, size="4rem", color=color, text_color="white")
        ui.label(title).classes("text-h6")
        ui.label(description)

        ui.button(button_text, on_click=lambda: ui.navigate.to(idx_dict[button_text])).props(
            "dense icon-right='arrow_right' rounded padding='0.6em 1.2em'").classes("self-end")

# def build_card_record(title: str, description: str):
#     with ui.card().classes("min-w-[300px]"):
#         ui.label(title).classes("text-h6")
#         ui.label(description)
