from ex4nicegui import deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui

data = deep_ref({"a": 1, "b": 2, "c": [0, 1, 2]})


rxui.label("test").bind_visible(lambda: data.value["c"][-1] == "v")


def onclick():
    data.value["c"].append("v")


ui.button("click me", on_click=onclick)
ui.run()