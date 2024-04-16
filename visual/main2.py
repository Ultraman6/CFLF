from nicegui import ui

echart = ui.echart(
    {
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": ["A", "B"], "inverse": True},
        "grid": {
            "width": "100%",
            "height": "100%",
            "left": "0",
            "top": "0",
            "right": "0",
            "bottom": "0",
        },
        "series": [
            {"type": "bar", "name": "Alpha", "data": [0.1, 0.2]},
            {"type": "bar", "name": "Beta", "data": [0.3, 0.4]},
        ],
    }
).classes("h-[20rem] w-[20rem] outline")

ui.run()