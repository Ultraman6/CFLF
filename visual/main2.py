from nicegui import ui
from random import random

echart = ui.echart({
    "tooltip": {
        "trigger": 'axis',
        "axisPointer": {
            "type": 'shadow'
        }
    },
    "legend": {
        "data": ['Product A', 'Product B', 'Product C']
    },
    "grid": {
        "left": '3%',
        "right": '4%',
        "bottom": '3%',
        "containLabel": True
    },
    "xAxis": {
        "type": 'category',
        "data": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    },
    "yAxis": {
        "type": 'value'
    },
    "series": [
        {
            "name": 'Product A',
            "type": 'line',
            "data": [320, 332, 301, 334, 390, 330]
        },
        {
            "name": 'Product B',
            "type": 'line',
            "data": [120, 132, 101, 134, 90, 230]
        },
        {
            "name": 'Product C',
            "type": 'line',
            "data": [220, 182, 191, 234, 290, 330]
        }
    ]
}
)

def update():
    echart.options['series'][0]['data'][0] = random()
    echart.update()

ui.button('Update', on_click=update)

ui.run()