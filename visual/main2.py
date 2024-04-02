import random

from ex4nicegui import deep_ref, to_ref, on
from ex4nicegui.reactive import rxui
from nicegui import ui


data = {"a": {'params': ['1', '2'], 'watch': 'k'}, "b":  {'params': ['3', '4'], 'watch': 'k'}}
dict_ref = deep_ref({'k': 10})
data_ref = deep_ref({'a': {'1': 10, '2': 10}, 'b': {'3': 10, '4': 10}})

def watch_from(k1, k2, mapping):
    print('watch')
    for k in mapping:
        data_ref.value[k1][k] = dict_ref.value[k2]

for k, v in data.items():
    on(lambda v=v: dict_ref.value[v['watch']])(watch_from(k, v['watch'], v['params']))




def onclick():
    dict_ref.value['k'] = random.randint(0, 10)


ui.button("click me", on_click=onclick)
ui.run(port=8085)