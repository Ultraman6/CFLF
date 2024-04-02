import random
from typing import List

from ex4nicegui import deep_ref, to_ref, on, Ref
from ex4nicegui.reactive import rxui
from nicegui import ui
from functools import partial, partialmethod


data = [1, 2, 3, 4, 5]
data_ref = deep_ref([data, data])


@rxui.vfor(data_ref)
def _(s):
    print(s.get())
    rxui.label(text=rxui.vmodel(s.get()))

def add():
    data_ref.value.append(data)

def change():
    i = random.randint(0, len(data_ref.value) - 1)
    data_ref.value[i][random.randint(0, 4)] = random.randint(0, 100)

rxui.button('add', on_click=add)
rxui.button('change', on_click=change)
ui.run()


# data = 1
# data_ref = deep_ref([data, ])
#
#
# @rxui.vfor(data_ref)
# def _(s):
#     rxui.label(text=s.get())
#
# def add():
#     data_ref.value.append(data)
#
# def change():
#     i = random.randint(0, len(data_ref.value) - 1)
#     data_ref.value[i] = random.randint(0, 100)
#
# rxui.button('add', on_click=add)
# rxui.button('change', on_click=change)
# ui.run()


# nums = deep_ref([0] * 10)
#
#
# def clickable_num(num: Ref[int]):
#     def onclick():
#         num.value = 0 if num.value == 1 else 1
#
#     label = (
#         rxui.label(num)
#         .on("click", onclick)
#         .tooltip("点我啊")
#         .tailwind.cursor("pointer")
#         .user_select("none")
#         .outline_color("blue-100")
#         .outline_width("4")
#         .outline_style("double")
#         .padding("p-1")
#     )
#
#     return label
#
#
# rxui.label(nums)
# # rxui.label(lambda: "玩完了，所有都是1").bind_visible(
# #     lambda: len(nums.value) == sum(nums.value)
# # )
#
#
# @rxui.vfor(nums)
# def _(s):
#     clickable_num(rxui.vmodel(s.get()))
#
# ui.run()