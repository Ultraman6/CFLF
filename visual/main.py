import os
from functools import partial
from typing import Optional
from ex4nicegui import deep_ref
from ex4nicegui.reactive import rxui
from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from nicegui import Client, app, ui, events, context
from tortoise import Tortoise
from visual.pages.frameworks import FramWindow
from visual.models import User
from visual.parts.authmiddleware import init_db, AuthMiddleware, close_db
from visual.parts.constant import idx_dict, unrestricted_page_routes, state_dict
from visual.parts.func import to_base64, han_fold_choice, my_vmodel, locked_page_height
from visual.parts.lazy.lazy_card import build_card

async def detect_use():
    user = await User.get_or_none(username=app.storage.user["user"]['username'])
    if user is None:
        app.storage.user.clear()
        ui.navigate.to('/hall')
    else:
        return user

@ui.page('/')
async def main_page() -> None:
    await detect_use()
    main = FramWindow()
    main.create_main_window()
    locked_page_height()

# 个人设置界面
@ui.page('/self')
async def self_page() -> None:

    async def try_set(k: str=None) -> None:
        state, mes = await user.update(k, user_ref[k])
        if state:
            app.storage.user.update({'user': dict(user), 'authenticated': True})
        ui.notify(mes, color=state_dict[state])

    # 检查连接，获取model对象
    user = await detect_use()
    user_ref = user.get_dict()
    with ui.card().classes('absolute-center'):
        with ui.row():
            rxui.input('用户名', value=user_ref['uname'])
            ui.button('提交', on_click=lambda: try_set('uname'))
        with ui.row():
            rxui.input('密码', password=True, password_toggle_button=True, value=user_ref['pwd'])
            ui.button('提交', on_click=lambda: try_set('pwd'))
        with ui.column():
            ui.label('个人信息')
            with ui.row():
                for v in user_ref['profile'].values():
                    if v['options']:
                        rxui.select(label=v['name'], options=v['options'], value=v['default']).classes('w-full')
                    elif v['format']:
                        rxui.number(label=v['name'], format=v['format'], value=v['default']).classes('w-full')
                ui.button('提交', on_click=lambda: try_set('profile'))

        with ui.column():
            ui.label('本地路径')
            with ui.row():
                for v in user_ref['local_path'].values():
                    with ui.card().tight():
                        rxui.label(v['default'])
                        rxui.button(text=v['name'], on_click=partial(han_fold_choice, ref=v['default'])).classes('w-full')
                ui.button('提交', on_click=lambda: try_set('local_path'))

        with ui.column():
            ui.label('头像')
            upload = ui.upload(on_upload=lambda e: on_upload(e), on_rejected=lambda e: ui.notify(f'{e.type} Rejected!'),
                               auto_upload=True).style("display:none").props('accept=.jpg, .jpeg, .png, .gif, .svg, .webp, .bmp, .ico')
            with ui.row():
                with ui.avatar().on('click', lambda: upload.run_method("pickFiles")):
                    rxui.image(source=user_ref['avatar'])
                def on_upload(e: events.UploadEventArguments):
                    user_ref['avatar'].value = to_base64(e.content.read())
                    ui.notify(f'Uploaded {e.name}')
                ui.button('提交', on_click=lambda: try_set('avatar'))

        with ui.row():
            ui.button('返回主页', on_click=lambda: ui.navigate.to('/'))


@ui.page('/hall')
async def hall() -> Optional[RedirectResponse]:

    ui.query("body").classes("bg-[#f7f8fc]")
    context.get_client().content.tailwind.align_items("center")
    ui.label("欢迎使用CFLF").classes("text-h4")
    ui.label("这是一个联邦学习领域入门级的可视化实验平台，可以帮助你快速上手FL领域相关实验").classes("text-body2 mb-12")
    with ui.row():
        build_card("login", "欢迎回来", "快来输入你的账号密码吧", "登录", color="blue-200")
        build_card("how_to_reg", "欢迎加入", "快来输入你的基本信息吧", "注册", color="rose-200")
        build_card("help_outline", "欢迎加入", "快来解决你的疑问吧", "答疑", color="rose-200")
    return None

@ui.page('/doubt')
async def doubt() -> Optional[RedirectResponse]:
    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')
    with ui.card().classes('absolute-center'):
        ui.label('疑难解答界面')
        with ui.row():
            ui.button('返回大厅', on_click=lambda: ui.navigate.to('/hall'))
    return None

@ui.page('/login')
async def login() -> Optional[RedirectResponse]:
    async def try_login() -> None:  # 改为异步函数
        # 当前字段检查
        state, mes, order = await User.login(login_info.value['uname'], login_info.value['pwd'])
        if state:
            app.storage.user.update({'user': order, 'authenticated': True})
            ui.navigate.to(app.storage.user.get('referrer_path', '/'))
        ui.notify(mes, color=state_dict[state])

    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')

    login_info = deep_ref({'uname': '', 'pwd': ''})
    ui.query("body").classes("bg-[#f7f8fc]")
    context.get_client().content.tailwind.align_items("center")
    with ui.card().classes('absolute-center'):
        rxui.input('用户名', value=my_vmodel(login_info.value, 'uname')).on('keydown.enter', try_login)
        rxui.input('密码', value=my_vmodel(login_info.value, 'pwd'), password=True, password_toggle_button=True).on('keydown.enter', try_login)
        ui.button('登录', on_click=try_login)
        with ui.row():
            ui.button('去注册', on_click=lambda: ui.navigate.to('/register'))
            ui.button('返回大厅', on_click=lambda: ui.navigate.to('/hall'))
    return None


@ui.page('/register')
async def register() -> Optional[RedirectResponse]:
    async def try_register() -> None:
        # 创建新用户并保存到数据库
        state, *info = await User.register(sign_info)
        if state:
            mes, order = info
            app.storage.user.update({'user': order, 'authenticated': True})
            # 导航到用户原来想要去的页面或首页
            ui.navigate.to(app.storage.user.get('referrer_path', '/'))
        else:
            mes = info
        ui.notify(mes, color=state_dict[state])

    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')

    sign_info = User.get_tem_dict()
    print(sign_info['profile']['edu']['default'].value)
    with ui.card().classes('absolute-center overflow-auto'):
        rxui.input('Username', value=sign_info['uname'])
        rxui.input('Password', value=sign_info['pwd'], password=True, password_toggle_button=True)
        with ui.column():
            ui.label('个人信息')
            with ui.row():
                for v in sign_info['profile'].values():
                    if v['options']:
                        rxui.select(label=v['name'], options=v['options'], value=v['default']).classes('w-full')
                    elif v['format']:
                        rxui.number(label=v['name'], format=v['format'], value=v['default']).classes('w-full')

        with ui.column():
            ui.label('本地路径')
            with ui.row():
                for v in sign_info['local_path'].values():
                    with ui.card().tight():
                        rxui.label(v['default'])
                        rxui.button(text=v['name'], on_click=partial(han_fold_choice, ref=v['default'])).classes('w-full')

        with ui.column():
            ui.label('头像')
            upload = ui.upload(on_upload=lambda e: on_upload(e), on_rejected=lambda e: ui.notify(f'{e.type} Rejected!'),
                               auto_upload=True).style("display:none").props('accept=.jpg, .jpeg, .png, .gif, .svg, .webp, .bmp, .ico')
            with ui.avatar().on('click', lambda: upload.run_method("pickFiles")):
                rxui.image(source=sign_info['avatar']).classes('w-full h-full')

            def on_upload(e: events.UploadEventArguments):
                sign_info['avatar'].value = to_base64(e.content.read())
                ui.notify(f'Uploaded {e.name}')

        with ui.row():
            ui.button('注册', on_click=try_register)
            ui.button('去登录', on_click=lambda: ui.navigate.to('/login'))
            ui.button('返回大厅', on_click=lambda: ui.navigate.to('/hall'))

    return None

# 加载数据库与中间件
app.on_startup(init_db)
app.on_shutdown(close_db)
app.add_middleware(AuthMiddleware)
os.environ['NICEGUI_STORAGE_PATH'] = 'running/storage'
ui.run(storage_secret='THIS_NEEDS_TO_BE_CHANGED')