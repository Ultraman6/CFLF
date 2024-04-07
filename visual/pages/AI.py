#!/usr/bin/env python3
from typing import List, Tuple

from ex4nicegui import deep_ref, to_raw
from ex4nicegui.reactive import rxui
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from nicegui import context, ui, app

from visual.models import User
from visual.parts.func import my_vmodel
from visual.parts.log_callback_handler import NiceGuiLogElementCallbackHandler


@ui.refreshable
def ai_interface():
    api_key = app.storage.user["user"]["ai_config"]['api_key']
    if api_key != '':
        llm = ConversationChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, verbose=False))
        messages: List[Tuple[str, str]] = []
        thinking: bool = False

        @ui.refreshable
        def chat_messages() -> None:
            for name, text in messages:  # 名字不一致，对话对不齐！！！
                ui.chat_message(text=text, name=name, sent=name == app.storage.user["user"]['username'])
            if thinking:
                ui.spinner(size='3rem').classes('self-center')
            if context.get_client().has_socket_connection:
                ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')

        async def send() -> None:
            nonlocal thinking
            message = text.value
            messages.append((app.storage.user["user"]['username'], text.value))
            thinking = True
            text.value = ''
            chat_messages.refresh()

            response = await llm.arun(message, callbacks=[NiceGuiLogElementCallbackHandler(log)])
            messages.append(('Bot', response))
            thinking = False
            chat_messages.refresh()

        # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
        ui.query('.q-page').classes('flex')
        ui.query('.nicegui-content').classes('w-full')
        with ui.tabs().classes('w-full') as tabs:
            chat_tab = ui.tab('Chat')
            logs_tab = ui.tab('Logs')
        with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
            with ui.tab_panel(chat_tab).classes('items-stretch'):
                chat_messages()
            with ui.tab_panel(logs_tab):
                log = ui.log().classes('w-full h-full')

        with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto'):
            with ui.row().classes('w-full items-center'):
                placeholder = 'message' if api_key != 'not-set' else \
                    '请先输入输入OpenAI的API KEY!'
                text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                    .classes('w-full self-center').on('keydown.enter', send)
                ui.button('上传附件', on_click=lambda: ui.notify('暂不支持上传附件！')).props('flat dense')
            ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
                .classes('text-xs self-end mr-8 m-[-1em] text-primary')
    else:
        ui.label('请先去配置api key！！！')


def ai_config():
    config_ref = deep_ref(app.storage.user["user"]["ai_config"])
    rxui.input(label='配置openai的api_key', value=my_vmodel(config_ref.value, 'api_key'))
    ui.button('保存AI配置', on_click=lambda: put())
    async def put():
        await User.filter(id=app.storage.user["user"]['id']).update(**{'ai_config': to_raw(config_ref.value)})
        ai_interface.refresh()
        ui.notify('AI配置成功！', color='positive')
