#!/usr/bin/env python3
import asyncio
import os
from functools import partial
from typing import List, Tuple
from ex4nicegui import deep_ref, to_raw
from ex4nicegui.reactive import rxui
from nicegui import context, ui, app, events, Client, run
from manager.save import Filer
from visual.models import User
from visual.modules.subscribers import self_record, view_mapping
from visual.parts.authmiddleware import ConfigPromptBuilder
from visual.parts.chat import ChatApp
from visual.parts.constant import path_dict
from visual.parts.embeddings import Embedding
from visual.parts.func import my_vmodel, han_fold_choice, build_task_loading
from visual.parts.lazy.lazy_panels import lazy_tab_panels


@ui.refreshable
def ai_interface():
    models = ['gpt-3.5-turbo', 'gpt-4-1106-preview']
    info_ref = {'tem':[], 'algo':[], 'res':[]}
    api_key = app.storage.user["user"]["ai_config"]['api_key']
    chat_his_path = app.storage.user["user"]["ai_config"]['chat_history']
    embedding_files = app.storage.user["user"]["ai_config"]['embedding_files']
    index_files = app.storage.user["user"]["ai_config"]['index_files']
    prompt = ConfigPromptBuilder()
    chat_app = ChatApp(api_key, chat_his_path, embedding_files, index_files)
    embedding = Embedding(api_key, embedding_files, index_files)
    async def open_chat():
        btn.set_visibility(False)
        async def send() -> None:
            message = prompt.build_prompt(info_ref, textarea.value)
            textarea.value = ""
            await chat_app.send(message)

        @ui.refreshable
        def embeddinglist():
            if not os.path.exists(embedding_files):
                os.makedirs(embedding_files)
            embedding_filenames = [f for f in os.listdir(embedding_files)]
            ui.label("Your Uploaded Files").bind_visibility(embedding_switch, "value")
            with ui.column().classes("h-1/2 overflow-y-auto bg-white cursor-pointer").bind_visibility_from(embedding_switch,"value"):
                with ui.element('q-list').props('bordered separator').classes("overflow-y-auto w-full"):
                    for filename in embedding_filenames:
                        with ui.element('q-item').classes("pt-2"):
                            with ui.element('q-item-section'):
                                ui.label(filename)

        async def handle_new_chat():
            textarea.set_value("")
            await chat_app.clear()
            chat_app.chat_history_grid.refresh()

        async def handle_upload(e: events.UploadEventArguments):
            folder_path = "embedding_files"
            os.makedirs(folder_path, exist_ok=True)
            filename = e.name
            filedata = e.content.read()
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "wb") as f:
                f.write(filedata)
            await embedding.create_index()
            embeddinglist.refresh()


        with ui.page_sticky(position='top').classes('bg-white'):
            with ui.row().classes('w-full gap-0 h-full no-wrap mx-auto justify-around'):
                ui.button(on_click=lambda: drawer.set_visibility(not drawer.visible), icon='menu').classes(
                    'items-left').props('flat color=black')
                ui.label('大模型对话💬').on("click", lambda: ui.open("/")).classes(
                    "items-center w-full cursor-pointer text-black w-2/3 text-base font-semibold md:text-[2rem]")
            ui.label("").bind_text_from(chat_app, "current_chat_name").classes(
                "text-black overflow-auto text-elipsis h-full w-full")
        # 右侧内容
        with ui.column().classes("p-4 overflow-y-auto").style("grid-area:content") as drawer:
            with ui.column().classes("w-full justify-around"):
                with ui.column().classes("w-full items-center"):
                    embedding_switch = ui.switch("Chat with your Data", on_change=lambda e: chat_app.on_value_change(
                        embedding_switch=e.value)).bind_value_from(chat_app, "embedding_switch")
                    ui.button(icon="add", on_click=handle_new_chat, color="slate-400").props("rounded")
                with ui.expansion("Settings").classes("w-full"):
                    ui.label("Model").classes("pt-5")
                    ui.select(models, value=app.storage.user['user']['ai_configs'].get('last_model', models[0]),on_change=lambda e: chat_app.on_value_change(ename=e.value)).classes("bg-slate-200 w-full")
                    ui.label("Temperature").classes("pt-5")
                    ui.slider(min=0, max=2, step=0.1, value=0.1,on_change=lambda e: chat_app.on_value_change(etemp=e.value)).props("label-always")
                with ui.column().classes("w-full no-wrap justify-center items-center pt-5"):
                    with ui.row():
                        ui.label("Tokens Used:")
                        ui.label("").bind_text_from(chat_app, "tokens_used").classes("pb-2")
                    with ui.row():
                        ui.label("Total Cost:")
                        ui.label("").bind_text_from(chat_app, "total_cost").classes("pb-2")
                ui.button('保存配置', on_click=lambda: User.set_ai_config(app.storage.user['user']['id'], app.storage.user['user']['ai_configs'])).classes("w-full")
                ui.label("Chat History").classes("pt-4 pb-2 text-xl").bind_visibility_from(embedding_switch, "value",
                                                                                           value=False)
                chat_app.chat_history_grid()
                await embeddinglist()
                ui.label("Upload more Files").classes("pt-4 bp-4").bind_visibility_from(embedding_switch, "value")
                ui.upload(on_upload=handle_upload, multiple=True, auto_upload=True).classes("w-full").props(
                    'color=black accept=".pdf,.txt"').bind_visibility_from(embedding_switch, "value")

        with ui.column().classes('w-full items-stretch items-center justiy-center'):
            await chat_app.chat_messages()
        with ui.column().classes("items-center").style("grid-area:bottom-bar"):
            with ui.row().classes("items-center"):
                placeholder = 'message' if os.environ.get('OPEN_API_KEY') != 'not-set' else \
                    'Please provide your OPENAI key in the Python script first!'
                with ui.textarea(placeholder=placeholder).classes("min-w-[50vw]").props("desen outlined autogrow").on('key.enter', send) as textarea:
                    ui.button(color='blue-8', on_click=send, icon='send').props('flat dense').bind_visibility_from(textarea, 'value')
                ui.button("附件", on_click=lambda: show_record_dialog(info_ref)).props("flat")
            ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)').classes('text-xs self-end mr-8 m-[-1em] text-primary')
        ui.add_css(".q-page-sticky.my-widthfull > div{width: 50%;}")
    btn = ui.button('开启对话', on_click=open_chat)
    # asyncio.run(open_chat())


def show_record_dialog(info_ref):
    user_path = app.storage.user["user"]["local_path"]
    path_reader = {}
    module_dict = {}
    dialog_dict = {}
    table_dict = {}
    grid_dict = {}
    info_dict = {}
    with ui.dialog().props('maximized') as dialog, ui.card().classes('w-full'):
        columns = [
            {'name': 'name', 'label': '实验名称', 'field': 'name'},
            {'name': 'time', 'label': '完成时间', 'field': 'time'},
        ]
        rxui.button('关闭窗口', on_click=dialog.close)
        def view(e: events.GenericEventArguments, k: str) -> None:
            if e.args['file_name'] in dialog_dict[k]:
                dialog_dict[k][e.args['file_name']].open()
            else:
                dialog = view_mapping(k, info_dict[k][e.args['file_name']])
                dialog.open()
                dialog_dict[k][e.args['file_name']] = dialog

        def select(e: events.GenericEventArguments, k: str) -> None:
            file_name = e.args['file_name']
            for item in info_ref[k]:
                if item['file_name'] == file_name:
                    ui.notify('已经选择过了！！！', color='negative')
                    return
            else:
                info_ref[k].append(info_dict[k][file_name])
                grid_dict[k].update()

        def delete(e: events.GenericEventArguments, k: str) -> None:
            file_name = e.args['file_name']
            print(file_name)
            print(info_ref[k])
            for item in info_ref[k]:
                if item['file_name'] == file_name:
                    info_ref[k].remove(item)
                    break
            grid_dict[k].update()

        with rxui.column().classes('w-full items-center'):
            with ui.tabs().classes('w-full') as tabs:
                for k, v in user_path.items():
                    module_dict[k] = ui.tab(path_dict[k]['name'])
                    path_reader[k] = Filer(v)
                    dialog_dict[k] = {}
                    info_dict[k] = {}
                    for item in path_reader[k].his_list:
                        file_name = item['file_name']
                        info_dict[k][file_name] = path_reader[k].load_task_result(file_name)

            with lazy_tab_panels(tabs).classes('w-full'):
                for k, v in module_dict.items():
                    with ui.tab_panel(v):
                        grid_dict[k] = ui.table(columns=columns, rows=info_ref[k])
                        grid_dict[k].add_slot("body", r'''
                            <q-tr :props="props">
                                <q-td key="name">
                                    {{ props.row.name }}
                                </q-td>
                                <q-td key="time">
                                    {{ props.row.time }}
                                </q-td>
                                <q-td key="file_name">
                                    <q-btn flat color="primary" label="删除"
                                        @click="() => $parent.$emit('delete', props.row)">
                                    </q-btn> 
                                </q-td>
                            </q-tr>
                        ''',
                        )
                        grid_dict[k].on('delete', lambda e, k=k: delete(e, k))

                        table_dict[k] = ui.table(columns=columns, rows=path_reader[k].his_list).props('grid')
                        table_dict[k].add_slot('item', r'''
                            <q-card flat bordered :props="props" class="m-1">
                                <q-card-section class="text-center">
                                    <strong>{{ props.row.name }}</strong>
                                </q-card-section>
                                <q-separator />
                                <q-card-section class="text-center">
                                    <strong>{{ props.row.time }}</strong>
                                </q-card-section>
                                <q-card-section class="text-center">
                                    <q-btn flat color="primary" label="查看"
                                        @click="() => $parent.$emit('view', props.row)">
                                    </q-btn> 
                                    <q-btn flat color="delete" label="选择" @click="show(props.row)"
                                        @click="() => $parent.$emit('select', props.row)">
                                    </q-btn> 
                                </q-card-section>
                            </q-card>
                        ''')
                        table_dict[k].on('view', lambda e, k=k: view(e, k))
                        table_dict[k].on('select', lambda e, k=k: select(e, k))
    dialog.open()


def ai_config():
    async def put():
        await User.set_ai_config(app.storage.user["user"]['id'], config)
        ai_interface.refresh()
        ui.notify('AI配置成功！', color='positive')

    config = app.storage.user["user"]["ai_config"]
    config_ref = deep_ref(config)
    rxui.input(label='配置openai的api_key', value=my_vmodel(config_ref.value, 'api_key'))
    with ui.row():
        with rxui.card().tight():
            rxui.button(text='对话记录存放路径', icon='file',on_click=
            partial(han_fold_choice, my_vmodel(config_ref.value, 'chat_history'))).classes('w-full')
            rxui.label(lambda: config_ref.value['chat_history'])
        with rxui.card().tight():
            rxui.button(text='对话文件存放路径', icon='file',on_click=
            partial(han_fold_choice, my_vmodel(config_ref.value, 'embedding_files'))).classes('w-full')
            rxui.label(lambda: config_ref.value['embedding_files'])
        with rxui.card().tight():
            rxui.button(text='文件索引存放路径', icon='file',on_click=
            partial(han_fold_choice, my_vmodel(config_ref.value, 'index_files'))).classes('w-full')
            rxui.label(lambda: config_ref.value['index_files'])
    ui.button('保存AI配置', on_click=put)

