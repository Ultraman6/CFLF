from datetime import datetime
from functools import partial
from ex4nicegui import on, deep_ref
from ex4nicegui.reactive import rxui, local_file_picker
from nicegui import ui, app
from nicegui.functions.refreshable import refreshable_method
from manager.save import Filer
from visual.models import User
from visual.parts.constant import path_dict
from visual.parts.func import han_fold_choice


class RecordManager:
    def __init__(self, key, opter):
        self.opter = opter
        self.key = key
        self.dir_ref = deep_ref(app.storage.user["user"]['local_path'][key])
        self.filer = Filer(self.dir_ref.value)

        @on(self.dir_ref)
        async def _():
            self.filer.set_save_dir(self.dir_ref.value)
            dirs = app.storage.user["user"]['local_path']
            dirs[self.key] = self.dir_ref.value
            await User.filter(id=app.storage.user["user"]['id']).update(**{'local_path': dirs})

    async def save_fold_choice(self):
        init = self.dir_ref.value
        fp = local_file_picker(mode='dir', dir=self.dir_ref.value)
        fp.open()
        fp.bind_ref(self.dir_ref)
        if not self.dir_ref.value and self.dir_ref.value == '':
            self.dir_ref.value = init

    def read_record(self, record):
        if self.key == 'tem':
            self.opter.read_tem(record)
        elif self.key == 'algo':
            self.opter.read_algo(record)
        elif self.key == 'res':
            self.opter.read_res(record)

    def han_save(self):
        if self.key == 'algo':
            self.filer.save_task_result(self.opter.exp_ref.value['name'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        self.opter.exp_args)
        elif self.key == 'tem':
            self.filer.save_task_result(self.opter.exp_ref.value['name'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        self.opter.algo_args)
        self.dialog_content.refresh()
        self.dialog.close()
        self.dialog.open()

    def show_panel(self):
        with ui.dialog() as self.dialog:
            self.card = ui.card()

        self.dialog_content()
        with ui.row():
            with rxui.card().tight():
                rxui.button(text=path_dict[self.key]['name'] + '存放路径', icon='file',
                            on_click=partial(han_fold_choice, self.dir_ref)).classes('w-full')
                rxui.label(self.dir_ref)
            rxui.button('历史' + path_dict[self.key]['name'], on_click=self.dialog.open)
            rxui.button('保存' + path_dict[self.key]['name'], on_click=self.han_save).bind_visible(self.key != 'res')

    def delete(self, filename):
        self.filer.delete_task_result(filename)
        self.filer.read_pkl_files()
        self.dialog_content.refresh()

    def click(self, cof_info):
        algo_info = self.filer.load_task_result(cof_info['file_name'])
        self.read_record(algo_info)
        self.dialog.close()

    @refreshable_method
    def dialog_content(self):
        with ui.column().classes('w-full') as grid:
            if len(self.filer.his_list) == 0:
                ui.label('无历史信息')
            else:
                for cof_info in self.filer.his_list:
                    with rxui.row():
                        (
                            rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                            .on("click", lambda cof_info=cof_info: self.click(cof_info))
                            .tooltip("点击选择")
                            .tailwind.cursor("pointer")
                            .outline_color("blue-100")
                            .outline_width("4")
                            .outline_style("double")
                            .padding("p-1")
                        )
                        rxui.button(icon='delete',
                                    on_click=lambda cof_info=cof_info: self.delete(cof_info['file_name']))
        self.card.clear()
        grid.move(self.card)
