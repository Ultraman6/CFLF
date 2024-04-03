from ex4nicegui import to_ref, on
from ex4nicegui.reactive import rxui
from nicegui import ui
from nicegui.functions.refreshable import refreshable_method

from manager.save import Filer
from visual.parts.constant import record_dict


class RecordManager:
    def __init__(self, dir_dict):
        self.configer, self.resulter = None, None
        self.dir_ref, self.filers = {}, {}
        for k, v in dir_dict.items():
            print(k, v)
            self.dir_ref[k] = to_ref(v)
            self.filers[k] = Filer(v)
            print(self.filers[k].his_list)
            @on(lambda k=k: self.dir_ref[k].value)
            def _():
                self.filers[k].set_save_dir(self.dir_ref[k].value)

    def show_dialog(self, key):
        if key == 'algo':
            return self.show_algo
        elif key == 'tem':
            return self.show_tem
        elif key == 'res':
            return self.show_res

    @refreshable_method
    def show_algo(self):
        with ui.dialog() as dialog, ui.card():
            with rxui.grid(columns=1):
                if len(self.filers['algo'].his_list) == 0:
                    rxui.label('无历史信息')
                else:
                    for cof_info in self.filers['algo'].his_list:
                        with rxui.row():
                            (
                                rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                                .on("click", lambda cof_info=cof_info: click(cof_info))
                                .tooltip("点击选择")
                                .tailwind.cursor("pointer")
                                .outline_color("blue-100")
                                .outline_width("4")
                                .outline_style("double")
                                .padding("p-1")
                            )
                            def click(cof_info):
                                algo_info = self.filers['algo'].load_task_result(cof_info['file_name'])
                                self.configer.read_algorithm(algo_info)
                                dialog.close()

                            rxui.button(icon='delete', on_click=lambda cof_info=cof_info: delete(cof_info['file_name']))
                            def delete(filename):
                                self.filers['algo'].delete_task_result(filename)
                                self.filers['algo'].read_pkl_files()
                                self.show_algo.refresh()
                                dialog.open()
        dialog.open()

    @refreshable_method
    def show_tem(self):
        with ui.dialog() as dialog, ui.card():
            with rxui.grid(columns=1):
                if len(self.filers['algo'].his_list) == 0:
                    rxui.label('无历史信息')
                else:
                    for cof_info in self.filers['tem'].his_list:
                        with rxui.row():
                            (
                                rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                                .on("click", lambda cof_info=cof_info: click(cof_info))
                                .tooltip("点击选择")
                                .tailwind.cursor("pointer")
                                .outline_color("blue-100")
                                .outline_width("4")
                                .outline_style("double")
                                .padding("p-1")
                            )

                            def click(cof_info):
                                algo_info = self.filers['tem'].load_task_result(cof_info['file_name'])
                                self.configer.read_template(algo_info)
                                dialog.close()

                            rxui.button(icon='delete', on_click=lambda cof_info=cof_info: delete(cof_info['file_name']))
                            def delete(filename):
                                self.filers['tem'].delete_task_result(filename)
                                self.filers['tem'].read_pkl_files()
                                self.show_tem.refresh()
                                dialog.open()
        dialog.open()

    @refreshable_method
    def show_res(self):
        with ui.dialog() as dialog, ui.card():
            with rxui.grid(columns=1):
                if len(self.filers['res'].his_list) == 0:
                    rxui.label('无历史信息')
                else:
                    for cof_info in self.filers['res'].his_list:
                        with rxui.row():
                            (
                                rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                                .on("click", lambda cof_info=cof_info: click(cof_info))
                                .tooltip("点击选择")
                                .tailwind.cursor("pointer")
                                .outline_color("blue-100")
                                .outline_width("4")
                                .outline_style("double")
                                .padding("p-1")
                            )

                            def click(cof_info):
                                algo_info = self.filers['res'].load_task_result(cof_info['file_name'])
                                self.resulter.add_res(algo_info)
                                dialog.close()

                            rxui.button(icon='delete', on_click=lambda cof_info=cof_info: delete(cof_info['file_name']))

                            def delete(filename):
                                self.filers['res'].delete_task_result(filename)
                                self.filers['res'].read_pkl_files()
                                self.show_res.refresh()
                                dialog.open()
        dialog.open()



# save_reader = RecordManager(record_dict)
# rxui.button(on_click=lambda: save_reader.show_dialog('algo')())
#
# ui.run()
