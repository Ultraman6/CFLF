from nicegui import ui


class preview_ui:
    @ui.refreshable_method
    def __init__(self, exp_args, algo_args):
        with ui.card():
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in self.algo_args.items():
                        ui.label(f'{key}: {value}')
                ui.button('展示算法模板', on_click=dialog.open)
                ui.button('保存算法模板', on_click=self.save_algo_args)
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in self.exp_args.items():
                        if key == 'algo_params':
                            for item in value:
                                with ui.card():
                                    ui.label(item['algo'])
                                    for k, v in item['params'].items():
                                        ui.label(f'{k}: {v}')
                        else:
                            ui.label(f'{key}: {value}')
                ui.button('展示实验配置', on_click=dialog.open)
                ui.button('保存实验配置', on_click=self.save_exp_args)
        with ui.grid(columns=3).classes('w-full'):
            ui.button('装载实验对象', on_click=lambda e: self.assemble_experiment(e, loading_box))
            loading_box = ui.row()
            ui.button('查看数据划分', on_click=self.show_distribution)
        self.draw_distribution()

    def save_algo_args(self, algo_args):
        ui.notify('save_algo_args')

    def save_exp_args(self, exp_args):
        print(exp_args)
        ui.notify('save_exp_args')
