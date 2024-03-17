from nicegui import ui


class preview_ui:
    @ui.refreshable_method
    def __init__(self, exp_args, algo_args):
        with ui.card():  # 刷新API有问题，但仍可使用
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in algo_args.items():
                        ui.label(f'{key}: {value}')
                ui.button('show_algo_args', on_click=dialog.open)
                ui.button('save_algo_args', on_click=self.save_algo_args(algo_args))
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in exp_args.items():
                        if key == 'algo_params':
                            for item in value:
                                with ui.card():
                                    ui.label(item['algo'])
                                    for k, v in item['params'].items():
                                        ui.label(f'{k}: {v}')
                        else:
                            ui.label(f'{key}: {value}')
                ui.button('show_exp_args', on_click=dialog.open)
                ui.button('save_exp_args', on_click=self.save_exp_args(exp_args))

    def save_algo_args(self, algo_args):
        ui.notify('save_algo_args')

    def save_exp_args(self, exp_args):
        print(exp_args)
        ui.notify('save_exp_args')
