from nicegui import ui
from visual.modules.consequence import res_ui
from visual.modules.preview import preview_ui
from visual.modules.running import run_ui
from visual.parts.lazy.lazy_stepper import lazy_stepper
from visual.modules.configuration import config_ui


def convert_to_list(mapping):
    mapping_list = []
    for key, value in mapping.items():
        mapping_list.append({'id': key, 'value': value})
    return mapping_list


def convert_to_dict(mapping_list):
    mapping = {}
    for item in mapping_list:
        mapping[item['id']] = item['value']
    return mapping


def build_task_loading(message: str, is_done=False):
    with ui.row().classes("flex-center"):
        if not is_done:
            ui.spinner(color="negative")
        else:
            ui.icon("done", color="positive")
        with ui.row():
            ui.label(message)


class experiment_page:
    algo_args = None
    exp_args = None
    experiment = None
    visual_data_infos = {}
    args_queue = []
    def __init__(self):  # 这里定义引用，传进去同步更新
        with lazy_stepper(keep_alive=False).props('vertical').classes('w-full') as self.stepper:
            with ui.step('参数配置'):
                self.cf_ui = config_ui()
                ui.notify(f"创建页面:{'参数配置'}")
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.args_fusion_step)

            step_pre = self.stepper.step('配置预览')
            @step_pre.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                with ui.card().classes('w-full'):  # 刷新API有问题，但仍可使
                    self.get_pre_ui()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.task_fusion_step)
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

            step_run = self.stepper.step('算法执行')
            @step_run.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                self.get_run_ui()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.res_fusion_step)
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

            step_run = self.stepper.step('结果分析')
            @step_run.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                self.get_res_ui()
                with ui.stepper_navigation():
                    ui.button('Done', on_click=lambda: ui.notify('Yay!', type='positive'))
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

    @ui.refreshable_method
    def get_pre_ui(self):
        self.pre_ui = preview_ui(self.exp_args, self.algo_args)

    @ui.refreshable_method
    def get_run_ui(self):
        self.run_ui = run_ui(self.pre_ui.experiment)

    @ui.refreshable_method
    def get_res_ui(self):
        self.res_ui = res_ui(self.run_ui.experiment, self.cf_ui, self.pre_ui)

    def args_fusion_step(self):
        self.algo_args, self.exp_args = self.cf_ui.get_fusion_args()
        if len(self.exp_args['algo_params']) == 0:
            ui.notify('请添加算法参数')
            return
        if hasattr(self, 'pre_ui'):
            self.get_pre_ui.refresh()
            self.pre_ui.refresh_need.refresh()
        self.stepper.next()

    def task_fusion_step(self):
        if hasattr(self, 'pre_ui'):
            if self.pre_ui.experiment is None:
                ui.notify('请装载实验对象')
                return
        if hasattr(self, 'run_ui'):
            self.get_run_ui.refresh()
            self.run_ui.refresh_need.refresh()
        self.stepper.next()

    def res_fusion_step(self):
        if hasattr(self, 'res_ui'):
            self.get_res_ui.refresh()
            self.res_ui.show_panels.refresh()
            self.res_ui.draw_res.refresh()
        self.stepper.next()


