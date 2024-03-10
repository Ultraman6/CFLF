from visual.modules.configuration import config_ui
from nicegui import ui


def create() -> None:
    @ui.page('/experiment')
    def experiment_page():
        with theme.frame('- 实验平台 -'):
            args = args_parser()
            cf_ui = config_ui()
            cf_ui.create_config_ui(args)

    @ui.page('/personal')
    def personal_page():
        with theme.frame('- Example B -'):
            message('Example B')

    @ui.page('/religion')
    def religion_page():
        with theme.frame('- Example B -'):
            message('Example B')
