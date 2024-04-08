# 中间件用户jwt认证
from nicegui import Client, app
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from fastapi import Request
from tortoise import Tortoise

unrestricted_page_routes = {'/hall', '/login', '/register', '/doubt'}
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not app.storage.user.get('authenticated', False):
            if request.url.path in Client.page_routes.values() and request.url.path not in unrestricted_page_routes:
                app.storage.user['referrer_path'] = request.url.path  # remember where the user wanted to go
                return RedirectResponse('/hall')
        return await call_next(request)

async def init_db() -> None:
    await Tortoise.init(db_url='sqlite://running/database/db.sqlite3', modules={'models': ['visual.models']})
    await Tortoise.generate_schemas()
async def close_db() -> None:
    await Tortoise.close_connections()


class ConfigPromptBuilder:
    def __init__(self):
        """初始化一个固定模式，用于构建prompt字符串。"""
        self.prompt_pattern = "Experiment Configurations:\n{}"

    def build_prompt(self, configs: dict, question: str) -> str:
        """根据配置信息和用户问题构建prompt字符串。
        :param configs: 包含不同种类记录的字典。
        :param question: 用户提出的问题。
        :return: 完整的prompt字符串。
        """
        # 按实验名称整理配置信息
        experiments = self.organize_configs_by_experiment(configs)
        # 构建配置信息字符串
        config_strings = []
        for exp_name, categories in experiments.items():
            config_strings.append(f"\nExperiment: {exp_name}\n")
            for category, records in categories.items():
                config_strings.append(f"  Category: {category}\n")
                for record in records:
                    config_strings.append(f"    {self.record_to_string(record)}\n")

        # 添加用户问题
        prompt = f"\nQuestion:\n{question}"
        prompt += self.prompt_pattern.format(''.join(config_strings))

        return prompt

    def organize_configs_by_experiment(self, configs: dict) -> dict:
        """将配置信息按实验名称分类整理。"""
        experiments = {}
        for category, records in configs.items():
            for record in records:
                exp_name = record.get("name", "Unknown")
                experiments.setdefault(exp_name, {}).setdefault(category, []).append(record)
        return experiments

    @staticmethod
    def record_to_string(record: dict) -> str:
        """将单个记录转换为字符串表示，忽略'name'键。"""
        return ", ".join([f"{key}: {value}" for key, value in record.items() if key != 'name'])