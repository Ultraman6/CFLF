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