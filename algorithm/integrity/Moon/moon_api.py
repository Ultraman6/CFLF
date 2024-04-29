from algorithm.base.server import BaseServer
from .client import Client

class MOON_API(BaseServer):
    def __init__(self, task):
        super().__init__(task)
        self.client_class = Client