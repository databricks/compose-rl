import asyncio
from typing import Any


class Buffer:
    def __init__(self, buffer_size: int = 1):
        self.buffer = asyncio.Queue(maxsize=buffer_size)

    async def put(self, struct: dict[str, Any]):
        raise NotImplementedError

    async def get(self, struct: dict[str, Any]):
        raise NotImplementedError
