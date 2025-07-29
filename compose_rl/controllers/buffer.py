from typing import Any

class Buffer:
    """Placeholder class for Async RL"""

    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size
        self.buffer = []

    def put(self, struct: dict[str, Any]):
        raise NotImplementedError

    def get(self, struct: dict[str, Any]):
        raise NotImplementedError
