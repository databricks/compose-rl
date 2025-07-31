from typing import Any, Optional

class Buffer:
    """Placeholder class for Async RL"""

    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size
        self.buffer = []

    def put(self, struct: dict[str, Any]):
        raise NotImplementedError

    def get(self, struct: dict[str, Any]):
        raise NotImplementedError

    def pop(self, struct: Optional[dict[str, Any]] = None):
        raise NotImplementedError

    def is_full(self):
        return len(self.buffer) >= self.buffer_size
