import asyncio
import threading
from typing import TypeVar, Awaitable


T = TypeVar("T")


def run_async_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from synchronous code safely.

    - If there's no running event loop in this thread, uses asyncio.run.
    - If there is a running loop (e.g., Jupyter, web frameworks), it spins up
      a short-lived helper thread and runs asyncio.run there, then blocks until
      completion. Exceptions are propagated to the caller.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        result_container: dict[str, T] = {}
        exception_box: list[BaseException] = []

        def _runner():
            try:
                result_container["result"] = asyncio.run(coro)
            except BaseException as exc:  # propagate later in caller thread
                exception_box.append(exc)

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()

        if exception_box:
            raise exception_box[0]
        return result_container["result"]

    # No active loop: safe to run directly
    return asyncio.run(coro)


