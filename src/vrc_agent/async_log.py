from __future__ import annotations

import asyncio
from typing import Optional

_SENTINEL = object()
_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


async def start_log_worker() -> None:
    global _queue, _worker_task, _loop
    if _worker_task is not None and not _worker_task.done():
        return
    _loop = asyncio.get_running_loop()
    _queue = asyncio.Queue()
    _worker_task = asyncio.create_task(_run_worker())


async def stop_log_worker() -> None:
    global _queue, _worker_task
    if _queue is None or _worker_task is None:
        return
    _queue.put_nowait(_SENTINEL)
    try:
        await asyncio.wait_for(_worker_task, timeout=2.0)
    except Exception:
        _worker_task.cancel()
        try:
            await _worker_task
        except Exception:
            pass
    finally:
        _queue = None
        _worker_task = None


def log(message: str) -> None:
    global _queue, _loop
    if _queue is None or _loop is None:
        print(message)
        return

    try:
        running_loop = asyncio.get_running_loop()
        if running_loop is _loop:
            _queue.put_nowait(message)
            return
    except RuntimeError:
        # Called from non-async thread.
        pass

    _loop.call_soon_threadsafe(_queue.put_nowait, message)


async def _run_worker() -> None:
    assert _queue is not None
    while True:
        item = await _queue.get()
        if item is _SENTINEL:
            break
        print(item, flush=True)
