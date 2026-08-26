"""asyncio 兼容工具。

Python 3.9 才有 asyncio.to_thread；JetPack 4.6 是 Python 3.8，
用 run_in_executor 提供等价实现，保证 3.8 也能跑。
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


async def to_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Awaitable:
    """等价 asyncio.to_thread，兼容 Python 3.8。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
