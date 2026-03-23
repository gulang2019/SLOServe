from __future__ import annotations

import inspect
from typing import Any


async def shutdown_engine_instance(engine: Any) -> None:
    if engine is None:
        return

    shutdown_async = getattr(engine, "shutdown_async", None)
    if callable(shutdown_async):
        await shutdown_async()
        return

    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return

    result = shutdown()
    if inspect.isawaitable(result):
        await result
