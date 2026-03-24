from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    DEFAULT_ENGINE_SHUTDOWN_TIMEOUT_S = max(
        float(os.getenv("SLOSSERVE_ENGINE_SHUTDOWN_TIMEOUT_S", "5.0")),
        0.0,
    )
except ValueError:
    DEFAULT_ENGINE_SHUTDOWN_TIMEOUT_S = 5.0


def collect_engine_process_pids(engine: Any) -> list[int]:
    if engine is None:
        return []

    pids: list[int] = []
    seen_pids: set[int] = set()
    seen_objects: set[int] = set()
    stack = [engine]

    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen_objects:
            continue
        seen_objects.add(obj_id)

        processes = getattr(obj, "processes", None)
        if processes is not None:
            try:
                for proc in processes:
                    pid = getattr(proc, "pid", None)
                    if isinstance(pid, int) and pid > 0 and pid not in seen_pids:
                        seen_pids.add(pid)
                        pids.append(pid)
            except TypeError:
                pass

        for attr in ("engine_core", "resources", "engine_manager"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)

    return pids


def _kill_process_tree(pid: int) -> None:
    try:
        from vllm.utils import kill_process_tree
    except Exception:
        kill_process_tree = None

    if kill_process_tree is not None:
        try:
            kill_process_tree(pid)
            return
        except ProcessLookupError:
            return
        except Exception:
            logger.exception(
                "Failed to kill process tree for engine pid=%s via vLLM helper",
                pid,
            )

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def force_kill_engine_processes(engine: Any) -> list[int]:
    pids = collect_engine_process_pids(engine)
    for pid in pids:
        try:
            _kill_process_tree(pid)
        except Exception:
            logger.exception("Failed to force kill engine pid=%s", pid)
    return pids


async def _run_sync_shutdown(shutdown: Any, timeout_s: float | None) -> None:
    result_box: dict[str, Any] = {}
    finished = threading.Event()

    def _runner() -> None:
        try:
            result_box["result"] = shutdown()
        except BaseException as exc:  # pragma: no cover - defensive
            result_box["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(
        target=_runner,
        name="engine-shutdown",
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_running_loop()
    deadline = None if timeout_s is None else loop.time() + timeout_s
    while not finished.is_set():
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            await asyncio.sleep(min(0.05, remaining))
        else:
            await asyncio.sleep(0.05)

    error = result_box.get("error")
    if error is not None:
        raise error

    result = result_box.get("result")
    if inspect.isawaitable(result):
        if deadline is None:
            await result
        else:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            await asyncio.wait_for(result, timeout=remaining)


async def shutdown_engine_instance(
    engine: Any,
    *,
    timeout_s: float | None = DEFAULT_ENGINE_SHUTDOWN_TIMEOUT_S,
    force_kill_on_timeout: bool = True,
) -> None:
    if engine is None:
        return

    shutdown_async = getattr(engine, "shutdown_async", None)
    try:
        if callable(shutdown_async):
            result = shutdown_async()
            if timeout_s is None:
                await result
            else:
                await asyncio.wait_for(result, timeout=timeout_s)
            return

        shutdown = getattr(engine, "shutdown", None)
        if not callable(shutdown):
            return

        await _run_sync_shutdown(shutdown, timeout_s)
    except asyncio.TimeoutError:
        if not force_kill_on_timeout:
            raise
        killed_pids = force_kill_engine_processes(engine)
        if killed_pids:
            logger.warning(
                "Engine shutdown timed out after %.3fs; killed managed engine "
                "processes %s",
                timeout_s if timeout_s is not None else -1.0,
                killed_pids,
            )
            return
        raise
