import asyncio
import threading
from types import SimpleNamespace

import pytest

from SLOsServe.router import api_server_ray
from SLOsServe.router import engine_shutdown as engine_shutdown_module
from SLOsServe.router.engine_shutdown import shutdown_engine_instance


@pytest.mark.asyncio
async def test_shutdown_engine_instance_prefers_async_shutdown():
    calls = []

    class FakeEngine:
        async def shutdown_async(self):
            calls.append("async")

        def shutdown(self):
            calls.append("sync")

    await shutdown_engine_instance(FakeEngine())

    assert calls == ["async"]


@pytest.mark.asyncio
async def test_shutdown_engine_instance_falls_back_to_sync_shutdown():
    calls = []

    class FakeEngine:
        def shutdown(self):
            calls.append("sync")

    await shutdown_engine_instance(FakeEngine())

    assert calls == ["sync"]


@pytest.mark.asyncio
async def test_shutdown_engine_instance_force_kills_managed_processes_on_timeout(
    monkeypatch,
):
    killed = []
    blocker = threading.Event()

    class FakeEngine:
        resources = SimpleNamespace(
            engine_manager=SimpleNamespace(
                processes=[SimpleNamespace(pid=12345)],
            ),
        )

        def shutdown(self):
            blocker.wait()

    monkeypatch.setattr(
        engine_shutdown_module,
        "_kill_process_tree",
        lambda pid: killed.append(pid),
    )

    await shutdown_engine_instance(FakeEngine(), timeout_s=0.01)

    assert killed == [12345]


@pytest.mark.asyncio
async def test_engine_worker_shutdown_awaits_async_engine_shutdown():
    calls = []

    class FakeEngine:
        async def shutdown_async(self):
            calls.append("async")

        def shutdown(self):
            raise AssertionError("sync shutdown should not be used")

    worker_cls = api_server_ray.EngineWorker.__ray_actor_class__
    worker = worker_cls.__new__(worker_cls)
    worker._energy_profiler = None
    worker.engine = FakeEngine()
    worker._mux_task = asyncio.create_task(asyncio.sleep(60))

    await worker_cls.shutdown(worker)

    assert calls == ["async"]
    assert worker._mux_task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_engine_actor_handle_kills_ray_actor_after_timeout(
    monkeypatch,
):
    killed = []

    class FakeShutdownHandle:
        async def remote(self):
            await asyncio.sleep(60)

    actor_handle = SimpleNamespace(shutdown=FakeShutdownHandle())

    monkeypatch.setattr(
        api_server_ray,
        "_engine_actor_shutdown_timeout_s",
        lambda: 0.01,
    )
    monkeypatch.setattr(
        api_server_ray.ray,
        "kill",
        lambda actor: killed.append(actor),
    )

    await api_server_ray._shutdown_engine_actor_handle(
        actor_handle,
        "engine_actor[0]",
    )

    assert killed == [actor_handle]
