import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from motivation import bench_api_server
from SLOsServe.router import api_server_ray


@pytest.mark.parametrize(
    ("raw_clients", "expected"),
    [
        (None, None),
        ("", None),
        ("0-3", "r0,r1,r2,r3"),
        ("0:4", "r0,r1,r2,r3"),
        ("0,2,4", "r0,r2,r4"),
        ("r0,r1", "r0,r1"),
        ("8001:2", "http://localhost:8001,http://localhost:8002"),
        (
            "http://localhost:8001,http://localhost:8002",
            "http://localhost:8001,http://localhost:8002",
        ),
    ],
)
def test_normalize_router_clients_arg_preserves_replica_semantics(
    raw_clients,
    expected,
):
    assert bench_api_server._normalize_router_clients_arg(raw_clients) == expected


def test_parse_clients_arg_splits_logical_replica_labels():
    assert api_server_ray._parse_clients_arg(None) == []
    assert api_server_ray._parse_clients_arg("r0, r1 ,, r2") == ["r0", "r1", "r2"]
    assert api_server_ray._parse_clients_arg("0-3") == ["r0", "r1", "r2", "r3"]
    assert api_server_ray._parse_clients_arg("0:4") == ["r0", "r1", "r2", "r3"]
    assert api_server_ray._parse_clients_arg("8001:2") == [
        "http://localhost:8001",
        "http://localhost:8002",
    ]


@pytest.mark.asyncio
async def test_update_config_rejects_tensor_parallel_size_mismatch(monkeypatch):
    fake_pool = SimpleNamespace(update_config=MagicMock())

    monkeypatch.setattr(api_server_ray, "request_pool", fake_pool)
    monkeypatch.setattr(
        api_server_ray,
        "args",
        SimpleNamespace(tensor_parallel_size=2, mock_connector=False),
        raising=False,
    )

    class FakeRequest:
        async def json(self):
            return {"tensor_parallel_size": 4}

    response = await api_server_ray.update_config(FakeRequest())
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert payload == {
        "error": "tensor_parallel_size mismatch",
        "requested_tensor_parallel_size": 4,
        "configured_tensor_parallel_size": 2,
    }
    fake_pool.update_config.assert_not_called()


@pytest.mark.asyncio
async def test_update_clients_restarts_when_replica_order_changes(monkeypatch):
    started = {}

    def fake_start_engine(clients):
        started["clients"] = clients

    async def fake_routing_loop():
        return None

    fake_pool = SimpleNamespace(clients=["r0", "r1"])

    monkeypatch.setattr(api_server_ray, "request_pool", fake_pool)
    monkeypatch.setattr(api_server_ray, "engine_actors", [])
    monkeypatch.setattr(api_server_ray, "engine_tasks", [])
    monkeypatch.setattr(api_server_ray, "routing_loop_task", None)
    monkeypatch.setattr(api_server_ray, "start_engine", fake_start_engine)
    monkeypatch.setattr(
        api_server_ray,
        "routing_loop_with_error_monitoring",
        fake_routing_loop,
    )

    class FakeRequest:
        async def json(self):
            return {"clients": "r1,r0"}

    response = await api_server_ray.update_clients(FakeRequest())

    assert response.status_code == 200
    assert started["clients"] == ["r1", "r0"]
    assert fake_pool.clients == ["r1", "r0"]


@pytest.mark.asyncio
async def test_update_clients_does_not_restart_when_clients_match(monkeypatch):
    started = {"called": False}
    ready_waits = []

    def fake_start_engine(clients):
        started["called"] = True

    class FakeWaitHandle:
        async def remote(self):
            ready_waits.append("ready")
            return True

    fake_pool = SimpleNamespace(clients=["r0", "r1"])

    monkeypatch.setattr(api_server_ray, "request_pool", fake_pool)
    monkeypatch.setattr(
        api_server_ray,
        "engine_actors",
        [SimpleNamespace(engine_actor=SimpleNamespace(wait_until_ready=FakeWaitHandle()))],
    )
    monkeypatch.setattr(api_server_ray, "engine_tasks", [])
    monkeypatch.setattr(api_server_ray, "routing_loop_task", None)
    monkeypatch.setattr(api_server_ray, "start_engine", fake_start_engine)

    class FakeRequest:
        async def json(self):
            return {"clients": "0-1"}

    response = await api_server_ray.update_clients(FakeRequest())

    assert response.status_code == 200
    assert started["called"] is False
    assert ready_waits == ["ready"]
    assert fake_pool.clients == ["r0", "r1"]


@pytest.mark.asyncio
async def test_update_config_waits_for_ready_before_fanout(monkeypatch):
    ready_waits = []
    updated = []

    class FakeWaitHandle:
        def __init__(self, actor_id):
            self.actor_id = actor_id

        async def remote(self):
            ready_waits.append(self.actor_id)
            return True

    class FakeUpdateHandle:
        def __init__(self, actor_id):
            self.actor_id = actor_id

        async def remote(self, request_json):
            updated.append((self.actor_id, request_json))
            return None

    fake_pool = SimpleNamespace(
        clients=["r0", "r1"],
        router=SimpleNamespace(update_json=lambda request_json, _: request_json),
        enable_rescheduling=False,
        update_config=MagicMock(),
    )

    monkeypatch.setattr(api_server_ray, "request_pool", fake_pool)
    monkeypatch.setattr(
        api_server_ray,
        "engine_actors",
        [
            SimpleNamespace(engine_actor=SimpleNamespace(
                wait_until_ready=FakeWaitHandle(0),
                update_config=FakeUpdateHandle(0),
            )),
            SimpleNamespace(engine_actor=SimpleNamespace(
                wait_until_ready=FakeWaitHandle(1),
                update_config=FakeUpdateHandle(1),
            )),
        ],
    )
    monkeypatch.setattr(
        api_server_ray,
        "args",
        SimpleNamespace(tensor_parallel_size=1, mock_connector=False),
        raising=False,
    )

    class FakeRequest:
        async def json(self):
            return {
                "n_devices": 2,
                "routing_policy": "round_robin",
                "routing_kwargs": {},
                "tensor_parallel_size": 1,
            }

    response = await api_server_ray.update_config(FakeRequest())

    assert response.status_code == 200
    assert sorted(ready_waits) == [0, 1]
    assert [actor_id for actor_id, _ in updated] == [0, 1]
    fake_pool.update_config.assert_called_once()


def test_start_engine_uses_tensor_parallel_gpu_reservations(monkeypatch):
    queue_sizes = []
    launches = []
    ready_waits = []

    class FakeQueue:
        def __init__(self, maxsize):
            queue_sizes.append(maxsize)

    class FakeActorHandle:
        def __init__(self, actor_id):
            self.actor_id = actor_id
            self.wait_until_ready = self

        def remote(self):
            ready_waits.append(self.actor_id)
            return f"ready-{self.actor_id}"

    class FakeEngineWorker:
        def options(self, **kwargs):
            launches.append(("options", kwargs))

            class FakeOptions:
                def remote(self_inner, *args, **remote_kwargs):
                    launches.append(("remote", args, remote_kwargs))
                    actor_id = len([kind for kind, *_ in launches if kind == 'remote'])
                    return FakeActorHandle(actor_id)

            return FakeOptions()

    monkeypatch.setattr(api_server_ray, "args", SimpleNamespace(
        log_level="INFO",
        ray_address=None,
        tensor_parallel_size=2,
        vllm_port_base=31000,
        vllm_port_stride=32,
        worker_env=[],
        mock_engine=False,
        model_name="demo-model",
        mock_connector=False,
    ), raising=False)
    monkeypatch.setattr(api_server_ray.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(api_server_ray.ray, "get", lambda obj: obj)
    monkeypatch.setattr(api_server_ray, "ExecPlanBus", SimpleNamespace(
        remote=lambda: "execplan-bus"))
    monkeypatch.setattr(api_server_ray, "RayQueue", FakeQueue)
    monkeypatch.setattr(api_server_ray, "EngineWorker", FakeEngineWorker())
    monkeypatch.setattr(api_server_ray, "execplan_bus_actor", None)
    monkeypatch.setattr(api_server_ray, "engine_actors", None)
    monkeypatch.setattr(api_server_ray, "engine_tasks", None)

    api_server_ray.start_engine(["r0", "r1", "r2"])

    option_calls = [entry[1] for entry in launches if entry[0] == "options"]
    remote_calls = [
        (entry[1], entry[2]) for entry in launches if entry[0] == "remote"
    ]

    assert option_calls == [
        {"num_gpus": 2},
        {"num_gpus": 2},
        {"num_gpus": 2},
    ]
    assert queue_sizes == [8192, 8192, 8192]
    assert len(remote_calls) == 3
    for replica_id, (args, kwargs) in enumerate(remote_calls):
        assert args == ("demo-model", False)
        assert kwargs["device_id"] == replica_id
        assert kwargs["tensor_parallel_size"] == 2
        assert kwargs["vllm_port"] == 31000 + replica_id * 32
        assert kwargs["execplan_bus"] == "execplan-bus"
        assert isinstance(kwargs["output_queue"], FakeQueue)
    assert ready_waits == [1, 2, 3]
    assert len(api_server_ray.engine_actors) == 3
