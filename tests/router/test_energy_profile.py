import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from SLOsServe.router import api_server_ray
from motivation.energy_measure import EnergyHistoryRecorder


def test_parse_energy_csv_sorts_gpu_columns_numerically(tmp_path):
    csv_path = tmp_path / "worker.energy.csv"
    csv_path.write_text(
        "ts,J_gpu10,J_gpu2,J_gpu0,J_total,W_gpu10,W_gpu2,W_gpu0,W_total,"
        "MHz_gpu10,MHz_gpu2,MHz_gpu0\n"
        "1.0,1.5,2.5,3.5,7.5,15,25,35,75,1010,1020,1030\n")

    events, per_gpu, total = api_server_ray._parse_energy_csv(str(csv_path))

    assert per_gpu == [3.5, 2.5, 1.5]
    assert total == 7.5
    assert events == [
        {
            "event_type": "energy",
            "timestamp": 1.0,
            "device_id": 0,
            "energy": 3.5,
            "power": 35.0,
            "mhz": 1030.0,
        },
        {
            "event_type": "energy",
            "timestamp": 1.0,
            "device_id": 1,
            "energy": 2.5,
            "power": 25.0,
            "mhz": 1020.0,
        },
        {
            "event_type": "energy",
            "timestamp": 1.0,
            "device_id": 2,
            "energy": 1.5,
            "power": 15.0,
            "mhz": 1010.0,
        },
    ]


def test_engine_worker_energy_start_is_deferred_until_explicit_start():
    worker_cls = api_server_ray.EngineWorker.__ray_actor_class__
    worker = worker_cls.__new__(worker_cls)
    worker.engine = SimpleNamespace(update_config=AsyncMock())
    worker._profile_events = [{"event_type": "stale"}]
    worker._energy_store_prefix = None
    worker._energy_profiler = SimpleNamespace(
        stop=MagicMock(),
        restart=MagicMock(),
    )

    asyncio.run(worker_cls.update_config(worker, {
        "store_prefix": "/shared/run-a",
    }))

    worker.engine.update_config.assert_awaited_once()
    assert worker._profile_events == []
    worker._energy_profiler.stop.assert_called_once_with()
    worker._energy_profiler.restart.assert_not_called()
    assert worker._energy_store_prefix == "/shared/run-a"

    started = asyncio.run(worker_cls.start_energy_profiling(worker))

    assert started is True
    worker._energy_profiler.restart.assert_called_once_with("/shared/run-a")


def test_dump_profile_events_aggregates_worker_energy(tmp_path, monkeypatch):
    output_file = tmp_path / "profile_events.jsonl"
    admission_file = tmp_path / "admission_history.jsonl"
    worker_event_file = tmp_path / "profile_events.device0.jsonl"
    energy_csv = tmp_path / "worker0.energy.csv"
    worker_event_file.write_text(json.dumps([
        {"event_type": "batch", "timestamp": 2.0, "batch_id": 7},
    ]))
    dump_remote = AsyncMock(return_value={
        "profile_events_path": str(worker_event_file),
        "profile_events": [
            {"event_type": "batch", "timestamp": 2.0, "batch_id": 7},
        ],
        "energy_csv_path": str(energy_csv),
        "per_gpu_joules": [4.5],
        "total_joules": 4.5,
        "physical_gpu_ids": [3],
    })

    fake_actor = SimpleNamespace(
        engine_actor=SimpleNamespace(
            dump_profile_events=SimpleNamespace(remote=dump_remote)))
    fake_pool = SimpleNamespace(
        clients=["worker-0"],
        _profile_events=[{"event_type": "arrival", "timestamp": 1.0}],
        admission_history=[{"request_id": "r0"}],
        sync=AsyncMock(),
    )

    class FakeRequest:
        async def json(self):
            return {
                "filename": str(output_file),
                "admission_filename": str(admission_file),
                "timeout": 3.0,
            }

    monkeypatch.setattr(api_server_ray, "engine_actors", [fake_actor])
    monkeypatch.setattr(api_server_ray, "request_pool", fake_pool)
    monkeypatch.setattr(
        api_server_ray,
        "_build_worker_dump_path",
        lambda filename, device_id: str(worker_event_file),
    )

    response = asyncio.run(api_server_ray.dump_profile_events(FakeRequest()))
    payload = json.loads(response.body)

    fake_pool.sync.assert_awaited_once_with(timeout=3.0)
    dump_remote.assert_awaited_once_with(str(worker_event_file))
    assert payload["energy_consumption"] == 4.5
    assert payload["per_gpu_energy_consumption"] == [4.5]
    assert payload["energy_csv_files"] == [str(energy_csv)]
    assert payload["physical_gpu_ids"] == [[3]]
    assert payload["worker_event_files"] == [str(worker_event_file)]

    dumped_events = json.loads(output_file.read_text())
    assert dumped_events == [
        {"event_type": "arrival", "timestamp": 1.0},
        {"event_type": "batch", "timestamp": 2.0, "batch_id": 7, "device_id": 0},
    ]
    assert json.loads(admission_file.read_text()) == [{"request_id": "r0"}]


def test_energy_history_recorder_stop_flushes_last_sample():
    recorder = EnergyHistoryRecorder(meter=SimpleNamespace(), csv_path=None)
    recorder._last_t = 1.0
    recorder._last_J = [0.0]

    flushed = []

    def fake_sample_once():
        flushed.append(True)

    recorder._sample_once = fake_sample_once

    recorder.stop()

    assert flushed == [True]
