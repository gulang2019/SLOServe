from __future__ import annotations

import ast
import csv
import importlib.util
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from SLOsServe.beam import (BeamProfile, BeamScheduler, PrefillRequest,
                           SimulatedGpu)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "3rdparty/BEAM/vllm/v1/core/sched"
A100_PROFILE = (ROOT / "3rdparty/BEAM/benchmarks/energy/offline_profile_results"
                / "dvfs_profile_NVIDIA A100 80GB PCIe_Qwen_Qwen2.5-14B_tp1_pp1_one.csv")
needs_upstream = pytest.mark.skipif(
    not (UPSTREAM / "scheduler.py").exists() or not A100_PROFILE.exists(),
    reason="BEAM checkout not available")


@pytest.fixture
def profile_path(tmp_path):
    # Two clocks; sizes < 128 are decode-only rows, as BEAM profiles them.
    path = tmp_path / "dvfs_profile_NVIDIA A100 80GB PCIe_test_tp1_pp1_one.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clock", "batch_size", "decode_tokens", "total_ctx_len",
                         "time_taken", "energy_consumption", "num_trials",
                         "valid"])
        for clock, time_scale, watts in [(1200, 1, 150), (600, 2, 50)]:
            for batch in (1, 16, 128, 256, 512):
                for ctx in (2048, 40000):
                    latency = (0.001 + batch * 0.00001) * time_scale
                    if ctx == 40000:
                        latency *= 1.2
                    writer.writerow([clock, batch, 60, ctx, latency,
                                     latency * watts * 1000, 25, 1])
    return str(path)


# ---- Parity with upstream BEAM -------------------------------------------

def _upstream_energy_simulator(path: Path):
    spec = importlib.util.spec_from_file_location(
        "beam_upstream_energy_model", UPSTREAM / "energy_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnergySimulator(str(path), num_pp=1, num_tp=1,
                                  gpu_name="NVIDIA A100 80GB PCIe",
                                  model_name="Qwen_Qwen2.5-14B")


def _upstream_scheduler_methods(clock):
    tree = ast.parse((UPSTREAM / "scheduler.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    wanted = {"beam_schedule_tbt_based_s1", "beam_schedule_tbt_based_s2",
              "_previous_prefill_interrupt_moderation"}
    body = [node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"math": math, "logger": Mock(), "Request": object,
                 "time": SimpleNamespace(time=lambda: clock[0],
                                         monotonic=lambda: clock[0])}
    exec(compile(ast.Module(body=body, type_ignores=[]),
                 str(UPSTREAM / "scheduler.py"), "exec"), namespace)
    return namespace


@needs_upstream
def test_lookups_match_upstream_energy_simulator():
    upstream = _upstream_energy_simulator(A100_PROFILE)
    ours = BeamProfile(str(A100_PROFILE))
    assert ours.clocks == [int(c) for c in upstream.available_clocks]
    assert ours.chunk_sizes == upstream.available_chunk_sizes
    assert ours.default_clock_high == upstream.default_clock_high
    assert ours.default_clock == upstream.default_clock
    assert ours.scheduling_overhead == upstream.scheduling_overhead
    assert (ours.scheduling_overhead_decode_only
            == upstream.scheduling_overhead_decode_only)
    rng = random.Random(0)
    for clock in ours.clocks:
        for size in list(range(0, 520, 7)) + [600, 900, 1100]:
            assert ours.get_energy(size, clock) == pytest.approx(
                upstream.get_energy(size, clock))
            for ctx in (None, 0, 2048, rng.uniform(0, 60000), 40000):
                assert ours.get_time_taken(size, clock, ctx) == pytest.approx(
                    upstream.get_time_taken(size, clock, ctx))


@needs_upstream
def test_s1_s2_decisions_match_upstream():
    clock = [1000.0]
    methods = _upstream_scheduler_methods(clock)
    simulator = _upstream_energy_simulator(A100_PROFILE)
    profile = BeamProfile(str(A100_PROFILE))
    rng = random.Random(1)
    compared = 0
    for _ in range(1500):
        now = clock[0] = rng.uniform(1000, 2000)
        num_running = rng.choice([0, 1, 5, 30, 90, 127, 200, 400, 700])
        state = dict(
            max_num_scheduled_tokens=rng.choice([128, 256, 320, 512, 8192]),
            last_prefill_timestamp=now - rng.uniform(0, 2),
            last_prefill_ttft=rng.uniform(0, 2))
        tokens = rng.choice([0, 1, 50, 127, 300, 512, 513, 1500, 6000,
                             rng.randint(-100, 9000)])
        tbt_slo = rng.uniform(0.01, 0.15)
        ttft_slo = rng.uniform(0.02, 3.0)
        arrivals = [now - rng.uniform(0, 2) for _ in range(rng.randint(0, 3))]
        dvfs_only = rng.random() < 0.1

        upstream = SimpleNamespace(
            energy_simulator=simulator, pp_size=1,
            running=[SimpleNamespace(num_tokens_with_spec=0)] * num_running,
            **state)
        upstream._previous_prefill_interrupt_moderation = (
            lambda ttft, latency: methods["_previous_prefill_interrupt_moderation"](
                upstream, ttft, latency))
        try:
            chunk, s1_clock = methods["beam_schedule_tbt_based_s1"](
                upstream, tbt_slo, ttft_slo, tokens, dvfs_only,
                [SimpleNamespace(arrival_time=a) for a in arrivals])
        except ZeroDivisionError:
            continue  # upstream crashes; the port skips that candidate
        max_queuing = max([now - a for a in arrivals] + [0.0])
        ours = BeamScheduler(profile, Mock(), now=0.0, initial_chunk=0,
                             use_s1_dvfs_only=dvfs_only)
        ours.max_num_scheduled_tokens = state["max_num_scheduled_tokens"]
        ours.last_prefill_timestamp = state["last_prefill_timestamp"]
        ours.last_prefill_ttft = state["last_prefill_ttft"]
        decision = ours.s1(tbt_slo=tbt_slo, ttft_budget=ttft_slo - max_queuing,
                           num_waiting_prefill_tokens=tokens,
                           num_running=num_running,
                           max_queuing_time=max_queuing, now=now)
        assert (decision["chunk_size"], decision["clock_mhz"]) == (chunk, s1_clock)
        assert ours.last_prefill_ttft == pytest.approx(upstream.last_prefill_ttft)
        assert ours.last_prefill_timestamp == upstream.last_prefill_timestamp

        ctx = rng.uniform(0, 80000)
        upstream.running = [SimpleNamespace(num_tokens_with_spec=ctx / max(1, num_running))
                            ] * num_running
        _, s2_clock = methods["beam_schedule_tbt_based_s2"](
            upstream, tbt_slo, num_running)
        assert ours.s2(tbt_slo=tbt_slo, num_running=num_running,
                       total_ctx_len=sum(r.num_tokens_with_spec
                                         for r in upstream.running)
                       )["clock_mhz"] == s2_clock
        compared += 1
    assert compared > 1000


# ---- Profile, ground truth and actuators ---------------------------------

def test_profile_units_and_ground_truth(profile_path):
    p = BeamProfile(profile_path)
    assert p.clocks == [1200, 600]  # file order, as upstream iterates
    assert p.gpu_name == "NVIDIA A100 80GB PCIe"
    assert p.default_clock_high == 1200
    # Decisions round batch sizes up; ground truth interpolates.
    assert p.get_time_taken(64, 600, 2048) == pytest.approx(
        (0.001 + 128 * 0.00001) * 2)
    latency, energy = p.simulate(1, 2048, 600)
    assert latency == pytest.approx(0.00202)
    assert energy == pytest.approx(0.00202 * 50)  # mJ on disk -> J
    mid, _ = p.simulate(1, (2048 + 40000) / 2, 600)
    assert mid == pytest.approx(0.00202 * 1.1)
    between, _ = p.simulate(72, 2048, 600)
    assert p.simulate(16, 2048, 600)[0] < between < p.simulate(128, 2048, 600)[0]


@pytest.mark.skipif(not A100_PROFILE.exists(), reason="profile not available")
def test_fit_matches_simulated_gpu():
    from SLOsServe.perf_model import PerfModel

    profile = BeamProfile(str(A100_PROFILE))
    slow = PerfModel("Qwen/Qwen2.5-7B-Instruct",
                     profile.fit_perf_model_params(810))
    fast = PerfModel("Qwen/Qwen2.5-7B-Instruct",
                     profile.fit_perf_model_params(1410))
    gpu = SimulatedGpu(profile, clock=810, switch_delay_s=0,
                       prefill_overhead_s=0.012, decode_overhead_s=0.003)
    for batch in ([(0, 256)], [(2048, 1)] * 16, [(0, 200)] + [(1000, 1)] * 8):
        truth = gpu.run_batch(batch, 0.0).duration_s
        assert slow.get_batch_time(batch) == pytest.approx(truth, rel=0.15)
        assert slow.get_batch_time(batch) > fast.get_batch_time(batch)
    with pytest.raises(ValueError, match="absent"):
        profile.fit_perf_model_params(905)


def test_simulated_gpu_switches_clock_mid_batch(profile_path):
    profile = BeamProfile(profile_path)
    gpu = SimulatedGpu(profile, clock=1200, switch_delay_s=0.001,
                       prefill_overhead_s=0.012, decode_overhead_s=0.003)
    fast, fast_energy = profile.simulate(512, 0, 1200)
    slow, slow_energy = profile.simulate(512, 0, 600)
    gpu.set_clock(600, 0.0)
    run = gpu.run_batch([(0, 512)], 0.0)
    done_fast = 0.001 / fast
    assert run.clock_mhz == 1200 and run.end_clock_mhz == 600
    assert run.gpu_time_s == pytest.approx(0.001 + (1 - done_fast) * slow)
    assert run.energy_j == pytest.approx(done_fast * fast_energy
                                         + (1 - done_fast) * slow_energy)
    assert run.duration_s == pytest.approx(run.gpu_time_s + 0.012)
    decode = gpu.run_batch([(2048, 1)], 1.0)
    assert decode.duration_s == pytest.approx(decode.gpu_time_s + 0.003)


def test_nvml_actuator_locks_on_a_thread_and_releases(monkeypatch):
    from SLOsServe.beam import NvmlClock
    nvml = Mock()
    monkeypatch.setattr("vllm.utils.import_pynvml", lambda: nvml)
    actuator = NvmlClock(3)
    actuator.set_clock(1200, 0.0)
    deadline = time.time() + 2
    while not nvml.nvmlDeviceSetGpuLockedClocks.called and time.time() < deadline:
        time.sleep(0.001)
    nvml.nvmlDeviceSetGpuLockedClocks.assert_called_once_with(
        nvml.nvmlDeviceGetHandleByIndex.return_value, 1200, 1200)
    # Once at construction (permission check), once to release the lock.
    actuator.close()
    assert nvml.nvmlDeviceResetGpuLockedClocks.call_count == 2
    nvml.nvmlShutdown.assert_called_once()


def test_nvml_actuator_fails_fast_without_clock_permission(monkeypatch):
    from SLOsServe.beam import NvmlClock
    nvml = Mock()
    nvml.nvmlDeviceResetGpuLockedClocks.side_effect = PermissionError("root")
    monkeypatch.setattr("vllm.utils.import_pynvml", lambda: nvml)
    with pytest.raises(PermissionError):
        NvmlClock(0)


# ---- Upstream state machine ----------------------------------------------

def _after(beam, *, now, scheduled_prefill=0, prefill=(), num_running=1,
           tbt_slo=1.0):
    return beam.after_schedule(
        prefill_unfinished_tokens=scheduled_prefill,
        has_prefill_requests=bool(prefill), prefill_requests=list(prefill),
        idle_ttft_budget=10.0, num_running=num_running,
        total_ctx_len=2048.0 * num_running, tbt_slo=tbt_slo, now=now)


def test_s1_s2_on_arrival_and_decode_clock_after_five_steps(profile_path):
    actuator = Mock()
    beam = BeamScheduler(BeamProfile(profile_path), actuator, now=0.0,
                         initial_chunk=512)
    beam.add_request(300)
    assert beam.num_waiting_prefill_tokens == 300
    first = _after(beam, now=1.0, scheduled_prefill=300,
                   prefill=[PrefillRequest(0.0, 10.0)])
    assert [d["phase"] for d in first] == ["s1", "s2"]
    assert beam.num_waiting_prefill_tokens == 0
    s1, s2 = first
    assert s1["applied"] and not s2["applied"]
    actuator.set_clock.assert_called_once_with(s1["clock_mhz"], 1.0)
    # No new arrival: no S1/S2. The S2 clock waits for 5 decode-only steps.
    for step in range(4):
        assert _after(beam, now=2.0 + step) == []
    fifth = _after(beam, now=6.0)
    assert [d["phase"] for d in fifth] == ["decode_clock"]
    assert fifth[0]["clock_mhz"] == s2["clock_mhz"]
    assert _after(beam, now=7.0) == []  # only on the 5th, as upstream


def test_s1_fallbacks_follow_upstream(profile_path):
    beam = BeamScheduler(BeamProfile(profile_path), Mock(), now=0.0,
                         initial_chunk=512)
    no_tbt = beam.s1(tbt_slo=1e-6, ttft_budget=100, num_waiting_prefill_tokens=4000,
                     num_running=1, max_queuing_time=0, now=1.0)
    assert (no_tbt["chunk_size"], no_tbt["clock_mhz"]) == (128, 1200)
    assert not no_tbt["feasible"]
    beam.last_prefill_ttft = 0
    neither = beam.s1(tbt_slo=1e-6, ttft_budget=-1, num_waiting_prefill_tokens=4000,
                      num_running=1, max_queuing_time=0, now=2.0)
    assert neither["chunk_size"] == 256  # TTFT fallback takes precedence


def test_s1_keeps_adjacent_chunks_during_a_prefill(profile_path):
    beam = BeamScheduler(BeamProfile(profile_path), Mock(), now=0.0,
                         initial_chunk=128)
    beam.last_prefill_timestamp, beam.last_prefill_ttft = 0.0, 5.0
    decision = beam.s1(tbt_slo=1, ttft_budget=100, num_waiting_prefill_tokens=4000,
                       num_running=0, max_queuing_time=0, now=1.0)
    assert decision["candidate_chunk_sizes"] == [128, 256]


# ---- Scheduler wiring -----------------------------------------------------

@pytest.fixture
def scheduler(profile_path):
    from vllm.v1.core.sched.scheduler_adm_ctrl import SchedulerAdmCtrl
    s = SchedulerAdmCtrl.__new__(SchedulerAdmCtrl)
    s.scheduler_config = SimpleNamespace(
        scheduling_policy="beam", max_num_batched_tokens=8192,
        beam_profile_path=profile_path, beam_backend="mock",
        mock_profile_clock_mhz=None, beam_device_id=None,
        beam_use_s1=True, beam_use_s2=True, beam_use_s1_dvfs_only=False,
        beam_prefill_overhead_s=0.012, beam_decode_overhead_s=0.003,
        beam_switch_delay_s=0.01, slo_tpot=0.05)
    s.parallel_config = SimpleNamespace(tensor_parallel_size=1,
                                        pipeline_parallel_size=1,
                                        data_parallel_size=1)
    s.vllm_config = SimpleNamespace(is_simulation=False)
    s.timer = SimpleNamespace(current_time=lambda: 100.0)
    s._profile_events = []
    s._configure_beam()
    return s


def test_mock_beam_uses_simulated_gpu_never_nvml(scheduler, monkeypatch):
    monkeypatch.setattr("SLOsServe.beam.NvmlClock",
                        Mock(side_effect=AssertionError("GPU touched")))
    scheduler._configure_beam()
    assert scheduler._beam_actuator is scheduler.sim_gpu
    assert scheduler.sim_gpu.clock == 1200  # boost clock before any lock
    assert scheduler.max_num_scheduled_tokens == 512  # profile coverage
    assert not scheduler.scheduler_config.allow_rejection


def test_fixed_clock_baseline_gets_simulated_gpu(scheduler):
    scheduler.scheduler_config.scheduling_policy = "atfc"
    scheduler.scheduler_config.mock_profile_clock_mhz = 600
    scheduler._configure_beam()
    assert scheduler._beam is None and scheduler._beam_actuator is None
    assert scheduler.sim_gpu.clock == 600
    scheduler.scheduler_config.mock_profile_clock_mhz = None
    scheduler._configure_beam()
    assert scheduler.sim_gpu is None


def test_beam_atfc_keeps_batch_and_obeys_planner_budget(scheduler):
    scheduler.scheduler_config.scheduling_policy = "beam_atfc"
    scheduler._configure_beam()
    scheduler.requests = {"r": SimpleNamespace(num_computed_tokens=2048,
                                               num_prompt_tokens=10)}
    scheduler.atfc_planner = SimpleNamespace(
        _perf_model=SimpleNamespace(get_batch_time=lambda batch: 0.0045))
    scheduled = {"r": 1}
    scheduler._beam_after_schedule(scheduled)
    assert scheduled == {"r": 1}
    event = scheduler._profile_events[-1]["extra_args"]
    assert event["policy"] == "beam_atfc" and event["backend"] == "mock"
    assert event["clock_mhz"] == 1200 and event["feasible"]


def test_real_uses_explicit_device_mapping(scheduler, monkeypatch):
    import vllm.v1.core.sched.scheduler_adm_ctrl as module
    actuator = Mock()
    factory = Mock(return_value=actuator)
    monkeypatch.setattr("SLOsServe.beam.NvmlClock", factory)
    monkeypatch.setattr(module, "current_platform", SimpleNamespace(
        device_id_to_physical_device_id=lambda device: device + 4))
    scheduler.scheduler_config.beam_backend = "real"
    scheduler.scheduler_config.beam_device_id = 1
    scheduler._configure_beam()
    factory.assert_called_once_with(5)
    assert scheduler.sim_gpu is None
    scheduler.scheduler_config.scheduling_policy = "atfc"
    scheduler._configure_beam()
    actuator.close.assert_called_once()
    assert scheduler._beam is None


def test_rejects_unsupported_parallelism(scheduler):
    scheduler.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1, PP=1"):
        scheduler._configure_beam()


@pytest.mark.parametrize("policy", ["beam", "beam_atfc", "atfc"])
def test_mock_engine_runs_on_profiled_gpu(policy, profile_path, monkeypatch):
    import asyncio
    from SLOsServe.router.mock_engine import MockEngineCore
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    # Exercise the actor's implementation locally, without starting Ray or GPUs.
    monkeypatch.setattr("SLOsServe.beam.NvmlClock",
                        Mock(side_effect=AssertionError("GPU touched")))
    post_init = VllmConfig.__post_init__
    outputs = []
    try:
        engine = MockEngineCore.__ray_metadata__.modified_class(
            model_name="Qwen/Qwen2.5-7B-Instruct", mock_connector=True,
            output_queue=SimpleNamespace(put_nowait=outputs.extend), device_id=0)
    finally:
        VllmConfig.__post_init__ = post_init

    async def run():
        await engine.update_config({
            "scheduling_policy": policy, "beam_profile_path": profile_path,
            "mock_profile_clock_mhz": 600,
            # The mock worker must override even an explicit real setting.
            "beam_backend": "real", "max_num_batched_tokens": 512,
            "enable_chunked_prefill": True, "chunked_prefill_enabled": True,
            "slo_tpot": 0.1, "max_decoding_length": 8,
            "admission_max_decoding_length": 8, "admission_mode": "arrival",
            "allow_rejection": policy != "beam", "routing_overhead": 0,
        })
        now = time.time()
        req = Request(
            request_id="beam-smoke", prompt_token_ids=list(range(1024)),
            multi_modal_kwargs=None, multi_modal_hashes=None,
            multi_modal_placeholders=None, pooling_params=None,
            eos_token_id=None, arrival_time=now,
            block_hasher=engine.request_block_hasher,
            sampling_params=SamplingParams(max_tokens=8, ignore_eos=True,
                extra_args={"slo_ttft": 10, "slo_tpot": 0.1,
                            "router_arrival_time": now}))
        assert engine.scheduler.add_request(req)
        engine.n_arrived_reqs = 1
        task = asyncio.create_task(engine.loop())
        try:
            async def completed():
                while engine.n_finished_reqs < 1:
                    if task.done():
                        await task
                        raise AssertionError("mock loop stopped early")
                    await asyncio.sleep(0.005)
            await asyncio.wait_for(completed(), timeout=10)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        batches = [e for e in engine._profile_events if e["event_type"] == "batch"]
        assert batches and all(e["sim_energy_j"] > 0 for e in batches)
        assert all(e["sim_idle_power_w"] == 70.0 for e in batches)
        assert sum(len(o.get("new_token_ids", [])) for o in outputs) == 8
        decisions = [e["extra_args"] for e in engine._profile_events
                     if e["event_type"] == "beam_decision"]
        if policy == "beam":
            assert [d["phase"] for d in decisions] == ["s1", "s2", "decode_clock"]
            assert batches[0]["sim_clock_mhz"] == 1200
        elif policy == "beam_atfc":
            assert {d["phase"] for d in decisions} == {"batch"}
        else:
            assert decisions == []
            assert {e["sim_clock_mhz"] for e in batches} == {600}
        await engine.update_config({"scheduling_policy": "vllm",
                                    "mock_profile_clock_mhz": None})
        assert engine.scheduler._beam is None and engine.scheduler.sim_gpu is None
        engine.scheduler.shutdown()

    asyncio.run(run())


# ---- Benchmark and analysis ----------------------------------------------

def _problems(monkeypatch, tmp_path, policy, routing="round_robin"):
    from motivation import bench_api_server as bench
    from Dataset.dataset import Request
    monkeypatch.setattr(bench, "_load_trace_inputs", lambda *a, **kw: (
        [Request(input_length=32, output_length=8)], [0.0], [("test", "test")]))
    return bench.build_problems(
        model_name="Qwen/Qwen2.5-7B-Instruct", trace="azure_chat_23",
        ttft_slo_scale=3, slo_tpot=0.1, profit="constant",
        scheduling_policy=policy, routing_policy=routing,
        n_device=1, tensor_parallel_size=1, window="0:1", load_scale=1,
        experiment_dir=str(tmp_path))


@pytest.mark.parametrize("policy", ["beam", "beam_atfc"])
def test_benchmark_requires_shared_profiled_gpu(policy, profile_path, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("SLOSSERVE_BEAM_PROFILE", profile_path)
    monkeypatch.delenv("SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ", raising=False)
    with pytest.raises(ValueError, match="SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ"):
        _problems(monkeypatch, tmp_path, policy)
    monkeypatch.setenv("SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ", "1200")
    kwargs = _problems(monkeypatch, tmp_path, policy)[0].scheduling_kwargs
    assert kwargs["scheduling_policy"] == policy
    assert kwargs["beam_profile_path"] == profile_path
    assert kwargs["mock_profile_clock_mhz"] == 1200
    assert kwargs["enable_admission"] == (policy == "beam_atfc")
    assert kwargs["max_num_batched_tokens"] == 512


def test_benchmark_shared_mock_profile_for_atfc(profile_path, tmp_path,
                                                 monkeypatch):
    monkeypatch.setenv("SLOSSERVE_BEAM_PROFILE", profile_path)
    monkeypatch.setenv("SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ", "1200")
    problem = _problems(monkeypatch, tmp_path, "atfc", "slosserve_planner")[0]
    assert problem.scheduling_kwargs["beam_profile_path"] == profile_path
    assert problem.scheduling_kwargs["mock_profile_clock_mhz"] == 1200
    assert problem.scheduling_kwargs["max_num_batched_tokens"] == 512
    assert problem.routing_kwargs["beam_profile_path"] == profile_path
    assert problem.routing_kwargs["mock_profile_clock_mhz"] == 1200


def test_profiled_energy_replaces_state_based_estimate():
    from motivation.events_analysis import Batch, _compute_state_based_power_series

    def batch(device, start, elapsed, gpu_time, energy):
        return Batch(event_type="batch", timestamp=start + elapsed,
                     device_id=device, batch_id=0, req_ids=["r"],
                     num_computed_tokens=[0], num_scheduled_tokens={"r": 1},
                     elapsed=elapsed, scheduling_overhead=0.0,
                     sim_gpu_time_s=gpu_time, sim_energy_j=energy,
                     sim_idle_power_w=60.0)

    events = [batch(0, 0.0, 0.5, 0.4, 100.0), batch(1, 1.0, 1.0, 1.0, 250.0)]
    summary = _compute_state_based_power_series(
        events, n_device=2, window_size=1.0, start_time=0.0, end_time=2.0)
    assert summary["source"] == "profile"
    idle = 60.0 * 2.0 * 2
    extra = (100.0 - 60.0 * 0.4) + (250.0 - 60.0 * 1.0)
    assert summary["total_energy_joules"] == pytest.approx(idle + extra)
    assert summary["per_device_energy_joules"][1] == pytest.approx(120.0 + 190.0)


def test_rr_shard_offset_selects_later_shards():
    from motivation.bench_api_server import _slice_rr_workload
    requests = list(range(12))
    kept, arrivals, indices = _slice_rr_workload(
        requests, [i * 0.5 for i in requests], 4, 1, shard_offset=3)
    assert indices == [3, 7, 11] and kept == indices
    assert arrivals == [1.5, 3.5, 5.5]  # original timestamps
    with pytest.raises(ValueError, match="exceeds"):
        _slice_rr_workload(requests, [0.0] * 12, 4, 2, shard_offset=3)


def test_benchmark_beam_on_real_gpus_keeps_default_slos(profile_path, tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("SLOSSERVE_BEAM_PROFILE", profile_path)
    monkeypatch.delenv("SLOSSERVE_MOCK_PROFILE_CLOCK_MHZ", raising=False)
    monkeypatch.setenv("SLOSSERVE_BEAM_BACKEND", "real")
    kwargs = _problems(monkeypatch, tmp_path, "beam")[0].scheduling_kwargs
    assert kwargs["beam_profile_path"] == profile_path
    assert "mock_profile_clock_mhz" not in kwargs
