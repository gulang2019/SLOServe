"""BEAM's S1/S2 DVFS scheduler, ported to SLOServe engines (TP=PP=1).

Upstream: 3rdparty/BEAM/vllm/v1/core/sched/scheduler.py (S1/S2 state machine)
and energy_model.py (EnergySimulator). BeamProfile and BeamScheduler follow
that code line by line, including its lookup rules and tie-breaking; the
docstrings of each method list the few places where SLOServe's interface
forces a different input. SimulatedGpu is the ground truth for mock engines:
it is not used for any decision.

Profile energy is in mJ on disk and J here.
"""
from __future__ import annotations

import csv
import math
import threading
from dataclasses import dataclass
from pathlib import Path

NUM_CONSECUTIVE_DECODE_ONLY_TO_SET_CLOCK = 5
CHUNK_SIZES = (128, 256, 320, 384, 448, 512)


def gpu_name_from_profile_path(path: str) -> str:
    """dvfs_profile_<GPU name>_<model>_tp<N>_pp<M>_one.csv -> GPU name."""
    name = Path(path).name
    prefix = "dvfs_profile_"
    return name[len(prefix):].split("_", 1)[0] if name.startswith(prefix) else ""


class BeamProfile:
    """BEAM's EnergySimulator lookups plus a ground-truth model for mocks."""

    def __init__(self, path: str, gpu_name: str | None = None):
        self.path = str(Path(path).resolve())
        self.gpu_name = (gpu_name if gpu_name is not None
                         else gpu_name_from_profile_path(self.path))
        self.min_clock = 800 if "A6000" in self.gpu_name else 500
        self.scheduling_overhead = 0.012  # upstream: "Conservative estimate."
        self.scheduling_overhead_decode_only = 0.003
        self.dvfs_delay = 0.01

        rows = []
        with open(self.path, newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append((int(row["clock"]), int(row["batch_size"]),
                             float(row["total_ctx_len"]),
                             float(row["time_taken"]),
                             float(row["energy_consumption"]) / 1000.0))
        if not rows:
            raise ValueError(f"BEAM profile {self.path} has no rows")
        # pandas unique() order: upstream iterates clocks in file order and
        # keeps the first strict minimum, so ties resolve identically.
        self.clocks = [c for c in dict.fromkeys(r[0] for r in rows)
                       if c >= self.min_clock]
        if not self.clocks:
            raise ValueError(f"BEAM profile {self.path} has no usable clock")
        # profiled_batch_sizes spans every clock, as upstream.
        self.batch_sizes = sorted({r[1] for r in rows})
        self._samples: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for clock, size, ctx, latency, energy in rows:
            if clock >= self.min_clock:
                self._samples.setdefault((clock, size), []).append(
                    (ctx, latency, energy))
        self.batch_energy_map: dict[int, dict[int, float]] = {}
        self.batch_time_map: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        for (clock, size), samples in self._samples.items():
            self.batch_energy_map.setdefault(clock, {})[size] = max(
                s[2] for s in samples)
            # min()/max() return the first extreme, like idxmin()/idxmax().
            lo = min(samples, key=lambda s: s[0])
            hi = max(samples, key=lambda s: s[0])
            self.batch_time_map.setdefault(clock, {})[size] = (
                lo[0], hi[0], lo[1], hi[1])
        self.chunk_sizes = [c for c in CHUNK_SIZES if c in self.batch_sizes]
        if not self.chunk_sizes:
            raise ValueError("BEAM profile needs a chunk size in 128..512")
        high, default = (1410, 1050) if "A100" in self.gpu_name else (1800, 1260)
        # beam-baseline.patch snaps the hard-coded defaults to profiled clocks.
        nearest = lambda c: min(self.clocks, key=lambda a: abs(a - c))
        self.default_clock_high = nearest(high)
        self.default_clock = nearest(default)

    # ---- EnergySimulator (decisions) ------------------------------------
    def find_smallest_batch_size_ge(self, target: int) -> int:
        for size in self.batch_sizes:
            if size >= target:
                return size
        raise ValueError(f"No batch size >= {target} found")

    def _find_surrounding_batch_sizes(self, target: int) -> tuple[int, int]:
        sizes = self.batch_sizes
        for i, size in enumerate(sizes):
            if size >= target:
                return (sizes[i - 1] if i > 0 else size), size
        return sizes[0], sizes[-1]

    @staticmethod
    def _interpolate_time_for_ctx_len(info, total_ctx_len) -> float:
        min_ctx, max_ctx, min_time, max_time = info
        if total_ctx_len is None:
            return max_time
        clipped = max(min_ctx, min(max_ctx, total_ctx_len))
        if max_ctx == min_ctx:
            return max_time
        alpha = (clipped - min_ctx) / (max_ctx - min_ctx)
        return min_time + alpha * (max_time - min_time)

    def _lookup(self, table, clock, size):
        if clock not in table:
            raise ValueError(f"No data for clock {clock}")
        if size not in table[clock]:
            raise ValueError(f"No data for batch size {size} at {clock} MHz")
        return table[clock][size]

    def get_energy(self, batch_size: int, clock: int) -> float:
        if batch_size <= 512:
            return self._lookup(self.batch_energy_map, clock,
                                self.find_smallest_batch_size_ge(batch_size))
        lower, upper = self._find_surrounding_batch_sizes(batch_size)
        e_lower = self._lookup(self.batch_energy_map, clock, lower)
        if lower == upper:
            return e_lower
        e_upper = self._lookup(self.batch_energy_map, clock, upper)
        alpha = (batch_size - lower) / (upper - lower)
        return e_lower + alpha * (e_upper - e_lower)

    def get_time_taken(self, batch_size: int, clock: int,
                       total_ctx_len: float | None = None) -> float:
        if batch_size <= 512:
            info = self._lookup(self.batch_time_map, clock,
                                self.find_smallest_batch_size_ge(batch_size))
            return float(self._interpolate_time_for_ctx_len(info, total_ctx_len))
        lower, upper = self._find_surrounding_batch_sizes(batch_size)
        t_lower = self._interpolate_time_for_ctx_len(
            self._lookup(self.batch_time_map, clock, lower), total_ctx_len)
        if lower == upper:
            return float(t_lower)
        t_upper = self._interpolate_time_for_ctx_len(
            self._lookup(self.batch_time_map, clock, upper), total_ctx_len)
        alpha = (batch_size - lower) / (upper - lower)
        return float(t_lower + alpha * (t_upper - t_lower))

    # ---- Ground truth (mock execution only) ------------------------------
    def _at_size(self, clock: int, size: int, ctx: float) -> tuple[float, float]:
        samples = self._samples[(clock, size)]
        lo = min(samples, key=lambda s: s[0])
        hi = max(samples, key=lambda s: s[0])
        if hi[0] == lo[0]:
            return hi[1], hi[2]
        alpha = (max(lo[0], min(hi[0], ctx)) - lo[0]) / (hi[0] - lo[0])
        return (lo[1] + alpha * (hi[1] - lo[1]),
                lo[2] + alpha * (hi[2] - lo[2]))

    def simulate(self, tokens: int, total_ctx_len: float,
                 clock: int) -> tuple[float, float]:
        """Measured (latency s, energy J) of one forward pass at `clock`.

        Linear in batch size between profiled sizes and in context between
        the measured contexts (clipped); linear extrapolation past the
        largest size. Unlike get_time_taken, this does not round batch sizes
        up, so BEAM's decisions are conservative against it, as on hardware.
        """
        sizes = [s for s in self.batch_sizes if (clock, s) in self._samples]
        if not sizes:
            raise ValueError(f"No data for clock {clock}")
        tokens = max(tokens, sizes[0])
        if tokens >= sizes[-1]:
            lower, upper = (sizes[-2], sizes[-1]) if len(sizes) > 1 else (
                sizes[-1], sizes[-1])
        else:
            i = next(i for i, s in enumerate(sizes) if s >= tokens)
            lower, upper = (sizes[i - 1] if i > 0 else sizes[i]), sizes[i]
        t_lo, e_lo = self._at_size(clock, lower, total_ctx_len)
        if lower == upper:
            return t_lo, e_lo
        t_hi, e_hi = self._at_size(clock, upper, total_ctx_len)
        alpha = (tokens - lower) / (upper - lower)
        return t_lo + alpha * (t_hi - t_lo), e_lo + alpha * (e_hi - e_lo)

    def fit_perf_model_params(self, clock: int, *,
                              prefill_overhead_s: float = 0.012,
                              decode_overhead_s: float = 0.003) -> list[float]:
        """Fit SLOServe's affine PerfModel to SimulatedGpu at a fixed clock.

        Samples cover decode-only batches and prefill chunks mixed with
        decodes, including the per-step overhead the mock charges, so ATFC
        plans against the same machine the mock executes.
        """
        import numpy as np
        from scipy.optimize import nnls

        if clock not in self.batch_time_map:
            raise ValueError(f"Clock {clock} MHz is absent from {self.path}")
        x, y = [], []
        for n_decode in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128):
            for ctx_per_req in (256, 1024, 2048, 4096):
                ctx = n_decode * ctx_per_req
                latency, _ = self.simulate(n_decode, ctx, clock)
                x.append([n_decode, n_decode, ctx, 0.0, 1.0])
                y.append(latency + decode_overhead_s)
        for chunk in (16, 32, 64, 128, 192, 256, 320, 384, 448, 512):
            for n_decode in (0, 8, 32):
                if n_decode >= chunk:
                    continue
                for ctx_per_req in (512, 2048):
                    ctx = n_decode * ctx_per_req
                    latency, _ = self.simulate(chunk, ctx, clock)
                    x.append([chunk, n_decode + 1, ctx, 0.0, 1.0])
                    y.append(latency + prefill_overhead_s)
        params, _ = nnls(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return [float(p) for p in params]


@dataclass(frozen=True)
class BatchExecution:
    duration_s: float  # forward pass plus per-step overhead
    gpu_time_s: float
    energy_j: float  # board energy over gpu_time_s
    clock_mhz: int  # clock when the batch started
    end_clock_mhz: int


class SimulatedGpu:
    """Mock GPU: profile timing and energy at the clock in effect.

    BEAM locks clocks on a background thread and does not wait for it, so a
    requested clock takes effect switch_delay_s later, possibly mid-batch.
    """

    def __init__(self, profile: BeamProfile, *, clock: int,
                 switch_delay_s: float, prefill_overhead_s: float,
                 decode_overhead_s: float):
        if clock not in profile.batch_time_map:
            raise ValueError(f"Unprofiled clock: {clock} MHz")
        if any(not math.isfinite(x) or x < 0 for x in
               (switch_delay_s, prefill_overhead_s, decode_overhead_s)):
            raise ValueError("Simulated GPU delays must be finite and >= 0")
        self.profile = profile
        self.clock = clock
        self.switch_delay_s = switch_delay_s
        self.prefill_overhead_s = prefill_overhead_s
        self.decode_overhead_s = decode_overhead_s
        self._pending: list[tuple[float, int]] = []

    def set_clock(self, clock: int, now: float) -> None:
        if clock not in self.profile.batch_time_map:
            raise ValueError(f"Unprofiled clock: {clock} MHz")
        self._pending.append((now + self.switch_delay_s, clock))

    def _settle(self, now: float) -> None:
        while self._pending and self._pending[0][0] <= now:
            self.clock = self._pending.pop(0)[1]

    def run_batch(self, batch: list[tuple[int, int]],
                  now: float) -> BatchExecution:
        """batch: (num_computed_tokens, num_scheduled_tokens) per request."""
        self._settle(now)
        start_clock = self.clock
        tokens = sum(n for _, n in batch)
        ctx = sum(c for c, _ in batch)
        t, remaining, energy = now, 1.0, 0.0
        while remaining > 0:
            latency, batch_energy = self.profile.simulate(tokens, ctx, self.clock)
            end = t + remaining * latency
            if self._pending and self._pending[0][0] < end:
                switch_at, clock = self._pending.pop(0)
                done = (switch_at - t) / latency
                energy += batch_energy * done
                remaining -= done
                t, self.clock = switch_at, clock
                continue
            energy += batch_energy * remaining
            t, remaining = end, 0.0
        has_prefill = any(n > 1 for _, n in batch)
        overhead = (self.prefill_overhead_s if has_prefill
                    else self.decode_overhead_s)
        return BatchExecution(t - now + overhead, t - now, energy,
                              start_clock, self.clock)


class NvmlClock:
    """upstream lock_gpu_clock: NVML calls on a thread, never awaited.

    Like upstream apply_dvfs, a TP group locks every one of its GPUs.
    """

    def __init__(self, physical_device_ids: list[int]):
        from vllm.utils import import_pynvml
        self.nvml = import_pynvml()
        self.nvml.nvmlInit()
        self.handles = [self.nvml.nvmlDeviceGetHandleByIndex(i)
                        for i in physical_device_ids]
        # Locks are applied on threads whose errors nobody sees; fail here,
        # synchronously, if this process may not control the clocks at all.
        for handle in self.handles:
            self.nvml.nvmlDeviceResetGpuLockedClocks(handle)
        self.locked = False

    def set_clock(self, clock: int, now: float) -> None:
        for handle in self.handles:
            threading.Thread(target=self.nvml.nvmlDeviceSetGpuLockedClocks,
                             args=(handle, clock, clock), daemon=True).start()
        self.locked = True

    def close(self) -> None:
        try:
            if self.locked:
                for handle in self.handles:
                    self.nvml.nvmlDeviceResetGpuLockedClocks(handle)
                self.locked = False
        finally:
            self.nvml.nvmlShutdown()


def cheapest_clock_within(profile: BeamProfile, *, tokens: int,
                          total_ctx_len: float, has_prefill: bool,
                          budget_s: float) -> tuple[int, bool]:
    """beam_atfc: lowest-energy clock whose BEAM estimate fits budget_s.

    Not part of upstream BEAM; returns (default_clock_high, False) when no
    clock fits.
    """
    overhead = (profile.scheduling_overhead if has_prefill
                else profile.scheduling_overhead_decode_only)
    best = None
    for clock in profile.clocks:
        latency = profile.get_time_taken(
            tokens, clock, None if has_prefill else total_ctx_len)
        energy = profile.get_energy(tokens, clock)
        if latency + overhead <= budget_s and (best is None or energy < best[0]):
            best = (energy, clock)
    return (best[1], True) if best else (profile.default_clock_high, False)


@dataclass(frozen=True)
class PrefillRequest:
    """A request scheduled with more than one token this step."""
    queuing_time_s: float  # now - arrival_time
    ttft_budget_s: float  # time left before its TTFT deadline


class BeamScheduler:
    """Upstream Scheduler's BEAM state: S1 on arrival, S2 after 5 decodes.

    SLOServe's TTFT SLO is per request. Upstream accepts a candidate when
    ttft + max_queuing_time <= ttft_slo; here ttft <= min over the step's
    prefill requests of their remaining budget (identical when every request
    shares one SLO). With no prefill request scheduled, upstream's queuing
    term is 0, so the budget is the smallest full TTFT SLO still pending.
    """

    def __init__(self, profile: BeamProfile, actuator, *, now: float,
                 initial_chunk: int, use_s1: bool = True, use_s2: bool = True,
                 use_s1_dvfs_only: bool = False):
        self.profile = profile
        self.actuator = actuator
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.use_s1_dvfs_only = use_s1_dvfs_only
        self.max_num_scheduled_tokens = initial_chunk
        self.num_waiting_prefill_tokens = 0
        self.new_request_arrived = False
        self.num_consecutive_decode_only = 0
        self.last_prefill_timestamp = now
        self.last_prefill_ttft = 0.0
        self.prefill_clock = profile.default_clock
        self.decode_clock = profile.default_clock
        self.current_clock: int | None = None  # apply_dvfs's CURRENT_CLOCK

    def add_request(self, num_prompt_tokens: int) -> None:
        self.num_waiting_prefill_tokens += num_prompt_tokens
        self.new_request_arrived = True

    def apply_dvfs(self, clock: int, now: float) -> bool:
        if self.current_clock == clock:
            return False
        self.actuator.set_clock(clock, now)
        self.current_clock = clock
        return True

    def s1(self, *, tbt_slo: float, ttft_budget: float,
           num_waiting_prefill_tokens: int, num_running: int,
           max_queuing_time: float, now: float) -> dict:
        """beam_schedule_tbt_based_s1 (PP=1, disagg_mode=False)."""
        p = self.profile
        tokens = num_waiting_prefill_tokens
        overhead = p.scheduling_overhead
        chunk_sizes = p.chunk_sizes
        # Drastic chunk-size changes during a prefill create bubbles.
        if now - self.last_prefill_timestamp < self.last_prefill_ttft:
            current = self.max_num_scheduled_tokens
            if current in p.chunk_sizes:
                i = p.chunk_sizes.index(current)
                chunk_sizes = p.chunk_sizes[max(0, i - 1):i + 2]
        if self.use_s1_dvfs_only:
            chunk_sizes = [256]

        best = None  # (energy, chunk, clock, ttft)
        has_ttft_adhering = has_tbt_adhering = False
        avg_decodes_per_chunk = num_running
        for chunk_size in chunk_sizes:
            for clock in p.clocks:
                per_chunk_energy = p.get_energy(chunk_size, clock)
                per_chunk_latency = p.get_time_taken(chunk_size, clock)
                divisor = chunk_size - avg_decodes_per_chunk
                if divisor == 0:
                    # Upstream raises ZeroDivisionError here; skip instead of
                    # stopping the engine.
                    continue
                num_chunks_floor = math.floor(tokens / divisor)
                if chunk_size >= tokens:
                    energy = p.get_energy(tokens, clock)
                    latency = p.get_time_taken(tokens, clock)
                    tbt = ttft = latency + overhead
                else:
                    leftover = tokens % chunk_size
                    leftover_energy = p.get_energy(leftover, clock)
                    leftover_latency = p.get_time_taken(leftover, clock)
                    tbt = per_chunk_latency + overhead
                    ttft = (num_chunks_floor * (per_chunk_latency + overhead)
                            + leftover_latency)
                    energy = per_chunk_energy * num_chunks_floor + leftover_energy
                ttft_adhering = ttft <= ttft_budget
                tbt_adhering = tbt <= tbt_slo
                has_ttft_adhering |= ttft_adhering
                has_tbt_adhering |= tbt_adhering
                if (ttft_adhering and tbt_adhering
                        and (best is None or energy < best[0])):
                    best = (energy, chunk_size, clock, ttft)

        self.last_prefill_timestamp = now
        if best is not None:
            energy, chunk, clock, ttft = best
            self.last_prefill_ttft = ttft + max_queuing_time
            feasible = True
        else:
            # Upstream leaves chunk_size at its last loop value when both
            # kinds of adhering choice exist but never together.
            chunk = chunk_sizes[-1]
            if not has_tbt_adhering:
                chunk = 128
            if not has_ttft_adhering:
                chunk = 256
            clock = p.default_clock_high
            energy = None
            self.last_prefill_ttft = math.ceil(tokens / chunk) * (
                p.get_time_taken(chunk, clock) + overhead)
            feasible = False
        return {"phase": "s1", "chunk_size": chunk, "clock_mhz": clock,
                "feasible": feasible, "estimated_energy_j": energy,
                "num_waiting_prefill_tokens": tokens,
                "ttft_budget_s": ttft_budget, "tbt_slo_s": tbt_slo,
                "candidate_chunk_sizes": list(chunk_sizes)}

    def s2(self, *, tbt_slo: float, num_running: int,
           total_ctx_len: float) -> dict:
        """beam_schedule_tbt_based_s2 with PP=1 (one microbatch)."""
        p = self.profile
        best = None  # (energy, clock)
        for clock in p.clocks:
            tbt = (p.get_time_taken(num_running, clock, total_ctx_len)
                   + p.scheduling_overhead_decode_only)
            energy = p.get_energy(num_running, clock)
            if tbt <= tbt_slo and (best is None or energy < best[0]):
                best = (energy, clock)
        clock = best[1] if best else p.default_clock_high
        return {"phase": "s2", "clock_mhz": clock, "feasible": best is not None,
                "estimated_energy_j": best[0] if best else None,
                "num_running": num_running, "total_ctx_len": total_ctx_len,
                "tbt_slo_s": tbt_slo}

    def after_schedule(self, *, prefill_unfinished_tokens: int,
                       has_prefill_requests: bool,
                       prefill_requests: list[PrefillRequest],
                       idle_ttft_budget: float, num_running: int,
                       total_ctx_len: float, tbt_slo: float,
                       now: float) -> list[dict]:
        """End of upstream schedule(): counters, S1/S2, decode-clock setter.

        prefill_unfinished_tokens: tokens scheduled this step for requests
        whose prefill was unfinished before the step.
        """
        self.num_waiting_prefill_tokens -= prefill_unfinished_tokens
        if has_prefill_requests:
            self.num_consecutive_decode_only = 0
        else:
            self.num_consecutive_decode_only += 1

        decisions = []
        if self.new_request_arrived:
            self.new_request_arrived = False
            if self.use_s1:
                if prefill_requests:
                    ttft_budget = min(r.ttft_budget_s for r in prefill_requests)
                    max_queuing = max(0.0, max(r.queuing_time_s
                                               for r in prefill_requests))
                else:
                    ttft_budget, max_queuing = idle_ttft_budget, 0.0
                decision = self.s1(
                    tbt_slo=tbt_slo, ttft_budget=ttft_budget,
                    num_waiting_prefill_tokens=self.num_waiting_prefill_tokens,
                    num_running=num_running, max_queuing_time=max_queuing,
                    now=now)
                self.max_num_scheduled_tokens = decision["chunk_size"]
                self.prefill_clock = decision["clock_mhz"]
                decision["applied"] = self.apply_dvfs(self.prefill_clock, now)
                decisions.append(decision)
            if self.use_s2:
                decision = self.s2(tbt_slo=tbt_slo, num_running=num_running,
                                   total_ctx_len=total_ctx_len)
                self.decode_clock = decision["clock_mhz"]
                decision["applied"] = False  # deferred to the decode setter
                decisions.append(decision)
        if self.num_consecutive_decode_only == NUM_CONSECUTIVE_DECODE_ONLY_TO_SET_CLOCK:
            decisions.append({"phase": "decode_clock",
                              "clock_mhz": self.decode_clock,
                              "applied": self.apply_dvfs(self.decode_clock, now)})
        return decisions

    def close(self) -> None:
        close = getattr(self.actuator, "close", None)
        if close is not None:
            close()
