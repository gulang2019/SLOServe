#!/usr/bin/env python3
# energy_meter_demo.py
# Records per-GPU final energy (J). Works with NVML energy counter if available,
# otherwise integrates power in a background thread.

import time
import threading
from typing import List, Tuple, Optional
import time, threading, csv
# Optional: cupy for the demo workload. If unavailable, we'll fall back.
try:
    import cupy as cp
    import numpy as np
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

class EnergyMeter:
    """
    Accessible, mid-run energy meter for GPUs.
    - NVML total energy counter (preferred, NVIDIA)
    - Background power sampling fallback (NVIDIA/AMD)
    You can call .read() mid-run and AFTER stop(); final snapshot is persisted.
    """
    def __init__(self, devices: Optional[List[int]] = None, sample_hz: float = 10.0):
        self.devices = devices
        self.sample_hz = float(sample_hz)
        self._lock = threading.Lock()
        self._running = False
        self._sampler_thread = None

        # Backends
        self._nvml = None
        self._rocm = None

        # Device handles/ids
        self._handles = []     # backend-specific
        self._uuids = []       # strings for logging

        # Counters / accumulators
        self._counter_supported = False  # prefers counter-diff path
        self._start_counters = None      # list of starting energy counters (J)
        self._energy_J = None            # list of integrated Joules (sampling)
        self._last_sample_t = None
        self._last_freqs_MHz: Optional[List[Optional[float]]] = None

        # Persisted results
        self._final_snapshot: Optional[Tuple[List[float], float]] = None
        self._start_time = None
        self._end_time = None

    # -------- Public API --------
    def start(self):
        """Initialize backend, capture baselines, start sampler if needed."""
        self._final_snapshot = None
        self._start_time = time.time()
        self._end_time = None

        self._init_backend()
        self._choose_devices()
        self._last_freqs_MHz = [None for _ in self._handles]
        if not self._handles:
            raise RuntimeError("No GPUs detected for energy measurement.")
        # Prime frequency cache if supported
        if self._handles:
            try:
                self._last_freqs_MHz = self._read_freqs_MHz()
            except Exception:
                pass

        # Try energy counter mode first (preferred)
        if self._try_counter_baseline():
            self._counter_supported = True
            self._running = True
            return

        # Fallback: start sampling thread
        self._counter_supported = False
        self._energy_J = [0.0 for _ in self._handles]
        self._running = True
        self._last_sample_t = time.perf_counter()
        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._sampler_thread.start()

    def read(self) -> Tuple[List[float], float]:
        """
        Get (per_gpu_joules, total_joules) since start().
        Works mid-run and AFTER stop() (returns final snapshot then).
        """
        if self._running:
            return self._snapshot_now()
        if self._final_snapshot is not None:
            return self._final_snapshot
        raise RuntimeError("EnergyMeter has not been started.")

    def read_freqs_MHz(self) -> List[Optional[float]]:
        """
        Returns the most recent SM/core frequency in MHz for each GPU.
        If the meter is stopped, returns the last cached reading.
        """
        if self._running:
            freqs = self._read_freqs_MHz()
            self._last_freqs_MHz = list(freqs)
            return freqs
        if self._last_freqs_MHz is not None:
            return list(self._last_freqs_MHz)
        raise RuntimeError("EnergyMeter has not been started.")

    def stop(self) -> Tuple[List[float], float]:
        """Stop measurement and return final (per_gpu_J, total_J)."""
        if not self._running:
            # Already stopped: return the final snapshot if we have it
            return self._final_snapshot or ([], 0.0)

        # Get the final reading BEFORE flipping running flag
        per, tot = self._snapshot_now()
        self._final_snapshot = (list(per), float(tot))
        self._end_time = time.time()
        try:
            self._last_freqs_MHz = self._read_freqs_MHz()
        except Exception:
            pass

        # Stop sampler
        self._running = False
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None

        self._shutdown_backend()
        return self._final_snapshot

    # Context manager sugar
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # -------- Backend setup --------
    def _init_backend(self):
        # Prefer NVML if present
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            self._nvml = nvml
            return
        except Exception:
            self._nvml = None

        # Try ROCm SMI
        try:
            import rocm_smi as rsm
            rsm.rsmi_init(0)
            self._rocm = rsm
            return
        except Exception:
            self._rocm = None

        raise RuntimeError("No supported GPU telemetry backend found (NVML/ROCm).")

    def _shutdown_backend(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
        if self._rocm is not None:
            try:
                self._rocm.rsmi_shut_down()
            except Exception:
                pass
            self._rocm = None

    def _choose_devices(self):
        self._handles, self._uuids = [], []
        if self._nvml is not None:
            nvml = self._nvml
            n = nvml.nvmlDeviceGetCount()
            idxs = list(range(n)) if self.devices is None else self.devices
            for i in idxs:
                h = nvml.nvmlDeviceGetHandleByIndex(i)
                self._handles.append(h)
                try:
                    self._uuids.append(nvml.nvmlDeviceGetUUID(h).decode())
                except Exception:
                    self._uuids.append(f"NVIDIA-GPU-{i}")
        elif self._rocm is not None:
            rsm = self._rocm
            n = rsm.rsmi_num_monitor_devices()
            idxs = list(range(n)) if self.devices is None else self.devices
            for i in idxs:
                self._handles.append(i)
                self._uuids.append(f"AMD-GPU-{i}")

    # -------- Counter path (preferred) --------
    def _try_counter_baseline(self) -> bool:
        """Returns True if counter-diff mode is available and baseline captured."""
        try:
            counters = self._read_counters_J()
            if counters is None:
                return False
            self._start_counters = counters
            return True
        except Exception:
            return False

    def _read_counters_J(self) -> Optional[List[float]]:
        """
        Read absolute energy counters in Joules for each device.
        Returns None if not supported.
        """
        if self._nvml is not None:
            nvml = self._nvml
            out = []
            for h in self._handles:
                try:
                    # millijoules since boot
                    mJ = nvml.nvmlDeviceGetTotalEnergyConsumption(h)
                    out.append(mJ / 1000.0)
                except Exception:
                    # Not supported on this SKU/driver
                    return None
            return out
        elif self._rocm is not None:
            rsm = self._rocm
            out = []
            for d in self._handles:
                try:
                    # Some AMD parts expose energy counters; if not, return None.
                    uj = rsm.rsmi_dev_energy_count_get(d)[0]  # microjoules
                    out.append(uj / 1_000_000.0)
                except Exception:
                    return None
            return out
        return None

    # -------- Sampling path (fallback) --------
    def _sampler_loop(self):
        dt_target = 1.0 / max(1e-6, self.sample_hz)
        while self._running:
            t1 = time.perf_counter()
            self._sample_once()
            t2 = time.perf_counter()
            # Sleep the remainder to target frequency
            sleep_s = max(0.0, dt_target - (t2 - t1))
            time.sleep(sleep_s)

    def _sample_once(self):
        now = time.perf_counter()
        watts = self._read_power_W()
        freqs = self._read_freqs_MHz()
        with self._lock:
            dt = max(1e-4, now - (self._last_sample_t or now))  # seconds
            self._last_sample_t = now
            for i, W in enumerate(watts):
                if W is not None:
                    self._energy_J[i] += W * dt
            if freqs is not None:
                self._last_freqs_MHz = list(freqs)

    def _read_power_W(self) -> List[Optional[float]]:
        out = []
        if self._nvml is not None:
            nvml = self._nvml
            for h in self._handles:
                try:
                    mw = nvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
                    out.append(mw / 1000.0)
                except Exception:
                    out.append(None)
            return out
        elif self._rocm is not None:
            rsm = self._rocm
            for d in self._handles:
                try:
                    # Average power in microwatts
                    uw = rsm.rsmi_dev_power_ave_get(d, 0)[0]
                    out.append(uw / 1_000_000.0)
                except Exception:
                    out.append(None)
            return out
        return [None] * len(self._handles)

    def _read_freqs_MHz(self) -> List[Optional[float]]:
        """
        Read the instantaneous SM/core frequency in MHz for each GPU.
        Returns per-device None if unsupported.
        """
        if not self._handles:
            return []
        out: List[Optional[float]] = []
        if self._nvml is not None:
            nvml = self._nvml
            clk_type = getattr(nvml, "NVML_CLOCK_SM", None)
            for h in self._handles:
                freq = None
                if clk_type is not None:
                    try:
                        freq = float(nvml.nvmlDeviceGetClockInfo(h, clk_type))
                    except Exception:
                        freq = None
                out.append(freq)
            return out
        elif self._rocm is not None:
            # Python ROCm-SMI bindings do not expose a stable frequency API yet.
            # Placeholder for future support; return None entries for now.
            return [None] * len(self._handles)
        return [None] * len(self._handles)

    # -------- Snapshot utility --------
    def _snapshot_now(self) -> Tuple[List[float], float]:
        if self._counter_supported:
            now = self._read_counters_J()
            per = [max(0.0, n - s) for n, s in zip(now, self._start_counters)]
            return per, sum(per)
        else:
            with self._lock:
                per = list(self._energy_J)
            return per, sum(per)

    # -------- Utility --------
    @property
    def uuids(self) -> List[str]:
        return list(self._uuids)



class EnergyHistoryRecorder:
    """
    Periodically samples EnergyMeter.read() and logs:
      - wall time (s since epoch)
      - window energy per GPU (J)
      - window average power per GPU (W)
      - EU clocks (MHz) if available
      - totals (J, W)
    """
    def __init__(self, meter, interval_s: float = 0.5, csv_path: Optional[str] = None):
        self.meter = meter
        self.interval_s = float(interval_s)
        self.csv_path = csv_path
        self._last_t: Optional[float] = None
        self._last_J: Optional[List[float]] = None
        self._last_freqs: Optional[List[Optional[float]]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fh = None
        self._writer = None
        self._latest_sample: Optional[dict] = None

    def start(self):
        # init CSV
        if self.csv_path:
            self._fh = open(self.csv_path, "w", newline="")
            headers = ["ts"] + \
                      [f"J_gpu{i}" for i,_ in enumerate(self.meter.uuids)] + ["J_total"] + \
                      [f"W_gpu{i}" for i,_ in enumerate(self.meter.uuids)] + ["W_total"] + \
                      [f"MHz_gpu{i}" for i,_ in enumerate(self.meter.uuids)]
            self._writer = csv.writer(self._fh)
            self._writer.writerow(headers)
            self._fh.flush()

        # baseline snapshot
        J_now, _Jtot = self.meter.read()
        self._last_J = list(J_now)
        self._last_t = time.time()
        try:
            self._last_freqs = self.meter.read_freqs_MHz()
        except Exception:
            self._last_freqs = [None for _ in self.meter.uuids]

        # background sampler
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            try:
                self._sample_once()
            except Exception:
                # best-effort logging; don't crash serving
                pass

    def _sample_once(self):
        assert self._last_t is not None and self._last_J is not None
        J_now, Jtot_now = self.meter.read()                # cumulative Joules since start()
        t_now = time.time()
        dt = max(1e-6, t_now - self._last_t)

        # window energy (J) and average power (W = J/s)
        J_win = [max(0.0, jn - jl) for jn, jl in zip(J_now, self._last_J)]
        W_win = [j / dt for j in J_win]
        Jtot_win = sum(J_win)
        Wtot_win = Jtot_win / dt
        try:
            freqs = self.meter.read_freqs_MHz()
        except Exception:
            freqs = [None for _ in self.meter.uuids]
        self._last_freqs = list(freqs)

        # write row
        if self._writer:
            freq_row = [f if f is not None else "" for f in freqs]
            row = [t_now] + J_win + [Jtot_win] + W_win + [Wtot_win] + freq_row
            self._writer.writerow(row)
            self._fh.flush()

        # advance baseline
        self._last_J = list(J_now)
        self._last_t = t_now
        self._latest_sample = {
            "timestamp": t_now,
            "joules": list(J_win),
            "total_joules": Jtot_win,
            "watts": list(W_win),
            "total_watts": Wtot_win,
            "mhz": list(freqs),
            "dt": dt,
        }

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._fh:
            self._fh.close()
            self._fh = None

    @property
    def latest_sample(self) -> Optional[dict]:
        """Return the most recent sampling window stats (copies lists)."""
        if self._latest_sample is None:
            return None
        sample = self._latest_sample
        return {
            "timestamp": sample["timestamp"],
            "dt": sample["dt"],
            "joules": list(sample["joules"]),
            "total_joules": sample["total_joules"],
            "watts": list(sample["watts"]),
            "total_watts": sample["total_watts"],
            "mhz": list(sample["mhz"]),
        }


# -------------------- Demo workload & CLI --------------------

def demo_workload():
    """
    Example workload:
      - If CuPy is available: do a GPU matmul
      - Else: CPU sleep as a placeholder
    """
    if HAS_CUPY:
        n = 1024
        print("Creating random matrices on GPU...")
        A = cp.asarray((cp.random.random((n, n))).astype(cp.float32))
        B = cp.asarray((cp.random.random((n, n))).astype(cp.float32))
        cp.cuda.Stream.null.synchronize()

        t0 = time.time()
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        print(f"GPU matmul {A.shape} x {B.shape} finished in {t1 - t0:.4f} s")
        _ = C  # keep reference so it's not optimized away
    else:
        print("CuPy not found; running CPU fallback (sleep 2s).")
        time.sleep(2.0)

def main():
    print("Starting energy measurement...")
    with EnergyMeter(sample_hz=10)  as meter:
        demo_workload()

    # After context: final snapshot is still available
    per_gpu, total = meter.read()
    print("\n✅ Final energy consumption (Joules):")
    for i, val in enumerate(per_gpu):
        print(f"GPU {i}: {val:.2f} J")
    print(f"Total: {total:.2f} J")
    freqs = meter.read_freqs_MHz()
    print("\nℹ️ Last recorded SM clock:")
    for i, freq in enumerate(freqs):
        if freq is None:
            print(f"GPU {i}: n/a")
        else:
            print(f"GPU {i}: {freq:.1f} MHz")

if __name__ == "__main__":
    main()
