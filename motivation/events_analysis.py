import json 
from dataclasses import dataclass, field
from typing import List, Dict, Set
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.metrics import r2_score
from typing import Tuple, Any, Optional
import matplotlib


from motivation.common import PerfModel
filepath = 'events/Qwen_Qwen2.5-7B-Instruct_pd_sharegpt_chat_azure_chat_0-500_0.1.jsonl'


matplotlib.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 14,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.grid": True
})

@dataclass(kw_only=True)
class Event:
    event_type: str
    timestamp: float
    device_id: int = 0
"""
"event_type": "energy",
"timestamp": float(row["ts"]),                      # seconds since epoch
"energy": float(row["J_total"]),                   # window energy (J)
"power":  float(row["W_total"]),                   # window avg power (W)
"gpu_ids": list(gpu_ids),                          # e.g., ["GPU-uuid0", "GPU-uuid1", ...]
"gpu_energy": row[j_cols].to_numpy(dtype=float).tolist(),  # per-GPU window energy (J)
"gpu_power":  row[w_cols].to_numpy(dtype=float).tolist(),  # per-GPU window avg power (W)
"""
@dataclass(kw_only=True)
class Energy(Event):
    energy: float
    power: float
    
@dataclass(kw_only=True)
class EngineStep(Event):
    num_requests: int
    elapsed: float
    
@dataclass(kw_only=True)
class ProcessInputQueue(Event):
    elapsed: float
    num_requests: int

@dataclass(kw_only=True)
class GlobalArrival(Event):
    request_id: str

@dataclass(kw_only=True)
class Batch(Event):
    batch_id: int
    req_ids: List[str]
    num_computed_tokens: List[int]
    num_scheduled_tokens: Dict[str, int]
    elapsed: float
    scheduling_overhead: float
    estimated_time: float = 0
    
    
    @property
    def num_reqs(self):
        return len(self.req_ids)
    
    @property
    def total_current_length(self):
        return sum(self.num_scheduled_tokens.values(), start = 0)
    
    @property
    def total_multiply(self):
        return sum([a * self.num_scheduled_tokens[req_id] for req_id, a in \
            zip(self.req_ids, self.num_computed_tokens) if req_id in self.num_scheduled_tokens], start = 0)
    
    @property
    def total_length(self):
        return sum(self.num_computed_tokens) + sum(self.num_scheduled_tokens.values(), start = 0)
    
    @property
    def total_past_tokens(self):
        return sum(self.num_computed_tokens)
    
    @property
    def max_computed_length(self):
        return max(self.num_computed_tokens, default = 0)
    
    @property
    def decode_only(self):
        return max(self.num_scheduled_tokens.values(), default = 0) == 1
    
    @property
    def prefill_only(self):
        return min(self.num_scheduled_tokens.values(), default = 0) > 1
    
    @property
    def mixed(self):
        return not self.decode_only and not self.prefill_only
    
    @property
    def type(self):
        if self.decode_only:
            return 'decode_only'
        elif self.prefill_only:
            return 'prefill_only'
        elif self.mixed:
            return 'mixed'
        
        return 'unknown'
    
    def classify_tokens(self, reqs: Set[str]):
        n_tokens_in_reqs = 0
        for req_id, num_scheduled_tokens in self.num_scheduled_tokens.items():
            if req_id in reqs:
                n_tokens_in_reqs += num_scheduled_tokens
        return n_tokens_in_reqs, sum(self.num_scheduled_tokens.values(), start=0)

@dataclass(kw_only=True)
class ScheduleProblem(Event):
    batch_id: int
    is_feasible: bool
    reqs: List[Dict[str, Any]]
    num_free_blocks: int
    estimated_time: float
    batch_schedule: List[Dict[str, Any]]
    accepted_ids: List[str]
    overhead: float = 0

@dataclass(kw_only=True)
class ReqState(Event):
    request_id: str
    state: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    ddl: float

@dataclass(kw_only=True)
class KVXferReady(Event):
    request_id: str

@dataclass(kw_only=True)
class Arrival(Event):
    request_id: str
    prompt_tokens: int
    num_cached_tokens: int = 0
    max_tokens: int = 0
    prefill_ddl: float = 0
    profit: float = 0
    prefill_only: bool = False
    decode_only: bool = False
    zero_load_ttft: float = 0
    add_req_time: float = 0 
    
@dataclass(kw_only=True)
class Rescheduling(Event):
    request_id: str
    prefill_device_id: int
    decode_device_id: int
    
@dataclass(kw_only=True)
class Finish(Event):
    request_id: str
    finish_reason: str
    scheduling_overhead: float = 0.0

@dataclass(kw_only=True)
class Routing(Event):
    routing_overhead: float
    schedules: Dict[str, Dict[str, Any]]
    
@dataclass(kw_only=True)
class RouterArrival(Event):
    request_id: str
    prefill_ddl: float
    profit: float
    prompt_tokens: int
    max_tokens: int
    zero_load_ttft: float = 0
    
@dataclass(kw_only=True)
class Dispatch(Event):
    type: str
    request_id: str
    prefill_device_id: int
    decode_device_id: int
    
@dataclass(kw_only=True)
class RouterDecision(Event):
    request_id: str
    prefill_device_id: int
    decode_device_id: int

@dataclass
class ReqSchedule:
    batch_id: int
    num_scheduled_tokens: int
    elapsed: float
    timestamp: float
    device_id: int

@dataclass(kw_only=True)
class RequestInstance:
    req_id: str
    prompt_tokens: int = 0
    num_cached_tokens: int = 0
    arrival_time: float = 0
    profit: float = 0
    schedules: List[ReqSchedule] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    output_tokens: int = 0
    prefill_device_id: int = -1
    decode_device_id: int = -1
    cache_hit_rate: float = 0
    prefill_ddl: float = 0
    ttft_normalized_laxity: float = 0
    tpot_laxities: List[float] = field(default_factory=list)
    expected_finish_time: List[float] = field(default_factory=list)
    finish_reason: str | None = None
    slo_violation: str | None = None
    kv_xfer_delay: float = 0
    zero_load_ttft: float = 0


    @property
    def is_finished(self):
        if self.finish_reason is None:
            for event in self.events:
                if event.event_type == 'finish':
                    self.finish_reason = event.finish_reason
                    if event.finish_reason == 'rejected':
                        if event.device_id == self.decode_device_id:
                            self.finish_reason += '-decode'
                        if event.device_id == self.prefill_device_id:
                            self.finish_reason += '-prefill'
        return self.finish_reason == 'length'

    def get_stats(self, slo_ttft_fn, slo_tpot):
        
        slo_ttft = slo_ttft_fn(self)
        required_tokens = [(self.arrival_time + slo_ttft, self.prompt_tokens - self.num_cached_tokens)]
        
        for i in range(self.output_tokens - 1):
            required_tokens.append((self.arrival_time + slo_ttft + (i + 1) * slo_tpot, 1))
            
        # required_tokens = [(self.prefill_ddl + overhead, self.prompt_tokens - self.num_cached_tokens)]
        
        # for i in range(self.output_tokens - 1):
        #     required_tokens.append((self.prefill_ddl + overhead + (i + 1) * slo_tpot, 1))
        # self.expected_finish_time = list(zip(*required_tokens))[0]

        # vLLM has some laggings, lets correct it
        lag = 0
        cutoff = 0.10
        for i, sch in enumerate(self.schedules):
            sch.timestamp -= lag
            if i > 0 and self.schedules[i - 1].timestamp + cutoff < sch.timestamp:
                delta = sch.timestamp - self.schedules[i - 1].timestamp - cutoff
                lag += delta
                sch.timestamp -= delta

        i = 0
        num_scheduled_tokens = 0
        timestamp = self.arrival_time
        for j, (t, required_token) in enumerate(required_tokens):
            while i < len(self.schedules) and num_scheduled_tokens < required_token:
                num_scheduled_tokens += self.schedules[i].num_scheduled_tokens
                timestamp = self.schedules[i].timestamp
                i += 1
            num_scheduled_tokens -= required_token
            self.timestamps.append(timestamp)
            self.expected_finish_time.append(t)
            if j == 0:
                self.ttft_normalized_laxity = (timestamp - t)
                # if self.ttft_normalized_laxity < 0:
                #     print(f'req {self.req_id} has negative ttft_normalized_laxity: {timestamp - t}')
            else:
                self.tpot_laxities.append(timestamp - t)
            
        self.cache_hit_rate = self.num_cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0
        
        for event in self.events:
            idx = 0
            while idx < len(self.events):
                event = self.events[idx]
                if event.event_type == 'kv_xfer_ready':
                    self.kv_xfer_delay = event.timestamp - self.events[idx - 1].timestamp
                    break
                idx += 1
    
    def violate_slo(self):
        if self.slo_violation is not None:
            return self.slo_violation
        if not self.is_finished:
            self.slo_violation = self.finish_reason
            return self.finish_reason
        if self.ttft_normalized_laxity > 0:
            self.slo_violation = 'ttft'
            return 'ttft'
        if max(self.tpot_laxities, default=0) > 0:
            self.slo_violation = 'tpot'
            return 'tpot'
        self.slo_violation = 'none'
        return 'none'
    
    def is_violate_slo(self):
        return self.violate_slo() != 'none'
    
    @property
    def e2e_latency(self):
        return self.schedules[-1].timestamp + self.schedules[-1].elapsed - self.arrival_time

    @property
    def max_tpot_laxity(self):
        return max(self.tpot_laxities, default=0)

def _ls_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Ordinary least squares with small Tikhonov for numerical stability
    # Solves min ||X w - y||_2
    ridge = 1e-9
    return np.linalg.lstsq(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]

def fit(batches: List[Batch], max_iters: int = 20, tol: float = 1e-6) -> Callable[[Batch], float]:
    """
    Fits parameters a,b,c and d,e for:
        time = max( a * total_multiply + b * total_current_length + c,
                    d + e * total_length )
    Returns a callable f(batch) -> predicted_time.
    """
    if not batches:
        raise ValueError("No batches to fit.")

    # Extract features/targets
    # Be robust to "total_multiple" vs "total_multiply" naming in the prompt.
    def total_mult(b: Batch) -> float:
        return getattr(b, "total_multiple", getattr(b, "total_multiply"))

    x1 = np.array([[total_mult(b), b.total_current_length, 1.0] for b in batches], dtype=float)  # [a, b, c]
    x2 = np.array([[1.0, b.total_length] for b in batches], dtype=float)                         # [d, e]
    y  = np.array([b.elapsed for b in batches], dtype=float)

    # Initialize by fitting both branches to y independently
    w1 = _ls_fit(x1, y)                # [a, b, c]
    w2 = _ls_fit(x2, y)                # [d, e]

    prev_obj = np.inf

    for _ in range(max_iters):
        # Compute branch predictions
        pred1 = x1 @ w1
        pred2 = x2 @ w2
        pred  = np.maximum(pred1, pred2)

        # Objective: squared error to observed times
        obj = float(np.mean((pred - y) ** 2))

        # Convergence check
        if abs(prev_obj - obj) <= tol * max(1.0, prev_obj):
            break
        prev_obj = obj

        # Assign active branch per sample
        active1 = pred1 >= pred2
        active2 = ~active1

        # Refit each branch **only on samples where it is active**.
        # If one branch has too few points, fall back to global fit.
        if np.sum(active1) >= 3:
            w1 = _ls_fit(x1[active1], y[active1])
        else:
            w1 = _ls_fit(x1, y)

        if np.sum(active2) >= 3:
            w2 = _ls_fit(x2[active2], y[active2])
        else:
            w2 = _ls_fit(x2, y)

    # Build the predictor
    a, b, c = w1.tolist()
    d, e    = w2.tolist()

    def predictor(batch: Batch) -> float:
        tmul = getattr(batch, "total_multiple", getattr(batch, "total_multiply"))
        v1 = a * tmul + b * batch.total_current_length + c
        v2 = d + e * batch.total_length
        return max(v1, v2)

    print('fitted_model: ')
    print(f'time = max({a:.6f} * total_multiply + {b:.6f} * total_current_length + {c:.6f}, {d:.6f} + {e:.6f} * total_length)')
    predicted_times = [predictor(batch) for batch in batches]
    r2 = r2_score(predicted_times, y)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize = (4, 16), tight_layout = True)
    ax1.scatter(y, predicted_times)
    ax1.set_xlabel('real_times')
    ax1.set_ylabel('predicted_times')
    # Make sure x and y axes are at the same scale for the main R2 scatterplot
    lims = [
        min(min(y), min(predicted_times)),
        max(max(y), max(predicted_times)),
    ]
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.plot(lims, lims, '--r', linewidth=1)  # Line y=x for reference
    ax1.set_title('R2 = ' + str(r2))
    for attr, ax in zip(['total_multiply', 'total_current_length', 'total_length', 'max_computed_length'], [ax2, ax3, ax4, ax5]):
        ax.scatter([getattr(batch, attr) for batch in batches], y)
        ax.set_xlabel(attr)
        ax.set_ylabel('real times')
    fig.savefig('figs/batch_time_fit.png')
    print('Saved figs/batch_time_fit.png')
    # compute R2
    print('R2', r2)
    return predictor, (a, b, c, d, e)

def analyze_events(filepath, start_time = None, verbose = False):
    events = []
    if isinstance(filepath, str):
        with open(filepath, 'r') as f:
            raw_events = json.load(f)
            for raw_event in raw_events:
                if raw_event['event_type'] == 'batch':
                    events.append(Batch(**raw_event))
                elif raw_event['event_type'] == 'arrival':
                    events.append(Arrival(**raw_event))
                elif raw_event['event_type'] == 'finish':
                    events.append(Finish(**raw_event))
                elif raw_event['event_type'] == 'routing':
                    events.append(Routing(**raw_event))
                elif raw_event['event_type'] == 'arrival-router':
                    events.append(RouterArrival(**raw_event))
                elif 'dispatch' in raw_event['event_type']:
                    raw_event['type'] = raw_event['event_type'].split('-')[1]
                    raw_event['event_type'] = 'dispatch'
                    events.append(Dispatch(**raw_event))
                elif raw_event['event_type'] == 'router_decision':
                    events.append(RouterDecision(**raw_event))
                elif raw_event['event_type'] == 'global_arrival':
                    events.append(GlobalArrival(**raw_event))
                elif raw_event['event_type'] == 'kv_xfer_ready':
                    events.append(KVXferReady(**raw_event))
                elif raw_event['event_type'] == 'rescheduling':
                    events.append(Rescheduling(**raw_event))
                elif raw_event['event_type'] == 'schedule_problem':
                    events.append(ScheduleProblem(**raw_event))
                elif raw_event['event_type'] == 'req_state':
                    events.append(ReqState(**raw_event))
                elif raw_event['event_type'] == 'engine_step':
                    events.append(EngineStep(**raw_event))
                elif raw_event['event_type'] == 'process_input':
                    events.append(ProcessInputQueue(**raw_event))
                elif raw_event['event_type'] == 'energy':
                    events.append(Energy(**raw_event))
                else: 
                    events.append(Event(**raw_event))
    elif isinstance(filepath, list):
        for event in filepath:
            if event['event_type'] == 'batch':
                events.append(Batch(**event))
            elif event['event_type'] == 'arrival':
                events.append(Arrival(**event))
            elif event['event_type'] == 'finish':
                events.append(Finish(**event))
            elif event['event_type'] == 'routing':
                events.append(Routing(**event))
            elif event['event_type'] == 'arrival-router':
                events.append(RouterArrival(**event))
            elif 'dispatch' in event['event_type']:
                event['type'] = event['event_type'].split('-')[1]
                event['event_type'] = 'dispatch'
                events.append(Dispatch(**event))
            elif event['event_type'] == 'router_decision':
                events.append(RouterDecision(**event))
            elif event['event_type'] == 'global_arrival':
                events.append(GlobalArrival(**event))
            elif event['event_type'] == 'kv_xfer_ready':
                events.append(KVXferReady(**event))
            elif event['event_type'] == 'rescheduling':
                events.append(Rescheduling(**event))
            elif event['event_type'] == 'schedule_problem':
                events.append(ScheduleProblem(**event))
            elif event['event_type'] == 'req_state':
                events.append(ReqState(**event))
            elif event['event_type'] == 'engine_step':
                events.append(EngineStep(**event))
            elif event['event_type'] == 'process_input':
                events.append(ProcessInputQueue(**event))
            elif event['event_type'] == 'energy':
                events.append(Energy(**event))
            else: 
                events.append(Event(**event))
    events = sorted(events, key=lambda x: x.timestamp)
        
    reqs = {}
    imcomplete_reqs = 0
    for event in events: 
        if hasattr(event, 'request_id'):
            if event.request_id not in reqs:
                reqs[event.request_id] = RequestInstance(
                req_id=event.request_id, 
                arrival_time = -1.0,
                prompt_tokens = -1,
                output_tokens = -1)
            req = reqs[event.request_id]
            # if event.event_type == 'arrival-router' and req.arrival_time < 0:
            #     req.zero_load_ttft = event.zero_load_ttft
            #     req.arrival_time = event.timestamp
            if event.event_type == 'arrival' and req.arrival_time < 0:
                req.zero_load_ttft = event.zero_load_ttft
                req.arrival_time = event.timestamp
            if hasattr(event, 'prefill_ddl'):
                req.prefill_ddl = event.prefill_ddl
            if event.event_type == 'router_decision':
                req.prefill_device_id = event.prefill_device_id
                req.decode_device_id = event.decode_device_id
            if event.event_type == 'arrival-router':
                req.prompt_tokens = event.prompt_tokens
                req.output_tokens = event.max_tokens
            if event.event_type == 'arrival':
                req.num_cached_tokens = event.num_cached_tokens
                req.prompt_tokens = event.prompt_tokens
                if not event.prefill_only:
                    req.output_tokens = event.max_tokens
            if event.event_type == 'arrival':
                if event.prefill_only: 
                    # req.prompt_tokens = event.prompt_tokens
                    req.prefill_device_id = event.device_id
                elif event.decode_only:
                    # req.output_tokens = event.max_tokens
                    req.decode_device_id = event.device_id
                else:
                    # req.prompt_tokens = event.prompt_tokens
                    # req.output_tokens = event.max_tokens
                    req.prefill_device_id = req.decode_device_id = event.device_id
            req.events.append(event)
    
        if event.event_type == 'batch':
            for req_id, num_computed_tokens in zip(
                event.req_ids,
                event.num_computed_tokens,
            ):
                if req_id not in event.num_scheduled_tokens:
                    continue
                num_scheduled_tokens = event.num_scheduled_tokens[req_id]
                if req_id in reqs:
                    reqs[req_id].schedules.append(ReqSchedule(
                        batch_id=event.batch_id,
                        num_scheduled_tokens=num_scheduled_tokens,
                        elapsed=event.elapsed,
                        timestamp=event.timestamp,
                        device_id=event.device_id,
                    ))
                else: 
                    imcomplete_reqs += 1
    if verbose:
        print(f'imcomplete_reqs: {imcomplete_reqs}, total_reqs: {len(reqs)}')
    from collections import defaultdict
    import numpy as np
    
    if verbose:
        print('- latency breakdown -')
        latencies = defaultdict(list)
        for req in reqs.values():
            last_event = ('global_arrival', req.arrival_time)
            for event in req.events:
                event_name = event.event_type
                if event_name == 'dispatch':
                    event_name += '-' + event.type
                if event_name == 'arrival':
                    if event.prefill_only:
                        event_name = 'prefill_arrival'
                    elif event.decode_only:
                        event_name = 'decode_arrival'
                if event_name == 'finish':
                    event_name = 'finish-' + event.finish_reason
                latencies[f'{last_event[0]}-{event_name}'].append((event.timestamp - last_event[1], req.prompt_tokens))
                last_event = (event_name, event.timestamp)
        for key, value in latencies.items():
            value, prompt_tokens = zip(*value)
            print(f'{key}: {np.mean(value):.4f} +- {np.std(value):.4f} p99: {np.percentile(value, 99):.4f}, p50: {np.percentile(value, 50):.4f}, p90: {np.percentile(value, 90):.4f}, p20: {np.percentile(value, 20):.4f}')
            fig, ax = plt.subplots()
            # ax.scatter(prompt_tokens, value)
            # ax.set_xlabel('Prompt Tokens')
            # ax.set_ylabel('Latency (s)')
            # ax.set_title(f'{key}')
            # ax.grid(True)
            # ax.legend()
            # fig.savefig(f'debug-{key}.png')
            # print(f'saved to debug-{key}.png')
        print('-' * 20)
    return events, reqs

def calc_n_active_servers(events: list, window_size: float = 1.0, ax = None, label: str = "", color: str = "tab:blue"):
    """
    Plots the exact number of active servers (distinct device ids running batches) in each time window for 'batch' events,
    using vlines and hlines for a stepwise plot.
    Args:
        events: List of Event objects (must have event_type, timestamp, elapsed, device_id).
        ax: matplotlib Axes to plot on.
        window_size: Size of each bin in seconds.
    """
    import numpy as np

    # Gather all batch events
    batch_events = [event for event in events if event.event_type == 'batch']

    batch_events = sorted(batch_events, key=lambda x: x.timestamp - x.elapsed)

    t0 = min(event.timestamp - event.elapsed for event in batch_events)
    tN = max(event.timestamp for event in batch_events)
    bins = np.arange(t0, tN + window_size, window_size)

    # For each window, determine which batches overlap and count unique device ids
    num_active_devices_per_bin = []
    for start, end in zip(bins[:-1], bins[1:]):
        device_ids = set()
        for event in batch_events:
            batch_start = event.timestamp - event.elapsed
            batch_end = event.timestamp
            # Overlap if batch runs into window [start, end)
            if batch_end > start and batch_start < end:
                device_ids.add(event.device_id)
        num_active_devices_per_bin.append(len(device_ids))

    # Draw stepwise hlines and vlines
    prev_y = num_active_devices_per_bin[0] if num_active_devices_per_bin else 0
    if ax:
        for i, (x1, x2, y) in enumerate(zip(bins[:-1], bins[1:], num_active_devices_per_bin)):
            ax.hlines(y, x1, x2, linewidth=2, label=label if i == 0 else "", color=color)
            if i > 0 and prev_y != y:
                ax.vlines(x1, prev_y, y, linewidth=2, color=color)
            prev_y = y

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("# Active Servers")
        ax.grid(True)
        if label:
            ax.legend()
    return np.mean(num_active_devices_per_bin)

def calc_num_effective_tokens(
    events: list,
    reqs: dict[str, RequestInstance],
    window_size: float = 1.0,
    ax: plt.Axes | None = None,
    label: str = "",
    color: str = "tab:blue"
):
    """
    Plot the number of "effective" scheduled tokens per window,
    where effective tokens are those scheduled for requests that did NOT violate SLO.
    Plots total scheduled tokens and effective scheduled tokens over time.
    Args:
        events: List of Event objects (must have event_type, timestamp, elapsed, device_id, num_scheduled_tokens [dict]).
        reqs: Mapping of req_id -> RequestInstance, with proper SLO stats already computed.
        ax: matplotlib Axes to plot on.
        window_size: Size of each bin in seconds.
        label: Label for legend for effective tokens line.
        color: Color for effective tokens.
    """
    import numpy as np

    # Precompute which req_ids did NOT violate SLO
    from collections import Counter
    id_to_effective = {req.req_id: req.violate_slo() == 'none' for req in reqs.values()}
    # Gather all batch events
    batch_events = [event for event in events if event.event_type == 'batch']

    batch_events = sorted(batch_events, key=lambda x: x.timestamp - x.elapsed)
    if not batch_events:
        ax.set_title("No batch events")
        return

    t0 = min(event.timestamp - event.elapsed for event in batch_events)
    tN = max(event.timestamp for event in batch_events)
    bins = np.arange(t0, tN + window_size, window_size)

    num_tokens_per_bin = []
    num_effective_per_bin = []
    num_waste_per_bin = []
    for start, end in zip(bins[:-1], bins[1:]):
        total_tokens = 0
        effective_tokens = 0
        for event in batch_events:
            batch_start = event.timestamp - event.elapsed
            batch_end = event.timestamp
            if batch_start > start and batch_end < end:
                # For each served req in batch, sum its tokens
                for req_id, num_scheduled_tokens in event.num_scheduled_tokens.items():
                    if req_id in id_to_effective:
                        total_tokens += num_scheduled_tokens
                    if id_to_effective.get(req_id, False):
                        effective_tokens += num_scheduled_tokens
        num_tokens_per_bin.append(total_tokens)
        num_effective_per_bin.append(effective_tokens)
        num_waste_per_bin.append(total_tokens - effective_tokens)
    centers = (bins[:-1] + bins[1:]) / 2
    
    if ax:
        # ax.step(centers, num_tokens_per_bin, where="mid", color="tab:gray", linestyle="--", linewidth=2, label="Total scheduled tokens")
        # ax.step(centers, num_effective_per_bin, where="mid", color=color, linestyle="-", linewidth=2, label=(label or "Effective scheduled tokens"))
        ax.step(centers, num_waste_per_bin, where="mid", linestyle="-", linewidth=2, label=label)
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("# Scheduled Tokens")
        # ax.grid(True)
        ax.legend(fontsize=18)
    return np.mean(num_effective_per_bin) / np.mean(num_tokens_per_bin)


def compare_schedulers(
    results: List[Dict[str, Any]],
    slo_ttft_fn,
    slo_tpot,
    prefix = 'compare_schedulers',
):
    import matplotlib.pyplot as plt
    import re
    # Set a larger global font size for all text in the figure
    plt.rcParams.update({'font.size': 20})
    # Define colors explicitly for clarity and consistency
    color_reqs = 'tab:black'
    color_violation_list = [
        'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    num_graphs = len(results) + 1
    fig, axes = plt.subplots(num_graphs, figsize=(16, 5 * num_graphs))
    # Plot rolling loads (Requests) on left y-axis
    ax = axes[0]
    reqs = results[0]['reqs']
    min_intervals, max_intervals = get_min_servers(list(reqs.values()), slo_ttft_fn, slo_tpot)
    window_starts, window_ends, min_prefill_servers, min_decode_servers, min_tot_servers = zip(*min_intervals)
    _, _, max_prefill_servers, max_decode_servers, max_tot_servers = zip(*max_intervals)
    prev_we, prev_ms = None, None
    for ws, we, ms in zip(window_starts, window_ends, min_tot_servers):
        ax.hlines(ms, ws, we, color='black', linewidth=2, label='lower bound' if ws == window_starts[0] else "")
        if prev_we is not None and prev_we == ws:  # connect contiguous intervals
            ax.vlines(ws, prev_ms, ms, color='black', linewidth=2)
        prev_we, prev_ms = we, ms
    prev_we, prev_ms = None, None
    for ws, we, ms in zip(window_starts, window_ends, max_tot_servers):
        ax.hlines(ms, ws, we, color='black', linewidth=2, label='upper bound' if ws == window_starts[0] else "", linestyle="--")
        if prev_we is not None and prev_we == ws:  # connect contiguous intervals
            ax.vlines(ws, prev_ms, ms, color='black', linewidth=2, linestyle="--")
        prev_we, prev_ms = we, ms
    # for result in results:
    #     plot_num_active_servers(result['events'], ax.twinx(), label = result['scheduler_name'])
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylabel('# Servers Required', fontsize=24)
    ax.tick_params(axis='y', labelsize=18)

    # ---- Annotate Scheduling & Routing ----
    # Try to deduce scheduling & routing from results. Assume first scheduler_name is representative.
    def split_scheduler_name(scheduler_name):
        # We expect formats like 'slosserve-edf/round_robin' or similar
        if '/' in scheduler_name:
            left, right = scheduler_name.split('/', 1)
            left = left.replace('-', ' ').replace('_', ' ').strip().upper()
            right = right.replace('-', ' ').replace('_', ' ').strip().upper()
            return left, right
        else:
            return scheduler_name, ""

    # Plot violation rates for each scheduler on right y-axis, using different colors
    for idx, (ax, result) in enumerate(zip(axes[1:], results)):
        reqs = result['reqs']
        scheduler_name = result['scheduler_name']
        time, num_violations = compute_window_stats(list(reqs.values()), slo_ttft_fn, slo_tpot)
        print(f'{scheduler_name} num_violations: {sum(num_violations)}')
        prev_time = None
        prev_value = None
        for i in range(len(time)):
            if i == len(time) - 1:
                t_next = time[i] + (time[i] - time[i-1] if i > 0 else 1)
            else:
                t_next = time[i + 1]
            # Draw horizontal line for this interval
            ax.hlines(num_violations[i], time[i], t_next, color='black', linewidth=2, label=scheduler_name if i == 0 else "")
            # Draw a vertical line to connect this to previous value (for step look)
            if prev_time is not None:
                ax.vlines(time[i], prev_value, num_violations[i], color='black', linewidth=2)
            prev_time = t_next
            prev_value = num_violations[i]
        ax.set_title(scheduler_name)
        ax.tick_params(axis='y', labelsize=18)
        sched, routing = split_scheduler_name(scheduler_name)
        ax.set_ylabel(f'# Violation', fontsize=24)
        ax.set_ylim(0, 50)
        ax.grid(True)

        # Use only a single (shared) twin axis, and set its ylabel and ylim only once
        ax_effective = ax.twinx()
        plot_num_effective_tokens(result['events'], reqs, ax_effective)
        # ax_effective.set_ylabel("Num scheduled tokens (per window)")
        ax_effective.set_ylim(0, 60000)

    # Make the overall figure title font size larger if desired
    fig.suptitle('Scheduler Comparison: Minimum Servers & SLO Violations', fontsize=30)
    # Save figure
    fig.savefig(f'figs/{prefix}.comparison.png')
    print(f'Saved figs/{prefix}.comparison.png')
    
    return fig

def count_intervals(intervals: list[tuple[float, float, float, str]], 
                    window: float,
                    mode: str = 'min'):
    import numpy as np
    import math
    start_time = intervals[0][0]
    end_time = intervals[-1][1]
    windows = [[window_start, window_start + window, 0, 0, 0] for window_start in np.arange(start_time, end_time, window)]

    for s, e, time, type in intervals:
        first_window_idx = math.floor((s - start_time) / window)
        last_window_idx = math.ceil((e - start_time) / window)
        if mode == 'min' and first_window_idx != last_window_idx - 1:
            continue
        if first_window_idx >= len(windows) or last_window_idx >= len(windows):
            continue
        for window_idx in range(first_window_idx, last_window_idx + 1):
            if type == 'P':
                windows[window_idx][2] += time
            else:
                windows[window_idx][3] += time
            windows[window_idx][4] += time
    if mode == 'max':
        for window in windows:
            window[4] = window[2] + math.ceil(window[3])
    
    return windows

def get_min_servers(requests: List[RequestInstance], slo_ttft_fn, slo_tpot, perf_model: PerfModel, window = 1):
    max_decode_batch_size = perf_model.get_max_decode_batch_size(slo_tpot)
    intervals = []
    for request in requests:
        ttft_slo = slo_ttft_fn(request)
        intervals.append((request.arrival_time, request.arrival_time + ttft_slo, request.zero_load_ttft, 'P'))
        for i in range(request.output_tokens - 1):
            intervals.append((request.arrival_time + ttft_slo + slo_tpot * i, request.arrival_time + ttft_slo + slo_tpot * (i + 1), 1 / max_decode_batch_size * slo_tpot, 'D'))
    intervals = sorted(intervals, key=lambda x: x[0])
    
    min_intervals = count_intervals(intervals, window, mode = 'min')
    max_intervals = count_intervals(intervals, window, mode = 'max')
    
    return min_intervals, max_intervals

def get_rolling_loads(requests: List[RequestInstance], slo_ttft_fn, slo_tpot, normalize = False):
    events = []
    
    for req in requests:
        slo_ttft = slo_ttft_fn(req.prompt_tokens) 
        events.append((req.arrival_time, req.prompt_tokens / slo_ttft))
        events.append((req.arrival_time + slo_ttft, -req.prompt_tokens / slo_ttft))
        events.append((req.arrival_time + slo_ttft, 1 / slo_tpot))
        events.append((req.arrival_time + slo_ttft + slo_tpot * req.output_tokens, -1 / slo_tpot))
    
    events = sorted(events, key=lambda x: x[0])
    start_time = events[0][0]
    rolling_loads = [(0,0)]
    
    for event in events:
        rolling_loads.append((event[0] - 1e-6 - start_time, rolling_loads[-1][1]))
        rolling_loads.append((event[0] - start_time, rolling_loads[-1][1] + event[1]))

    return rolling_loads

def compute_window_stats(reqs: List[RequestInstance], slo_ttft_fn, slo_tpot):
    events = []
    for req in reqs:
        req.get_stats(slo_ttft_fn, slo_tpot)        
        violate = req.violate_slo()
        # if violate == 'tpot': 
        #     print(f'req {req.req_id} violates tpot, tpots: {req.tpot_laxities}, batch ids: {[s.batch_id for s in req.schedules]}, timestamps: {[s.timestamp for s in req.schedules]}')
            
        # if violate == 'ttft':
        #     print(f'req {req.req_id} violates ttft, ttft: {req.ttft_normalized_laxity}, batch ids: {[s.batch_id for s in req.schedules]}, timestamps: {[s.timestamp for s in req.schedules]}')
        # print(f'req.request_id: {req.req_id}, slo_ttft: {slo_ttft_fn(req.prompt_tokens)}, req.ttft: {req.ttft_normalized_laxity}')
        event = {
            'arrival_time': req.arrival_time,
            'tpot_violation': violate == 'tpot',
            'ttft_violation': violate == 'ttft',
            'any_violation': violate != False,
            'unfinished': not req.is_finished,
            'ttft_normalized_laxity': req.ttft_normalized_laxity,
            'max_tpot_laxity': max(req.tpot_laxities, default=0),
            'prompt_tokens': req.prompt_tokens,
            'output_tokens': req.output_tokens,
            'total_tokens': req.prompt_tokens + req.output_tokens,
        }
        events.append(event)
        
    events = sorted(events, key=lambda x: x['arrival_time'])
    window_size = 1
    event_idx = 0
    window_start = events[0]['arrival_time']

    window_stats = []

    while window_start + window_size < events[-1]['arrival_time']:
        window_end = window_start + window_size
        start_idx = event_idx
        while event_idx < len(events) and events[event_idx]['arrival_time'] < window_end:
            event_idx += 1

        window_events = events[start_idx:event_idx]
        num_reqs = len(window_events)
        num_violations = sum(e['any_violation'] for e in window_events)
        max_ttft_normalized_laxity = max((e['ttft_normalized_laxity'] for e in window_events), default=0)
        max_tpot = max((e['max_tpot_laxity'] for e in window_events), default=0)
        total_prompt_tokens = sum(e['prompt_tokens'] for e in window_events)
        total_output_tokens = sum(e['output_tokens'] for e in window_events)
        total_tokens = sum(e['total_tokens'] for e in window_events)

        window_stats.append((
            num_reqs,
            num_violations,
            max_ttft_normalized_laxity,
            max_tpot,
            total_prompt_tokens,
            total_output_tokens,
            total_tokens
        ))
        window_start = window_end
    num_reqs, num_violations, max_ttft, max_tpot, total_prompt_tokens, total_output_tokens, total_tokens = zip(*window_stats)
    time = np.arange(len(num_reqs)) * window_size 
    return time, num_violations

def compute_effective_tokens(
    all_events: List[Event],
    slo_attained_reqs: Set[str],
    window_size: float = 1,
):
    batches = [event for event in all_events if event.event_type == 'batch']
    n_effective_tokens, n_total_tokens = zip(*[event.classify_tokens(slo_attained_reqs) for event in batches])
    n_effective_tokens = np.array(n_effective_tokens)
    n_total_tokens = np.array(n_total_tokens)
    times = [event.timestamp for event in batches]
    # print('times', times[:10])
    # Aggregate n_effective_tokens and n_total_tokens by window_size, using event[k] starts by times[k]
    # Bin by window_size, using times as the start of each window
    binned_effective = []
    binned_total = []
    binned_times = []
    if len(times) > 0:
        min_time = times[0]
        max_time = times[-1]
        num_windows = int(np.ceil((max_time - min_time) / window_size))
        for i in range(num_windows):
            window_start = min_time + i * window_size
            window_end = window_start + window_size
            idxs = [k for k, t in enumerate(times) if window_start <= t < window_end]
            if idxs:
                binned_effective.append(np.sum(n_effective_tokens[idxs]))
                binned_total.append(np.sum(n_total_tokens[idxs]))
                binned_times.append(window_start)
            else:
                binned_effective.append(0)
                binned_total.append(0)
                binned_times.append(window_start)
    n_effective_tokens = np.array(binned_effective)
    n_total_tokens = np.array(binned_total)
    times = np.array(binned_times)

    return times, n_effective_tokens, n_total_tokens

def draw_laxity(
    reqs: Dict[str, RequestInstance],
    ax: plt.Axes,
    slo_ttft_fn: Callable[[int], float],
    slo_tpot: float,
    color: str = 'tab:blue',
):
    from dataclasses import asdict
    import numpy as np
    ttft_points = []
    for req in reqs.values():
        req.get_stats(slo_ttft_fn, slo_tpot)
        if not req.is_finished:
            continue
        timestamps = np.array(req.timestamps)
        expected_finish_times = np.array(req.expected_finish_time)
        laxities = timestamps - expected_finish_times
        if min(laxities) < -17.5:
            with open('req.json', 'w') as f:
                json.dump(asdict(req), f, indent=4)
        ttft_points.append((timestamps[0], laxities[0]))
        ax.plot(timestamps, laxities, color=color, zorder=0)
    ttft_times, ttft_laxities = zip(*ttft_points)
    import numpy as np
    print(f'ttft laxities: {np.mean(ttft_laxities)}')
    ax.scatter([x[0] for x in ttft_points], [x[1] for x in ttft_points], color='blue', marker='o', zorder=10)
    ax.grid(True)
    ax.set_yscale('symlog', linthresh=0.05)
    ax.set_ylabel('Laxity (s)', fontsize=18)
    ax.set_xlabel('Time (s)', fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label='Laxity = 0')
    ax.legend(loc='upper right', fontsize=13)


def draw_min_servers(reqs: Dict[str, RequestInstance],
                     ax: plt.Axes,
                     slo_ttft_fn: Callable[[int], float],
                     slo_tpot: float,
                     perf_model: PerfModel,
                     ):
    min_intervals, max_intervals = get_min_servers(list(reqs.values()), slo_ttft_fn, slo_tpot, perf_model)
    window_starts, window_ends, min_prefill_servers, min_decode_servers, min_tot_servers = zip(*min_intervals)
    _, _, max_prefill_servers, max_decode_servers, max_tot_servers = zip(*max_intervals)
    prev_we, prev_ms = None, None
    # Draw as a shade (filled region)
    import numpy as np
    print('MEAN MIN SERVERS', np.mean(min_tot_servers), 'STD MIN SERVERS', np.std(min_tot_servers), 'MAX MIN SERVERS', np.max(min_tot_servers))
    print('MEAN MAX SERVERS', np.mean(max_tot_servers), 'STD MAX SERVERS', np.std(max_tot_servers), 'MAX MAX SERVERS', np.max(max_tot_servers))
    import numpy as np
    ws = np.array(window_starts)
    we = np.array(window_ends)
    ms = np.array(min_tot_servers)
    # INSERT_YOUR_CODE

    # Count the fraction of time intervals that require each number of servers.
    import collections

    # (ms: array of min_tot_servers, ws, we: window start/end)
    # Build (duration, num_servers) list
    durations = we - ws
    server_duration = collections.defaultdict(float)  # num_servers -> total_time

    for num_servers, duration in zip(ms, durations):
        server_duration[num_servers] += duration

    total_time = sum(durations)
    if total_time > 0:
        print("Fraction of time requiring N servers:")
        for n in sorted(server_duration):
            frac = server_duration[n] / total_time
            # print(f"  {n:.0f} servers: {frac:.2%} of the time")
    else:
        print("Warning: total_time is zero, cannot compute fraction of time.")

    # Draw a step-like shaded region
    # We will create the endpoints for fill_between
    x = []
    y = []
    for start, end, val in zip(ws, we, ms):
        x.extend([start, end])
        y.extend([val, val])
    ax.fill_between(x, y, step='post', color='b', alpha=0.15, label='min servers')
    # ax.plot(x, y, color='b', linewidth=2)
    # ax.set_ylabel('Min Servers')
    # ax.set_xlabel('Time (s)')
    # ax.tick_params(axis='x')
    # ax.tick_params(axis='y')
    # ax.legend(loc='upper right', fontsize=13)
    # ax.grid(True)
    return x, y
    
def draw_laxity_and_min_servers():
    filepaths = {
        'sarathi': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/sarathi_round_robin_0.3_1_anytime_3.0_0.025.events.jsonl',
        'vllm': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/vllm_round_robin_0.3_1_anytime_3.0_0.025.events.jsonl',
        'slosserve-edf': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.3_1_anytime_3.0_0.025.events.jsonl',
    }
    fig, axes = plt.subplots(len(filepaths) + 1, figsize=(10, 16), sharex=True, tight_layout = True)

    model_name = 'Qwen/Qwen2.5-7B-Instruct'
    ttft_slo_scale = 3.0
    slo_tpot = 0.025
    slo_ttft_overhead = 0.05
    perf_model = PerfModel.get_perf_model(model_name)
    slo_ttft_fn = lambda x: perf_model.get_batch_time([(0, x)]) * ttft_slo_scale + slo_ttft_overhead
    for i, (scheduler_name, filepath) in enumerate(filepaths.items()):
        events, reqs = analyze_events(filepath)
        if i == 0:
            ax = axes[i]
            draw_min_servers(reqs, ax, slo_ttft_fn, slo_tpot, perf_model)
        draw_laxity(reqs, axes[i+1], slo_ttft_fn, slo_tpot, color = 'red')
        axes[i+1].set_title(scheduler_name, fontsize=22)
    for ax in axes: 
        ax.set_xlim(140, 190)
    for ax in axes[1:]:
        ax.set_ylim(-10, 10)
    fig.savefig(f'laxity_and_min_servers.png', dpi=300, bbox_inches='tight')
 
def _compute_window_series(events: list[dict], window_size: float = 1.0):
    """
    Compute per-window series used by plots.
    Returns (time, num_reqs, num_violations, max_ttft, max_tpot,
             total_prompt_tokens, total_output_tokens, total_tokens, avg_kv_xfer_delay)
    """
    import numpy as np
    if not events:
        return (np.array([]),) * 9
    events = sorted(events, key=lambda x: x['arrival_time'])
    event_idx = 0
    window_start = events[0]['arrival_time']
    window_stats = []
    while window_start + window_size < events[-1]['arrival_time']:
        window_end = window_start + window_size
        start_idx = event_idx
        while event_idx < len(events) and events[event_idx]['arrival_time'] < window_end:
            event_idx += 1
        window_events = events[start_idx:event_idx]
        num_reqs = len(window_events)
        num_violations = sum(e['violation'] != False for e in window_events)
        max_ttft_normalized_laxity = max((e['ttft_normalized_laxity'] for e in window_events), default=0)
        max_tpot = max((e['max_tpot_laxity'] for e in window_events), default=0)
        total_prompt_tokens = sum(e['prompt_tokens'] for e in window_events)
        total_output_tokens = sum(e['output_tokens'] for e in window_events)
        total_tokens = sum(e['total_tokens'] for e in window_events)
        avg_kv_xfer_delay = (sum(e['kv_xfer_delay'] for e in window_events) / max(num_reqs, 1)) if num_reqs else 0.0
        window_stats.append((
            num_reqs,
            num_violations,
            max_ttft_normalized_laxity,
            max_tpot,
            total_prompt_tokens,
            total_output_tokens,
            total_tokens,
            avg_kv_xfer_delay
        ))
        window_start = window_end
    if not window_stats:
        return (np.array([]),) * 9
    num_reqs, num_violations, max_ttft, max_tpot, total_prompt_tokens, total_output_tokens, total_tokens, avg_kv_xfer_delay = zip(*window_stats)
    num_reqs = np.array(num_reqs)
    num_violations = np.array(num_violations)
    max_ttft = np.array(max_ttft)
    max_tpot = np.array(max_tpot)
    total_prompt_tokens = np.array(total_prompt_tokens)
    total_output_tokens = np.array(total_output_tokens)
    total_tokens = np.array(total_tokens)
    avg_kv_xfer_delay = np.array(avg_kv_xfer_delay)
    time = np.arange(len(num_reqs)) * window_size
    return time, num_reqs, num_violations, max_ttft, max_tpot, total_prompt_tokens, total_output_tokens, total_tokens, avg_kv_xfer_delay

def _plot_line(ax, x, y, label: str, color: str, ylabel: str | None = None, linewidth: float = 2.0, stat: str = 'mean'):
    import numpy as np
    if x.size == 0 or y.size == 0:
        return {f'{label}_{stat}': float('nan')}
    ax.plot(x, y, color=color, label=label, linewidth=linewidth)
    if ylabel:
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis='y', labelcolor=color)
    if stat == 'mean':
        v = float(np.mean(y))
    elif stat == 'max':
        v = float(np.max(y))
    elif stat == 'min':
        v = float(np.min(y))
    else:
        v = float(np.mean(y))
    return {f'{label}_{stat}': v}

def _plot_step_hlines(ax, xs: list[float], xe: list[float], ys: list[float], label: str, color: str, ylabel: str | None = None, stat: str = 'mean'):
    import numpy as np
    if not xs:
        return {f'{label}_{stat}': float('nan')}
    prev_y = ys[0]
    for i, (x1, x2, y) in enumerate(zip(xs, xe, ys)):
        ax.hlines(y, x1, x2, linewidth=2, label=label if i == 0 else "", color=color)
        if i > 0 and prev_y != y:
            ax.vlines(x1, prev_y, y, linewidth=2, color=color)
        prev_y = y
    if ylabel:
        ax.set_ylabel(ylabel)
    v = float(np.mean(ys)) if stat == 'mean' else float(np.max(ys))
    return {f'{label}_{stat}': v}

def _plot_fraction(ax, x, num: np.ndarray, den: np.ndarray, label: str, color: str, ylabel: str | None = None, stat: str = 'mean'):
    import numpy as np
    eps = 1e-12
    y = num / np.maximum(den, eps)
    return _plot_line(ax, x, y, label, color, ylabel=ylabel, stat=stat)

def calc_batch_size_distribution(
    events: List["Event"],
    ax: Optional[plt.Axes] = None,
    *,
    # --- plotting controls ---
    color: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = 0.35,
    linewidth: float = 1.25,
    # --- binning controls for cross-call consistency ---
    bins: int = 100,
    bin_edges: Optional[np.ndarray] = None,  # <- pass this from a previous call to keep identical widths
    data_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    use_stairs: bool = True,   # contiguous step histogram (no bar gaps)
    show: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot a batch-size distribution that plays nicely when overlaid multiple times.
    - Transparent fill + crisp outline so overlaps are visible.
    - Stable bar widths across calls when you pass the same `bin_edges`.
    - Returns stats *and* the `bin_edges` actually used so you can reuse them.
    """
    batches = [e for e in events if e.event_type == "batch"]
    batch_sizes = np.array([b.total_current_length for b in batches], dtype=float)

    # Figure out bin edges (so they can be reused by future calls)
    if bin_edges is None:
        lo, hi = (np.min(batch_sizes), np.max(batch_sizes)) if data_range is None else data_range
        bin_edges = np.arange(lo, hi + 9, 10)

    # Compute histogram first so we control both the fill and the outline
    counts, edges = np.histogram(batch_sizes, bins=bin_edges, density=density)

    # Make / reuse axis
    if ax is None:
        _, ax = plt.subplots()

    # Draw as contiguous "stairs" (fills perfectly, consistent widths)
    if use_stairs:
        # Filled area (transparent)
        ax.stairs(counts, edges, fill=True, alpha=alpha, label=label, **kwargs)
        # Crisp outline on top
        ax.stairs(counts, edges, fill=False, linewidth=linewidth, **kwargs)
    else:
        # Fallback to hist with fixed bins (still consistent if you pass bin_edges)
        ax.hist(batch_sizes, bins=edges, density=density,
                histtype="stepfilled", alpha=alpha, color=color, label=label, **kwargs)
        ax.hist(batch_sizes, bins=edges, density=density,
                histtype="step", linewidth=linewidth, color=color, **kwargs)

    # Styling
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Density" if density else "Frequency")
    ax.set_title("Batch Size Distribution")
    ax.set_yscale("log")
    if label is not None:
        ax.legend(frameon=False)

    stats = {
        "mean": float(np.mean(batch_sizes)),
        "max": float(np.max(batch_sizes)),
        "min": float(np.min(batch_sizes)),
        "std": float(np.std(batch_sizes)),
        "median": float(np.median(batch_sizes)),
        "percentile_90": float(np.percentile(batch_sizes, 90)),
        "percentile_95": float(np.percentile(batch_sizes, 95)),
        "percentile_99": float(np.percentile(batch_sizes, 99)),
        "bin_edges": edges,     # <- return these for reuse
    }

    if show:
        plt.tight_layout()

    return stats
    
def analyze_slo_violation(reqs: Dict[str, RequestInstance],
                          all_events: List[Event],
                          model_name: str,
                          ttft_slo_scale: float,
                          slo_tpot: float,
                          length_pattern: str,
                          slo_ttft_overhead: float = 0.0,
                          prefix = 'trace_analysis',
                          draw = False,
                          verbose = False):
    if verbose:
        input_lengths = [req.prompt_tokens - req.num_cached_tokens for req in reqs.values()]
        output_lengths = [req.output_tokens for req in reqs.values()]
        total_lengths = [req.prompt_tokens + req.output_tokens for req in reqs.values()]
        print('input_lengths mean', np.mean(input_lengths))
        print('output_lengths mean', np.mean(output_lengths))
        print('total_lengths mean', np.mean(total_lengths))
        print('input_lengths median', np.median(input_lengths))
        print('output_lengths median', np.median(output_lengths))
        print('total_lengths median', np.median(total_lengths))
        print('input_lengths p99', np.percentile(input_lengths, 99))
        print('output_lengths p99', np.percentile(output_lengths, 99))
        print('total_lengths p99', np.percentile(total_lengths, 99))
        print('total_lengths', np.percentile(total_lengths, 99))
    
    perf_model = PerfModel.get_perf_model(model_name, length_pattern)
    slo_ttft_fn = lambda req: req.zero_load_ttft * ttft_slo_scale + slo_ttft_overhead
    # slo_ttft_fn = lambda req: req.prefill_ddl - req.arrival_time 
    events = []
    
    slo_attained_reqs = set()
    # Use explicit dicts for each event for clarity
    if verbose:
        print('--analyze_slo_violation--')
    for req in reqs.values():
        req.get_stats(slo_ttft_fn, slo_tpot)        
        violate = req.violate_slo()
        
        event = {
            'arrival_time': req.arrival_time,
            'violation': violate,
            # 'tpot_violation': violate == 'tpot',
            # 'ttft_violation': violate == 'ttft',
            # 'any_violation': violate != False,
            # 'unfinished': not req.is_finished,
            'ttft_normalized_laxity': req.ttft_normalized_laxity,
            'max_tpot_laxity': max(req.tpot_laxities, default=0),
            'prompt_tokens': req.prompt_tokens,
            'total_tokens': req.prompt_tokens + req.output_tokens,
            'output_tokens': req.output_tokens,
            'kv_xfer_delay': req.kv_xfer_delay,
        }
        events.append(event)
        if violate == 'none':
            slo_attained_reqs.add(req.req_id)
            
    events = sorted(events, key=lambda x: x['arrival_time'])
    
    violations = [e['violation'] for e in events]
    from collections import Counter
    violations = Counter(violations)
    total = sum(violations.values())
    violations = {k: v / total for k, v in violations.items()}
    attainment_rate = violations.pop('none', 0)
    try:
        rejection_rate = len(list(req for req in reqs.values() if  'reject' in req.finish_reason)) / len(reqs)
    except:
        rejection_rate = 0
    
    print('attainment rate', attainment_rate)
    print('violations', violations)
    print('rejection rate', rejection_rate)
        
    profit = sum(req.profit for req in reqs.values() if not req.is_violate_slo) / len(reqs)

    average_cache_hit_rate = sum(req.cache_hit_rate for req in reqs.values()) / len(reqs)
    
    print('average cache hit rate', average_cache_hit_rate)

    window_size = 1
    time, num_reqs_series, num_violations, max_ttft, max_tpot, total_prompt_tokens, total_output_tokens, total_tokens, avg_kv_xfer_delay = _compute_window_series(events, window_size)


    if not draw:
        return {
            'slo_attainment_rate': attainment_rate,
            'violations': violations,
            'average_cache_hit_rate': average_cache_hit_rate,
            'max_ttft_laxity': np.percentile([req.ttft_normalized_laxity for req in reqs.values()], 99),
            'max_tpot_laxity': np.percentile([req.max_tpot_laxity for req in reqs.values()], 99),
            'profit': profit,
            'extra_metrics': {
                'average_effective_tokens_ratio': float(calc_num_effective_tokens(all_events, reqs, window_size = 1.0)),
                'average_n_active_servers': float(calc_n_active_servers(all_events, window_size = 1.0)),
                'max_num_reqs': float(np.max(num_reqs_series)),
                'max_num_reqs_under_slo': float(np.max(num_reqs_series - num_violations)),
                'rejection_rate': rejection_rate,
            }
        }


    # Create figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), tight_layout=True)

    # Colors for consistency
    color_reqs = 'tab:blue'
    color_violation = 'tab:green'
    color_ttft = 'tab:red'
    color_tpot = 'tab:orange'
    color_prompt_tokens = 'tab:purple'
    color_output_tokens = 'tab:brown'
    color_total_tokens = 'tab:gray'
    color_kv_xfer_delay = 'tab:pink'
    subplot_stats = {}
    # 1. requests and violations
    ax1 = axes[0, 0]
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax1, time, num_reqs_series, 'requests', color_reqs, ylabel='Requests', stat='mean'))
    ax1.tick_params(axis='y', labelcolor=color_reqs)
    ax1_right = ax1.twinx()
    ax1_right.set_ylabel('Violation Count', color=color_violation)
    subplot_stats.update(_plot_line(ax1_right, time, num_violations, 'violations', color_violation, ylabel='Violation Count', stat='mean'))
    # compose legend
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_title('Requests & Violation Rate')

    # 2. requests & max TTFT
    ax2 = axes[0, 1]
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax2, time, num_reqs_series, 'requests_ax2', color_reqs, ylabel='Requests', stat='mean'))
    ax2.tick_params(axis='y', labelcolor=color_reqs)
    ax2_right = ax2.twinx()
    ax2_right.set_ylabel('Max TTFT', color=color_ttft)
    subplot_stats.update(_plot_line(ax2_right, time, max_ttft, 'max_ttft', color_ttft, ylabel='Max TTFT', stat='max'))
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_title('Requests & Max TTFT')

    # 3. requests & max TPOT
    ax3 = axes[0, 2]
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax3, time, num_reqs_series, 'requests_ax3', color_reqs, ylabel='Requests', stat='mean'))
    ax3.tick_params(axis='y', labelcolor=color_reqs)
    ax3_right = ax3.twinx()
    ax3_right.set_ylabel('Max TPOT', color=color_tpot)
    subplot_stats.update(_plot_line(ax3_right, time, max_tpot, 'max_tpot', color_tpot, ylabel='Max TPOT', stat='max'))
    ax3.legend(loc='upper right', frameon=True)
    ax3.set_title('Requests & Max TPOT')

    # 4. requests & total prompt tokens
    ax4 = axes[1, 0]
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax4, time, num_reqs_series, 'requests_ax4', color_reqs, ylabel='Requests', stat='mean'))
    ax4.tick_params(axis='y', labelcolor=color_reqs)
    ax4_right = ax4.twinx()
    ax4_right.set_ylabel('Total Prompt Tokens', color=color_prompt_tokens)
    subplot_stats.update(_plot_line(ax4_right, time, total_prompt_tokens, 'total_prompt_tokens', color_prompt_tokens, ylabel='Total Prompt Tokens', stat='mean'))
    ax4.legend(loc='upper right', frameon=True)
    ax4.set_title('Requests & Total Prompt Tokens')

    # 5. requests & total output tokens
    ax5 = axes[1, 1]
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax5, time, num_reqs_series, 'requests_ax5', color_reqs, ylabel='Requests', stat='mean'))
    ax5.tick_params(axis='y', labelcolor=color_reqs)
    ax5_right = ax5.twinx()
    ax5_right.set_ylabel('Total Output Tokens', color=color_output_tokens)
    subplot_stats.update(_plot_line(ax5_right, time, total_output_tokens, 'total_output_tokens', color_output_tokens, ylabel='Total Output Tokens', stat='mean'))
    ax5.legend(loc='upper right', frameon=True)
    ax5.set_title('Requests & Total Output Tokens')

    # 6. requests & KV xfer delay
    ax6 = axes[1, 2]
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Requests', color=color_reqs)
    subplot_stats.update(_plot_line(ax6, time, num_reqs_series, 'requests_ax6', color_reqs, ylabel='Requests', stat='mean'))
    ax6.tick_params(axis='y', labelcolor=color_reqs)
    ax6_right = ax6.twinx()
    ax6_right.set_ylabel('KV Xfer Delay', color=color_kv_xfer_delay)
    subplot_stats.update(_plot_line(ax6_right, time, avg_kv_xfer_delay, 'kv_xfer_delay', color_kv_xfer_delay, ylabel='KV Xfer Delay', stat='mean'))
    ax6.legend(loc='upper right', frameon=True)
    ax6.set_title('Requests & Total Tokens')
    
    # 7. min servers and violations
    ax = axes[2, 0]
    min_intervals, max_intervals = get_min_servers(list(reqs.values()), slo_ttft_fn, slo_tpot, perf_model)
    window_starts, window_ends, min_prefill_servers, min_decode_servers, min_tot_servers = zip(*min_intervals)
    _, _, max_prefill_servers, max_decode_servers, max_tot_servers = zip(*max_intervals)
    subplot_stats.update(_plot_step_hlines(ax, list(window_starts), list(window_ends), list(min_tot_servers), 'min_servers', 'b', ylabel='Minimum Servers', stat='mean'))
    # Plot max servers as horizontal lines per window and connect with verticals
    # prev_we, prev_mx = None, None
    # for ws, we, mx in zip(window_starts, window_ends, max_tot_servers):
    #     ax.hlines(mx, ws, we, color='r', linewidth=2, linestyle="--", label='max' if ws == window_starts[0] else "")
    #     if prev_we is not None and prev_we == ws:  # connect contiguous intervals
    #         ax.vlines(ws, prev_mx, mx, color='r', linewidth=2, linestyle="--")
    #     prev_we, prev_mx = we, mx
    ax.tick_params(axis='y', labelcolor=color_reqs)
    ax_right = ax.twinx()
    ax.grid(True)
    ax_right.set_ylabel('Violation Count', color=color_violation)
    if time.size:
        t_starts = list(time)
        t_ends = list(time + window_size)
        subplot_stats.update(_plot_step_hlines(ax_right, t_starts, t_ends, list(num_violations), 'violations_ax7', color_violation, ylabel='Violation Count', stat='mean'))
    ax_right.tick_params(axis='y', labelcolor=color_violation)
    # Compose legend from all lines
    # We'll just use the first two as before to avoid double legend entries.
    handles, labels = [], []
    # Add a dummy line for min (blue) and max (red) servers if not already present, for visual legend clarity
    handles.append(plt.Line2D([0], [0], color='b', linewidth=2, label='min'))
    handles.append(plt.Line2D([0], [0], color='r', linewidth=2, linestyle='--', label='max'))
    handles.append(plt.Line2D([0], [0], color=color_violation, linewidth=2, label='# Violation'))
    ax.legend(handles, [h.get_label() for h in handles], loc='upper right', frameon=True)
    ax.set_title('Infinite Loads & SLO Violation Rate (Intervals)')
    
    # 8. fraction of effective tokens & violations
    color_effective_tokens = 'tab:pink'
    batches = [event for event in all_events if event.event_type == 'batch']
    n_effective_tokens, n_total_tokens = zip(*[event.classify_tokens(slo_attained_reqs) for event in batches])
    n_effective_tokens = np.array(n_effective_tokens)
    n_total_tokens = np.array(n_total_tokens)
    times = [event.timestamp for event in batches]
    # print('times', times[:10])
    # Aggregate n_effective_tokens and n_total_tokens by window_size, using event[k] starts by times[k]
    # Bin by window_size, using times as the start of each window
    binned_effective = []
    binned_total = []
    binned_times = []
    if len(times) > 0:
        min_time = times[0]
        max_time = times[-1]
        num_windows = int(np.ceil((max_time - min_time) / window_size))
        for i in range(num_windows):
            window_start = min_time + i * window_size
            window_end = window_start + window_size
            idxs = [k for k, t in enumerate(times) if window_start <= t < window_end]
            if idxs:
                binned_effective.append(np.sum(n_effective_tokens[idxs]))
                binned_total.append(np.sum(n_total_tokens[idxs]))
                binned_times.append(window_start)
            else:
                binned_effective.append(0)
                binned_total.append(0)
                binned_times.append(window_start)
    n_effective_tokens = np.array(binned_effective)
    n_total_tokens = np.array(binned_total)
    times = np.array(binned_times)
    ax8 = axes[2, 1]
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Fraction of Effective Tokens', color=color_effective_tokens)
    ax8_right = ax8.twinx()
    ax8_right.set_ylabel('SLO Violation Rate', color=color_violation)
    subplot_stats.update(_plot_line(ax8_right, time, num_violations, 'violations_ax8', color_violation, ylabel='Violation Count', stat='mean'))
    ax8_right.tick_params(axis='y', labelcolor=color_violation)
    subplot_stats.update(_plot_fraction(ax8, times, n_effective_tokens, n_total_tokens, 'fraction_effective_tokens', color_effective_tokens, ylabel='Fraction of Effective Tokens', stat='mean'))
    ax8.tick_params(axis='y', labelcolor=color_effective_tokens)
    ax8.set_title('Fraction of Effective Tokens & SLO Violation Rate')
    
    # 9. absolute tokens & violations
    ax9 = axes[2, 2]
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Absolute Effective Tokens', color=color_effective_tokens)
    ax9_right = ax9.twinx()
    ax9_right.set_ylabel('SLO Violation Rate', color=color_violation)
    subplot_stats.update(_plot_line(ax9_right, time, num_violations, 'violations_ax9', color_violation, ylabel='Violation Count', stat='mean'))
    ax9_right.tick_params(axis='y', labelcolor=color_violation)
    subplot_stats.update(_plot_line(ax9, times, n_effective_tokens, 'absolute_effective_tokens', color_effective_tokens, ylabel='Absolute Effective Tokens', stat='mean'))
    subplot_stats.update(_plot_line(ax9, times, n_total_tokens, 'absolute_total_tokens', color_total_tokens, ylabel='Absolute Total Tokens', stat='mean'))
    ax9.legend(loc='upper right', frameon=True)
    ax9.tick_params(axis='y', labelcolor=color_effective_tokens)
    ax9.set_title('Absolute Effective Tokens & SLO Violation Rate')

    fig.suptitle('Requests and SLO/Token Metrics Over Time', fontsize=20)
    fig.savefig(f'{prefix}.png', dpi=300, bbox_inches='tight')
    print(f'Saved {prefix}.png')
    
    return {
        'slo_attainment_rate': attainment_rate,
        'violations': violations,
        'average_cache_hit_rate': average_cache_hit_rate,
            'figure': f'{prefix}.png',
        'max_ttft_laxity': np.percentile([req.ttft_normalized_laxity for req in reqs.values()], 99),
            'max_tpot_laxity': np.percentile([req.max_tpot_laxity for req in reqs.values()], 99),
        'profit': profit,
        'subplot_stats': subplot_stats,
        'extra_metrics': {
                'average_effective_tokens_ratio': float(calc_num_effective_tokens(all_events, reqs, window_size = 1.0)),
                'average_n_active_servers': float(calc_n_active_servers(all_events, window_size = 1.0)),
                'max_num_reqs': float(np.max(num_reqs_series)),
                'max_num_reqs_under_slo': float(np.max(num_reqs_series - num_violations)),
                'rejection_rate': rejection_rate,
            }
        
    }

def analyze_overprovisioning(filepath, loads):
    import numpy as np
    import matplotlib.pyplot as plt
    data = []
    for load in loads:
        events, reqs = analyze_events(filepath.format(load=load))
        slo_violations = [req.violate_slo() != False for req in reqs.values()]
        e2e_latencies = [req.e2e_latency for req in reqs.values()]
        data.append((load, sum(slo_violations) / len(slo_violations), sum(e2e_latencies) / len(e2e_latencies)))
    
    loads, slo_violations, e2e_latencies = zip(*sorted(data, key=lambda x: x[0]))
    loads = np.array(loads)
    slo_violations = np.array(slo_violations)
    e2e_latencies = np.array(e2e_latencies)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#1f77b4'
    color2 = '#ff7f0e'

    ax1.set_xlabel('Load')
    ax1.set_ylabel('SLO Violation Rate', color=color1, fontsize=12)
    ln1 = ax1.plot(loads, slo_violations, marker='o', color=color1, label='SLO Violation Rate', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    hline1 = ax1.axhline(0.01, color='red', linestyle='--', linewidth=2, label='1% Violation')
    hline2 = ax1.axhline(0.05, color='red', linestyle='-.', linewidth=2, label='5% Violation')
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg. E2E Latency (s)', color=color2, fontsize=12)
    ln2 = ax2.plot(loads, e2e_latencies, marker='s', color=color2, label='Avg. E2E Latency', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(e2e_latencies) * 1.1)

    # Combine legends
    lns = ln1 + ln2 + [hline1, hline2]
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left', frameon=True, fontsize=11)

    fig.suptitle('Overprovisioning: SLO Violations & E2E Latency vs Load', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'figs/overprovisioning.png', dpi=300, bbox_inches='tight')
    print(f'Saved figs/overprovisioning.png')

def analyze_results(filepath,
                    ttft_slo_scale: float,
                    slo_tpot: float,
                    slo_ttft_overhead: float = 0.0):
    import pandas as pd
    import os
    df = pd.read_json(f'{filepath}/results.jsonl', lines=True)
    # df = df[df['n_device'] == n_device]
    df = df[df['ttft_slo_scale'] == ttft_slo_scale]
    df = df[df['slo_tpot'] == slo_tpot]
    df = df.drop_duplicates(subset=['load_scale', 'n_device', 'ttft_slo_scale', 'slo_tpot', 'scheduling_policy', 'routing_policy'], keep = 'last')
    
    results = []
    import concurrent.futures

    def process_row(row):
        import os  # For multiprocess picklability (re-import in child)
        event_file = row['event_file']
        if not os.path.exists(event_file):
            event_file = event_file.replace('events.jsonl', '0.events.jsonl')
            if not os.path.exists(event_file):
                return None
        events, reqs = analyze_events(event_file)
        result = analyze_slo_violation(
            reqs, events,
            model_name='Qwen/Qwen2.5-7B-Instruct',
            ttft_slo_scale=row['ttft_slo_scale'],
            slo_ttft_overhead = slo_ttft_overhead,
            slo_tpot=row['slo_tpot'], draw=False
        )
        return (
            row['scheduling_policy'],
            row['load_scale'],
            1 - result['slo_attainment_rate'],
            result['slo_attainment_rate'],
            row['slo_violation_rate'],
            result['max_ttft_laxity'],
            result['max_tpot_laxity'],
        )

    # Convert DataFrame rows to dicts for picklability
    rows = [row._asdict() if hasattr(row, '_asdict') else row for _, row in df.iterrows()]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        for future in concurrent.futures.as_completed(future_to_row):
            out = future.result()
            if out is not None:
                scheduling_policy, load_scale, slo_violation, slo_attainment_rate, slo_violation_rate, max_ttft_laxity, max_tpot_laxity = out
                print('Scheduling Policy', scheduling_policy)
                print('Load Scale', load_scale)
                print('SLO Attainment Rate', slo_attainment_rate, 'before', 1 - slo_violation_rate)
                print('--------------------------------')
                results.append((scheduling_policy, load_scale, slo_violation, max_ttft_laxity, max_tpot_laxity))
    df = pd.DataFrame(results, columns=['scheduling_policy', 'load_scale', 'slo_violation', 'max_ttft_laxity', 'max_tpot_laxity'])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))
    ax1.set_xlabel('Load Scale')
    ax1.set_ylabel('SLO Attainment Rate')
    ax2.set_xlabel('Load Scale')
    ax2.set_ylabel('Max TTFT Laxity')
    ax3.set_xlabel('Load Scale')
    ax3.set_ylabel('Max TPOT Laxity')
    for scheduling_policy, tdf in df.groupby('scheduling_policy'):
        tdf = tdf.sort_values(by='load_scale')
        ax1.plot(tdf['load_scale'], tdf['slo_violation'], marker='o', label=scheduling_policy)
        ax2.plot(tdf['load_scale'], tdf['max_ttft_laxity'], marker='o', label=scheduling_policy)
        ax3.plot(tdf['load_scale'], tdf['max_tpot_laxity'], marker='o', label=scheduling_policy)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig.savefig(f'slo_violation.png', dpi=300, bbox_inches='tight')
    print(f'Saved slo_violation.png')


def plot_throughput_survival(events, ax=None, label=None, color=None):
    """
    Plot empirical throughput survival f(tpt) = Pr(T > tpt) for given events.
    Each 'event' must have attributes:
        - event_type == 'batch'
        - num_scheduled_tokens: dict
        - elapsed: float (sec)
        - timestamp: float (optional, ignored here)

    Parameters
    ----------
    events : list
        List of event objects or dicts.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure/ax is created.
    label : str, optional
        Label for the line (e.g., scheduler name).
    color : str, optional
        Line color.
    """
    # --- extract throughput per batch ---
    throughputs = []
    for e in events:
        if getattr(e, "event_type", None) == "batch":
            try:
                n_tokens = sum(e.num_scheduled_tokens.values())
                if e.elapsed and e.elapsed > 0:
                    throughputs.append(n_tokens / e.elapsed)
            except Exception:
                pass
    if not throughputs:
        print("No valid batch events.")
        return ax

    # --- empirical survival ---
    vals = np.sort(throughputs)
    n = len(vals)
    ccdf = np.arange(n - 1, -1, -1, dtype=float) / n

    # --- plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(vals, ccdf, where="post", label=label, color=color)
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("f(tpt) = Pr(T > tpt)")
    ax.set_title("Throughput Survival (CCDF)")
    ax.grid(True, alpha=0.4)
    if label:
        ax.legend()
    return ax

def plot_batch_size_survival(
    events,
    ax=None,
    *,
    label=None,
    color=None,
    time_window=None,   # e.g., (t0, t1) to restrict by timestamp
    event_type="batch", # change if your batch events use another tag
    return_data=False   # if True, returns (sorted_sizes, ccdf)
):
    """
    Plot empirical survival (CCDF) of batch sizes: f(b) = Pr(BatchSize > b).

    Parameters
    ----------
    events : list
        Iterable of event objects with:
          - event_type (str) == 'batch' (configurable)
          - num_scheduled_tokens (dict)  -> batch size = sum(values)
          - timestamp (float) (optional, only used if time_window is given)
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; new figure/axis created if None.
    label : str, optional
        Legend label (e.g., scheduler name).
    color : str, optional
        Line color.
    time_window : tuple(float, float), optional
        (t0, t1) inclusive bounds on event.timestamp.
    event_type : str
        Event type to treat as a batch event (default "batch").
    return_data : bool
        If True, return (sorted_batch_sizes, ccdf) arrays.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plotted curve.
    (optional) (np.ndarray, np.ndarray)
        (sorted_sizes, ccdf) if return_data=True.
    """
    # --- collect batch sizes ---
    sizes = []
    use_window = isinstance(time_window, (tuple, list)) and len(time_window) == 2
    t0, t1 = (time_window or (None, None))

    for e in events:
        if getattr(e, "event_type", None) != event_type:
            continue
        if use_window:
            ts = getattr(e, "timestamp", None)
            if ts is None or ts < t0 or ts > t1:
                continue
        try:
            bs = sum(getattr(e, "num_scheduled_tokens").values())
        except Exception:
            continue
        if bs is not None:
            sizes.append(bs)

    if not sizes:
        print("No batch sizes found for plotting.")
        return ax if not return_data else (ax, (np.array([]), np.array([])))

    # --- empirical survival (right-continuous step) ---
    vals = np.sort(np.asarray(sizes))
    n = len(vals)
    ccdf = np.arange(n - 1, -1, -1, dtype=float) / n  # P(BatchSize > b)

    # --- plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(vals, ccdf, where="post", label=label, color=color)
    ax.set_xlabel("Batch size (tokens)")
    ax.set_ylabel("f(b) = Pr(BatchSize > b)")
    ax.set_title("Batch Size Survival (CCDF)")
    ax.grid(True, alpha=0.4)
    if label:
        ax.legend()

    if return_data:
        return ax, (vals, ccdf)
    return ax

def draw_bs_comparison():
    # qwen7b_code_filepaths = {
    #     'vLLM (OPT)': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/vllm_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    #     'Ours': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    #     'Sarathi': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/sarathi_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    # }
    
    gemma26b_chat_filepaths = [
        ('vLLM', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/vllm_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ('Sarathi', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/sarathi_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ('QLM', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/qlm_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ('Ours', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
    ]
    
    # qwen7b_chat_filepaths = {
    #     'vLLM': 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime/vllm_round_robin_1.0_1_anytime_5.0_0.1.events.jsonl',
    #     'Ours': 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime/slosserve-edf_round_robin_1.0_1_anytime_5.0_0.1.events.jsonl',
    #     'Sarathi': 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime/sarathi_round_robin_1.0_1_anytime_5.0_0.1.events.jsonl',
    #     'QLM': 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1201_anytime/qlm_round_robin_1.0_1_anytime_5.0_0.1.events.jsonl',
    # }
    
    import numpy as np
    
    def calc_batch_stats(event):
        batch_size = sum(event.num_scheduled_tokens.values())
        elapsed = event.elapsed
        tpt = batch_size / elapsed 
        return batch_size, elapsed, tpt
    
    def compute_batch_sizes(events, time_window = [120, 130]):
        # Extract batch size for each batch event
        return [sum(event.num_scheduled_tokens.values()) 
                for event in events if event.event_type == 'batch' and event.timestamp >= time_window[0] and event.timestamp <= time_window[1]]

    # Set commonly used larger font sizes
    label_fontsize = 22
    title_fontsize = 24
    legend_fontsize = 18
    tick_fontsize = 18

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
    fig, ax = plt.subplots(figsize=(4, 4))
    for scheduler_name, filepath in gemma26b_chat_filepaths:
        events, reqs = analyze_events(filepath)
        batch_sizes = compute_batch_sizes(events)
        batch_sizes = [bs for bs in batch_sizes if bs is not None]
        if not batch_sizes:
            continue
        values = np.sort(batch_sizes)
        counts = np.arange(len(values)-1, -1, -1) / float(len(values))
        # ax2.hist(batch_sizes, density=True, label=scheduler_name)
        ax.step(values, counts, where="post", label=scheduler_name)
        # plot_batch_size_survival(events, ax=ax4, label=scheduler_name)
        # plot_throughput_survival(events, ax=ax3, label=scheduler_name)
    ax.vlines(x=212,ymin=0,ymax=1,color='red',linestyle='--',linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=label_fontsize)
    ax.set_ylabel('Pr (BatchSize > b)', fontsize=label_fontsize)
    # ax.set_title('Batch Size CCDF', fontsize=title_fontsize)
    # ax2.set_xlabel('Batch Size', fontsize=label_fontsize)

    ax.legend(fontsize=legend_fontsize)
    # ax2.legend(fontsize=legend_fontsize)
    # ax1.grid()
    # ax2.grid()

    # ax3.set_xlabel('Throughput (tokens/sec)', fontsize=label_fontsize)
    # ax3.set_ylabel('f(tpt) = Pr(T > tpt)', fontsize=label_fontsize)
    # ax3.set_title('Throughput Survival', fontsize=title_fontsize)
    # ax3.legend(fontsize=legend_fontsize)
    # ax3.grid()

    # ax4.set_xlabel('Batch Size', fontsize=label_fontsize)
    # ax4.set_ylabel('f(b) = Pr(BatchSize > b)', fontsize=label_fontsize)
    # ax4.set_title('Batch Size Survival', fontsize=title_fontsize)
    # ax4.legend(fontsize=legend_fontsize)
    # ax4.grid()

    # Make tick labels larger for all axes
    # for ax in [ax1]:
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
    fig.savefig(f'bs_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'bs_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved bs_comparison.png')

def draw_reqeust_arrivals(trace_name, window_start: int, window_end: int):
    from Dataset.dataset import ArrivalTimes
    arrivals = ArrivalTimes.load(trace_name, window_start = window_start, window_end = window_end)
    arrivals = [(t, 1) for t in arrivals.arrival_times]
    fig, ax = plt.subplots(figsize=(8, 3), tight_layout = True)
    ax.tick_params(axis='both', which='major', labelsize=18)
    arg_values = plot_windowed_average_step(
        arrivals, window_size=4.0, ax=ax, label = 'num_requests', 
        color = 'black', reduction = 'sum')
    ax.set_xlabel('Time (s)', fontsize=18)
    ax.set_ylabel('# Arrival / Window', fontsize=15)
    ax.grid(True)

    # Remove the legend if it exists (forcefully disables it even if plotting function adds one)
    ax.get_legend().remove() if ax.get_legend() is not None else None
    fig.savefig(f'request_arrivals.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'request_arrivals.pdf', dpi=300, bbox_inches='tight')
    
    arg_values = sorted(arg_values)
    arg_values = np.cumsum(arg_values)
    arg_values /= arg_values[-1]
    arg_values = list(arg_values)[::-1]
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout = True)
    ax.step(np.linspace(0, 1, len(arg_values)), arg_values, label = 'num_requests', color = 'black')
    ax.set_xlabel('Time Fraction', fontsize=18)
    ax.set_ylabel('Request Fraction', fontsize=18)
    ax.grid(True)
    ax.hlines(0.2, 0, 1, color = 'red', linestyle = '--')
    ax.vlines(0.17, 0, 0.2, color = 'red', linestyle = '--')
    ax.set_xticks([0, 0.5, 0.80, 1])
    ax.annotate("17%", (0.12, 0.0), fontsize=16, ha='center', va='bottom', color = 'red')
    # ax.annotate("23%", (0.23, 0.0), fontsize=16, ha='center', va='bottom', color = 'red')
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    fig.savefig(f'request_arrivals_cumulative.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'request_arrivals_cumulative.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved request_arrivals_cumulative.png')

def draw_token_waste_comparison():
    # filepaths = {
    #     'vLLM': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/vllm_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    #     'Ours': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    #     'Sarathi': 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/sarathi_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl',
    # }
    # model_name = 'Qwen/Qwen2.5-7B-Instruct'
    # perf_model = PerfModel.get_perf_model(model_name)
    # ttft_slo_scale = 2.5
    # slo_tpot = 0.025
    # slo_ttft_overhead = 0.026
    
    filepaths = {
        ''
    }
    
    gemma26b_chat_filepaths = [
        ('vLLM', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/vllm_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ('Sarathi', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/sarathi_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ('QLM', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/qlm_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'), 
        ('Ours', 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'),
        ]
    model_name = 'gemma/gemma-3-27b-it'
    perf_model = PerfModel.get_perf_model(model_name)
    ttft_slo_scale = 3.0
    slo_tpot = 0.05
    slo_ttft_overhead = 0.04
    
    slo_ttft_fn = lambda req: req.zero_load_ttft * ttft_slo_scale + slo_ttft_overhead
    
    events = []
    
    fig, ax = plt.subplots(figsize=(8,6), tight_layout = True)
    drawn = False
    twin_ax = ax.twinx()
    n_req = 0
    n_violation = 0
    for scheduler_name, filepath in gemma26b_chat_filepaths:
        events, reqs = analyze_events(filepath, verbose = False)
        reqs = {k: v for k, v in reqs.items() if v.arrival_time >= 80 and v.arrival_time <= 120}
        for req in reqs.values():
            req.zero_load_ttft = perf_model.get_batch_time([(0, req.prompt_tokens)])
        for req in reqs.values():
            req.get_stats(slo_ttft_fn, slo_tpot)
        if req.arrival_time >= 90 and req.arrival_time <= 100:
            n_req += 1
            if not req.violate_slo() == 'none':
                n_violation += 1
        from collections import Counter
        print(scheduler_name)
        print(Counter([req.violate_slo() for req in reqs.values()]))
        fraction = calc_num_effective_tokens(events, reqs, window_size = 1.5, ax=ax, label=scheduler_name)
        print(scheduler_name, fraction)
        if not drawn:
            draw_min_servers(reqs, twin_ax, slo_ttft_fn, slo_tpot, perf_model)
            drawn = True
    print('n_req', n_req, 'n_violation', n_violation, 'fraction')
    # twin_ax.set_xlabel('Time (s)')
    twin_ax.set_ylabel('Load')
    ax.set_ylabel('# Wasted Tokens')
    ax.set_xlabel('Time (s)')
    twin_ax.set_xlim(80, 150)
    twin_ax.tick_params(axis='y', left=False, right=False, labelleft=False)
    twin_ax.spines['right'].set_visible(False)
    # twin_ax.set_yticklabels([]) 
    twin_ax.hlines(0.8, 80, 150, color='red', linestyle='--', linewidth=2)
    twin_ax.text(135, 0.65, "High Load", color="black", fontsize=20, weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    twin_ax.text(135, 0.9, "Over load", color="black", fontsize=20, weight="bold", ha="center", bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    ax.legend(
        loc='center left',          # anchor legend box to the left-center of its bbox
        bbox_to_anchor=(1.02, 0.5), # (x, y) coordinates relative to the axes; 1.02 puts it just outside the right edge
        frameon=True,
        framealpha=1.0,
        fontsize=20,
        edgecolor='black',
        ncol=1                      # usually 1 column looks cleaner on the side
    )
    ax.tick_params(axis='y', labelsize=18)
    fig.savefig(f'token_waste_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'token_waste_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f'Saved token_waste_comparison.png')

def analyze_auto_scaling(reqs: Dict[str, RequestInstance]):
    n_schedulings = []
    rescheduling_time_gaps = []
    pairs = [('finish', 'rescheduling'), ('rescheduling', 'dispatch'), ('dispatch', 'arrival'), ('rescheduling', 'finish')]
    from collections import defaultdict
    times = defaultdict(list)
    for req in reqs.values():
        # calculate the time between two reschedulings
        # n_reschedulings = 0 # number of reschedulings
        rescheduling_times = []
        for i, event in enumerate(req.events):
            for x,y in pairs:
                if event.event_type == x:
                    for j in range(i+1, len(req.events)):
                        if y in req.events[j].event_type:
                            times[f'{x}->{y}'].append(req.events[j].timestamp - event.timestamp)
                            break
                    
            if event.event_type == 'rescheduling':
                rescheduling_times.append(event.timestamp)
        rescheduling_time_gaps.extend([x-y for x, y in zip(rescheduling_times[1:], rescheduling_times[:-1])])
        n_schedulings.append(len(rescheduling_times))
    from collections import Counter
    print(Counter(n_schedulings))
    if len(n_schedulings) > 0:
        print('Average number of schedulings', np.mean(n_schedulings), '+-', np.std(n_schedulings))
        print('Average rescheduling time gap', np.mean(rescheduling_time_gaps), '+-', np.std(rescheduling_time_gaps))
    for k, v in times.items():
        print(k, np.mean(v), '+-', np.std(v))
    return rescheduling_time_gaps, n_schedulings
def plot_windowed_average_step(time_value_pairs, window_size=5.0, ax=None, step_where="post", label = "None", reduction = 'mean', **kwargs):
    """
    Plot average value over time in a step graph using a sliding window.

    Args:
        time_value_pairs: list of (time, value) tuples (sorted ascending)
        window_size: width of sliding window in same time unit (default: 5.0)
        ax: optional matplotlib Axes to draw on
        step_where: 'pre', 'post', or 'mid' (controls step alignment)
    """
    # Convert to numpy arrays
    data = np.array(time_value_pairs)
    times, values = data[:, 0], data[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Compute averages at discrete centers
    centers = np.linspace(times.min(), times.max(), 200)
    avg_values = []
    for t in centers:
        mask = (times >= t - window_size / 2) & (times <= t + window_size / 2)
        avg_values.append(values[mask].sum() if reduction == 'sum' else values[mask].mean() if np.any(mask) else np.nan)

    # Remove NaN gaps (optional)
    centers = np.array(centers)
    avg_values = np.array(avg_values)
    valid = ~np.isnan(avg_values)
    centers, avg_values = centers[valid], avg_values[valid]

    # Plot step line
    ax.step(centers, avg_values, where=step_where, linewidth=2, label=label, **kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Average value")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return avg_values

def analyze_high_load_delay():
    # filepath = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo_req-0.4_1.0_4_anytime_3.0_0.025.events.jsonl"
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime/slosserve-edf_round_robin_0.4_1_anytime_3.0_0.1.events.jsonl'
    events, reqs = analyze_events(filepath)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    slo_ttft_fn = lambda req: req.prefill_ddl - req.arrival_time
    perf_model = PerfModel.get_perf_model('Qwen/Qwen2.5-7B-Instruct')
    # draw_min_servers(reqs, ax, slo_ttft_fn, 0.025, perf_model)
    from collections import defaultdict
    gaps = defaultdict(list)
    
    zero_load_ttfts = []
    for req in reqs.values():
        zero_load_ttfts.append((req.arrival_time, req.zero_load_ttft))
    zero_load_ttfts.sort(key=lambda x: x[0])
    zero_load_ttfts = [x[1] for x in zero_load_ttfts]
    print('zero_load_ttfts mean', np.mean(zero_load_ttfts), '+-', np.std(zero_load_ttfts))
    print('zero_load_ttfts median', np.median(zero_load_ttfts))
    print('zero_load_ttfts p99', np.percentile(zero_load_ttfts, 99))
    print('zero_load_ttfts p1', np.percentile(zero_load_ttfts, 1))
    print('zero_load_ttfts p99', np.percentile(zero_load_ttfts, 99))
    num_requests = []
    
    for event in events:
        if event.event_type == 'batch':
            gaps['batch'].append((event.timestamp, event.elapsed))
            gaps['scheduling_overhead'].append((event.timestamp, event.scheduling_overhead))
        if event.event_type == 'process_input':
            gaps['process_input'].append((event.timestamp, event.elapsed))
        if event.event_type == 'engine_step':
            gaps['engine_step'].append((event.timestamp, event.elapsed))
            num_requests.append((event.timestamp, event.num_requests))
    # Create a table summarizing the mean, P90, and P99 for each key
    import pandas as pd
    table_rows = []
    for k in ['batch', 'scheduling_overhead', 'process_input', 'engine_step']:
        _, times = zip(*gaps[k])
        times = np.array(times)
        row = {
            'Event': k,
            'Mean': np.mean(times),
            'P90': np.percentile(times, 90),
            'P99': np.percentile(times, 99)
        }
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    print("\nEvent Delay Table (seconds):")
    print(df.to_string(index=False, float_format="%.6f"))
        
    plot_windowed_average_step(num_requests, window_size=1.0, ax=ax, label = 'num_requests', color = 'black')
    ax.set_ylabel('Number of Requests', fontsize=18)
    def check_event(event):
        # if event.event_type in ['dispatch', , 'rescheduling']:
        #     return [(event.event_type, event.timestamp)]
        if event.event_type in ['arrival-router', 'dispatch', 'rescheduling']:
            return [(event.event_type, event.timestamp)]
        if event.event_type == 'arrival':
            assert isinstance(event, Arrival)
            return [('arrival-scheduler', event.add_req_time), ('added_to_scheduler', event.timestamp)]
        if event.event_type == 'finish':
            return [('finish-' + event.finish_reason, event.timestamp)]
        return []
    n_resch, n_resch_slo_attained = 0, 0
    for req in reqs.values():
        # we only consider when a rejection and rerouting happens
        event_pairs = sum([check_event(event) for event in req.events], [])
        event_pairs.sort(key=lambda x: x[1])
        idx = len(event_pairs) - 1
        if any(x[0] == 'rescheduling' for x in event_pairs):
            n_resch += 1
            if req.violate_slo() == 'none':
                n_resch_slo_attained += 1
        # if any(x[0] == 'finish-length' for x in event_pairs):
        #     continue
        # while idx >= 0 and event_pairs[idx][0] != 'rescheduling':
        #     idx -= 1
        # if idx < 0:
        #     continue
        # event_pairs = event_pairs[:idx+1]
        for i in range(len(event_pairs) - 1):
            gaps[f'{event_pairs[i][0]}->{event_pairs[i+1][0]}'].append(
                (event_pairs[i+1][1], event_pairs[i+1][1] - event_pairs[i][1]))
    # print('n_resch', n_resch, 'n_resch_slo_attained', n_resch_slo_attained, 'ratio', n_resch_slo_attained / n_resch)
    
    for k, v in gaps.items():
        v, times = zip(*v)
        print(k, np.mean(times), '+-', np.std(times))
                    
    ax.set_xlim(30, 80)
    twinx = ax.twinx()
    V = []
    for k, v in gaps.items():
        print(k, len(v), v[0])
        # V.extend(list(v))
        # V.sort(key=lambda x: x[0])
        plot_windowed_average_step(v, window_size=0.2, ax=twinx, label = k)
    # plot_windowed_average(gaps, window_size=1.0, ax=twinx)
    twinx.set_ylabel('Average Delay', fontsize=18)
    twinx.set_ylim(0, 0.05)
    fig.savefig(f'high_load_delay.png', dpi=300, bbox_inches='tight')
    

def update_expected():
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/new_results.jsonl'
    import json
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    # example = data[-10:]
    import concurrent.futures

    def process_example(example):
        print('*' * 100)
        events, reqs = analyze_events(example['event_file'])

        results = analyze_slo_violation(reqs, events,
                            model_name='Qwen/Qwen2.5-7B-Instruct',
                            length_pattern='sharegpt_code',
                            ttft_slo_scale=3,
                            slo_tpot=0.025,
                            prefix='debug-events',
                            slo_ttft_overhead=0.05,
                            draw=False)
        print(example['scheduling_policy'], example['n_device'], results['slo_attainment_rate'], 1 - example['slo_violation_rate'])
        example['slo_violation_rate'] = 1 - results['slo_attainment_rate']
        print('*' * 100)
        return example

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(process_example, data))
    with open(filepath, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
        
def update_expected_chat():
    filepath = 'tmp1.json'
    import json
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    # example = data[-10:]
    import concurrent.futures

    def process_example(example):
        print('*' * 100)
        events, reqs = analyze_events(example['event_file'])

        results = analyze_slo_violation(reqs, events,
                            model_name='Qwen/Qwen2.5-7B-Instruct',
                            length_pattern='azure_chat_23',
                            ttft_slo_scale=5,
                            slo_tpot=0.1,
                            prefix='debug-events',
                            slo_ttft_overhead=0.03,
                            draw=False)
        print(example['scheduling_policy'], example['n_device'], results['slo_attainment_rate'], 1 - example['slo_violation_rate'])
        example['slo_violation_rate'] = 1 - results['slo_attainment_rate']
        print('*' * 100)
        return example

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(process_example, data))
    with open(filepath, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')

def map_times_to_servers(step_x, step_y, query_times):
    """Given a piecewise-constant step defined by (step_x, step_y),
    return the server value at each query time."""
    xs = np.asarray(step_x, dtype=float)
    ys = np.asarray(step_y, dtype=float)
    qt = np.asarray(query_times, dtype=float)
    # index of the step just before or at time t
    idx = np.searchsorted(xs, qt, side='right') - 1
    idx = np.clip(idx, 0, len(ys) - 1)
    return ys[idx]

# --- 1) Build a step function: active requests over time ---
def build_active_requests_step(arrival_finish_times):
    """
    Returns arrays (event_times, counts_after_event), where counts_after_event[i]
    is the number of active requests immediately AFTER processing the event at event_times[i].
    Departures at the same timestamp are processed before arrivals (more realistic).
    """
    events = []  # (time, delta) where delta in {-1, +1}
    for a, f in arrival_finish_times:
        # Treat departures before arrivals at the same timestamp
        events.append((float(a), +1))
        events.append((float(f), -1))
    # Sort by time, then departures first
    events.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))

    event_times = []
    counts_after = []
    cur = 0
    for t, d in events:
        cur += d
        event_times.append(t)
        counts_after.append(cur)
    return np.asarray(event_times), np.asarray(counts_after, dtype=int)

def count_at_times(event_times, counts_after, query_times):
    """
    For each query time, return the active request count (piecewise-constant step).
    """
    qt = np.asarray(query_times, dtype=float)
    idx = np.searchsorted(event_times, qt, side='right') - 1
    idx = np.clip(idx, -1, len(counts_after) - 1)
    # If idx == -1, it means time is before first event -> 0 active
    result = np.where(idx >= 0, counts_after[idx], 0)
    return result

def energy_per_second_vs_concurrency(
    arrival_finish_times: List[Tuple[float, float]],
    times: List[float],
    powers: List[float],
    bin_width: float = 1.0,
):
    """
    Compute (bin_centers, power_per_sec, n_requests) with no external deps.

    - Integrates piecewise-linearly between consecutive (times, powers) samples
      and distributes energy into fixed-width bins (default 1s).
    - Concurrency is evaluated at the bin centers using a sweep over arrivals/finishes.

    Returns:
        bin_centers: List[float]    # time at center of each bin [s]
        power_per_sec: List[float]  # average power in that bin [W] == energy/bin_width [J/s]
        n_requests: List[int]       # concurrent active requests at bin center
    """
    if not times or not powers or len(times) != len(powers):
        return [], [], []

    # --------- Step 1: sort power samples by time ---------
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    powers = [powers[i] for i in order]

    t_min = times[0]
    t_max = times[-1]
    if t_max <= t_min:
        return [], [], []

    # Align bins to multiples of bin_width covering [t_min, t_max)
    import math
    start_edge = math.floor(t_min / bin_width) * bin_width
    end_edge   = math.ceil(t_max / bin_width) * bin_width
    nbins = int(round((end_edge - start_edge) / bin_width))
    print('nbins', nbins)
    energy_bins = [0.0] * nbins  # Joules in each bin

    # --------- Step 2: distribute segment energy into bins (linear interp) ---------
    # For each segment [t0, t1] with powers p0->p1, integrate piecewise into bins.
    for i in range(len(times) - 1):
        t0, t1 = float(times[i]), float(times[i + 1])
        p0, p1 = float(powers[i]), float(powers[i + 1])
        if t1 <= t0:
            continue

        # Linear power: p(t) = p0 + m*(t - t0)
        m = (p1 - p0) / (t1 - t0)

        # Clip segment to [start_edge, end_edge)
        a = max(t0, start_edge)
        b = min(t1, end_edge)
        if b <= a:
            continue

        # Iterate through overlapped bins
        # current sub-interval starts at 'a'
        cur = a
        while cur < b - 1e-12:
            # end of current bin
            bin_idx = int((cur - start_edge) // bin_width)
            bin_right_edge = start_edge + (bin_idx + 1) * bin_width
            sub_end = min(b, bin_right_edge)

            # power at sub-interval endpoints
            pa = p0 + m * (cur - t0)
            pb = p0 + m * (sub_end - t0)

            # Trapezoid integral over [cur, sub_end]
            sub_energy = 0.5 * (pa + pb) * (sub_end - cur)  # Joules

            if 0 <= bin_idx < nbins:
                energy_bins[bin_idx] += sub_energy

            cur = sub_end

    # Convert Joules/bin to average power (W) per bin
    power_per_sec = [e / bin_width for e in energy_bins]

    # --------- Step 3: concurrency at bin centers via sweep ---------
    # Build events: (time, delta), departures before arrivals at same time
    events = []
    for a, f in arrival_finish_times:
        # Skip malformed entries
        if f is None or a is None:
            continue
        a = float(a); f = float(f)
        # Treat zero-length safely
        if f < a:
            f = a
        events.append((f, -1))
        events.append((a, +1))
    events.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))

    # Bin centers to query
    bin_centers = [start_edge + (k + 0.5) * bin_width for k in range(nbins)]

    # Sweep once
    n_requests = []
    cur = 0
    j = 0
    ne = len(events)
    for t in bin_centers:
        while j < ne and events[j][0] <= t:
            cur += events[j][1]
            j += 1
        if cur < 0:  # just in case
            cur = 0
        n_requests.append(cur)

    return bin_centers, power_per_sec, n_requests

def align_energy_times(powers, power_times, times):
    power = powers[0]
    energies_aligned = []
    idx = 0
    for t, t_next in zip(times[:-1], times[1:]):
        energy = 0.0
        while idx < len(powers) and power_times[idx] < t:
            idx += 1
            power = powers[idx]
        t0 = t 
        while idx < len(powers) and power_times[idx] < t_next:
            energy += power * (power_times[idx] - t0)
            power = powers[idx]
            t0 = power_times[idx]
            idx += 1
        energy += power * (t_next - t0)
        energies_aligned.append(energy)
    return energies_aligned

def count_tokens_by_time(events, bin_size = 1.0):
    import math
    from collections import defaultdict
    batches = [event for event in events if event.event_type == 'batch']
    token_by_time = defaultdict(int)
    start_time = batches[0].timestamp
    end_time = batches[-1].timestamp
    
    
    round_down_time = lambda t: math.floor(t / bin_size) * bin_size
    round_up_time = lambda t: math.ceil(t / bin_size) * bin_size
    round_down_idx = lambda t: int(math.floor(t / bin_size))
    times = np.arange(round_down_time(start_time), round_up_time(end_time) + bin_size, bin_size)
    counts = [0] * len(times)
    past_tokens = [0] * len(times)
    for b in batches:
        assert isinstance(b, Batch)
        num_tokens = b.total_current_length
        num_past_tokens = b.total_past_tokens
        counts[round_down_idx(b.timestamp)] += num_tokens
        past_tokens[round_down_idx(b.timestamp)] += num_past_tokens
    
    return times, counts, past_tokens

def fit_linear(x, y):
    # --- Fit simple linear regression y = a*x + b ---
    n = len(x)
    if n > 1:
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x)**2 for xi in x)
        a = num / den if den != 0 else 0.0
        b = mean_y - a * mean_x
    else:
        a, b = 0.0, y[0] if y else 0.0

    # --- Create regression line values ---
    x_line = [min(x), max(x)]
    y_line = [a * xi + b for xi in x_line]
    return x_line, y_line, a, b

def analyze_energy_consumption():
    import pandas as pd
    import matplotlib.pyplot as plt
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime_0.0/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_chat_23_3978:4579_anytime_0.0/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.events.jsonl'
    events, reqs = analyze_events(filepath)
    fig, ax = plt.subplots(figsize=(10, 5))
    ttft_slo_scale = 3
    slo_tpot = 0.025
    slo_ttft_overhead = 0.05
    from motivation.common import PerfModel
    perf_model = PerfModel.get_perf_model('Qwen/Qwen2.5-7B-Instruct')
    slo_ttft_fn = lambda req: req.zero_load_ttft * ttft_slo_scale + slo_ttft_overhead
    x,y=draw_min_servers(reqs, ax, slo_ttft_fn, slo_tpot, perf_model)
    energy_events = [event for event in events if event.event_type == 'energy']
    power_times = [event.timestamp for event in energy_events][10:-10]
    powers = [event.power for event in energy_events][10:-10]
    twinx = ax.twinx()
    twinx.plot(power_times, powers, label = 'Power Consumption')
    
    twinx.set_xlabel('Time (s)')
    twinx.set_ylabel('Power (W)')
    twinx.legend()
    # plt.show()
    fig.savefig('energy_consumption_by_time.png', dpi=300, bbox_inches='tight')
    fig.savefig('energy_consumption_by_time.pdf', dpi=300, bbox_inches='tight')
    print('Saved energy_consumption_by_time.png')
    print('Saved energy_consumption_by_time.pdf')
    
    arrival_finish_times = []
    for req_id, req in reqs.items():
        assert isinstance(req, RequestInstance)
        if req.is_finished:
            finish_time = req.schedules[-1].timestamp + req.schedules[-1].elapsed
            arrival_finish_times.append((req.arrival_time, finish_time))
    print(power_times[:10])
    # exit(0)
    # print(times[0], times[-10:])
    bin_centers, pps, nreq = energy_per_second_vs_concurrency(arrival_finish_times, power_times, powers, bin_width = 0.5)
    
    
    fig, ax = plt.subplots(figsize=(7,5), tight_layout = True)
    ax.plot(bin_centers, pps, label = 'Energy Consumption')
    twinx = ax.twinx()
    twinx.plot(bin_centers, nreq, label = 'Concurrent Requests', color = 'red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy Consumption (W)')
    twinx.set_ylabel('Concurrent Requests')
    ax.legend()
    twinx.legend()
    fig.savefig('energy_consumption_vs_concurrency.png', dpi=300, bbox_inches='tight')
    fig.savefig('energy_consumption_vs_concurrency.pdf', dpi=300, bbox_inches='tight')
    print('Saved energy_consumption_vs_concurrency.png')
    print('Saved energy_consumption_vs_concurrency.pdf')
    
    fig, (ax, ax1, ax2) = plt.subplots(1,3, figsize=(14,5), tight_layout = True)
    times, counts, past_tokens = count_tokens_by_time(events, bin_size = 1.0)
    energies = align_energy_times(pps, bin_centers, times)
    counts = counts[:-1]
    ax.scatter(nreq, pps)
    ax.set_xlabel('Concurrent Requests')
    ax.set_ylabel('Power (J/s)')
    
    ax1.scatter(counts, energies)
    ax1.set_xlabel('Throughput (token/s)')
    ax1.set_ylabel('Power (J/s)')
    
    energy_per_token = np.array(energies) / np.array(counts)
    ax2.scatter(counts, energy_per_token)
    ax2.set_xlabel('Throughput (token/s)')
    ax2.set_ylabel('Energy per Token (J/token)')
    
    x_line, y_line, a, b = fit_linear(counts, energies)
    
    ax1.legend()
    ax2.legend()
    fig.savefig('energy_consumption_vs_throughput.png', dpi=300, bbox_inches='tight')
    fig.savefig('energy_consumption_vs_throughput.pdf', dpi=300, bbox_inches='tight')
    print('Saved energy_consumption_vs_throughput.png')
    # ax.plot(times, counts, label = 'Current Tokens')
    # ax.plot(times, past_tokens, label = 'Past Tokens')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Number of Tokens')
    # twinx = ax.twinx()
    # twinx.plot(bin_centers, pps, label = 'Energy Consumption', color = 'red')
    # twinx.set_ylabel('Energy Consumption (W)')
    # ax.legend()
    # twinx.legend()
    fig.savefig('tokens_vs_energy_by_time.png', dpi=300, bbox_inches='tight')
    fig.savefig('tokens_vs_energy_by_time.pdf', dpi=300, bbox_inches='tight')
    print('Saved tokens_vs_energy_by_time.png')
    print('Saved tokens_vs_energy_by_time.pdf')
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5), tight_layout = True)
    
    x = nreq
    y = pps

    
    
    x_line, y_line, a, b = fit_linear(x, y)
    # --- Plot scatter and regression ---
    fig, ax = plt.subplots(figsize=(7,4), tight_layout = True)
    ax.scatter(x, y, s=10, alpha=1.0, color = 'blue')
    ax.plot(x_line, y_line, 'r--', linewidth=4,
        label=f"Linear fit: y = {a:.2f}x + {b:.2f}")
    
    ax.scatter(nreq, pps, s=10, alpha=1.0)
    ax.set_xlabel("# concurrent requests")
    ax.set_ylabel("Energy / s (W)")
    # Draw the annotation for static (idle) power
    ax.annotate(
        'Static Power: 53 W',
        xy=(0.0, 53.0),           # data coords
        xytext=(10, 53.0 + 12),  # move text into the plot
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
        fontsize=24, ha='left', va='bottom'
    )
    ax.set_ylim(0, 210)
    ax.legend(fontsize=20)
    # ax.set_title("Energy per Second vs. Concurrency")
    ax.grid(True, alpha=0.3)
    ax.scatter([0.0], [53.0],
           marker='*', color='red', s=300,  # s = size in points^2
           edgecolor='black', linewidth=1.0,
           zorder=5, label='Static Power Point')
    fig.savefig("energy_per_second_vs_concurrency.png", dpi=300)
    fig.savefig("energy_per_second_vs_concurrency.pdf", dpi=300)
    print('Saved energy_per_second_vs_concurrency.png')
    print('Saved energy_per_second_vs_concurrency.pdf')
    exit(0)
    
    ## Build step function from arrivals/finishes
    event_times, counts_after = build_active_requests_step(arrival_finish_times)

    # Map each energy sample to concurrent #requests
    reqs_at_energy = count_at_times(event_times, counts_after, times)

    # --- 2) Plot: #requests vs. energy (scatter + aggregated summary) ---
    df_req_energy = pd.DataFrame({
        "n_requests": reqs_at_energy.astype(int),
        "energy": energies,     # rename to "power" if that’s actually power samples
        "time": times
    }).sort_values("time")
    
    
    order = np.argsort(times)
    times = np.array(times)[order]
    power = np.array(energies)[order]  # your instantaneous power array

    # Compute time deltas and per-interval average power
    dt = np.diff(times)
    p_mid = 0.5 * (power[1:] + power[:-1])  # average power in each small interval
    energy_joules = p_mid * dt              # energy for each interval in Joules

    # Define 1-second bins for aggregating
    bin_width = 1.0
    bins = np.arange(times.min(), times.max() + bin_width, bin_width)

    # Integrate total energy within each 1s bin
    energy_per_sec, _ = np.histogram(times[:-1], bins=bins, weights=energy_joules)
    t_centers = (bins[:-1] + bins[1:]) / 2

    # Convert to mean power (J/s = W)
    power_per_sec = energy_per_sec / bin_width

    # --- Step 2. Compute concurrent requests at each bin center ---
    reqs_per_sec = count_at_times(event_times, counts_after, t_centers)

    # --- Step 3. Assemble into DataFrame ---
    df_energy_load = pd.DataFrame({
        "time": t_centers,
        "n_requests": reqs_per_sec,
        "energy_per_s": power_per_sec,   # J/s = W
    })

    # --- Step 4. Plot Energy (Power) vs Concurrency ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_energy_load["n_requests"], df_energy_load["energy_per_s"], s=10, alpha=0.4)
    ax.set_xlabel("# concurrent requests")
    ax.set_ylabel("Average power (W)")
    ax.set_title("Energy Consumption per Second vs. Concurrency")
    ax.grid(True, alpha=0.3)

    # --- Step 5. Optional: overlay smoothed mean trend ---
    grp = df_energy_load.groupby("n_requests")["energy_per_s"]
    mean_p = grp.mean()
    std_p = grp.std()
    ax.errorbar(mean_p.index, mean_p.values, yerr=std_p.values, fmt='-o', capsize=3, color='red', label="mean ± std")
    ax.legend()

    plt.tight_layout()
    plt.savefig("energy_per_second_vs_concurrency.png", dpi=300)
    exit(0)
    
    
    df_req_energy = df_req_energy.set_index("time")
    df_req_energy["energy"] = df_req_energy["energy"].rolling(window=1, min_periods=1).mean()
    df_req_energy = df_req_energy.reset_index()
    # (A) Raw scatter: #requests vs. energy
    fig_scatter, ax_sc = plt.subplots(figsize=(6.5, 5))
    ax_sc.scatter(df_req_energy["n_requests"], df_req_energy["energy"], s=10, alpha=0.35)
    ax_sc.set_xlabel("# concurrent requests")
    ax_sc.set_ylabel("Energy")  # or "Power" depending on your units
    ax_sc.set_title("Instantaneous Energy vs. #Concurrent Requests")
    ax_sc.grid(True, alpha=0.25)
    fig_scatter.tight_layout()
    fig_scatter.savefig("requests_vs_energy_scatter.png", dpi=300)

    # (B) Aggregated mean ± std per #requests (cleaner trend)
    grp = df_req_energy.groupby("n_requests")["energy"]
    levels = grp.mean().index.values
    mean_e = grp.mean().values
    std_e  = grp.std().fillna(0.0).values

    fig_agg, ax_ag = plt.subplots(figsize=(6.5, 5))
    ax_ag.errorbar(levels, mean_e, yerr=std_e, fmt='-o', capsize=3)
    ax_ag.set_xlabel("# concurrent requests")
    ax_ag.set_ylabel("Energy")  # or "Power"
    ax_ag.set_title("Energy vs. #Concurrent Requests (mean ± 1 std)")
    ax_ag.grid(True, alpha=0.25)
    fig_agg.tight_layout()
    fig_agg.savefig("requests_vs_energy_agg.png", dpi=300)

    # (C) Optional: boxplot to show distribution per request level
    counts = df_req_energy["n_requests"].value_counts()
    keep_levels = counts[counts >= 5].index  # require ≥5 samples to avoid tiny bins
    df_box = df_req_energy[df_req_energy["n_requests"].isin(keep_levels)]

    fig_box, ax_box = plt.subplots(figsize=(7.5, 5))
    df_box.boxplot(column="energy", by="n_requests", ax=ax_box)
    ax_box.set_xlabel("# concurrent requests")
    ax_box.set_ylabel("Energy")  # or "Power"
    ax_box.set_title("Energy distribution by #Concurrent Requests")
    ax_box.grid(True, alpha=0.25)
    plt.suptitle("")  # remove pandas auto-title
    fig_box.tight_layout()
    fig_box.savefig("requests_vs_energy_box.png", dpi=300)

    print("Saved requests_vs_energy_scatter.png, requests_vs_energy_agg.png, requests_vs_energy_box.png")


if __name__ == '__main__':
    analyze_energy_consumption()
    exit(0)
    # analyze_high_load_delay()
    # draw_bs_comparison()
    # exit(0)
    # analyze_high_load_delay()
    # draw_reqeust_arrivals('azure_chat_23', 600, 1202)
    # draw_token_waste_comparison()
    # update_expected()
    # update_expected_chat()
    # exit(0)
    
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/vllm_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_chat:azure_code_23_3979:4580_anytime/vllm_round_robin_0.5_4_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_chat:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.5_4_anytime_3.0_0.025.events.jsonl'

    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_auto_scaling_1.0_4_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1200_anytime/slosserve-edf_auto_scaling_4.0_4_anytime_5.0_0.1.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_auto_scaling_1.0_4_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4579_anytime/vllm_round_robin_0.5_1_anytime_3.0_0.05.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_auto_scaling_1.0_4_anytime_3.0_0.025.events.jsonl'
    
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime/sarathi_round_robin_0.4_1_anytime_10.0_0.03.events.jsonl'
    filepath = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime/qlm_round_robin_0.4_1_anytime_2.0_0.025.events.jsonl"
    filepath = "experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4578_anytime/sarathi_round_robin_0.4_1_anytime_3.0_0.05.events.jsonl"
    # filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo-1.0_1.0_1_anytime_3.0_0.025.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:551_anytime/slosserve-edf_round_robin_11.0_1_anytime_3.0_0.1.events.jsonl'
    # filepath = 'experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat_per_device_0.9_4_anytime_5.0_0.1.events.jsonl'
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_per_device-0.04_1.5_4_anytime_3.0_0.025.events.jsonl'
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/results.jsonl'
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all-0.08_1.5_4_anytime_3.0_0.025.events.jsonl'
    filepath = 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/slosserve-edf_round_robin_2.5_4_anytime_3.0_0.025.events.jsonl'
    events, reqs = analyze_events(filepath)
    # analyze_auto_scaling(reqs)
    analyze_slo_violation(reqs, events, 
                          model_name = 'Qwen/Qwen2.5-7B-Instruct',
                          length_pattern = 'sharegpt_code',
                          ttft_slo_scale = 3,
                          slo_tpot = 0.025, 
                          prefix = 'debug-events', 
                          slo_ttft_overhead = 0.05,
                          draw = True)
    exit(0)
    rescheduling_time_gapss = []
    n_schedulingss = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not 'resch' in data['routing_policy']: continue
            try:
                events, reqs = analyze_events(data['event_file'], verbose = True)
                rescheduling_time_gaps, n_schedulings = analyze_auto_scaling(reqs)
                rescheduling_time_gapss.extend(rescheduling_time_gaps)
                n_schedulingss.extend(n_schedulings)
            except Exception as e:
                print(f'Error analyzing events for {filepath}: {e}')
    print(f'rescheduling_time_gaps: {np.mean(rescheduling_time_gapss)} +- {np.std(rescheduling_time_gapss)}')
    print(f'n_schedulings: {np.mean(n_schedulingss)} +- {np.std(n_schedulingss)}')
    exit(0)
    
    analyze_slo_violation(reqs, events, 
                          model_name = 'Qwen/Qwen2.5-7B-Instruct',
                          length_pattern = 'azure_chat_23',
                          ttft_slo_scale = 5,
                          slo_tpot = 0.1, 
                          prefix = 'debug-events', 
                          slo_ttft_overhead = 0.25,
                          draw = False)
    reqs = [req for req in reqs.values() if req.violate_slo() != "none"]
    fig, axes = plt.subplots(10,10,figsize=(20,20))
    for ax, req in zip(axes.flatten(), reqs):
        timestamps = [sch.timestamp for sch in req.schedules]
        expecteds = req.expected_finish_time
        if len(timestamps) != len(expecteds):
            length = min(len(timestamps), len(expecteds))
            timestamps = timestamps[:length]
            expecteds = expecteds[:length]
        ax.plot(expecteds, expecteds, label = 'expected')
        ax.plot(expecteds, timestamps, label = 'actual')
        ax.legend()
    fig.savefig('tpot_req.png', dpi=300, bbox_inches='tight')
    # fig.savefig('tpot_req.pdf', dpi=300, bbox_inches='tight')
    print('Saved tpot_req.png')
    print('Saved tpot_req.pdf')
    # exit(0)
    analyze_auto_scaling(reqs)
    

    results = analyze_slo_violation(reqs, events, 
                          model_name = 'Qwen/Qwen2.5-7B-Instruct',
                          length_pattern = 'sharegpt_code',
                          ttft_slo_scale = 3,
                          slo_tpot = 0.05, 
                          prefix = 'debug-events', 
                          slo_ttft_overhead = 0.02,
                          draw = False)

    print('slo_violation_rate', 1 - results['slo_attainment_rate'])
    
    # filepath = ''
    # analyze_results('experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime', 
    #                 ttft_slo_scale = 3, slo_tpot = 0.025, slo_ttft_overhead = 0.05