from dataclasses import dataclass
from typing import List, Optional
import os
import pickle
import dotenv
from functools import lru_cache
import numpy as np
import pandas as pd 
import math
import random
# dotenv.load_dotenv()
DATASET_DIR = 'assets/datasets'

print(f'DATASET_DIR:{DATASET_DIR}')

def generate_bursty_trace(
    duration: float = 600.0,   # total seconds
    base_rate: float = 5.0,    # mean requests per second (global average)
    burstiness_level: float = 0.0,  # 0.0 = homogeneous Poisson; 1.0 = very bursty
    burst_freq: float = 0.03,  # bursts per second (Hz). e.g., 0.02 => period 50s
    seed: int = 42
) -> List[float]:
    """
    Generate arrival timestamps (seconds) for a non-homogeneous Poisson process
    with steep, periodic bursts. The *average* arrival rate over the whole trace
    equals `base_rate` exactly (independent of burstiness_level).

    Model:
      λ(t) = λ_base + λ_burst * gate(t)
    where gate(t) ∈ [0,1] is a steep 'on' window each period; λ_base and λ_burst
    are chosen so that E[λ(t)] = base_rate for all parameter choices.

    burstiness_level in [0,1]:
      - Controls (i) peakiness (how much mass moves into bursts),
                (ii) duty cycle (shorter bursts when higher),
                (iii) edge steepness (sharper edges when higher).

    Returns:
      Strictly increasing list of arrival times in [0, duration).
    """

    assert duration > 0 and base_rate >= 0 and burst_freq > 0
    random.seed(seed)

    # === Map burstiness_level -> duty cycle and edge steepness ===
    # Shorter duty (narrower bursts) & steeper edges as burstiness goes up.
    # Duty ∈ [0.06, 0.25] roughly
    duty_min, duty_max = 0.06, 0.25
    duty = duty_max - (duty_max - duty_min) * max(0.0, min(1.0, burstiness_level))

    # Steepness k for fast sigmoids (higher => sharper edges)
    k_min, k_max = 20.0, 220.0
    k = k_min + (k_max - k_min) * max(0.0, min(1.0, burstiness_level))

    # Period from frequency
    period = 1.0 / burst_freq
    on_time = duty * period

    # === Rate decomposition that preserves mean(base_rate) ===
    # λ(t) = λ_base + λ_burst * gate(t), with mean(gate)=duty.
    # Choose λ_base = base_rate * (1 - burstiness_level)
    # and λ_burst so that mean is preserved:
    #   mean λ = λ_base + λ_burst * duty = base_rate  =>  λ_burst = base_rate*burstiness_level / duty
    lam_base = base_rate * (1.0 - burstiness_level)
    lam_burst = (base_rate * burstiness_level / duty) if duty > 0 else 0.0

    # Gate(t): near-square pulses with steep sigmoid edges
    def gate(t: float) -> float:
        # Phase within the period
        ph = t % period
        # Fast sigmoids to approximate a box of width on_time
        # up ≈ H(phase > 0), down ≈ H(phase < on_time)
        up = 1.0 / (1.0 + math.exp(min(max(-k * (ph - 1e-2), -100), 100)))
        down = 1.0 / (1.0 + math.exp(min(max(-k * ((on_time - ph) - 1e-2), -100), 100)))
        return up * down  # ~1 during burst window, ~0 otherwise

    def lam(t: float) -> float:
        return lam_base + lam_burst * gate(t)

    # Upper bound for thinning (max gate = 1)
    lam_max = lam_base + lam_burst
    if lam_max <= 0:
        return []

    # === Ogata thinning ===
    arrivals = []
    t = 0.0
    while t < duration:
        # propose next event from Homogeneous Poisson with rate lam_max
        u = random.random()
        w = -math.log(u) / lam_max
        t += w
        if t >= duration:
            break
        # accept with probability lam(t) / lam_max
        if random.random() <= lam(t) / lam_max:
            arrivals.append(t)

    return arrivals

# def generate_bursty_trace(
#     duration=600,           # total seconds
#     base_rate=5,            # mean requests per second
#     burstiness_level=0.0,
#     burst_freq=0.02,        # frequency (Hz) of bursts
#     seed=42
# ) -> list[float]:
#     np.random.seed(seed)
#     traces = []

#     timestamps = []
#     t = 0.0
#     while t < duration:
#         # instantaneous rate λ(t)
#         lam_t = base_rate * (1 + burstiness_level * np.sin(2 * np.pi * burst_freq * t))
#         # draw next inter-arrival time
#         delta_t = np.random.exponential(1.0 / max(lam_t, 1e-5))
#         t += delta_t
#         timestamps.append(t)
#     return timestamps

def _fit(lengths: List[int], dist: str):
    # INSERT_YOUR_CODE
    import numpy as np
    from scipy import stats

    # Collect all lengths

    data = np.array(lengths)
    data = data[data > 0]  # Remove zeros for fitting

    sorted_data = np.sort(data)
    if dist == 'exponential':
        # Fit exponential: lambda = 1/mean
        loc, scale = stats.expon.fit(data, floc=0)
        x = np.linspace(sorted_data.min(), sorted_data.max(), 100)
        # Pr{X > x} = 1 - CDF = exp(-x/scale)
        y = np.exp(-x/scale)
    elif dist == 'lognorm':
        # Fit lognormal
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        x = np.linspace(sorted_data.min(), sorted_data.max(), 100)
        y = 1 - stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    elif dist == 'gamma':
        # Fit gamma
        a, loc, scale = stats.gamma.fit(data, floc=0)
        x = np.linspace(sorted_data.min(), sorted_data.max(), 100)
        y = 1 - stats.gamma.cdf(x, a, loc=loc, scale=scale)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    return x, y

@dataclass(kw_only=True)
class Request: 
    input_length: int 
    output_length: int 
    cached_length: int = 0
    
    prompt: Optional[str] = None 
    thinking: Optional[str] = None 
    answer: Optional[str] = None 

    thinking_length: int = 0 

    @property
    def total_length(self):
        return self.input_length + self.output_length + self.thinking_length

@dataclass 
class Requests:
    name: str
    requests: List[Request]
    is_reasoning: bool = False  # Provide default for backward compatibility
    
    def get_requests(self, start: int, end: int, model: str):
        reqs = self.requests[start:end]
        # If all prompts are already present, return early
        if all(getattr(r, "prompt", None) for r in reqs):
            return reqs

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)

        # Find a string that encodes to exactly ONE token for this tokenizer.
        # Try a few safe candidates; pick the first that works.
        candidates = ["\n", " x", " a", ".", ",", " z"]
        one_token_str = None
        for s in candidates:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] is not None:
                one_token_str = s
                break
        if one_token_str is None:
            # Fallback: use a single regular token ID (e.g., eos) and decode it;
            # not ideal, but prevents empty prompts.
            fallback_id = getattr(tok, "eos_token_id", 0)
            one_token_str = tok.decode([fallback_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # Build dummy prompts so that tokenized length ≈ input_length
        for r in reqs:
            if getattr(r, "prompt", None):
                continue
            L = int(getattr(r, "input_length", 0) or 0)
            if L <= 0:
                r.prompt = ""  # nothing to do
                continue
            # Repeat a single-token string L times → should re-tokenize back to L tokens.
            # Use join to avoid accidental merges.
            # For " x" or " a", repetition like " x" * L is fine; for "\n" it’s also fine.
            # Joining is robust across tokenizers:
            if one_token_str.strip():
                # Insert spaces only if the candidate doesn’t already start with a space
                if one_token_str.startswith(" "):
                    text = one_token_str * L
                else:
                    text = one_token_str * L
            else:
                text = one_token_str * L

            r.prompt = text

        return reqs
    
    @staticmethod
    def merge(requests1: 'Requests', requests2: 'Requests'):
        return Requests(f'{requests1.name}-{requests2.name}', requests1.requests + requests2.requests)
    
    @staticmethod
    @lru_cache(maxsize=None)
    def load(name: str, max_tokens: int = None, window_start: int = 0, window_end: int = None): 
        path = os.path.join(DATASET_DIR, f'{name}.requests.pkl')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        assert obj.name == name
        print(f'Loaded {name} from {path}')
        if max_tokens is not None:
            obj.requests = list(filter(lambda r: (r.total_length + 1000) <= max_tokens, obj.requests))
        if window_end is not None:
            obj.requests = obj.requests[window_start:window_end]
        return obj
    
    def save(self): 
        path = os.path.join(DATASET_DIR, f'{self.name}.requests.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Saved {self.name} to {path}')
    
    
    
    def visualize(self, log_scale: bool = True, fit_with: str | None = None):
        import numpy as np
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(9, 3))

        def _plot_cdf(ax, data, label, log_scale):
            data = np.array(data)
            if log_scale:
                data = data[data > 0]  # Remove zeros for log scale
            if len(data) == 0:
                ax.set_title(f'{label}: No data')
                return
            sorted_data = np.sort(data)
            yvals = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            # To avoid log(0), remove yvals==0
            if log_scale:
                mask = yvals > 0
                sorted_data = sorted_data[mask]
                yvals = yvals[mask]
            ax.plot(sorted_data, yvals, marker='.', linestyle='none', label='Data')
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            
        def plot_cdf(ax, data, label, log_scale, fit_with=None):
            if len(data) == 0:
                ax.set_title(f'{label}: No data')
                return
            _plot_cdf(ax, data, label, log_scale)

            if fit_with is not None:
                try:
                    fit_result = _fit(data, fit_with)
                    ax.plot(fit_result[0], fit_result[1], label=fit_with)
                except Exception as e:
                    print(f'Error fitting {label} with {fit_with}: {e}')
                    pass
            mean_val = round(np.mean(data), 2)
            ax.legend()
            ax.set_title(f'{label}: Mean: {mean_val}')
            ax.set_xlabel(label)
            ax.set_ylabel('F(l) = Pr{{x > l}}')
        

        plot_cdf(ax1, [req.input_length for req in self.requests], 'Input Length', log_scale, fit_with)
        plot_cdf(ax2, [req.output_length for req in self.requests], 'Output Length', log_scale, fit_with)
        plot_cdf(ax3, [req.thinking_length for req in self.requests], 'Thinking Length', log_scale, fit_with)

        fig.suptitle(self.name)
        scale_str = "-log" if log_scale else ""
        fig_name = f'figs/requests_lengths-{self.name}{scale_str}.png'
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f'Saved {fig_name}')
        
class ArrivalTimes:
    name: str
    arrival_times: List[float]
    load_scale: float
    
    def __init__(self, name: str, arrival_times: List[float]): 
        self.name = name
        self.arrival_times = arrival_times
    
    def set_load_scale(self, load_scale: float):
        self.load_scale = load_scale
        self.arrival_times = [t / load_scale for t in self.arrival_times]
    
    def __len__(self): 
        return len(self.arrival_times)
    
    
    @staticmethod
    @lru_cache(maxsize=None)
    def load(name: str, load_scale: float = 1, window_start: int = 0, window_end: int = None):
        if name.startswith('bursty'):
            burstiness_level = name.split('_')[1]
            traces = generate_bursty_trace(burstiness_level=float(burstiness_level))
            obj = ArrivalTimes(name, traces)
        else:
            path = os.path.join(DATASET_DIR, f'{name}.arrival.pkl')
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        # assert obj.name == name
            print(f'Loaded {name} from {path}')
            if window_end is not None:
                obj.arrival_times = obj.arrival_times[window_start:window_end]
        obj.set_load_scale(load_scale)
        return obj
    
    def save(self): 
        path = os.path.join(DATASET_DIR, f'{self.name}.arrival.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Saved {self.name} to {path}')
        
    def visualize(self):
        import os
        import math
        import numpy as np
        import matplotlib.pyplot as plt

        os.makedirs("figs", exist_ok=True)

        # ---- sanitize & prepare ----
        t = np.asarray(self.arrival_times, dtype=float)
        t = np.sort(t)
        if t.size < 2:
            print("Not enough arrivals to analyze.")
            return

        # Normalize time to start at 0 for nicer plots
        t0 = t[0]
        t = t - t0
        T = t[-1] if t[-1] > 0 else 1.0
        n = t.size
        rate = (n - 1) / T

        # Interarrival statistics
        ia = np.diff(t)
        ia_mean = ia.mean()
        ia_std = ia.std(ddof=1) if ia.size > 1 else 0.0
        cv2 = (ia_std / ia_mean) ** 2 if ia_mean > 0 else float("nan")

        # ---- Helper: bin counts over time ----
        def bin_counts(bin_size=None, max_bins=2000):
            if bin_size is None:
                # target ~200-500 bins
                bin_size = max(T / 300.0, 1e-9)
            nbins = max(1, min(int(math.ceil(T / bin_size)), max_bins))
            edges = np.linspace(0, T, nbins + 1)
            counts, _ = np.histogram(t, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return counts.astype(float), centers, edges, bin_size

        counts, centers, edges, base_bin = bin_counts()

        # ---- IDC across window sizes ----
        # window sizes spaced log between ~base_bin and T/5
        min_w = max(base_bin, T / 500.0)
        max_w = max(min(T / 5.0, T), min_w * 2)
        W = np.unique(np.geomspace(min_w, max_w, num=20))
        idc_ws = []
        for w in W:
            # aggregate counts to window size w by grouping base bins
            g = max(1, int(round(w / base_bin)))
            m = (len(counts) // g) * g
            if m == 0:
                continue
            c_agg = counts[:m].reshape(-1, g).sum(axis=1)
            mu = c_agg.mean()
            var = c_agg.var(ddof=1) if c_agg.size > 1 else 0.0
            idc = (var / mu) if mu > 0 else np.nan
            idc_ws.append((g * base_bin, idc))
        idc_ws = np.array(idc_ws) if idc_ws else np.empty((0, 2))

        # ---- ACF of binned counts (up to 100 lags or len-2) ----
        def acf(x, nlags=100):
            x = np.asarray(x, dtype=float)
            x = x - x.mean()
            denom = np.dot(x, x)
            if denom <= 0:
                return np.zeros(nlags + 1)
            ac = np.correlate(x, x, mode="full")
            mid = len(ac) // 2
            ac = ac[mid:mid + nlags + 1] / denom
            return ac

        nlags = min(100, max(1, len(counts) - 2))
        acf_vals = acf(counts, nlags)

        # ---- Peak-over-threshold (POT) burst detection on base bins ----
        thr = np.quantile(counts, 0.99) if counts.size >= 100 else (counts.mean() + 3 * counts.std(ddof=1))
        thr = max(thr, counts.mean() + 2 * counts.std(ddof=1))  # be conservative
        above = counts >= thr
        bursts = []
        i = 0
        while i < len(above):
            if above[i]:
                j = i
                s = 0.0
                while j < len(above) and above[j]:
                    s += counts[j]
                    j += 1
                start_t = edges[i]
                end_t = edges[j] if j < len(edges) else T
                bursts.append((start_t, end_t, s, int(j - i)))
                i = j
            else:
                i += 1
        bursts.sort(key=lambda x: x[2], reverse=True)
        top_bursts = bursts[:5]

        # ---- Figure 1: Event raster with burst overlays + stats box ----
        fig1, ax1 = plt.subplots(figsize=(10, 2.8))
        ax1.eventplot(t, orientation='horizontal')
        ax1.set_xlabel("Time (relative)")
        ax1.set_yticks([])
        ax1.set_title(f"Arrival Events ({self.name})")

        # Shade top bursts
        for (bs, be, s, width_bins) in top_bursts:
            ax1.axvspan(bs, be, alpha=0.15)

        # Stats text
        idc_short = None
        if idc_ws.size > 0:
            # pick a representative small window (first)
            idc_short = idc_ws[0, 1]
        lines = [
            f"n={n}, T={T:.3g}, rate={rate:.3g}/s",
            f"IA mean={ia_mean:.3g}s, CV²={cv2:.3g}",
            f"IDC(small)={idc_short:.3g}" if idc_short is not None and np.isfinite(idc_short) else "IDC(small)=NA",
            f"99% thresh={thr:.3g} (counts/bin)",
            f"Top bursts (start–end; bins; mass):"
        ]
        for (bs, be, s, w) in top_bursts:
            lines.append(f"  [{bs:.3g}, {be:.3g}] ; {w} bins ; mass={s:.0f}")
        txt = "\n".join(lines)
        ax1.text(0.01, 0.05, txt, transform=ax1.transAxes, fontsize=9,
                va='bottom', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))

        fig1.savefig(f'figs/arrival_times-{self.name}-events.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # ---- Figure 2: Counts per base bin (time series) + threshold ----
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(centers, counts, linewidth=1.0)
        ax2.axhline(thr, linestyle='--')
        ax2.set_xlabel("Time (relative)")
        ax2.set_ylabel(f"Count / {base_bin:.3g}s")
        ax2.set_title(f"Binned Counts ({self.name})")
        fig2.savefig(f'figs/arrival_times-{self.name}-counts.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # ---- Figure 3: Interarrival histogram (log-x) ----
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ia_pos = ia[ia > 0]
        if ia_pos.size > 0:
            lo, hi = ia_pos.min(), ia_pos.max()
            if hi > lo:
                l_vals = np.logspace(np.log10(lo), np.log10(hi), 40)
            else:
                l_vals = np.linspace(lo, hi, 20)
            # Compute Pr{x > l} for each l
            pr_x_gt_l = np.array([(ia_pos > l).mean() for l in l_vals])
        else:
            l_vals = np.array([1])
            pr_x_gt_l = np.array([0])
        ax3.plot(l_vals, pr_x_gt_l, marker='o')
        ax3.set_xscale('log')
        ax3.set_xlabel("Interarrival threshold $l$ (s, log scale)")
        ax3.set_ylabel(r"$\Pr\{x > l\}$")
        ax3.set_yscale('log')
        ax3.set_title(f"Interarrival Tail Probability ({self.name})")
        fig3.savefig(f'figs/arrival_times-{self.name}-iat_hist.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # ---- Figure 4: IDC vs window size ----
        if idc_ws.size > 0:
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.plot(idc_ws[:,0], idc_ws[:,1], marker='o', linewidth=1.0)
            ax4.set_xscale('log')
            ax4.set_xlabel("Window size (s, log)")
            ax4.set_ylabel("IDC = Var/Mean of window counts")
            ax4.set_title(f"IDC Across Scales ({self.name})")
            fig4.savefig(f'figs/arrival_times-{self.name}-idc.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)

        # ---- Figure 5: ACF of counts ----
        if nlags >= 2:
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            lags = np.arange(nlags + 1)
            ax5.stem(lags, acf_vals)
            ax5.set_xlabel("Lag (bins)")
            ax5.set_ylabel("ACF")
            ax5.set_title(f"Autocorrelation of Binned Counts ({self.name})")
            fig5.savefig(f'figs/arrival_times-{self.name}-acf.png', dpi=300, bbox_inches='tight')
            plt.close(fig5)

        # ---- Textual judgment on burstiness ----
        # Simple heuristic: bursty if CV^2 > 1 or median IDC > 1
        idc_med = float(np.nanmedian(idc_ws[:,1])) if idc_ws.size else float('nan')
        is_bursty = (cv2 > 1.0) or (np.isfinite(idc_med) and idc_med > 1.0)
        pattern = "bursty" if is_bursty else "not strongly bursty"

        print(f"[{self.name}] Arrivals analyzed:")
        print(f"  n={n}, span T={T:.3g}s, rate={rate:.3g}/s")
        print(f"  Interarrival mean={ia_mean:.3g}s, std={ia_std:.3g}s, CV^2={cv2:.3g}")
        if idc_ws.size:
            print(f"  IDC median across scales={idc_med:.3g} ({'>' if idc_med>1 else '<='} 1 indicates {pattern})")
        print(f"  Threshold for bursts (counts/bin) ≈ {thr:.3g}")
        if top_bursts:
            print("  Top bursts (start, end, bins, mass):")
            for (bs, be, s, w) in top_bursts:
                print(f"    [{bs:.3g}, {be:.3g}]  bins={w}  mass={int(s)}")
        print(f"Saved figures to figs/: "
            f"events, counts, iat_hist, idc, acf.")
