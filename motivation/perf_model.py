from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import r2_score

from motivation.events_analysis import Batch, analyze_events

import numpy as np
from typing import Callable, Tuple, Optional, Dict

def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares with tiny ridge for numerical stability."""
    XT = X.T
    A = XT @ X + 1e-12 * np.eye(X.shape[1])
    b = XT @ y
    return np.linalg.solve(A, b)

def fit_piecewise_two_lines(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points_per_side: int = 8,
    candidates: Optional[np.ndarray] = None,
    continuous: bool = True,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, float]]:
    """
    Fit y ≈ piecewise linear in x with one breakpoint c.

    If continuous=True:
        y = y0 + m1*(x-c)   for x <= c
        y = y0 + m2*(x-c)   for x >  c
      (both lines meet at x=c, value y0)

    If continuous=False:
        y = a1 + b1*x       for x <= c
        y = a2 + b2*x       for x >  c
      (two independent lines)

    Returns:
      predictor(x_new) -> y_hat
      params dict (includes c and line parameters)
    """
    
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n != y.size:
        raise ValueError("x and y must have the same length")
    if n < 2 * min_points_per_side + 1:
        raise ValueError("Not enough points for the requested min_points_per_side")

    # Candidate breakpoints: midpoints between sorted unique x, respecting min_points_per_side
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    uniq = np.unique(xs)

    if candidates is None:
        # Use midpoints between adjacent unique xs
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = mids

    best_sse = np.inf
    best = None  # will store (c, params, sse)

    for c in candidates:
        # split indices with count guards
        left_mask  = xs <= c
        right_mask = xs >  c
        nl = int(np.sum(left_mask))
        nr = int(np.sum(right_mask))
        if nl < min_points_per_side or nr < min_points_per_side:
            continue

        xl, yl = xs[left_mask], ys[left_mask]
        xr, yr = xs[right_mask], ys[right_mask]

        if continuous:
            # Design matrix enforces continuity at x=c:
            # y = y0 + m1*(x-c) for left; y = y0 + m2*(x-c) for right
            # Stack into one system: [1, (x-c), 0] for left; [1, 0, (x-c)] for right
            X_left  = np.column_stack([np.ones_like(xl), xl - c, np.zeros_like(xl)])
            X_right = np.column_stack([np.ones_like(xr), np.zeros_like(xr), xr - c])
            X = np.vstack([X_left, X_right])
            y_all = np.concatenate([yl, yr])
            # params = [y0, m1, m2]
            params = _ols(X, y_all)
            y0, m1, m2 = params.tolist()

            # Compute SSE
            yhat_left  = y0 + m1 * (xl - c)
            yhat_right = y0 + m2 * (xr - c)
            sse = np.sum((yl - yhat_left)**2) + np.sum((yr - yhat_right)**2)

            if sse < best_sse:
                best_sse = sse
                best = ("continuous", c, dict(y0=y0, m1=m1, m2=m2), sse)
        else:
            # Two independent OLS fits
            Xl = np.column_stack([np.ones_like(xl), xl])  # [a1, b1]
            Xr = np.column_stack([np.ones_like(xr), xr])  # [a2, b2]
            wl = _ols(Xl, yl)
            wr = _ols(Xr, yr)
            a1, b1 = wl.tolist()
            a2, b2 = wr.tolist()

            yhat_left  = a1 + b1 * xl
            yhat_right = a2 + b2 * xr
            sse = np.sum((yl - yhat_left)**2) + np.sum((yr - yhat_right)**2)

            if sse < best_sse:
                best_sse = sse
                best = ("independent", c, dict(a1=a1, b1=b1, a2=a2, b2=b2), sse)

    if best is None:
        raise RuntimeError("No valid breakpoint candidate satisfied min_points_per_side.")

    mode, c, pars, sse = best

    # Build predictor
    if mode == "continuous":
        y0, m1, m2 = pars["y0"], pars["m1"], pars["m2"]
        def predictor(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, dtype=float)
            out = np.empty_like(x_new, dtype=float)
            mask = x_new <= c
            out[mask]  = y0 + m1 * (x_new[mask]  - c)
            out[~mask] = y0 + m2 * (x_new[~mask] - c)
            return out
        params_out = dict(mode=mode, c=c, y0=y0, m1=m1, m2=m2, sse=float(sse))
    else:
        a1, b1, a2, b2 = pars["a1"], pars["b1"], pars["a2"], pars["b2"]
        def predictor(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, dtype=float)
            out = np.empty_like(x_new, dtype=float)
            mask = x_new <= c
            out[mask]  = a1 + b1 * x_new[mask]
            out[~mask] = a2 + b2 * x_new[~mask]
            return out
        params_out = dict(mode=mode, c=c, a1=a1, b1=b1, a2=a2, b2=b2, sse=float(sse))

    return predictor, params_out



def fit(batches: List["Batch"], max_iters: int = 250, tol: float = 1e-6,
        l2: float = 1e-6, # L2 regularization
        plot_path: str = 'figs/batch_time_fit.png',
        predictor: str = 'piecewise',
        features: list[str] = ['total_current_length', 'num_reqs', 'total_length', 'total_multiply'],
        _ax = None,
        **kwargs
        ) -> Tuple[Callable[["Batch"], float], Tuple[float,float,float,float,float]]:
    """
    Fit time = max( a * total_current_length + b,
                    c + d * total_length )
    using a convex QP (preferred) or a smooth softmax fallback.

    Returns: (predictor_fn, (a,b,c,d,e))
    """
    if not batches:
        raise ValueError("No batches to fit.")
    batches = [b for b in batches if b.total_current_length > 0]
    p99_elapsed = np.percentile([b.elapsed for b in batches], 99)
    batches = [b for b in batches if b.elapsed <= p99_elapsed]

    # Feature extractors with naming robustness
    def total_mult(b: "Batch") -> float:
        return getattr(b, "total_multiple", getattr(b, "total_multiply"))

    # X1 = np.array([[ total_mult(b), b.total_current_length, 1.0 ] for b in batches], dtype=float)  # [a,b,c]
    X = np.array([[getattr(b, feature) for feature in features] for b in batches], dtype=float)                          # [d,e]
    y  = np.array([ b.elapsed - b.scheduling_overhead for b in batches ], dtype=float)

    n = y.size
    
    if predictor == 'linear':
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X.reshape(-1, len(features)), y)
        a = m.coef_
        d = m.intercept_
        def predictor(batch: "Batch") -> float:
            return sum(a[i] * getattr(batch, feature) for i, feature in enumerate(features)) + d
        yhat = np.array([predictor(b) for b in batches], dtype=float)
        r2 = r2_score(y, yhat)
        print('fitted_model:')
        print(f'time = {a} * {features} + {d}')
        print('R2', r2)
        average_gap = np.mean(np.abs(yhat - y))
        print('average_gap', average_gap)
    else: 
        yhat = np.array([predictor(b) for b in batches], dtype=float)
        r2 = r2_score(y, yhat)
        print('R2', r2)
        average_gap = np.mean(np.abs(yhat - y))
        print('average_gap', average_gap)
        
    # Plot
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(4,16), tight_layout=True)
    if _ax is not None: _ax.scatter(y, yhat, s=6, **kwargs)
    ax1.scatter(y, yhat, s=6)
    ax1.set_xlabel('real_times')
    ax1.set_ylabel('predicted_times')
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    ax1.plot(lims, lims, '--r', linewidth=1)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_title('R2 = ' + f'{r2:.4f}')
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.axis('equal')

    decode_indices = [idx for idx, b in enumerate(batches) if b.decode_only]
    prefill_indices = [idx for idx, b in enumerate(batches) if b.prefill_only]
    mixed_indices = [idx for idx, b in enumerate(batches) if b.mixed]
    
    

    for attr, ax in zip(
        ['total_multiply', 'total_current_length', 'total_length', 'max_computed_length', 'num_reqs'],
        [ax2, ax3, ax4, ax5, ax6]
    ):
        xs = [getattr(b, attr) if hasattr(b, attr) else getattr(b, 'total_multiple', getattr(b,'total_multiply')) for b in batches]
        
        ax.scatter(xs, yhat, s=6, label = 'predicted')
        ax.scatter(xs, y, s=6, label = 'real')
        for name, indices in zip(['decode', 'prefill', 'mixed'], [decode_indices, prefill_indices, mixed_indices]):
            xs_name = [xs[idx] for idx in indices]
            y_name = [y[idx] for idx in indices]
            ax.scatter(xs_name, y_name, s = 6, label = name)
        # xs_decode = [xs[idx] for idx in decode_indices]
        # y_decode = [y[idx] for idx in decode_indices]
        # ax.scatter(xs_decode, y_decode, s = 6, label = 'decode', color = 'red')
        ax.set_xlabel(attr); ax.set_ylabel('real times')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.axis('equal')
        ax.legend()

    import os
    os.makedirs(os.path.dirname(plot_path) or '.', exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    print(f'Saved {plot_path}')

    return predictor



def fit_model(filepaths: List[str], predictor = 'linear', ax = None, **kwargs):
    features = ['total_current_length', 'num_reqs', 'total_past_tokens']
    batches = []
    for filepath in filepaths:
        events, reqs = analyze_events(filepath)
        batches.extend([event for event in events if event.event_type == 'batch'])
    
    computed_tokens = [np.mean(b.num_computed_tokens) for b in batches if len(b.num_computed_tokens) > 0]
    current_tokens = [np.mean(list(b.num_scheduled_tokens.values())) for b in batches if len(b.num_scheduled_tokens) > 0]
    total_tokens = [np.mean(b.total_length) for b in batches if len(b.num_computed_tokens) > 0]
    max_computed_tokens = [np.mean(b.max_computed_length) for b in batches if len(b.num_computed_tokens) > 0]
    
    print(f'computed_tokens: {np.mean(computed_tokens)}, {np.std(computed_tokens)}')
    print(f'current_tokens: {np.mean(current_tokens)}, {np.std(current_tokens)}')
    print(f'total_tokens: {np.mean(total_tokens)}, {np.std(total_tokens)}')
    print(f'max_computed_tokens: {np.mean(max_computed_tokens)}, {np.std(max_computed_tokens)}')
    
    
    
    # def predictor(batch: "Batch") -> float:
    #     return 6.25e-5 * batch.total_current_length + 3.7e-5 * batch.num_reqs + 5.00e-8 * batch.total_past_tokens + 1.4e-2
    
    fit(batches, predictor=predictor, plot_path = 'all-1.png', features=features, _ax = ax, **kwargs)
        
    fit([b for b in batches if b.decode_only], predictor=predictor, plot_path = 'decode_only-1.png', features=features)
        
    fit([b for b in batches if b.prefill_only], predictor=predictor, plot_path = 'prefill_only-1.png', features=features)

if __name__ == '__main__':

    
        

    gemma27b_filepaths = [
        'experiments/Gemma-3-27B-IT_constant_azure_chat_23:azure_chat_23_600:1201_anytime/slosserve-edf_round_robin_0.5_1_anytime_5.0_0.1.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:600_anytime/slosserve-edf_round_robin_8.0_1_anytime_1.5_0.05.0.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/slosserve-edf_round_robin_0.5_1_anytime_1.5_0.05.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.4_1_anytime_3.0_0.05.events.jsonl'
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.events.jsonl'
    ]
    
    
    qwne7b_4_filepaths = [
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_2_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_4_anytime_3.0_0.025.1.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_auto_scaling_0.8_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_chat:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.5_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/slosserve-edf_lightest_first_retry_0.5_4_anytime_5.0_0.1.events.jsonl',
        'experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/slosserve-edf_lightest_first_retry_0.5_4_anytime_5.0_0.1.events.jsonl',
        'experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/slosserve-edf_round_robin_2.0_4_anytime_5.0_0.1.events.jsonl'
    ]
    
    qwen7b_chat_filepaths = [
        # 'experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/vllm_round_robin_1.0_4_anytime_3.0_0.025.events.jsonl'
        # 'experiments/Qwen-7B_constant_sharegpt_chat:azure_chat_23_600:800_anytime/slosserve-edf_round_robin_0.5_1_anytime_5.0_0.1.events.jsonl'
        'experiments/Qwen-7B_constant_azure_chat_23:azure_chat_23_600:1200_anytime/slosserve-edf_round_robin_1.0_1_anytime_5.0_0.1.events.jsonl'
    ]
    
    arxiv_filepaths = [
        'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation_450:550_anytime/vllm_round_robin_10.0_1_anytime_2.0_0.1.events.jsonl',
        'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_450:550_anytime/slosserve-edf_round_robin_10.0_1_anytime_2.0_0.1.events.jsonl'
    ]
    
    qwen7b_code_filepaths = [
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4579_anytime/sarathi_round_robin_1.2_4_anytime_3.0_0.025.events.jsonl',
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling-all-0.16_1.5_4_anytime_3.0_0.025.events.jsonl',
        'experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat_per_device_0.9_4_anytime_5.0_0.1.events.jsonl'
    ]
    
    gemma27b_4gpu_filepaths = [
        'experiments/Gemma-3-27B-IT_constant_azure_chat_23:azure_chat_23_601:1201_anytime/slosserve-edf_round_robin_4.0_4_anytime_5.0_0.1.events.jsonl'
    ]
    
    qwen7b_arxiv_filepaths = [
        'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-load_slo-1.0_1.0_1_anytime_3.0_0.025.events.jsonl',
    ]
    
    qwne7b_filepaths = [
        'experiments/Qwen-7B_constant_azure_chat_23:azure_code_23_3978:4579_anytime/slosserve-edf_auto_scaling_resch-all_chat-1.0_0.9_1_anytime_5.0_0.1.events.jsonl' # IMPORTANT
        # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.8_2_anytime_3.0_0.025.events.jsonl'
            # 'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:600_anytime/slosserve-edf_round_robin_8.0_1_anytime_1.5_0.05.0.events.jsonl',
            # 'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.3_1_anytime_3.0_0.025.events.jsonl',
            # 'experiments/Qwen-7B_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:800_anytime/slosserve-edf_round_robin_10.0_1_anytime_1.5_0.05.0.events.jsonl'
            ]
    # qwen7b_files = [
    #     'experiments/Qwen-7B_constant_sharegpt_code:azure_code_23_3978:4579_anytime/slosserve-edf_round_robin_1.0_5_anytime_3.0_0.025.events.jsonl'
    # ]
    gemma27b_filepaths = [
        'experiments/Gemma-3-27B-IT_constant_azure_chat_23:azure_chat_23_600:1201_anytime/slosserve-edf_round_robin_0.5_1_anytime_5.0_0.1.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_arxiv_summary:burstgpt_GPT-4_Conversation log_400:600_anytime/slosserve-edf_round_robin_8.0_1_anytime_1.5_0.05.0.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_chat:azure_chat_23_600:1200_anytime/slosserve-edf_round_robin_0.5_1_anytime_1.5_0.05.events.jsonl',
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_0.4_1_anytime_3.0_0.05.events.jsonl'
        # 'experiments/Gemma-3-27B-IT_constant_sharegpt_code:azure_code_23_3979:4580_anytime/slosserve-edf_round_robin_1.0_1_anytime_3.0_0.025.events.jsonl'
    ]
   
    def predictor(batch: "Batch") -> float:
        return 6.565e-5 * batch.total_current_length + 8.00e-8 * batch.total_past_tokens + 1.3e-2
    def gemma27b_predictor(batch: "Batch") -> float:
        return 7.1e-5 * batch.total_current_length + 3.82e-5 * batch.num_reqs + 9.380e-8 * batch.total_past_tokens + 1.8e-2
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    fit_model(gemma27b_filepaths, predictor = 'linear', ax = ax, label = '27B (H200)')
    
    fit_model(qwne7b_filepaths, predictor = 'linear', ax = ax, label = '7B (A100)')
    ax.legend(fontsize=20, loc='upper left')
    ax.plot([0, 0.1], [0, 0.1], 'k--', linewidth=2)
    ax.set_xlabel('Real Time (s)')
    ax.set_ylabel('Predicted Time (s)')
    fig.savefig('figs/perf_model.png', dpi=200)
    fig.savefig('figs/perf_model.pdf', dpi=200)