import copy
import json
import logging
from pathlib import Path

from SLOsServe.fitting_utils import (
    fit_linear_perf_model,
    sanitize_filename,
    save_prediction_scatter,
    write_json,
)
from SLOsServe.model_config import get_model_config

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PERF_MODEL_PATH = ASSETS_DIR / "perf_model.json"
PERF_MODEL_FIG_DIR = ASSETS_DIR / "perf_model_figs"

class PerfModel:
    def __init__(self, model_name, hardware_params: list[float]):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        assert len(hardware_params) == 5
        self.hardware_params = copy.deepcopy(hardware_params)
        self._online_average_delay = 0.0
        self._online_spike_slack = 0.0
        self._decay_factor = 0.95
        self._update_cnt = 0
    
    def get_batch_time(self, num_tokens: list[tuple[int, int]]) -> float:
        num_reqs = len(num_tokens)
        num_tot_tokens = sum([x[1] for x in num_tokens], start = 0)
        num_past_tokens = sum([x[0] for x in num_tokens], start = 0)
        num_decode_steps = 1
        return self.hardware_params[0] * num_tot_tokens \
            + self.hardware_params[1] * num_reqs \
            + self.hardware_params[2] * num_past_tokens + \
            self.hardware_params[3] * num_decode_steps + self.hardware_params[4] + self._online_spike_slack
    
    def update(self, batch: list[tuple[int, int]], elapsed: float):
        pass 
        return 
        estimated = self.get_batch_time(batch) - self._online_spike_slack
        self._online_average_delay = self._decay_factor * (self._online_average_delay) + \
            (1 - self._decay_factor) * (elapsed - estimated)
        self._online_spike_slack = max(self._online_average_delay, 0.0)
        if self._update_cnt % 100 == 0:
            logger.info(f'[PerfModel::Update]: {self._online_average_delay=}, {self._online_spike_slack=}')
        self._update_cnt += 1
        return 
    
    def get_bs(self, t: float, num_reqs: int, num_past_tokens: int = 0, num_decode_steps: int = 1) -> int:
        return int((t - self.hardware_params[4] - self._online_spike_slack \
            - self.hardware_params[3] * num_decode_steps - self.hardware_params[2] * num_past_tokens\
            - self.hardware_params[1] * num_reqs) / (self.hardware_params[0]))
    
    def get_max_decode_batch_size(self, t: float, average_context_length: float = 0.0) -> int:
        return int((t - self.hardware_params[4] - self.hardware_params[3] - self._online_spike_slack) / (self.hardware_params[0] + self.hardware_params[1] + self.hardware_params[2] * average_context_length))

    def copy_with_adjustments(
        self,
        *,
        scale: float = 1.0,
        constant_offset: float = 0.0,
    ) -> 'PerfModel':
        adjusted = copy.deepcopy(self)
        if scale != 1.0:
            adjusted.hardware_params = [
                param * scale for param in adjusted.hardware_params
            ]
        if constant_offset != 0.0:
            adjusted.hardware_params[4] += constant_offset
        return adjusted
    
    @staticmethod
    def get_perf_model(model_name: str, task: str = 'default') -> 'PerfModel':
        return PerfModel(model_name, get_hardware_params(model_name, task))
    
    def get_zero_load_ttft(self, input_length: int, cached_length: int = 0) -> float:
        return self.get_batch_time([(cached_length, input_length - cached_length)])

    def get_kv_mem_per_token(self):
        return self.model_config.get_token_cache_mem()

    def get_max_decode_length(self):
        return get_model_max_tokens(self.model_name)

    def fit(self,
            batch_times: list[tuple[list[tuple[int, int]], float]],
            tag: str, 
            viz=False,
            min_abs_num_reqs_coef: float = 1e-9):
        '''
        fit the linear regression model 
        @param batch_times list of batches: [[(past_len, currentlen)], measured_time]
        @param tag: the store prefix 
        @param viz: visualize or not 
        '''
        fit_result = fit_linear_perf_model(
            batch_times,
            min_abs_num_reqs_coef=min_abs_num_reqs_coef,
        )
        self.hardware_params = copy.deepcopy(fit_result["hardware_params"])
        upsert_hardware_params(self.model_name, tag, self.hardware_params)

        if viz:
            safe_name = sanitize_filename(f"{self.model_name}__{tag}")
            plot_path = PERF_MODEL_FIG_DIR / f"{safe_name}.png"
            fit_result["plot_path"] = str(
                save_prediction_scatter(
                    plot_path,
                    fit_result["measured_times"],
                    fit_result["predicted_times"],
                    title=f"{get_easy_name(self.model_name)} [{tag}]",
                )
            )

        logger.info(
            "[PerfModel.fit] model=%s tag=%s params=%s stats=%s",
            self.model_name,
            tag,
            self.hardware_params,
            fit_result["stats"],
        )
        return fit_result
        
DEFAULT_HW_PARAMS = {
    'default': [4.86e-5, 1.69e-5, 8e-8, 0, 1.4e-2],
    # 'Qwen/Qwen2.5-7B-Instruct': [4.86e-5, 3.7e-5, 5e-8, 0, 1.3e-2],
    'Qwen/Qwen2.5-7B-Instruct': {
        'default': [5.1e-05, 1.69e-5, 8e-8, 0, 1.4e-2],
        'sharegpt_code': [5.1e-05, 0.00, 8e-8, 0, 1.4e-02],
        'azure_chat_23': [5.55e-5, 0.00, 9e-8, 0, 1.4e-2],
        'arxiv_summary': [6.565e-05, 0.00, 8e-8, 0, 1.3e-02]}, # ChatBot
    'google/gemma-3-27b-it': {
        'default': [7.69e-5, 5.82e-5, 4.40e-8, 0, 1.9e-2],
        'azure_chat_23': [7.1e-5, 3.82e-5, 9.380e-8, 0, 1.8e-2],
        'azure_code_23': [3.62e-5, 3.31e-5, 9.7e-8, 0.0, 0.0176],
        'sharegpt_code': [3.62e-5, 3.31e-5, 9.7e-8, 0.0, 0.0176],
    },
    'meta-llama/Llama-3.1-70B': {
        'default': [6.2e-5, 3.7e-5, 5e-8, 0, 1.4e-2]
    },
}

HW_PARAMS = DEFAULT_HW_PARAMS


def _normalize_hw_param_list(params):
    if not isinstance(params, list) or len(params) != 5:
        return None
    try:
        return [float(param) for param in params]
    except (TypeError, ValueError):
        return None


def _load_hw_params() -> dict:
    registry = copy.deepcopy(DEFAULT_HW_PARAMS)
    if not PERF_MODEL_PATH.exists():
        return registry

    try:
        with PERF_MODEL_PATH.open("r", encoding="utf-8") as f:
            persisted = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load perf model file %s: %s", PERF_MODEL_PATH, exc)
        return registry

    if not isinstance(persisted, dict):
        return registry

    persisted_default = _normalize_hw_param_list(persisted.get("default"))
    if persisted_default is not None:
        registry["default"] = persisted_default

    for model_name, task_params in persisted.items():
        if model_name == "default":
            continue
        if isinstance(task_params, list):
            normalized = _normalize_hw_param_list(task_params)
            if normalized is not None:
                registry[model_name] = {"default": normalized}
            continue
        if not isinstance(task_params, dict):
            continue
        model_registry = copy.deepcopy(registry.get(model_name, {}))
        if not isinstance(model_registry, dict):
            model_registry = {}
        for task, params in task_params.items():
            normalized = _normalize_hw_param_list(params)
            if normalized is None:
                continue
            model_registry[task] = normalized
        if model_registry:
            registry[model_name] = model_registry
    return registry


def upsert_hardware_params(model_name: str, task: str, params: list[float]) -> Path:
    normalized = _normalize_hw_param_list(params)
    if normalized is None:
        raise ValueError("hardware params must be a length-5 numeric list")

    registry = _load_hw_params()
    model_registry = copy.deepcopy(registry.get(model_name, {}))
    if not isinstance(model_registry, dict):
        model_registry = {}
    model_registry[task] = normalized
    if "default" not in model_registry:
        existing_default = _normalize_hw_param_list(
            DEFAULT_HW_PARAMS.get(model_name, {}).get("default")
            if isinstance(DEFAULT_HW_PARAMS.get(model_name), dict) else None
        )
        model_registry["default"] = existing_default or normalized
    registry[model_name] = model_registry
    return write_json(PERF_MODEL_PATH, registry)


def get_hardware_params(model_name, task):
    hw_params = _load_hw_params()
    if model_name not in hw_params:
        return hw_params['default']
    if task not in hw_params[model_name]:
        return hw_params[model_name]['default']
    return hw_params[model_name][task]

get_easy_name = lambda model_name: {
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen-7B', 
    'facebook/opt-125m': 'OPT-125M',
    'google/gemma-7b-it': 'Gemma-7B-IT',
    'google/gemma-3-27b-it': 'Gemma-3-27B-IT',
    'meta-llama/Llama-3.1-70B': 'Llama-70B',
}.get(model_name, model_name)

get_model_max_tokens = lambda model_name: {
    'Qwen/Qwen2.5-7B-Instruct': 24000,
    'facebook/opt-125m': 2048,
    'google/gemma-7b-it': 8192,
    'google/gemma-3-27b-it': 8192,
    'meta-llama/Llama-3.1-70B': 24000,
}.get(model_name, None)

if __name__ == '__main__':
    perf_model = PerfModel.get_perf_model('Qwen/Qwen2.5-7B-Instruct')
    zero_load_decode_time = perf_model.get_batch_time([(0, 1)])
    prefill_time = perf_model.get_batch_time([(1e4, 1000)])
    max_decode_batch_size = perf_model.get_max_decode_batch_size(0.10, 2e3)
    max_decode_batch_size_tight = perf_model.get_max_decode_batch_size(0.025, 100)
    print(
        f'zero_load_decode_time: {zero_load_decode_time}',
        f'prefill_time: {prefill_time}',
        f'max_decode_batch_size: {max_decode_batch_size}',
        f'max_decode_batch_size_tight: {max_decode_batch_size_tight}',
    )
