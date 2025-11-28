from __future__ import annotations

import copy

class PerfModel:
    def __init__(self, hardware_params: list[float]):
        assert len(hardware_params) == 5
        self.hardware_params = copy.deepcopy(hardware_params)
    
    def get_batch_time(self, num_tokens: list[tuple[int, int]]) -> float:
        num_reqs = len(num_tokens)
        num_tot_tokens = sum([x[1] for x in num_tokens], start = 0)
        num_past_tokens = sum([x[0] for x in num_tokens], start = 0)
        num_decode_steps = 1
        return self.hardware_params[0] * num_tot_tokens \
            + self.hardware_params[1] * num_reqs \
            + self.hardware_params[2] * num_past_tokens + \
            self.hardware_params[3] * num_decode_steps + self.hardware_params[4]
    
    def get_bs(self, t: float, num_reqs: int, num_past_tokens: int = 0, num_decode_steps: int = 1) -> int:
        return int((t - self.hardware_params[4] \
            - self.hardware_params[3] * num_decode_steps - self.hardware_params[2] * num_past_tokens\
            - self.hardware_params[1] * num_reqs) / (self.hardware_params[0]))
    
    def get_max_decode_batch_size(self, t: float, average_context_length: float = 0.0) -> int:
        return int((t - self.hardware_params[4] - self.hardware_params[3]) / (self.hardware_params[0] + self.hardware_params[1] + self.hardware_params[2] * average_context_length))
    
    @staticmethod
    def get_perf_model(model_name: str, task: str = 'default') -> 'PerfModel':
        return PerfModel(get_hardware_params(model_name, task))
    
    def get_zero_load_ttft(self, input_length: int, cached_length: int = 0) -> float:
        return self.get_batch_time([(cached_length, input_length - cached_length)])

HW_PARAMS = {
    'default': [4.86e-5, 1.69e-5, 8e-8, 0, 1.4e-2],
    # 'Qwen/Qwen2.5-7B-Instruct': [4.86e-5, 3.7e-5, 5e-8, 0, 1.3e-2],
    'Qwen/Qwen2.5-7B-Instruct': {
        'default': [5.1e-05, 1.69e-5, 8e-8, 0, 1.4e-2],
        'sharegpt_code': [5.1e-05, 0.00, 8e-8, 0, 1.4e-02],
        'azure_chat_23': [5.55e-5, 0.00, 9e-8, 0, 1.4e-2],
        'arxiv_summary': [6.565e-05, 0.00, 8e-8, 0, 1.3e-02]}, # ChatBot
    'google/gemma-3-27b-it': {
        'default': [7.69e-5, 5.82e-5, 4.40e-8, 0, 1.9e-2],
        'azure_chat_23': [7.1e-5, 3.82e-5, 9.380e-8, 0, 1.8e-2]
    },
    'meta-llama/Llama-3.1-70B': {
        'default': [6.2e-5, 3.7e-5, 5e-8, 0, 1.4e-2]
    },
}

def get_hardware_params(model_name, task):
    if model_name not in HW_PARAMS:
        return HW_PARAMS['default']
    if task not in HW_PARAMS[model_name]:
        return HW_PARAMS[model_name]['default']
    return HW_PARAMS[model_name][task]

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
