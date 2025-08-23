from typing import List, Tuple
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import warnings

# PROFILES = {
#     ('humaneval', 'facebook/opt-6.7b', None): [5.9141031e-05, 2.768749e-03, 3.02998e-02],
#         # [0.0, 0.0, 0.024,
#         #   8e-5, 0.0, 0.012],
#     ('humaneval', 'facebook/opt-6.7b', 'facebook/opt-125m'): [5.9141031e-05, 2.768749e-03, 3.02998e-02],
#     ('sharegpt', 'facebook/opt-6.7b', None): [5.3141031e-05, 2.768749e-03, 3.02998e-02],
#         # [0.0, 0.0, 0.024,
#         #   8e-5, 0.0, 0.012],
#     ('sharegpt', 'facebook/opt-6.7b', 'facebook/opt-125m'): [5.3141031e-05, 2.768749e-03, 3.02998e-02],
#     ('azure', 'facebook/opt-6.7b', None): [5.3141031e-05, 2.768749e-03, 3.02998e-02],
#     ('azure', 'facebook/opt-6.7b', 'facebook/opt-125m'): [5.3141031e-05, 2.768749e-03, 3.02998e-02]
# }

PROFILES = {
    ('facebook/opt-6.7b', 'facebook/opt-125m', '1-1'): [6.87825911e-05,1.24858655e-06,2.48005651e-02,7.52283729e-05,3.46460621e-03,3.03278716e-03],
    ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', '1-1'): [6.87825911e-05,1.24858655e-06,2.48005651e-02,7.52283729e-05,3.46460621e-03,3.03278716e-03],
    ('facebook/opt-13b', 'facebook/opt-125m', '2-1'): [7.212770050439044e-05, 0.0033406819163180985, 0.031731739725975795, 0.0, 0.0030136716729365584, 1.2387542377292403e-05],
    ('facebook/opt-13b', None, '2-1'): [7.212770050439044e-05, 0.0033406819163180985, 0.031731739725975795, 0.0, 0.0030136716729365584, 1.2387542377292403e-05],
    ('facebook/opt-30b', 'facebook/opt-125m', '4-1'): [9.314424265751298e-05, 0.0027965517836802886, 0.037879674659981134, 0.0, 0.0030136716729365584, 1.2387542377292403e-05],
    ('Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', '4-1'): [0.00010540079657808384, 0.004520039697391629, 0.0451294531246859],
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', '1-1'): [0.00010540079657808384, 0.004520039697391629, 0.0451294531246859],
    ('Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', '4-1'): [0.00010540079657808384, 0.004520039697391629, 0.0451294531246859],
    ('facebook/opt-30b', None, '4-1'): [9.314424265751298e-05, 0.0027965517836802886, 0.037879674659981134, 0.0, 0.0030136716729365584, 1.2387542377292403e-05],
    ('Qwen/Qwen2.5-32B-Instruct', None, '4-1'): [0.00010540079657808384, 0.0, 0.0451294531246859],
    ('facebook/opt-6.7b', None, '1-1'): [6.87825911e-05,0.0,2.48005651e-02,7.52283729e-05,0.0,3.03278716e-03],
    ('facebook/opt-13b', None, '1-1'): [6.87825911e-05,0.0,2.48005651e-02,7.52283729e-05,0.0,3.03278716e-03],
    ('facebook/opt-125m', None, '1-1'): [4.544639279046835e-05, 0.0, 0.017011914494666814],
    ('facebook/opt-125m', None, '1-1'): [4.544639279046835e-05, 0.0, 0.017011914494666814],
    ('deepseek-ai/DeepSeek-R1', None, '1-1'): [6.87825911e-05,0.0,2.48005651e-02,7.52283729e-05,0.0,3.03278716e-03],
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', None, '1-1'): [0.0002915977341226002, 0.0, 0.018044674471652762, 0.0, 0.0, 1.2387542377292403e-05],
    ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', None, '1-1'): [6.044639279046835e-05, 
                                                        0.00, 
                                                        0.014011914494666814,
                                                        0,0,0.030],
    ('ToolBench/ToolLLaMA-2-7b-v2', None, '1-1'): [6.572428379390161e-05, 0, 0.027524459279686265, 0, 0, 1.2387542377292403e-05],
    ('facebook/opt-13b (H100)', None, '1-1'): [4.1773664644533544e-05, 0, 0.027431231194497186, 0, 0.0, 1.2387542377292403e-05]
}

@dataclass 
class BatchTimer:
    # max_i {params[i] #token + params[i+1] decode_step + params[i+2]}
    params: List[float]
    
    @staticmethod
    def from_hardware(
        n_param: float,
        mem_bw: float,
        compute_speed: float
    ):
        raise NotImplementedError
        return BatchTimer(
            [2 / compute_speed * n_param, 0, 0.,n_param / mem_bw]
        )

    
    @staticmethod 
    def from_model(
        model: str,
        spec_model: str | None = None,
        para_config: str = '1-1-1'
    ):
    
        tp_pp = '-'.join(para_config.split('-')[-2:])
        if ( 
            model, 
            spec_model, tp_pp) in PROFILES:
            return BatchTimer(PROFILES[(
            model,
            spec_model,
            tp_pp)])
        warnings.warn(f"batch timer not registered for {model}, {spec_model}, and {tp_pp}")
        return None
    
    
    def __post_init__(self):
        assert len(self.params) % 3 == 0 
        for i in range(0, len(self.params), 3):
            k1, k2, b = self.params[i:i+3]
            assert k1 >=0 
            assert k2 >= 0 
            assert b >= 0
            self.params[i] = max(self.params[i], 1e-7)
        # self.shift_bs = int((self.b2 - self.b1) / (self.k1 - self.k2))
        print(self)
    
    # def get_max_throughput(self):
    #     return 1 / self.k
    
    def __repr__(self):
        ret = f'T(b) = max('
        for i in range(0, len(self.params), 3): 
            ret += f'{self.params[i]} #Token + {self.params[i+1]}#decode steps + {self.params[i+2]}'
        ret += ')'
        return ret
    
    def __call__(self, bs: int, decode_steps: int = 0) -> float:
        return max(
            bs * self.params[i] + decode_steps * self.params[i+1] + self.params[i+2]
            for i in range(0, len(self.params), 3)
        )
        
    # def time_to_token(self, t: float) -> int:
    #     if not t >= self.b:
    #         print('t < b', t, self.b)
    #         raise RuntimeError
    #     return max(math.floor((t - self.b1) / self.k), self.shift_bs)

    def reverse(self, t: float, decode_steps = 0) -> float:
        bs = 100000
        for i in range(0, len(self.params), 3):
            k1, k2, b = self.params[i:i+3]
            bs = min(bs, (t - b - k2 * decode_steps) / k1)
            
        assert bs >= 0
        return bs
    
    
    @staticmethod
    def from_data(data: List):
        '''
        fit the batch timer w/ real data.
        '''
        from .performance_model import fit 
        print('fitting data...')
        return BatchTimer(
            *fit(data)
        )
        
        # Define the model
        # def max_model(x, k, b, b1):
        #     return np.maximum(k * x + b1, b)

        # # Define the loss function (MSE)
        # def loss_function(params, x, y):
        #     k, b, b1 = params
        #     predictions = max_model(x, k, b, b1)
        #     return np.mean((y - predictions) ** 2)

        # # Initial guess for k and b
        # initial_guess = [1e-4, 3e-2, 0]

        # data = sorted(data, key = lambda x: x[0])

        # x_data = np.array([x[0] for x in data], dtype = np.float64)
        # y_data = np.array([x[1] for x in data], dtype = np.float64)

        # # Fit the model using scipy.optimize
        # result = minimize(loss_function, initial_guess, args=(x_data, y_data))

        # # Extract the optimal parameters
        # optimal_k, optimal_b, optimal_b1 = result.x
        # print(f"Optimal k: {optimal_k}")
        # print(f"Optimal b: {optimal_b}")
        
        # # Plot the results
        # fig, ax = plt.subplots()
        # ax.scatter(x_data, y_data, label="Data", alpha=0.6)
        # ax.plot(x_data, max_model(x_data, optimal_k, optimal_b, optimal_b1), color="red", label="Fitted Model")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_title("Fitting y = max(kx, b)")
        # plt.legend()
        # fig_path = f'{prefix}-fit.png'
        # print('fit figure saved to ', fig_path)
        # fig.savefig(fig_path)
        
        # self.k, self.b, self.b1 = result.x
        # self.shift_bs = int(int((self.b - self.b1) / self.k) // 16 * 16)
        # print(f'T(b) = max({self.k} #Token + {self.b1}, {self.b}), shift = {self.shift_bs}')