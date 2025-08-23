from .profiler import ProfileDatapoint
from .struct import BatchSchedule
from dataclasses import is_dataclass, fields
import json 
from typing import List, Any
import torch
from itertools import accumulate
import pandas as pd
import tqdm
import os 

'''
def from_json(cls, json_dict):
    """
    Recursively loads a dataclass object from a JSON dictionary.
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} must be a dataclass type.")
    
    # Prepare field mappings
    field_types = {f.name: f.type for f in fields(cls)}
    
    # Initialize the dataclass
    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name not in json_dict:
            continue
        
        value = json_dict[field_name]
        
        if is_dataclass(field_type):  # Nested dataclass
            kwargs[field_name] = from_json(field_type, value)
        elif isinstance(value, list) and hasattr(field_type, "__args__"):  # List of dataclasses
            item_type = field_type.__args__[0]  # Get the type of list items
            kwargs[field_name] = [from_json(item_type, item) if is_dataclass(item_type) else item for item in value]
        elif isinstance(value, dict) and hasattr(field_type, "__args__"):  # Dicts with complex values
            key_type, val_type = field_type.__args__
            kwargs[field_name] = {key: from_json(val_type, val) if is_dataclass(val_type) else val for key, val in value.items()}
        else:  # Base types
            kwargs[field_name] = value
    
    return cls(**kwargs)

def load_profile_data(file_path: str) -> List[ProfileDatapoint]:
    """
    Load a list of ProfileDatapoint objects from a JSON file.

    Args:
        file_path (str): File path to load the data from.

    Returns:
        List[ProfileDatapoint]: List of loaded ProfileDatapoint objects.
    """
    try:
        # Read JSON file
        with open(file_path, 'r') as f:
            data_dicts = json.load(f)
        # Convert list of dictionaries to list of dataclass objects
        data = [from_json(ProfileDatapoint, datapoint) for datapoint in data_dicts]
        print(f"Data successfully loaded from {file_path}")
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
'''
'''
The performance model is 
T(Batch) = max(Sum_b(b.n_tokens) * k + b, b1) + max(Sum_b(b.n_tokens) * k + b, b1) 
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
'''
def regression(x_data: np.ndarray, 
               y_true: np.ndarray,
               num_funcs: int = 3,
               save: str| None = None):
    # Number of linear functions
    # Define the objective function
    def objective(params, iter=[0]):  # Use a mutable default argument to keep track of iterations
        # Split params into slopes (k) and intercepts (b)
        k = params[:num_funcs]
        b = params[num_funcs:]
        
        # Compute y_j = max(k_i * x_j + b_i) for all datapoints
        y_pred = np.array([max(k * x + b) for x in x_data])
                
        # Return the squared error
        return np.sum((y_pred - y_true) ** 2)

    # Initial guess for parameters (random initialization)
    initial_params = np.zeros(num_funcs * 2)

    # Optimize
    result = minimize(objective, initial_params, method='L-BFGS-B')

    # Extract optimized k and b
    k_opt = result.x[:num_funcs]
    b_opt = result.x[num_funcs:]

    # Final visualization
    y_pred_final = np.array([max(k_opt * x + b_opt) for x in x_data])
    # Visualization setup
    if save is not None:
        fig, ax = plt.subplots()
        ax.set_title("Fitting Process")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(x_data, y_true, label="True Values", color="red")
        line, = ax.plot([], [], label="Predicted Values", color="blue")
        ax.legend()
        line.set_xdata(x_data)
        line.set_ydata(y_pred_final)
        plt.pause(0.1)
        fig.savefig(f'{save}.png')
        print(f'fitted saved to {save}.png')

    # Output results
    print(f"Optimized k: {k_opt}")
    print(f"Optimized b: {b_opt}")
'''

def linear_model(x, a, b):
    """Linear function y = ax + b."""
    return a * x + b

def regression(x_data: np.ndarray, 
               y_true: np.ndarray,
               n_funcs: int = 1,
               initial_params: Any | None = None, 
               ax = None,
               label = 'True Values'):
    print('x_data', x_data.shape)
    n_metrics = x_data.shape[-1]  # Number of features in x_data

    def objective(params):
        # Reshape params back to (n_metrics, n_funcs)
        params = params.reshape(n_funcs, n_metrics).T
        # print('x_data', x_data.shape, 'params', params.shape)
        # print('params', params, x_data[0, :])
        # Compute predictions
        y_pred = np.max(x_data @ params, axis=-1)  # max across functions
        return np.mean((y_pred - y_true) ** 2)  # Mean Squared Error

    # Initial guess for parameters (flattened array)
    if initial_params is None:
        initial_params = np.zeros(shape=(n_metrics, n_funcs)).astype(np.float32).flatten() * 1e-3
    # initial_params = [1e-5, 0, 3e-2, 2e-2]
    # initial_params = [1e-5, 0, 3e-2, 2e-2, 0, 0, 0, 0]
    
    # Minimize the objective function
    result = minimize(objective, initial_params, method='L-BFGS-B')

    # Reshape the optimized parameters back to 2D
    # params_opt = result.x.reshape(n_funcs, n_metrics).T
    params_opt = np.array(initial_params).reshape(n_funcs, n_metrics).T
    
    # Compute the final predictions
    y_pred_final = np.max(x_data @ params_opt, axis=-1)

    # Visualization setup
    if ax is not None:
        # fig, ax = plt.subplots()
        ax.set_title("Fitting Process")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        # ax.set_xlabel('#Token')
        # ax.set_ylabel('Time')
        # ax.scatter(x_data[:, 0], y_pred_final, label = 'Pred Values')
        # ax.scatter(x_data[:, 0], y_true, label = 'True Values', color = "red")
        ax.scatter(y_pred_final, y_true, label = label)
        
        
        # fig.savefig(f'{save}.png')
        # fig.savefig(f'{save}.pdf')
        # print(f'Fitted plot saved to {save}.png')
    
    ss_res = np.sum((y_true - y_pred_final) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    mse = np.mean((y_pred_final - y_true) ** 2)
    is_ub = y_pred_final > y_true
    # Output results
    print(f"Optimized: {params_opt.T.reshape(-1).tolist()}")
    print(f'MSE', mse)
    print(f'r2', r2)
    print(f'ub rate', np.mean(is_ub))
    
    
    return mse, r2, params_opt

def predict(x, params):
    t = 0
    for i in range(0, len(params), 3):
        t = max(t, params[i]*x[0] + params[i+1]*x[1] + params[i+2]*x[2])
    return t

def get_metrics(sch: BatchSchedule):
    decode_steps= [req.n for req in sch.reqs if not req.is_prefill]
    return sch.get_effective_bs(), max(decode_steps, default = 0), 1
    # return sch.get_effective_bs(), 1
 
def fit(
    datapoints: List[ProfileDatapoint],  
):
    xs = [get_metrics(data.sch) for data in datapoints]
    ys = [data.e2e_time for data in datapoints]
    mse, r2, params = regression(np.array(xs), 
                                np.array(ys), 
                                n_funcs = 2,
                                initial_params= [1e-5, 0, 3e-2, 2e-2],
                                save = f'all-e2e-fit')
    print('params', params, 'mse', mse, 'r2', r2)
    
    return params 
# for ds in ('sharegpt', 'azure'):
#     data_path = f'profile/{ds}.json'
#     datapoints = load_profile_data(data_path)
#     xs = [get_metrics(data.sch) for data in datapoints]
#     ys = [data.e2e_time for data in datapoints]
#     mse, r2, params = regression(np.array(xs), np.array(ys), save = f'{ds}-e2e-fit')
#     print('params', params, 'mse', mse, 'r2', r2)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type = str, default = 'profile/humaneval.json')
    parser.add_argument('--tag', type = str, default = 'profile/humaneval.json')
    args = parser.parse_args()
    # filenames = [
    #     ('Chatting', 'result/sla_newest/profile/splitwise_conv-sharegpt-900.0:300.0-1.0-3.0:0.03:0.0:0.0:1.0$promax-dp-best_effort$1.0.json'),
    #     ('Coding', 'result/sla_newest/profile/splitwise_code-humaneval-1786.0:300.0-1.0-10.0:0.1:0.0:0.0:1.0$promax-dp-best_effort$4.0.json')
    # ]
    filenames = [(args.tag, args.file_path)]
    
    from .profiler import load_profile_data
    
    
    # print('drafter:')
    # ys = [data.draft_time for data in datapoints]
    # mse, r2, params = regression(np.array(xs), np.array(ys), save = f'all-drafter-fit')
    # print('params', params, 'mse', mse, 'r2', r2)

    # print('verifier:')
    # ys = [data.verifier_time for data in datapoints]
    # mse, r2, params = regression(np.array(xs), 
    #                             np.array(ys), 
    #                             save = f'all-verifier-fit', n_funcs=2, initial_params= [1e-5, 0, 3e-2, 2e-2])
    # print('params', params, 'mse', mse, 'r2', r2)

    print('e2e:')
    for (tag, filename) in filenames:
        
        print('filename', filename)
        assert os.path.exists(filename)
        datapoints = load_profile_data(filename)
        # print(datapoints)
        # print('EXAMPLE', datapoints[0])
        # datapoints = []
        # for data_path in args.file_path:
        #     datapoints.extend(load_profile_data(data_path))
        datapoints = [data for data in datapoints if data.bs > 1 and data.e2e_time > 0.019]
        # print('#batch', len(datapoints))
        # exit(0)
        bs_times = [(data.sch.get_effective_bs(), data.e2e_time) for data in datapoints]
        df = pd.DataFrame(bs_times, columns = ['batch_size', 'time'])
        # df.to_csv('batch_sizes/slos_serve.csv')
        bs_times = sorted(bs_times, key = lambda x: x[0])
        times = []
        for bs, t in bs_times: 
            while bs >= len(times) * 1:
                times.append(0)
            times[-1] += t
        times = list(accumulate(times))
        max_time = times[-1]
        times = [t / max_time for t in times]
        
        # print('average bs', np.mean(batch_sizes))
        # large_batch_sizes = [bs for bs in batch_sizes if bs >= 512]
        # print('lage bs portion', len(large_batch_sizes) / len(batch_sizes))
        # print('max bs', np.max(batch_sizes))
        # print('hist', np.histogram(batch_sizes, density = True))
        fig, ax = plt.subplots()
        # ax.hist([bs_time[0] for bs_time in bs_times])
        ax.plot(range(0, len(times) * 1, 1), times)
        # ax.hist(batch_sizes)
        fig.savefig('bs_dist.png')
        # fig, ax = plt.subplots()
        # ax.scatter([x.sch.get_effective_bs() for x in datapoints], [x.e2e_time for x in datapoints])
        # fig.savefig('batch.png')
        # exit(0)
        fig, (ax1, ax2) = plt.subplots(1,2,tight_layout = True)
        xs = [get_metrics(data.sch) for data in datapoints]
        # xs = [x for x in xs if x[0] > 0]
        ys = [data.e2e_time for data in datapoints]
        mse, r2, params = regression(np.array(xs),
                                    np.array(ys), 
                                    n_funcs = 1, 
                                    # initial_params=[6.572428379390161e-05, 0, 0.027524459279686265, -0.00032448190656012005, 0, 1.2387542377292403e-05],
                                    # initial_params=[4.242750451725976e-05, -0.00032393034607608266, -2.3358577478035366e-06, 0.0029551942913882295, 0.02478741149675072, 0.00298192548748054],
                                    initial_params = [0.00011540079657808384, 0.004520039697391629, 0.0451294531246859],
                                    ax = ax1,
                                    label = tag)
        print('params', params, 'mse', mse, 'r2', r2)
        ax1.plot([0, 0.17], [0, 0.17], label = 'x=y', linestyle = '--', color = 'red', linewidth = 2.5)
        ax1.grid()
        ax1.legend()
        ax1.set_xlim(0, 0.20)
        ax1.set_ylim(0, 0.20)
        ax1.set_xlabel('Estimated Batch Time (s)', fontsize = 12)
        ax1.set_title('')
        ax1.set_ylabel('Profiled Batch Time (s)', fontsize = 12)
        bss = [x.sch.get_effective_bs() for x in datapoints]
        params = params.reshape(-1)
        # params = [0.00010540079657808384, 0.004520039697391629, 0.04291294531246859]
        # print(xs)
        estimated_times = [predict(x, params) for x in xs]
        profiled_times = [x.sch.profiled_time_ft for x in datapoints]
        ax2.scatter(bss, profiled_times, label = 'profiled')
        ax2.scatter(bss, estimated_times, label = 'prediction')
        ax1.axis('equal')
        ax2.legend()
        ax2.grid()
        ax1.legend()
        fig.savefig(f'fit.pdf')
        fig.savefig(f'fit.png')
    print('params', params, 'mse', mse, 'r2', r2)