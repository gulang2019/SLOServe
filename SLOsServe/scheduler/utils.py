import numpy as np 
import time
from dataclasses import fields, is_dataclass, dataclass
import random

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

def poisson_process(lam, N):
    # Step 1: Sample inter-arrival times from an exponential distribution
    inter_arrival_times = np.random.exponential(1 / lam, N)
    
    # Step 2: Calculate event times
    event_times = np.cumsum(inter_arrival_times)
    return event_times

def poisson_geo_process(lam, n_req_at_once, N):
    '''
    lambda: the request rate
    p: coin prob.
    '''
    indices = np.array(sum(([i] * n for i, n in enumerate(np.random.geometric(1/n_req_at_once, size=(N,)))), [])[:N])
    
    event_times = np.cumsum(np.random.exponential(1 / lam, N))[indices]
    return event_times


class Timer:
    def current_time(self) -> float:
        return time.perf_counter()
    
    def start(self):
        self.start_time = time.perf_counter()
        self.times = {}
        self.last_s = 'START'
        
    def __call__(self, s: str):
        cur_time = time.perf_counter()
        self.times[f'{self.last_s}->{s}'] = self.times.get(s, 0) + cur_time - self.start_time
        self.start_time = cur_time         
        self.last_s = s
    
    def display(self):
        times = list(self.times.items())
        times = sorted(times, key = lambda x: x[1])
        e2e_time = sum((t[-1] for t in times), 0)
        times = [(k, round(v / e2e_time, 3)) for k, v in times]
        print(times)
    
    
@dataclass
class ExponentialBackoff:
    MAX_RETRIES: float = 5  # Maximum number of retries
    BASE_DELAY: float = 1  # Base delay in seconds
    MAX_DELAY: float = 16  # Maximum delay in seconds
    
    def exponential_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.
        :param attempt: The retry attempt number (starting from 1).
        """
        delay = min(self.BASE_DELAY * (2 ** (attempt - 1)), self.MAX_DELAY)
        delay_with_jitter = delay * random.uniform(0.8, 1.2)  # Add some randomness
        return delay_with_jitter