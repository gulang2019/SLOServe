from dataclasses import dataclass

@dataclass 
class RequestOutput:
    generated_text: str 
    is_end: bool 
    n_generated: int
    n_spec: int
    acc_rate: float = 0
    get_time: float = 0
    n_interpreted: int = 0
    issue_time: float = 0
    executor_get_delay: float = 0
    frontend_get_delay: float = 0
    n_get: float = 0
    n_alg_spec: float = 0
    
    def __post_init__(self):
        self.acc_rate = self.n_spec / self.n_generated

@dataclass
class RequestInput:
    prompt: str 
    max_new_tokens: int = 2048
    ignore_eos: bool = False
    temperature: float = 0.0