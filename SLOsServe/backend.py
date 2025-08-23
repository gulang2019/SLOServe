from enum import Enum 

class Backend(Enum):
    HUGGINGFACE = 0 
    DEEPSPEED = 1
    VLLM = 2
