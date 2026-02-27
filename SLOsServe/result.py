from Dataset.dataset import Request, Requests, ArrivalTimes
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class ExecutionResult:
    request: Request
    timestamps: List[float] # timestamps[0] is the arrival time of the request, timestamps[i] is the time to output the i-th token
    ttft: float  = field(init=False)
    ttfat: float = field(init=False)
    tpots: List[float] = field(init = False)
    normalized_ttft: float = field(init=False)
    normalized_ttfat: float = field(init=False)
    
    def __post_init__(self):
        # assert len(self.timestamps) == 1 + self.request.output_length + self.request.thinking_length
        # print(f'timestamps: {len(self.timestamps)}, thinking_length: {self.request.thinking_length}, output_length: {self.request.output_length}, input_length: {self.request.input_length}')
        # Add safeguards to prevent IndexError and handle edge cases
        if len(self.timestamps) < 2:
            self.ttft = 0.0
            self.normalized_ttft = 0.0
        else:
            self.ttft = self.timestamps[1] - self.timestamps[0]
            self.normalized_ttft = self.ttft / self.request.input_length

        thinking_idx = self.request.thinking_length
        if len(self.timestamps) > thinking_idx:
            self.ttfat = self.timestamps[thinking_idx] - self.timestamps[0]
            self.normalized_ttfat = 0.0
        else:
            self.ttfat = self.timestamps[-1] - self.timestamps[0]
            self.normalized_ttfat = self.ttfat / (self.request.thinking_length + self.request.input_length)
        
        self.tpots = []
        base_idx = self.request.thinking_length + 1
        for i in range(self.request.output_length - 1):
            idx1 = base_idx + i + 1
            idx0 = base_idx + i
            if idx1 < len(self.timestamps) and idx0 < len(self.timestamps):
                self.tpots.append(self.timestamps[idx1] - self.timestamps[idx0])