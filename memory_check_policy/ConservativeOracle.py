from bisect import bisect_right
from typing import List, Callable, Any
import random

def accept_conservative_oracle(
    N: int,
    M: float,
    m: float,
    L_in_old_list: List[int],
    L_in_new: int,
    L_out_old_list: List[int],
    L_out_new: int,
    P_list: List[int] = None,
    **kwargs: Any,
) -> bool:
    """
    conservative的oracle - 假设完全知道output len
    """
    L_out_old = [int(x) for x in L_out_old_list]
    L_in_old = [int(x) for x in L_in_old_list]
    L_in_new = int(L_in_new)
    L_out_new = int(L_out_new)
    
    sum_old_out = sum(L_out_old) if L_out_old else 0
    sum_old_in = sum(L_in_old) if L_in_old else 0
    
    peak_mem = m * (sum_old_out + sum_old_in + L_in_new + L_out_new)
    return peak_mem <= M
