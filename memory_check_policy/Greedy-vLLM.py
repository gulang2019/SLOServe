from bisect import bisect_right
from typing import List, Callable, Any
import random

def accept_greedy(
    N: int,
    M: float,
    m: float,
    t_token: float,
    L_in_old_list: List[int],
    L_in_new: int,
    **kwargs: Any,
) -> bool:
    """一律接受。"""
    return True