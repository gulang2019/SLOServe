from bisect import bisect_right
from typing import List, Callable, Any
import random

def accept_oracle_upper_bound(
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
    性能上界（Oracle / 最优情况估计）策略：
    """
    if L_out_new is None:
        L_out_new = 0
    P_list = P_list if P_list is not None else []
    n_old = len(L_in_old_list)
    if len(L_out_old_list) != n_old:
        return True

    P_eff = [int(x) for x in P_list]

    L_out_old = [int(x) for x in L_out_old_list]
    L_in_old = [int(x) for x in L_in_old_list]
    L_in_new = int(L_in_new)
    L_out_new = int(L_out_new)

    # 需要模拟的步数上界：直到所有请求 retire
    remaining_old = [max(L_out_old[i] - P_eff[i], 0) for i in range(n_old)]
    K = max([L_out_new] + remaining_old) if (n_old > 0 or L_out_new > 0) else 0

    peak_tokens = 0
    for k in range(0, K + 1):
        total = 0
        # 旧请求：若当前已生成 P_i + k < L_out_i 则仍活跃
        for i in range(n_old):
            if P_eff[i] + k < L_out_old[i]:
                total += L_in_old[i] + P_eff[i] + k
        # 新请求：若 k < L_out_new 则活跃
        if k < L_out_new:
            total += L_in_new + k
        if total > peak_tokens:
            peak_tokens = total

    peak_mem = m * peak_tokens
    return peak_mem <= M