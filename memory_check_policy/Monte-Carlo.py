from bisect import bisect_right
from typing import List, Callable, Any
import random


RUN_TIMES_DEFAULT = 200


def _sample_output_len(output_lengths: List[int]) -> int:
    """从经验分布中采样一个 output length。"""
    return int(random.choice(output_lengths))


def _sample_output_len_conditional(
    output_lengths: List[int],
    min_exclusive: int,
    sorted_output_lengths: List[int] = None,
) -> int:
    """
    从经验分布中按条件 L > min_exclusive 采样总 output length。

    对于仍在 batch 中、当前已生成了 P_i 个 token 的旧请求，需要满足真实总长度
    L_out_old > P_i；否则该请求不可能仍处于活跃状态。若经验样本中不存在满足条件
    的值，则返回 min_exclusive + 1。
    """
    if sorted_output_lengths is None:
        conditional_pool = [int(x) for x in output_lengths if int(x) > int(min_exclusive)]
        if conditional_pool:
            return int(random.choice(conditional_pool))
        return int(min_exclusive) + 1

    start = bisect_right(sorted_output_lengths, int(min_exclusive))
    if start < len(sorted_output_lengths):
        return int(random.choice(sorted_output_lengths[start:]))
    return int(min_exclusive) + 1


def accept_mc(
    N: int,     # 活跃请求数
    M: float,   # LLM server总存储空间
    m: float,   # 每token占用空间
    L_in_old_list: List[int],   # 活跃请求的input len
    L_in_new: int,      # 新到达请求的input len
    theta: float = None,    # 概率阈值
    output_lengths: List[int] = None,   # 请求的output len的历史分布
    P_list: List[int] = None,   # 当前活跃请求的decoded len
    run_times: int = RUN_TIMES_DEFAULT,     # mc模拟次数
    **kwargs: Any,
) -> bool:
    """
    基于蒙特卡洛模拟的 OOM 准入策略：

    - 每次试验中，对当前 N 个旧请求 + 1 个新请求的"未来 output token 数"做一次随机采样，
      然后按 Memory Model 模拟 decode，检查是否发生 OOM。
    - 运行 run_times 次试验，得到经验 OOM 概率 p_oom。
    - 若 p_oom <= theta 则 Accept，否则 Reject。

    use_initial_output:
      - True（默认）：使用 P_list 作为旧请求当前已生成的 output token 数；
      - False（--no-init-output）：忽略 P_list，视为旧请求从 0 开始。
    """
    if not output_lengths:
        return True
    P_list = P_list if P_list is not None else []
    sorted_output_lengths = sorted(int(x) for x in output_lengths)

    def one_trial() -> bool:
        """返回本次试验是否发生 OOM。"""
        # 初始已生成 tokens（P_eff）
        P_eff = [int(x) for x in P_list]

        # 对旧请求按条件分布采样
        L_out_old = [
            _sample_output_len_conditional(
                sorted_output_lengths,
                int(P_eff[i]),
                sorted_output_lengths=sorted_output_lengths,
            )
            for i in range(N)
        ]
        L_out_new = _sample_output_len(output_lengths)

        # 剩余需要生成的token数列表
        remaining_old = [max(int(L_out_old[i]) - int(P_eff[i]), 0) for i in range(N)]
        # 剩余需要生成的token数最大值
        K = max([int(L_out_new)] + remaining_old) if (N > 0 or L_out_new > 0) else 0

        peak_tokens = 0
        for k in range(0, K + 1):
            total = 0
            # 旧请求
            for i in range(N):
                if int(P_eff[i]) + k < int(L_out_old[i]):
                    total += int(L_in_old_list[i]) + int(P_eff[i]) + k
            # 新请求
            if k < int(L_out_new):
                total += int(L_in_new) + k
            if total > peak_tokens:
                peak_tokens = total

        mem_peak = m * peak_tokens
        return mem_peak > M

    oom_cnt = 0
    for _ in range(run_times):
        if one_trial():
            oom_cnt += 1
    p_oom = oom_cnt / max(run_times, 1)
    return p_oom <= theta
