#!/usr/bin/env python3
import argparse
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
VLLM_ROOT = ROOT / "3rdparty" / "vllm"
for path in (ROOT, VLLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vllm.distributed.device_communicators.pynccl import (  # noqa: E402
    PyNcclCommunicator,
)


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _worker(rank: int, world_size: int, master_addr: str, master_port: int,
            visible_devices: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    _log(rank, f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    _log(rank, f"device={torch.cuda.get_device_name(rank)}")

    group = None
    try:
        _log(rank, "calling init_process_group(backend='gloo')")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
        _log(rank, "initialized gloo process group")

        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        _log(rank, "created gloo subgroup for PyNcclCommunicator")

        _log(rank, f"constructing PyNcclCommunicator on {device}")
        comm = PyNcclCommunicator(group=group, device=device)
        _log(rank, "constructed PyNcclCommunicator")

        tensor = torch.tensor([rank + 1.0], device=device)
        _log(rank, f"before pynccl all_reduce tensor={tensor.item()}")
        out = comm.all_reduce(tensor)
        torch.cuda.synchronize(device)
        _log(rank, f"after pynccl all_reduce tensor={out.item()}")
    finally:
        if group is not None:
            del group
        if dist.is_initialized():
            dist.destroy_process_group()
            _log(rank, "destroyed process group")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal vLLM PyNcclCommunicator repro."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Comma-separated physical GPU ids to expose to the worker group.",
    )
    args = parser.parse_args()

    if args.world_size < 2:
        raise ValueError("--world-size must be at least 2")

    visible_devices = args.cuda_visible_devices
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    master_port = args.master_port or _pick_port()
    print(
        f"Launching vLLM PyNccl repro with world_size={args.world_size}, "
        f"master={args.master_addr}:{master_port}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices or '<inherit>'}",
        flush=True,
    )

    mp.spawn(
        _worker,
        args=(args.world_size, args.master_addr, master_port, visible_devices),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
