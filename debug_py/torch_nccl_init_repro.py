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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    _log(rank, "setting CUDA device")
    torch.cuda.set_device(rank)
    _log(rank, f"device={torch.cuda.get_device_name(rank)}")

    try:
        _log(rank, "calling init_process_group(backend='nccl')")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
        _log(rank, "initialized process group")

        tensor = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        _log(rank, f"before all_reduce tensor={tensor.item()}")
        dist.all_reduce(tensor)
        _log(rank, f"after all_reduce tensor={tensor.item()}")

        dist.barrier()
        _log(rank, "barrier completed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            _log(rank, "destroyed process group")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal 2-rank torch.distributed NCCL init/all-reduce repro."
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
        f"Launching torch NCCL repro with world_size={args.world_size}, "
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
