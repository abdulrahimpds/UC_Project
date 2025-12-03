"""
diagnostic script to profile spacenet7 training throughput.

usage:
    python scripts/diagnose_training_speed.py --config configs/SpaceNet7/sanity_check/TSViT.yaml --device 0 --max-steps 5

keep task manager's gpu tab open while this runs to observe utilization.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device
from data import get_dataloaders


def summarize(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: n/a"
    return f"{name}: min={min(values):.4f}s | mean={mean(values):.4f}s | max={max(values):.4f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="profile spacenet7 training speed")
    parser.add_argument("--config", required=True, help="path to yaml config")
    parser.add_argument("--device", default="0", help="comma-separated gpu ids (use -1 for cpu)")
    parser.add_argument("--max-steps", type=int, default=5, help="number of training batches to profile")
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",") if int(d) >= 0]
    allow_cpu = all(d < 0 for d in [int(d) for d in args.device.split(",")])

    device = get_device(device_ids if not allow_cpu else [], allow_cpu=allow_cpu)
    config = read_yaml(args.config)
    config["local_device_ids"] = device_ids if not allow_cpu else []

    print("=== configuration summary ===")
    print(f"config file       : {args.config}")
    print(f"target device     : {device}")
    print(f"max profiled steps: {args.max_steps}")
    print("=============================\n")

    dataloaders = get_dataloaders(config)
    model = get_model(config, device)
    model.eval()

    fetch_times: list[float] = []
    forward_times: list[float] = []
    total_step_times: list[float] = []
    gpu_mem: list[float] = []

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_fetch = time.perf_counter()
    for step, sample in enumerate(dataloaders["train"]):
        fetch_end = time.perf_counter()
        fetch_times.append(fetch_end - start_fetch)

        inputs = sample["inputs"].to(device, non_blocking=True)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        start_forward = time.perf_counter()
        with torch.no_grad():
            _ = model(inputs)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_mem.append(torch.cuda.max_memory_allocated(device) / (1024 ** 3))

        forward_duration = time.perf_counter() - start_forward
        total_duration = time.perf_counter() - fetch_end
        forward_times.append(forward_duration)
        total_step_times.append(total_duration)

        print(
            f"[step {step + 1}] fetch={fetch_times[-1]:.4f}s | forward={forward_duration:.4f}s | "
            f"total={total_duration:.4f}s | gpu_mem={gpu_mem[-1]:.3f} GB" if gpu_mem else ""
        )

        if step + 1 >= args.max_steps:
            break

        start_fetch = time.perf_counter()

    print("\n=== summary ===")
    print(summarize("data fetch", fetch_times))
    print(summarize("forward pass", forward_times))
    print(summarize("step total", total_step_times))
    if gpu_mem:
        print(f"peak gpu memory: {max(gpu_mem):.3f} GB")

    if torch.cuda.is_available() and device.type == "cuda":
        print("gpu utilization tip: keep task manager -> performance -> gpu open while running this script.")


if __name__ == "__main__":
    main()