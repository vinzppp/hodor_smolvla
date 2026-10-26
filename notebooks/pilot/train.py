#!/usr/bin/env python
"""Train one arm. Thin CLI over sg_pilot.train (the notebook calls the same function).

Example:
  python train.py --interface A2 --tasks VideoUnmaskSwap --seed 0 --steps 20000 --out runs/A2_s0
  python train.py --interface A3feat --tasks VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap \
      --seed 1 --steps 20000 --label-mode table --out runs/A3feat_all_s1
Requires the replay/graph cache built by the notebook (Steps 3-7) for every listed task.
"""

import argparse
import json
from pathlib import Path

import sg_pilot as sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", required=True, choices=sg.INTERFACES)
    ap.add_argument("--tasks", required=True, help="comma-separated RoboMME task names")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-mode", choices=["table", "text"], default="table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = sg.train(
        a.interface,
        [t.strip() for t in a.tasks.split(",") if t.strip()],
        a.seed,
        a.steps,
        Path(a.out),
        a.episodes_per_task,
        a.batch_size,
        a.lr,
        a.label_mode,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "loss"}, indent=1))


if __name__ == "__main__":
    main()
