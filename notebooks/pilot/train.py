#!/usr/bin/env python
"""Train one arm. Thin CLI over sg_pilot.train (the notebooks call the same function).

Example:
  python train.py --interface A2 --tasks VideoUnmaskSwap --seed 0 --steps 20000 --out runs/A2_s0
  python train.py --policy pi05 --interface A3 --tasks permanence --seed 1 --steps 80000 \
      --batch-size 32 --accum 2 --out runs/pi05_A3_s1
--tasks permanence = VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap (one multi-task model).
Requires the replay/graph cache for every listed task (notebook Steps 3-7, or tools/build_cache.py).
"""

import argparse
import json
from pathlib import Path

import sg_pilot as sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["smolvla", "pi05"], default="smolvla")
    ap.add_argument("--interface", required=True, choices=sg.INTERFACES)
    ap.add_argument("--tasks", required=True, help="comma-separated RoboMME task names, or 'permanence'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (batch x accum)")
    ap.add_argument("--lr", type=float, default=None, help="default: the policy's recipe")
    ap.add_argument("--label-mode", choices=["table", "text"], default="table")
    ap.add_argument("--siglip-cache", action="store_true", help="pi05: cache SigLIP tokens on disk")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = sg.PERMANENCE if a.tasks == "permanence" else a.tasks
    info = sg.train(
        a.interface,
        [t.strip() for t in tasks.split(",") if t.strip()],
        a.seed,
        a.steps,
        Path(a.out),
        a.episodes_per_task,
        a.batch_size,
        a.lr,
        a.label_mode,
        policy_name=a.policy,
        accum_steps=a.accum,
        siglip_cache=a.siglip_cache,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "loss"}, indent=1))


if __name__ == "__main__":
    main()
