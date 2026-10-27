#!/usr/bin/env python
"""Closed-loop evaluation of one checkpoint. Thin CLI over sg_pilot.evaluate.

Example:
  python eval.py --ckpt runs/A2_s0 --interface A2 --task VideoUnmaskSwap --quality Q0 --episodes 50 \
      --out results/A2_s0_VideoUnmaskSwap_Q0.json
  python eval.py ... --counterfactual        # graph names the wrong container for the hidden target
The policy family (smolvla, pi05) is read from the checkpoint's train_info.json unless --policy is given.
"""

import argparse
from pathlib import Path

import sg_pilot as sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--interface", required=True, choices=sg.INTERFACES)
    ap.add_argument("--task", required=True)
    ap.add_argument("--quality", default="Q0", choices=sg.QUALITY_LEVELS)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=1300)
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--policy", choices=["smolvla", "pi05"], default=None)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = sg.evaluate(
        Path(a.ckpt),
        a.interface,
        a.task,
        a.quality,
        a.episodes,
        Path(a.out),
        a.max_steps,
        a.counterfactual,
        episode_offset=a.episode_offset,
        policy_name=a.policy,
    )
    lo, hi = s["wilson95"]
    print(
        f"{a.interface} {a.task} {a.quality}: {s['successes']}/{s['episodes']} = {s['rate']:.3f} "
        f"[{lo:.3f}, {hi:.3f}]"
    )


if __name__ == "__main__":
    main()
