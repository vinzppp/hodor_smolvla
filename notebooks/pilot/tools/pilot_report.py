"""Pilot report: training curves and Q0 success with 95% Wilson intervals, overall and per test split.

    python tools/pilot_report.py [--task VideoUnmaskSwap] [--arms A0,A2] [--out /workspace/logs/pilot]

Reads runs/pilot_<arm>_s0/train_info.json, results/pilot_<arm>_Q0.json and the test motion labels
(cache/<task>_test/motion.json, from tools/memory_demand.py --split test). Writes <out>/pilot_loss.png.
Prints numbers only; the pilot decision is not made here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT))
import sg_pilot as sg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VideoUnmaskSwap")
    ap.add_argument("--arms", default="A0,A2")
    ap.add_argument("--out", default="/workspace/logs/pilot")
    a = ap.parse_args()
    arms = a.arms.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    print("training")
    for arm in arms:
        p = PILOT / "runs" / f"pilot_{arm}_s0" / "train_info.json"
        if not p.exists():
            print(f"  {arm}: not trained yet")
            continue
        info = json.loads(p.read_text())
        loss = np.array(info["loss"])
        w = 200
        smooth = np.convolve(loss, np.ones(w) / w, mode="valid") if len(loss) > w else loss
        ax.plot(loss, lw=0.3, alpha=0.3)
        ax.plot(np.arange(len(smooth)) + (len(loss) - len(smooth)), smooth, lw=1.2, label=f"{arm} ({w}-step mean)")
        tail = loss[-1000:]
        print(f"  {arm}: {info['steps']} steps, batch {info['batch_size']}, {info['seconds'] / 3600:.2f} h, "
              f"{info.get('steps_per_s_steady') or info['steps_per_s']:.2f} steps/s, peak GPU {info['max_mem_GB']:.1f} GB; "
              f"loss first 100 steps {loss[:100].mean():.3f}, last 1000 steps {tail.mean():.3f} (sd {tail.std():.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow-matching loss")
    ax.set_yscale("log")
    ax.set_title(f"pilot training, {a.task}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pilot_loss.png", dpi=130)
    print(f"  loss figure: {out / 'pilot_loss.png'}")

    motion = sg.load_motion(a.task)
    print(f"\nQ0 success, 95% Wilson interval (test motion labels: {'yes' if motion else 'missing'})")
    for arm in arms:
        p = PILOT / "results" / f"pilot_{arm}_Q0.json"
        if not p.exists():
            print(f"  {arm}: not evaluated yet")
            continue
        summ = json.loads(p.read_text())
        for name, r in sg.success_by_split(summ, motion).items():
            lo, hi = r["wilson95"]
            print(f"  {arm:3s} {name:14s} {r['k']:2d}/{r['n']:2d} = {r['rate']:.3f}  [{lo:.3f}, {hi:.3f}]")
        steps = [r["steps"] for r in summ["results"]]
        print(f"      steps per episode median {np.median(steps):.0f} (successes {np.median([r['steps'] for r in summ['results'] if r['success']] or [np.nan]):.0f}); "
              f"{summ['seconds_per_episode']:.0f} s per episode")


if __name__ == "__main__":
    main()
