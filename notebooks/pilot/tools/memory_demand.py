"""How much memory does VideoUnmaskSwap demand at t*? Plus the spread of the once-per-episode Q1/Q2 draws.

    python tools/memory_demand.py [--task VideoUnmaskSwap] [--out /workspace/logs/memory_check]

Per cached training episode (cache/<task>/ep*_replay.npz, built by the notebook in pilot mode):
  target      the active target at t* (the first target named; ActiveTarget)
  t_hide      first frame the target is hidden in both cameras (after having been seen)
  t_star      decision frame t* (first policy frame)
  disp_px     pixel distance of the target's true position (full oracle, projected) between t_hide and t*:
              how far the hidden target was carried by the swaps
  frozen_px   frozen rule's target error at t* (keep the last seen position)
  bound_px    bound rule's target error at t* (follow the covering container)
  gap_px      frozen_px - bound_px: what the bound memory buys over a static memory at t*
Rules and errors are computed exactly as notebook cell 6.2 / 6.4 (visibility-gated oracle, apply_rule,
target_error against the full oracle).

Q1/Q2 draws: sg.evaluate seeds each test episode's corruption with its episode index, so test episodes
0..N-1 use draw_corruption(level, seed=ep). The Q1 offset depends only on the seed; which episodes Q2
corrupts is a fixed share (sg.q2_fires); the wrong container also depends on the episode's scene.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sg_pilot as sg  # noqa: E402


MOVED_PX = 10.0  # moved vs back at start; training episodes leave 1-21 px empty (0 vs >= 21 px)


def episode_row(path: Path) -> dict:
    rep = sg.load_replay(path)
    return row_from(rep, sg.decision_frame(rep))


def test_row(task: str, ep: int, cache: Path) -> dict:
    """Test episodes have no demonstration: reset plays the conditioning video, and t* is the last reset
    frame (the first one the policy sees, as for training's policy_frame[0])."""
    path = cache / f"ep{ep:04d}_replay.npz"
    if path.exists():
        rep = sg.load_replay(path)
    else:
        rep = sg.replay_episode(task, ep, np.zeros((0, 8), np.float32), split="test")
        sg.save_replay(rep, path)
    return row_from(rep, len(rep.front_rgb) - 1)


def row_from(rep, t_star: int) -> dict:
    targets = sg.target_cube_nodes(rep.scene, rep.task_goal)
    active = sg.active_targets(rep, targets)
    tg = int(active[t_star])
    og, full_xyz = sg.oracle_graph(rep)
    gated = og.copy()
    gated[gated[..., sg.F_HIDDEN] == 1, sg.F_U : sg.F_Z + 1] = np.nan
    grip_w = rep.gripper_state.sum(1)
    frozen_g, _ = sg.apply_rule(gated, "frozen", grip_w, rep.ee_xyz, rep.K_front, rep.E_front)
    bound_g, _ = sg.apply_rule(gated, "bound", grip_w, rep.ee_xyz, rep.K_front, rep.E_front)
    pxf, _ = sg.target_error(frozen_g, full_xyz, active, rep.K_front, rep.E_front)
    pxb, _ = sg.target_error(bound_g, full_xyz, active, rep.K_front, rep.E_front)
    hidden = og[:, tg, sg.F_HIDDEN] == 1
    seen = np.nonzero(~hidden)[0]
    after = np.nonzero(hidden & (np.arange(len(hidden)) > (seen[0] if len(seen) else -1)))[0]
    t_hide = int(after[0]) if len(after) else -1
    disp = float("nan")
    if t_hide >= 0 and np.all(np.isfinite(full_xyz[[t_hide, t_star], tg])):
        uv0 = sg.project_points(full_xyz[t_hide, tg], rep.K_front, rep.E_front[t_hide])[0]
        uv1 = sg.project_points(full_xyz[t_star, tg], rep.K_front, rep.E_front[t_star])[0]
        disp = float(np.linalg.norm(uv1 - uv0))
    return dict(
        episode=rep.episode,
        n_targets=len(targets),
        n_containers=rep.scene["n_bins"],
        target=tg,
        target_text=rep.scene["objects"][tg]["text"],
        t_hide=t_hide,
        t_star=t_star,
        hidden_at_t_star=bool(hidden[t_star]),
        disp_px=disp,
        frozen_px=float(pxf[t_star]),
        bound_px=float(pxb[t_star]),
        gap_px=float(pxf[t_star] - pxb[t_star]),
        moved=bool(disp >= MOVED_PX),
        status=rep.status,
    )


def quart(x) -> str:
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return "n=0"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"n={len(x)}  median {med:6.1f}  quartiles [{q1:6.1f}, {q3:6.1f}]  max {x.max():6.1f}"


def summary(rows: list[dict], label: str) -> None:
    gap = np.array([r["gap_px"] for r in rows])
    frz = np.array([r["frozen_px"] for r in rows])
    print(f"\n{label}: {len(rows)} episodes")
    print(f"  gap > 10 px (frozen - bound at t*): {int(np.sum(gap > 10))} of {len(rows)}")
    print(f"  frozen error > 10 px at t*:         {int(np.sum(frz > 10))} of {len(rows)}")
    for k in ["disp_px", "frozen_px", "bound_px", "gap_px"]:
        print(f"  {k:10s} {quart([r[k] for r in rows])}")


def draws(n: int) -> None:
    # Q1: rng.normal(0, sigma, 2) per seed; Q2: rng.random() < p per seed (sg.draw_corruption)
    dummy_scene = {"objects": [{"node": 0, "label_id": 0}, {"node": 1, "label_id": 0},
                               {"node": 2, "label_id": 0}, {"node": 3, "label_id": 1}],
                   "cube_to_bin": {3: 0}}
    print(f"\nQ1/Q2 draws for test episodes 0..{n - 1} (seed = episode index, as in sg.evaluate)")
    for lvl in ["Q1_20", "Q1_50"]:
        sigma = float(lvl.split("_")[1])
        r = np.array([np.linalg.norm(sg.draw_corruption(lvl, s, [3], dummy_scene)[0]) for s in range(n)])
        # 2-D isotropic Gaussian offset -> Rayleigh radius: median sigma*sqrt(2 ln 2), mean sigma*sqrt(pi/2)
        print(f"  {lvl}: |offset| {quart(r)} px; expected median {sigma * math.sqrt(2 * math.log(2)):.1f}, "
              f"mean {sigma * math.sqrt(math.pi / 2):.1f}; observed mean {r.mean():.1f}; "
              f"episodes under 5 px: {int(np.sum(r < 5))}")
    r20 = np.array([sg.draw_corruption("Q1_20", s, [3], dummy_scene)[0] for s in range(n)])
    r50 = np.array([sg.draw_corruption("Q1_50", s, [3], dummy_scene)[0] for s in range(n)])
    print(f"  Q1_50 offset == 2.5 x Q1_20 offset for the same episode: {np.allclose(r50, 2.5 * r20)}")
    fired = {}
    for lvl in ["Q2_25", "Q2_50"]:
        f = [sg.draw_corruption(lvl, s, [3], dummy_scene)[1] is not None for s in range(n)]
        fired[lvl] = set(np.nonzero(f)[0])
        print(f"  {lvl}: corrupts {sum(f)} of {n} episodes (fixed share, sg.q2_fires): {sorted(fired[lvl])}")
    print(f"  Q2_25 firing episodes are a subset of Q2_50's: {fired['Q2_25'] <= fired['Q2_50']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VideoUnmaskSwap")
    ap.add_argument("--test-episodes", type=int, default=50)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="test: replay the test episodes' videos (no actions) and write cache/<task>_test/motion.json")
    ap.add_argument("--out", default="/workspace/logs/memory_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "test":
        test_cache = sg.CACHE_DIR / f"{a.task}_test"  # outside cache/<task>/, so training never reads it
        test_cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for ep in range(a.test_episodes):
            rows.append(test_row(a.task, ep, test_cache))
            r = rows[-1]
            print(f"test ep {ep:2d}  targets {r['n_targets']}  {r['target_text']:10s}  t* {r['t_star']:4d}  "
                  f"disp {r['disp_px']:6.1f}  frozen {r['frozen_px']:6.1f}  bound {r['bound_px']:6.1f}  "
                  f"{'moved' if r['moved'] else 'back at start'}", flush=True)
        with open(out / f"{a.task}_test_memory_demand.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        motion = {str(r["episode"]): dict(moved=r["moved"], disp_px=r["disp_px"], n_targets=r["n_targets"])
                  for r in rows}
        (test_cache / "motion.json").write_text(json.dumps(motion, indent=1))
        summary(rows, "test, all")
        summary([r for r in rows if r["moved"]], "test, moved")
        summary([r for r in rows if not r["moved"]], "test, back at start")
        print(f"\nmoved: {sum(r['moved'] for r in rows)} of {len(rows)}; motion labels: {test_cache / 'motion.json'}")
        return
    paths = sorted((sg.CACHE_DIR / a.task).glob("ep*_replay.npz"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    rows = []
    for p in paths:
        rows.append(episode_row(p))
        r = rows[-1]
        print(f"ep {r['episode']:3d}  targets {r['n_targets']}  {r['target_text']:10s}  hide {r['t_hide']:4d}  "
              f"t* {r['t_star']:4d}  disp {r['disp_px']:6.1f}  frozen {r['frozen_px']:6.1f}  "
              f"bound {r['bound_px']:6.1f}  gap {r['gap_px']:6.1f}", flush=True)
    with open(out / f"{a.task}_memory_demand.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows, "all")
    summary([r for r in rows if r["n_targets"] == 1], "one target")
    summary([r for r in rows if r["n_targets"] == 2], "two targets")
    bad = [r["episode"] for r in rows if not r["hidden_at_t_star"]]
    print(f"\nepisodes where the target is not hidden at t*: {bad or 'none'}")
    draws(a.test_episodes)
    print(f"\nper-episode table: {out / f'{a.task}_memory_demand.csv'}")


if __name__ == "__main__":
    main()
