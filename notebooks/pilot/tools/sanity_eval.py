"""Pipeline sanity checks for the closed-loop evaluation (changes no pipeline code).

    python tools/sanity_eval.py replay --episodes 0,1,2,3,4      # check 1
    python tools/sanity_eval.py policies --episodes 0,1,2,3,4    # check 2

replay:   recorded demo actions of training episodes, sent through evaluate's env and action path:
          the same env (sg.make_env, max_steps 1300), and each action normalised with the policy's own
          action statistics and passed back through the policy postprocessor (post), so env.step gets
          exactly what a policy that predicted the demo would produce. They should succeed.
policies: sg.evaluate for runs/pilot_<arm>_s0 at Q0 on training episodes, unchanged except that
          sg.make_env is redirected from the test split to the train split.
val:      sg.evaluate on the val split (make_env redirected) at Q0, with the policy's n_action_steps set
          right after sg.build_policy (actions executed per predicted chunk of 50; the queue length is read
          from config.n_action_steps at every policy.reset()).
Outputs: /workspace/logs/sanity/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT))
import sg_pilot as sg  # noqa: E402

OUT = Path("/workspace/logs/sanity")
TASK = "VideoUnmaskSwap"


def replay(episodes: list[int]) -> None:
    import h5py
    import torch
    from lerobot.processor.rename_processor import rename_stats

    policy, flow, pre, post, cfg, meta = sg.build_policy("cuda")
    st = rename_stats(meta.stats, sg.RENAME_MAP)["action"]  # what the normalizer/unnormalizer use
    mean, std = np.asarray(st["mean"], np.float32), np.asarray(st["std"], np.float32)
    print(f"action normalisation ({cfg.normalization_mapping['ACTION']}): mean {mean.round(3)} std {std.round(3)}")
    h5 = h5py.File(sg.CACHE_DIR / "h5" / f"record_dataset_{TASK}.h5", "r")
    rows = []
    for ep in episodes:
        acts = sg.policy_actions(sg.read_h5_episode(h5, ep))
        env = sg.make_env(TASK, "train", ep, max_steps=1300)  # evaluate's env, train split
        obs, info = env.reset()
        status, steps, maxdiff = "ongoing", 0, 0.0
        for a_rec in acts:
            norm = torch.from_numpy((a_rec - mean) / std)[None].to("cuda")
            with torch.no_grad():
                act = post(norm)  # evaluate: act = post(policy.select_action(...))
            a = sg._np(act).reshape(-1)[:8]  # evaluate: a = _np(act).reshape(-1)[:8]
            maxdiff = max(maxdiff, float(np.abs(a - a_rec).max()))
            obs, _, term, trunc, info = env.step(a)
            steps += 1
            if bool(sg._np(term).reshape(-1)[0]) or bool(sg._np(trunc).reshape(-1)[0]):
                status = info.get("status", "unknown")
                break
        env.close()
        rows.append(dict(episode=ep, status=status, steps=steps, recorded_actions=len(acts),
                         max_abs_diff_after_post=maxdiff, dtype=str(a.dtype)))
        print(f"replay ep {ep}: {status} after {steps} of {len(acts)} recorded actions; "
              f"max |post(normalise(a)) - a| = {maxdiff:.2e}; action dtype {a.dtype}", flush=True)
    (OUT / "replay.json").write_text(json.dumps(rows, indent=1))


def policies(episodes: list[int], arms: list[str]) -> None:
    orig = sg.make_env
    sg.make_env = lambda task, split, ep, max_steps=None: orig(task, "train", ep, max_steps)
    for arm in arms:
        for ep in episodes:  # one call per episode: evaluate numbers episodes from episode_offset
            s = sg.evaluate(PILOT / "runs" / f"pilot_{arm}_s0", arm, TASK, "Q0", 1,
                            OUT / f"{arm}_train_ep{ep}.json", episode_offset=ep)
            r = s["results"][0]
            print(f"{arm} train ep {ep}: {r['status']} in {r['steps']} steps; lifted {r['lifted_bin']} "
                  f"(true {r['true_bin']}); targets {r['targets']}", flush=True)


def val(episodes: list[int], arms: list[str], n_steps: list[int]) -> None:
    orig_env, orig_build = sg.make_env, sg.build_policy
    sg.make_env = lambda task, split, ep, max_steps=None: orig_env(task, "val", ep, max_steps)
    rows = []
    for n in n_steps:
        def build(*a, _n=n, **k):
            policy, flow, pre, post, cfg, meta = orig_build(*a, **k)
            cfg.n_action_steps = _n
            policy.get_base_model().config.n_action_steps = _n
            print(f"n_action_steps = {policy.get_base_model().config.n_action_steps}", flush=True)
            return policy, flow, pre, post, cfg, meta

        sg.build_policy = build
        for arm in arms:
            s = sg.evaluate(PILOT / "runs" / f"pilot_{arm}_s0", arm, TASK, "Q0", len(episodes),
                            OUT / f"val_{arm}_n{n}.json", episode_offset=episodes[0])
            rs = s["results"]
            k = sum(r["success"] for r in rs)
            lifts = [r for r in rs if r["lifted_bin"] is not None]
            right = sum(r["lifted_bin"] == r["true_bin"] for r in lifts)
            lo, hi = sg.wilson(k, len(rs))
            rows.append(dict(arm=arm, n_action_steps=n, k=k, n=len(rs), wilson95=[lo, hi], lifted=len(lifts),
                             first_lift_right=right, timeouts=sum(r["status"] == "ongoing" for r in rs),
                             seconds_per_episode=s["seconds_per_episode"]))
            print(f"VAL {arm} n_action_steps={n}: {k}/{len(rs)} [{lo:.3f}, {hi:.3f}]; lifted {len(lifts)}, "
                  f"first lift right {right}; timeouts {rows[-1]['timeouts']}; {s['seconds_per_episode']:.0f} s/episode",
                  flush=True)
    sg.build_policy = orig_build
    (OUT / "val_summary.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["replay", "policies", "val"])
    ap.add_argument("--n-action-steps", default="16,50")
    ap.add_argument("--episodes", default="0,1,2,3,4")
    ap.add_argument("--arms", default="A0,A2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [int(e) for e in a.episodes.split(",")]
    if a.check == "replay":
        replay(eps)
    elif a.check == "policies":
        policies(eps, a.arms.split(","))
    else:
        val(eps, a.arms.split(","), [int(n) for n in a.n_action_steps.split(",")])


if __name__ == "__main__":
    main()
