"""Build training caches for more tasks (multi-task training), slim by default.

    python tools/build_cache.py --task VideoUnmask --episodes 100 [--full] [--delete-h5]

Per episode: replay the HDF5 demo in the simulator (sg.replay_episode), build the oracle graph and the Q0
graph (visibility-gated oracle + bound rule, exactly as notebook cell "Step 7, cache"), then save
  cache/<task>/epNNNN_replay.npz  slim: only what training reads (sg.TRAIN_KEYS), already cut to the
                                  policy frames (policy_frame = 0..K-1), plus the episode metadata
  cache/<task>/epNNNN_graph.npz   Q0 at those frames
--full keeps the whole replay instead (depth, segmentation, poses of every frame: ~90 MB per episode
on disk vs ~35 MB slim), as the notebook does for VideoUnmaskSwap. No DINOv2 features are stored, so
A3feat cannot train on slim tasks.
--delete-h5 removes the extracted HDF5 (~15-25 GB per task) once all episodes are cached.
Existing episode files are kept, so an interrupted run continues where it stopped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sg_pilot as sg  # noqa: E402


def save_slim(rep, q0: np.ndarray, path: Path) -> None:
    pf = rep.policy_frame.astype(int)
    arrays = {
        "front_rgb": rep.front_rgb[pf],
        "wrist_rgb": rep.wrist_rgb[pf],
        "joint_state": rep.joint_state[pf],
        "gripper_state": rep.gripper_state[pf],
        "policy_frame": np.arange(len(pf)),
        "actions": rep.actions,
    }
    meta = dict(task=rep.task, episode=rep.episode, task_goal=rep.task_goal, scene=rep.scene, status=rep.status)
    np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrays)
    np.savez_compressed(path.with_name(path.name.replace("_replay", "_graph")), Q0=q0[pf])


def main():
    import h5py

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--delete-h5", action="store_true")
    a = ap.parse_args()
    out = sg.CACHE_DIR / a.task
    out.mkdir(parents=True, exist_ok=True)
    h5_path = sg.download_h5(a.task, sg.CACHE_DIR / "h5")
    h5 = h5py.File(h5_path, "r")
    eps = sg.h5_episode_indices(h5)[: a.episodes]
    t0, failed = time.time(), []
    for ep in eps:
        path = out / f"ep{ep:04d}_replay.npz"
        if path.exists() and path.with_name(path.name.replace("_replay", "_graph")).exists():
            continue
        rep = sg.replay_episode(a.task, ep, sg.policy_actions(sg.read_h5_episode(h5, ep)))
        if rep.status != "success":
            failed.append((ep, rep.status))
        og, _ = sg.oracle_graph(rep)
        og[og[..., sg.F_HIDDEN] == 1, sg.F_U : sg.F_Z + 1] = np.nan
        q0, _ = sg.apply_rule(og, "bound", rep.gripper_state.sum(1), rep.ee_xyz, rep.K_front, rep.E_front)
        if a.full:
            sg.save_replay(rep, path)
            np.savez_compressed(path.with_name(path.name.replace("_replay", "_graph")), Q0=q0)
        else:
            save_slim(rep, q0, path)
        n_done = len(list(out.glob("ep*_graph.npz")))
        print(f"{a.task} ep {ep}: {rep.status}, {len(rep.actions)} actions, "
              f"{len(rep.scene['objects'])} objects; {n_done}/{len(eps)} cached, {time.time() - t0:.0f} s", flush=True)
    h5.close()
    size = sum(f.stat().st_size for f in out.glob("ep*.npz")) / 1e9
    print(f"{a.task}: {len(list(out.glob('ep*_graph.npz')))} episodes cached, {size:.2f} GB; "
          f"replays not ending in success: {failed or 'none'}")
    if a.delete_h5:
        h5_path.unlink()
        print(f"deleted {h5_path}")


if __name__ == "__main__":
    main()
