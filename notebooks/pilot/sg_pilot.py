"""sg_pilot: scene-graph interfaces for SmolVLA on RoboMME (pilot).

Every function here is shared by the notebook (sg_interfaces_pilot.ipynb), train.py and eval.py,
so the notebook and the full grid run the same code.

Sources read while writing this module (pinned clones, 2026-10-26):
  RoboMME  github.com/RoboMME/robomme_benchmark @ 53673b3
    doc/env_format.md, doc/h5_data_format.md, scripts/dataset_replay.py,
    src/robomme/env_record_wrapper/{episode_config_resolver.py, DemonstrationWrapper.py},
    src/robomme/robomme_env/VideoUnmaskSwap.py
  LeRobot  github.com/huggingface/lerobot @ 5831498
    src/lerobot/envs/robomme.py, src/lerobot/policies/smolvla/*.py,
    src/lerobot/policies/factory.py, src/lerobot/scripts/lerobot_train.py, docs/source/robomme.mdx
  This repo: post_process_scene_cutie.py, extract_part_masks_cutie.py, dinov2_encoder.py, configs/*.yaml
Names marked "VERIFY" are checked against the installed packages in the notebook before use.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # scene-graph-robot/
PILOT_DIR = Path(__file__).resolve().parent  # notebooks/pilot/
FIG_DIR = PILOT_DIR / "figures"
CACHE_DIR = PILOT_DIR / "cache"
RUNS_DIR = PILOT_DIR / "runs"
RESULTS_DIR = PILOT_DIR / "results"

PERMANENCE_TASKS = ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"]
H5_REPO = "Yinpei/robomme_data_h5"  # raw HDF5 demos (readme.md, "Training data")
LEROBOT_REPO = (
    "lerobot/robomme"  # LeRobot v3.0 copy of the same demos (docs/source/robomme.mdx)
)
SMOLVLA_BASE = "lerobot/smolvla_base"

# Node layout used everywhere: [T, N, F]
F_ID, F_LABEL, F_U, F_V, F_X, F_Y, F_Z, F_HIDDEN = range(8)
NODE_FIELDS = ["id", "label_id", "u", "v", "x", "y", "z", "hidden"]
LABELS = {0: "container", 1: "cube"}

# --------------------------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------------------------
RUN_MODES = {
    # task, training episodes, training steps, evaluation episodes
    "smoke": dict(
        task="VideoUnmaskSwap",
        train_episodes=2,
        steps=10,
        eval_episodes=1,
        counterfactual_episodes=1,
    ),
    "pilot": dict(
        task="VideoUnmaskSwap",
        train_episodes=100,
        steps=None,
        eval_episodes=50,
        counterfactual_episodes=20,
    ),
}


def set_seed(seed: int = 0) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pass_fail(name: str, ok: bool, value: Any) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value}")


def source_line(obj) -> str:
    """'file:line' of a function or class, for the API-verification prints."""
    import inspect

    try:
        return f"{inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}"
    except (TypeError, OSError):
        return f"<no source for {obj!r}>"


# --------------------------------------------------------------------------------------------
# Step 1: environment
# --------------------------------------------------------------------------------------------
def versions() -> dict:
    import importlib

    out = {"python": sys.version.split()[0]}
    for mod in ["torch", "lerobot", "mani_skill", "robomme", "transformers", "peft", "cv2"]:
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "installed (no __version__)")
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"MISSING ({type(exc).__name__})"
    import torch

    out["cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out["gpu_mem_GB"] = (
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available()
        else None
    )
    return out


def cuda_forward_backward() -> float:
    """Tiny forward/backward on the GPU; returns the loss value."""
    import torch

    x = torch.randn(64, 32, device="cuda", requires_grad=True)
    w = torch.randn(32, 8, device="cuda", requires_grad=True)
    loss = (x @ w).pow(2).mean()
    loss.backward()
    return float(loss.item())


# --------------------------------------------------------------------------------------------
# Step 2-3: raw demos (HDF5) and simulator replay
# --------------------------------------------------------------------------------------------
def download_h5(task: str, local_dir: Path) -> Path:
    """Download record_dataset_<task>.h5 from the RoboMME HF dataset.

    The repo stores each task as record_dataset_<task>.h5.tar.xz holding the single .h5 file
    (verified 2026-10-26; Yinpei/robomme_data_h5 README.md and tarxz_h5.py). The archive is
    extracted next to itself and deleted; an already extracted .h5 is reused.
    """
    import tarfile

    from huggingface_hub import hf_hub_download

    name = f"record_dataset_{task}.h5"
    h5_path = Path(local_dir) / name
    if h5_path.exists():
        return h5_path
    archive = Path(
        hf_hub_download(
            H5_REPO, name + ".tar.xz", repo_type="dataset", local_dir=str(local_dir)
        )
    )
    with tarfile.open(archive, "r:xz") as tf:
        if tf.getnames() != [name]:
            raise ValueError(f"unexpected members in {archive.name}: {tf.getnames()}")
        tf.extractall(local_dir, filter="data")
    archive.unlink()
    return h5_path


def h5_episode_indices(h5file) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in h5file.keys() if k.startswith("episode_"))


def _decode(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return str(raw)


def h5_task_goals(h5file) -> dict[int, str]:
    """Instruction of every episode, read from episode_<i>/setup/task_goal only (no frames)."""
    return {
        ep: _decode(np.asarray(h5file[f"episode_{ep}"]["setup"]["task_goal"][()]).reshape(-1)[0])
        for ep in h5_episode_indices(h5file)
    }


def read_h5_episode(h5file, ep: int) -> dict:
    """All timesteps of one episode as arrays (doc/h5_data_format.md)."""
    g = h5file[f"episode_{ep}"]
    keys = sorted(
        (k for k in g.keys() if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1])
    )
    out: dict[str, list] = {
        k: []
        for k in [
            "front_rgb",
            "wrist_rgb",
            "front_depth",
            "joint_state",
            "gripper_state",
            "is_gripper_close",
            "eef_state",
            "front_camera_extrinsic",
            "joint_action",
            "is_video_demo",
            "grounded_subgoal",
            "simple_subgoal",
        ]
    }
    for k in keys:
        ts = g[k]
        obs, act, info = ts["obs"], ts.get("action"), ts.get("info")
        for name in [
            "front_rgb",
            "wrist_rgb",
            "front_depth",
            "joint_state",
            "gripper_state",
            "is_gripper_close",
            "eef_state",
            "front_camera_extrinsic",
        ]:
            out[name].append(np.asarray(obs[name][()]) if name in obs else None)
        out["joint_action"].append(
            np.asarray(act["joint_action"][()], np.float32)
            if act is not None and "joint_action" in act
            else None
        )
        out["is_video_demo"].append(
            bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0])
            if info is not None and "is_video_demo" in info
            else False
        )
        out["grounded_subgoal"].append(
            _decode(info["grounded_subgoal"][()])
            if info is not None and "grounded_subgoal" in info
            else ""
        )
        out["simple_subgoal"].append(
            _decode(info["simple_subgoal"][()])
            if info is not None and "simple_subgoal" in info
            else ""
        )
    setup = g["setup"]
    out["task_goal"] = [_decode(x) for x in np.asarray(setup["task_goal"][()]).reshape(-1)]
    out["seed"] = int(np.asarray(setup["seed"][()]).reshape(-1)[0])
    out["front_camera_intrinsic"] = np.asarray(setup["front_camera_intrinsic"][()], np.float32)
    out["wrist_camera_intrinsic"] = np.asarray(setup["wrist_camera_intrinsic"][()], np.float32)
    return out


def policy_actions(h5ep: dict) -> np.ndarray:
    """joint_action of non-video timesteps, in order (as scripts/dataset_replay.py builds them)."""
    acts = [
        a
        for a, v in zip(h5ep["joint_action"], h5ep["is_video_demo"])
        if (not v) and a is not None
    ]
    return np.stack(acts).astype(np.float32)


def subgoal_ranges(subgoals: list[str]) -> list[tuple[int, int, str]]:
    out, start = [], 0
    for i in range(1, len(subgoals) + 1):
        if i == len(subgoals) or subgoals[i] != subgoals[start]:
            out.append((start, i - 1, subgoals[start]))
            start = i
    return out


def make_env(task: str, split: str, episode_idx: int, max_steps: int | None = None):
    """RoboMME env with every privileged field switched on (doc/env_format.md, include_* flags)."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id=task, dataset=split, action_space="joint_angle", gui_render=False
    )
    return builder.make_env_for_episode(
        episode_idx,
        max_steps=max_steps,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )


def _np(x, dtype=None):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


def _seg_from_maniskill(mobs: dict, cam: str) -> np.ndarray:
    """[H, W] int segmentation id map. VERIFY key path: obs['sensor_data'][cam]['segmentation']."""
    seg = _np(mobs["sensor_data"][cam]["segmentation"])
    while seg.ndim > 2:
        seg = seg[0] if seg.shape[0] == 1 else seg[..., 0]
    return seg.astype(np.int32)


def scene_objects(env) -> dict:
    """Task objects from the unwrapped VideoUnmaskSwap-style env (robomme_env/VideoUnmaskSwap.py _load_scene)."""
    base = env.unwrapped
    bins = list(getattr(base, "spawned_bins", []))
    cubes = list(getattr(base, "spawned_dynamic_cubes", []))
    pairs = list(getattr(base, "cube_bin_pairs", []))
    colors = list(getattr(base, "color_names", []))
    objs = []
    for i, b in enumerate(bins):
        objs.append(
            dict(
                node=len(objs),
                name=getattr(b, "name", f"bin_{i}"),
                label_id=0,
                text="container",
                actor=b,
            )
        )
    for i, c in enumerate(cubes):
        color = colors[i] if i < len(colors) else f"cube{i}"
        objs.append(
            dict(
                node=len(objs),
                name=getattr(c, "name", f"cube_{i}"),
                label_id=1,
                text=f"{color} cube",
                color=color,
                actor=c,
            )
        )
    actor_to_node = {id(o["actor"]): o["node"] for o in objs}
    pair_nodes = {
        actor_to_node[id(c)]: actor_to_node[id(b)]
        for c, b in pairs
        if id(c) in actor_to_node and id(b) in actor_to_node
    }
    # segmentation ids per node (ManiSkill BaseEnv.segmentation_id_map: id -> Actor/Link)
    id_map = getattr(base, "segmentation_id_map", {}) or {}
    seg_ids = {o["node"]: [k for k, v in id_map.items() if v is o["actor"]] for o in objs}
    return dict(
        objects=objs,
        cube_to_bin=pair_nodes,
        seg_ids=seg_ids,
        id_map=id_map,
        n_bins=len(bins),
        color_names=colors,
    )


def actor_xyz(actor) -> np.ndarray:
    p = actor.pose.p
    return _np(p, np.float32).reshape(-1)[:3]


@dataclass
class Replay:
    task: str
    episode: int
    front_rgb: np.ndarray  # [T, H, W, 3] uint8
    wrist_rgb: np.ndarray  # [T, H, W, 3] uint8
    front_depth_m: np.ndarray  # [T, H, W] float32 metres
    wrist_depth_m: np.ndarray  # [T, H, W] float32 metres
    front_seg: np.ndarray  # [T, H, W] int32 segmentation ids
    wrist_seg: np.ndarray  # [T, H, W] int32
    K_front: np.ndarray  # [3, 3]
    K_wrist: np.ndarray  # [3, 3]
    E_front: np.ndarray  # [T, 3, 4] world -> camera (OpenCV), extrinsic_cv
    E_wrist: np.ndarray  # [T, 3, 4]
    joint_state: np.ndarray  # [T, 7]
    gripper_state: np.ndarray  # [T, 2] finger widths
    ee_xyz: np.ndarray  # [T, 3]
    is_video: np.ndarray  # [T] bool, True for the conditioning video shown before acting
    policy_frame: np.ndarray  # [K] frame index observed before policy action k
    actions: np.ndarray  # [K, 8] joint_action used for the replay
    task_goal: str
    scene: dict  # from scene_objects(), minus live actors
    status: str
    # [T, N, 3] actor pose positions (world, m) per node; NaN for the video frames before the last
    # reset frame (reset plays the video internally, so only its final state is observable)
    obj_xyz: np.ndarray | None = None


def _collect(obs: dict, store: dict) -> int:
    n = len(obs["front_rgb_list"])
    for i in range(n):
        store["front_rgb"].append(_np(obs["front_rgb_list"][i], np.uint8))
        store["wrist_rgb"].append(_np(obs["wrist_rgb_list"][i], np.uint8))
        store["front_depth"].append(
            _np(obs["front_depth_list"][i], np.float32).squeeze() / 1000.0
        )  # mm -> m
        store["wrist_depth"].append(
            _np(obs["wrist_depth_list"][i], np.float32).squeeze() / 1000.0
        )
        store["E_front"].append(
            _np(obs["front_camera_extrinsic_list"][i], np.float32).reshape(3, 4)
        )
        store["E_wrist"].append(
            _np(obs["wrist_camera_extrinsic_list"][i], np.float32).reshape(3, 4)
        )
        store["joint_state"].append(
            _np(obs["joint_state_list"][i], np.float32).reshape(-1)[:7]
        )
        store["gripper_state"].append(
            _np(obs["gripper_state_list"][i], np.float32).reshape(-1)[:2]
        )
        store["ee_xyz"].append(_np(obs["eef_state_list"][i], np.float32).reshape(-1)[:3])
        mobs = obs["maniskill_obs"][i]
        store["front_seg"].append(_seg_from_maniskill(mobs, "base_camera"))
        store["wrist_seg"].append(_seg_from_maniskill(mobs, "hand_camera"))
    return n


def replay_episode(
    task: str,
    episode: int,
    actions: np.ndarray,
    split: str = "train",
    max_steps: int | None = None,
) -> Replay:
    """Reset (which plays the conditioning video) and step the recorded actions, keeping every frame."""
    env = make_env(task, split, episode, max_steps=max_steps)
    keys = [
        "front_rgb",
        "wrist_rgb",
        "front_depth",
        "wrist_depth",
        "E_front",
        "E_wrist",
        "joint_state",
        "gripper_state",
        "ee_xyz",
        "front_seg",
        "wrist_seg",
    ]
    store: dict[str, list] = {k: [] for k in keys}
    obs, info = env.reset()
    n0 = _collect(obs, store)
    is_video = [True] * (n0 - 1) + [
        False
    ]  # dataset_replay.py marks all reset frames but the last as video
    K_front = _np(info["front_camera_intrinsic"], np.float32).reshape(3, 3)
    K_wrist = _np(info["wrist_camera_intrinsic"], np.float32).reshape(3, 3)
    goal = info.get("task_goal", "")
    goal = goal[0] if isinstance(goal, (list, tuple)) and goal else str(goal)
    scene = scene_objects(env)
    actors = [o["actor"] for o in scene["objects"]]

    def poses(n_frames: int) -> list[np.ndarray]:
        # the env returns n_frames images per call but exposes only the final pose
        p = np.stack([actor_xyz(a) for a in actors])
        return [np.full_like(p, np.nan)] * (n_frames - 1) + [p]

    obj_xyz = poses(n0)
    policy_frame, status = [], "ongoing"
    for a in actions:
        policy_frame.append(len(store["front_rgb"]) - 1)
        obs, _, term, trunc, info = env.step(a)
        n = _collect(obs, store)
        is_video += [False] * n
        obj_xyz += poses(n)
        if bool(_np(term).reshape(-1)[0]) or bool(_np(trunc).reshape(-1)[0]):
            status = info.get("status", "unknown")
            break
    env.close()
    scene_static = {k: v for k, v in scene.items() if k not in ("id_map",)}
    scene_static["objects"] = [
        {k: v for k, v in o.items() if k != "actor"} for o in scene["objects"]
    ]
    stack = {k: np.stack(v) for k, v in store.items()}
    return Replay(
        task=task,
        episode=episode,
        front_rgb=stack["front_rgb"],
        wrist_rgb=stack["wrist_rgb"],
        front_depth_m=stack["front_depth"],
        wrist_depth_m=stack["wrist_depth"],
        front_seg=stack["front_seg"],
        wrist_seg=stack["wrist_seg"],
        K_front=K_front,
        K_wrist=K_wrist,
        E_front=stack["E_front"],
        E_wrist=stack["E_wrist"],
        joint_state=stack["joint_state"],
        gripper_state=stack["gripper_state"],
        ee_xyz=stack["ee_xyz"],
        is_video=np.array(is_video),
        policy_frame=np.array(policy_frame),
        actions=actions[: len(policy_frame)],
        task_goal=goal,
        scene=scene_static,
        status=status,
        obj_xyz=np.stack(obj_xyz).astype(np.float32),
    )


def save_replay(rep: Replay, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        k: getattr(rep, k)
        for k in [
            "front_rgb",
            "wrist_rgb",
            "front_depth_m",
            "wrist_depth_m",
            "front_seg",
            "wrist_seg",
            "K_front",
            "K_wrist",
            "E_front",
            "E_wrist",
            "joint_state",
            "gripper_state",
            "ee_xyz",
            "is_video",
            "policy_frame",
            "actions",
            "obj_xyz",
        ]
    }
    meta = dict(
        task=rep.task,
        episode=rep.episode,
        task_goal=rep.task_goal,
        scene=rep.scene,
        status=rep.status,
    )
    np.savez_compressed(path, meta=json.dumps(meta, default=str), **arrays)


def load_replay(path: Path) -> Replay:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    meta["scene"]["cube_to_bin"] = {
        int(k): int(v) for k, v in meta["scene"]["cube_to_bin"].items()
    }
    meta["scene"]["seg_ids"] = {int(k): v for k, v in meta["scene"]["seg_ids"].items()}
    return Replay(**{k: z[k] for k in z.files if k != "meta"}, **meta)


PICK_Z = 0.15  # metres: RoboMME counts a container as picked up when its pose z exceeds this
# (robomme_env/utils/subgoal_evaluate_func.py is_bin_pickup)


def target_cube_nodes(scene: dict, task_goal: str) -> list[int]:
    """The cubes named in the instruction, in the order they must be picked.

    Some episodes ask for two: '... hiding the blue cube, finally pick up another container hiding
    the red cube'. The order of mention is the env's subgoal order (VideoUnmaskSwap._load_scene).
    """
    found = []
    for o in scene["objects"]:
        if o["label_id"] == 1 and o.get("color"):
            i = task_goal.find(f"{o['color']} cube")
            if i >= 0:
                found.append((i, o["node"]))
    if not found:
        raise ValueError(
            f"no cube colour of {[o.get('color') for o in scene['objects']]} found in {task_goal!r}"
        )
    return [n for _, n in sorted(found)]


class ActiveTarget:
    """The target currently asked for: the first until its container is picked up, then the next.

    A pick counts once the target's container (the cube rides with it) has pose z > PICK_Z, the
    test RoboMME itself uses to advance the subgoal. Latched: a container put back down stays picked.
    """

    def __init__(self, targets: list[int], cube_to_bin: dict):
        self.targets, self.bins = list(targets), [cube_to_bin[c] for c in targets]
        self.k = 0

    def update(self, obj_xyz: np.ndarray) -> int:
        """obj_xyz [N, 3] pose positions this frame (NaN rows ignored) -> active target node."""
        while self.k < len(self.targets) - 1 and obj_xyz[self.bins[self.k], 2] > PICK_Z:
            self.k += 1
        return self.targets[self.k]


def active_targets(rep: Replay, targets: list[int]) -> np.ndarray:
    """[T] active target node per replayed frame (ActiveTarget over the recorded poses)."""
    at = ActiveTarget(targets, rep.scene["cube_to_bin"])
    return np.array([at.update(rep.obj_xyz[t]) for t in range(len(rep.obj_xyz))])


def decision_frame(rep: Replay) -> int:
    """First frame the policy acts on: the remembered fact (which container) is needed here."""
    return int(rep.policy_frame[0])


# --------------------------------------------------------------------------------------------
# Step 4: geometry and the oracle graph
# --------------------------------------------------------------------------------------------
def cam_to_world(E_world_to_cam: np.ndarray) -> np.ndarray:
    """Invert a 3x4 OpenCV extrinsic (world -> camera) into a 4x4 camera -> world matrix."""
    R, t = E_world_to_cam[:3, :3], E_world_to_cam[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def project_points(xyz: np.ndarray, K: np.ndarray, E_world_to_cam: np.ndarray) -> np.ndarray:
    """World points [N, 3] -> pixel (u, v) [N, 2] with the pinhole model (u = column, v = row)."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    cam = (E_world_to_cam[:3, :3] @ xyz.T + E_world_to_cam[:3, 3:4]).T
    z = np.where(np.abs(cam[:, 2:3]) < 1e-9, 1e-9, cam[:, 2:3])
    uv = (K @ (cam / z).T).T[:, :2]
    return uv


def backproject_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    E_world_to_cam: np.ndarray,
    zmin: float = 0.05,
    zmax: float = 3.0,
) -> np.ndarray | None:
    """Mean world point of the masked pixels with valid depth; None if no valid pixel."""
    v, u = np.nonzero(mask & (depth_m > zmin) & (depth_m < zmax))
    if len(u) == 0:
        return None
    z = depth_m[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], 1)
    world = (cam_to_world(E_world_to_cam) @ pts.T).T[:, :3]
    return world.mean(0)


def oracle_graph(rep: Replay, min_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """oracle_graph [T, N, 8] from simulator segmentation + depth.

    Visible in the front camera: (u, v) = mask pixel centroid, xyz = back-projected mask mean.
    Visible only in the wrist camera: xyz from the wrist mask, (u, v) = its front projection.
    Hidden in both: hidden = 1 and xyz/u/v = NaN (the update rules in Step 6 fill these).
    Also returns full_xyz [T, N, 3]: hidden cubes take their paired container's xyz, because the
    simulator moves each cube with its container during the swap (VideoUnmaskSwap.step,
    lift_and_drop_objectA_onto_objectB).
    """
    objs = rep.scene["objects"]
    T, N = len(rep.front_rgb), len(objs)
    g = np.full((T, N, 8), np.nan, np.float32)
    for o in objs:
        g[:, o["node"], F_ID] = o["node"]
        g[:, o["node"], F_LABEL] = o["label_id"]
    for t in range(T):
        for o in objs:
            n, ids = o["node"], rep.scene["seg_ids"][o["node"]]
            mf = np.isin(rep.front_seg[t], ids)
            mw = np.isin(rep.wrist_seg[t], ids)
            if mf.sum() >= min_px:
                v, u = np.nonzero(mf)
                g[t, n, F_U], g[t, n, F_V] = u.mean(), v.mean()
                xyz = backproject_mask(rep.front_depth_m[t], mf, rep.K_front, rep.E_front[t])
                if xyz is not None:
                    g[t, n, F_X : F_Z + 1] = xyz
                g[t, n, F_HIDDEN] = 0
            elif mw.sum() >= min_px:
                xyz = backproject_mask(rep.wrist_depth_m[t], mw, rep.K_wrist, rep.E_wrist[t])
                if xyz is not None:
                    g[t, n, F_X : F_Z + 1] = xyz
                    g[t, n, F_U : F_V + 1] = project_points(xyz, rep.K_front, rep.E_front[t])[
                        0
                    ]
                g[t, n, F_HIDDEN] = 0
            else:
                g[t, n, F_HIDDEN] = 1
    full_xyz = g[..., F_X : F_Z + 1].copy()
    for cube, bin_ in rep.scene["cube_to_bin"].items():
        hid = g[:, cube, F_HIDDEN] == 1
        full_xyz[hid, cube] = g[hid, bin_, F_X : F_Z + 1]
    return g, full_xyz


# --------------------------------------------------------------------------------------------
# Step 6: hidden-object update rules (causal; the same object serves offline graphs and online eval)
# --------------------------------------------------------------------------------------------
@dataclass
class RuleParams:
    r_bind: float = 0.06  # metres: hidden object's last xy must lie within r of the cover's xy
    g_closed: float = 0.03  # gripper total width (m) below which the gripper counts as closed
    d_grip: float = 0.05  # metres: end-effector within d of the object for the gripper rule


class GraphTracker:
    """Fills hidden nodes frame by frame, using only frames <= t.

    rule = 'frozen': keep the last seen position.
    rule = 'bound' : at the frame an object disappears, bind it to the nearest other node visible at
                     that frame whose xy lies within r_bind (tie-break: nearest), or to the end effector
                     if the gripper is closed within d_grip; afterwards follow the partner with the offset
                     measured at binding time. No partner -> frozen.
    """

    def __init__(self, n_nodes: int, rule: str, params: RuleParams | None = None):
        assert rule in ("frozen", "bound")
        self.rule, self.p, self.n = rule, params or RuleParams(), n_nodes
        self.last = np.full((n_nodes, 3), np.nan)
        self.partner: dict[int, int | str] = {}
        self.offset: dict[int, np.ndarray] = {}
        self.prev_hidden = np.ones(n_nodes, bool)
        self.events: list[tuple[int, int, Any]] = []
        self.t = -1

    def update(
        self, frame: np.ndarray, gripper_width: float, ee_xyz: np.ndarray
    ) -> np.ndarray:
        """frame [N, 8] with NaN for hidden nodes -> filled [N, 8] (u, v left NaN for the caller to project)."""
        self.t += 1
        out = frame.copy()
        hidden = frame[:, F_HIDDEN] == 1
        vis_xyz = frame[:, F_X : F_Z + 1]
        # nodes that just disappeared: choose a partner (bound) or nothing (frozen)
        for n in np.nonzero(hidden & ~self.prev_hidden)[0]:
            self.partner.pop(n, None)
            if self.rule == "bound" and np.all(np.isfinite(self.last[n])):
                closed = gripper_width < self.p.g_closed
                if closed and np.linalg.norm(ee_xyz - self.last[n]) < self.p.d_grip:
                    self.partner[n], self.offset[n] = "ee", self.last[n] - ee_xyz
                else:
                    cands = [
                        (np.linalg.norm(vis_xyz[m, :2] - self.last[n, :2]), m)
                        for m in range(self.n)
                        if m != n and not hidden[m] and np.all(np.isfinite(vis_xyz[m]))
                    ]
                    cands = [c for c in cands if c[0] < self.p.r_bind]
                    if cands:
                        m = min(cands)[1]
                        self.partner[n], self.offset[n] = m, self.last[n] - vis_xyz[m]
                self.events.append((self.t, int(n), self.partner.get(n)))
        # visible nodes: record; release any binding
        for n in np.nonzero(~hidden)[0]:
            if np.all(np.isfinite(vis_xyz[n])):
                self.last[n] = vis_xyz[n]
            self.partner.pop(n, None)
        # hidden nodes: fill from partner (partners filled first if they are hidden themselves)
        for n in np.nonzero(hidden)[0]:
            p = self.partner.get(n)
            if p == "ee":
                self.last[n] = ee_xyz + self.offset[n]
            elif p is not None and np.all(np.isfinite(self.last[p])):
                self.last[n] = self.last[p] + self.offset[n]
            out[n, F_X : F_Z + 1] = self.last[n]
            out[n, F_U : F_V + 1] = np.nan
        self.prev_hidden = hidden
        return out


def apply_rule(
    graph: np.ndarray,
    rule: str,
    gripper_width: np.ndarray,
    ee_xyz: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    params: RuleParams | None = None,
) -> tuple[np.ndarray, list]:
    """Run GraphTracker over [T, N, 8]; hidden nodes get xyz from the rule and (u, v) by projection."""
    T, N, _ = graph.shape
    tr = GraphTracker(N, rule, params)
    out = np.empty_like(graph)
    for t in range(T):
        f = tr.update(graph[t], float(gripper_width[t]), ee_xyz[t])
        hid = f[:, F_HIDDEN] == 1
        ok = hid & np.all(np.isfinite(f[:, F_X : F_Z + 1]), 1)
        if ok.any():
            f[ok, F_U : F_V + 1] = project_points(f[ok, F_X : F_Z + 1], K, E[t])
        out[t] = f
    return out, tr.events


def target_error(
    graph: np.ndarray,
    full_xyz: np.ndarray,
    node: int | np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
):
    """Per-frame error of one node against the full oracle: (pixels [T], metres [T]).

    node is one node id, or a [T] array of ids (the active target per frame).
    """
    T = len(graph)
    nodes = np.broadcast_to(np.asarray(node), (T,))
    px, m = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T):
        node = int(nodes[t])
        if np.all(np.isfinite(full_xyz[t, node])) and np.all(
            np.isfinite(graph[t, node, F_X : F_Z + 1])
        ):
            m[t] = np.linalg.norm(graph[t, node, F_X : F_Z + 1] - full_xyz[t, node])
            uv_true = project_points(full_xyz[t, node], K, E[t])[0]
            if np.all(np.isfinite(graph[t, node, F_U : F_V + 1])):
                px[t] = np.linalg.norm(graph[t, node, F_U : F_V + 1] - uv_true)
    return px, m


# --------------------------------------------------------------------------------------------
# Step 7: graph-quality levels
# --------------------------------------------------------------------------------------------
QUALITY_LEVELS = ["Q0", "Q1_20", "Q1_50", "Q2_25", "Q2_50", "Q3"]


def draw_corruption(level: str, seed: int, targets: list[int], scene: dict):
    """The per-episode random draws of a quality level, shared by corrupt() and OnlineGraph.

    Returns (q1, q2): q1 = the (du, dv) pixel offset for Q1_s, else None; q2 = {target: wrong
    container} for Q2_p when the episode's Bernoulli draw fires, else None. Each target gets its own
    fixed wrong container, chosen among the containers that do not hide it.
    """
    rng = np.random.default_rng(seed)
    if level.startswith("Q1_"):
        return rng.normal(0, float(level.split("_")[1]), 2), None
    if level.startswith("Q2_") and rng.random() < float(level.split("_")[1]) / 100.0:
        bins = [o["node"] for o in scene["objects"] if o["label_id"] == 0]
        return None, {
            c: int(rng.choice([b for b in bins if b != scene["cube_to_bin"][c]]))
            for c in targets
        }
    return None, None


def _offset_node(f: np.ndarray, n: int, duv: np.ndarray, K: np.ndarray, E: np.ndarray) -> None:
    """Shift node n of one frame [N, 8] by duv pixels; 3D = back-projection at the node's depth."""
    xyz = f[n, F_X : F_Z + 1]
    if not np.all(np.isfinite(xyz)):
        return
    cam = E[:3, :3] @ xyz + E[:3, 3]
    u, v = project_points(xyz, K, E)[0] + duv
    ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * cam[2]
    f[n, F_X : F_Z + 1] = (cam_to_world(E) @ np.r_[ray, 1.0])[:3]
    f[n, F_U : F_V + 1] = (u, v)


def corrupt(
    gated_bound: np.ndarray,
    level: str,
    seed: int,
    active: np.ndarray,
    targets: list[int],
    scene: dict,
    K: np.ndarray,
    E: np.ndarray,
    perceived_bound: np.ndarray | None = None,
) -> np.ndarray:
    """Test-time corruptions of the active target. Input = visibility-gated oracle + bound (Q0).

    active [T] is the active target node per frame (ActiveTarget) and targets the episode's ordered
    target list; each frame corrupts only the active node, and only while it is hidden. Draws are made
    once per episode from the full target list (draw_corruption), as OnlineGraph does.
    Q1_s : Gaussian pixel offset (sigma s px); the 3D offset is the back-projection of that pixel
           offset at the target's depth.
    Q2_p : with probability p, the hidden target takes a wrong container's position.
    Q3   : the perceived graph with the bound rule.
    """
    g = gated_bound.copy()
    if level == "Q0":
        return g
    if level == "Q3":
        if perceived_bound is None:
            raise ValueError("Q3 needs the perceived graph with the bound rule")
        return perceived_bound.copy()
    if not (level.startswith("Q1_") or level.startswith("Q2_")):
        raise ValueError(level)
    q1, q2 = draw_corruption(level, seed, targets, scene)
    for t, tg in enumerate(active):
        tg = int(tg)
        if g[t, tg, F_HIDDEN] != 1:
            continue
        if q1 is not None:
            _offset_node(g[t], tg, q1, K, E[t])
        elif q2 is not None:
            g[t, tg, F_X : F_Z + 1] = g[t, q2[tg], F_X : F_Z + 1]
            g[t, tg, F_U : F_V + 1] = g[t, q2[tg], F_U : F_V + 1]
    return g


# --------------------------------------------------------------------------------------------
# Step 5: perceived graph with the repo's HODOR pipeline (adapter; repo files untouched)
# --------------------------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("SG_MODELS_DIR", "/workspace/models"))  # SAM + GroundingDINO weights
PERCEP_PYTHON = os.environ.get("SG_PERCEP_PYTHON", "/workspace/envs/sg-percep/bin/python")


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# Smallest box GroundingDINO may accept as a container, in px^2 of the 512 px detector image. Measured
# from simulator masks (episodes 0-1): cube boxes <= 728, container boxes 5th percentile >= 1800. Without
# it the container slots seed on cubes at frame 0, before any container is in view.
CONTAINER_MIN_AREA_512 = 1200


def make_robomme_perception_config(n_bins: int, colors: list[str], scale: int = 2) -> dict:
    """Config for HODORProcessor on RoboMME front frames (256 px, optionally upscaled for detection).

    A plain dict (configs/base.yaml merged with the task block): the perception worker wraps it in
    OmegaConf, so the notebook's environment does not need OmegaConf.
    Class order: containers first, then cubes (a container box enclosing an earlier cube mask would be
    rejected by detect_class's overlap filter). 'container.' is listed once per container so each
    container gets its own slot and Cutie processor (extract_part_masks_cutie.py detect_class/process).
    Area ranges are in pixels of the detector image (256 * scale).
    """
    import yaml

    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    # one phrase per class: GroundingDINO splits its caption at '.', so "green.cube." would be the two
    # phrases "green" and "cube" and match any cube
    classes = ["container."] * n_bins + [f"{c} cube." for c in colors]
    img = 256 * scale
    task = {
        "task": {
            "name": "robomme",
            "data_path": "",
            "language_instruction": "",
            "tracker": "cutie",
            "classes": classes,
            "num_objects": len(classes),
            "confidence_threshold": 0.3,
            "slot_by_class": {i: i + 1 for i in range(len(classes))},
        },
        "perception": {
            "num_processors": len(classes),
            "class_area_ranges": {
                i: [int(CONTAINER_MIN_AREA_512 * (scale / 2) ** 2), None] for i in range(n_bins)
            },
            "skip_non_first_frame": False,
            "global_area_min": int(0.0005 * img * img),
            "global_area_max": int(0.25 * img * img),
            "overlap_threshold": 0.7,
        },
        "data": {"depth_min_m": 0.05, "depth_max_m": 3.0},
        "paths": {
            "home_path": str(REPO_ROOT),
            # absolute paths: os.path.join(home_path, abs) keeps abs (extract_part_masks_cutie.py)
            "groundingdino_checkpoint": str(MODELS_DIR / "groundingdino_swint_ogc.pth"),
            "sam_checkpoint": str(MODELS_DIR / "sam_vit_h_4b8939.pth"),
        },
        "output": {
            "output_dir": str(CACHE_DIR / "hodor_out"),
            "save_annotations": False,
            "save_masks": False,
            "save_dino_full": False,
            "save_rerun": False,
        },
    }
    return _deep_merge(base, task)


def robomme_processor_class():
    """RoboMMEProcessor = HODORProcessor with RoboMME camera geometry. Built inside the perception
    worker (sg-percep venv), which holds the repo's perception stack.

    # changed: _depth_to_pointcloud: no DROID crop shift (cx - 150); extrinsics are a 4x4 camera->world
    #          OpenCV matrix (converted from RoboMME's 3x4 world->camera extrinsic_cv by the caller).
    # changed: _add_centroids: masks resized to the depth resolution with nearest neighbour; no
    #          800x800 resize and no 180-degree rotation (RoboMME frames are not rotated).
    # added:   container candidates whose box touches the top image edge are dropped before
    #          detect_class's filters: the robot enters the front view there and is container-sized.
    """
    sys.path.insert(0, str(REPO_ROOT))
    import torch.nn.functional as F
    from post_process_scene_cutie import HODORProcessor

    class TopEdgeFilter:
        """Wraps the shared GroundingDINO model; drops 'container' boxes with y0 <= margin px."""

        def __init__(self, model, margin: int = 2):
            self.model, self.margin = model, margin

        def predict_with_classes(self, image, classes, box_threshold, text_threshold):
            det = self.model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if classes and classes[0].startswith("container") and len(det.xyxy):
                det = det[det.xyxy[:, 1] > self.margin]
            return det

    class RoboMMEProcessor(HODORProcessor):
        def __init__(self, config):
            super().__init__(config)
            det = self.tracker.detector
            det.grounding_dino = TopEdgeFilter(det.grounding_dino)

        def _depth_to_pointcloud(self, depth, rgb, intrinsics, extrinsics, binary_mask):
            fx, fy, cx, cy = (
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            )
            v, u = np.nonzero(
                (depth > float(self.config.data.depth_min_m))
                & (depth < float(self.config.data.depth_max_m))
                & (binary_mask > 0)
            )
            if len(u) == 0:
                return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
            z = depth[v, u]
            pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], 1)
            world = (np.asarray(extrinsics) @ pts.T).T[:, :3]
            return world, rgb[v, u, :]

        def _add_centroids(
            self, binary_masks, bboxes, depth, hand_rgb_unrotated, intrinsics, extrinsics
        ):
            H, W = depth.shape
            for i in range(binary_masks.shape[1]):
                m = F.interpolate(
                    binary_masks[:, i : i + 1].float(), size=(H, W), mode="nearest"
                )[0, 0].numpy()
                pts, _ = self._depth_to_pointcloud(
                    depth, hand_rgb_unrotated, intrinsics, extrinsics, m
                )
                if len(pts) > 0:
                    bboxes[i + 1, 4:7] = pts.mean(axis=0)

    return RoboMMEProcessor


# Messages between the notebook and the perception worker: a JSON header plus raw array bytes, so
# the two venvs' numpy versions (1.26 and 2.x) never have to unpickle each other's arrays.
def pack_msg(meta: dict, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    arrays = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    specs = {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()}
    head = json.dumps({"meta": meta, "arrays": specs}).encode()
    return b"".join(
        [len(head).to_bytes(8, "little"), head] + [v.tobytes() for v in arrays.values()]
    )


def unpack_msg(buf: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    n = int.from_bytes(buf[:8], "little")
    head = json.loads(buf[8 : 8 + n])
    arrays, off = {}, 8 + n
    for k, (dt, shape) in head["arrays"].items():
        size = int(np.prod(shape)) * np.dtype(dt).itemsize
        arrays[k] = np.frombuffer(buf, dt, int(np.prod(shape)), off).reshape(shape).copy()
        off += size
    return head["meta"], arrays


@dataclass
class FrameResult:
    """What process_frame returns across the process boundary (HODOR's FrameOutput, minus dino_full)."""

    representation: np.ndarray  # [S+1, 768]: slot 0 = global CLS, 1..S = object features
    bboxes: np.ndarray  # [S+1, 7]: [row_min, row_max, col_min, col_max] (normalised) + centroid xyz
    label_mask: np.ndarray | None  # [H, W] int: slot index per pixel, S = background (if requested)
    seed_events: list  # GroundingDINO+SAM seedings that happened during this frame
    compute_s: float  # time spent inside the worker


class PerceptionWorker:
    """Client for perception_worker.py: one long-lived process in the sg-percep venv that loads
    GroundingDINO, SAM, Cutie and DINOv2 once and then serves frames over a Unix socket.

    Started on first use and shared by everything in this process (get_perception_worker()).
    """

    def __init__(self, python: str = PERCEP_PYTHON, timeout_s: float = 600.0):
        import secrets
        import subprocess
        import tempfile
        from multiprocessing.connection import Listener

        self.log_path = CACHE_DIR / "perception_worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        sock = str(Path(tempfile.mkdtemp(prefix="sgpw_")) / "sock")
        key = secrets.token_bytes(16)
        listener = Listener(sock, family="AF_UNIX", authkey=key)
        # the pod sets HF_HUB_ENABLE_HF_TRANSFER=1, but hf_transfer is not installed in sg-percep;
        # a Jupyter kernel exports its inline MPLBACKEND, which sg-percep's matplotlib rejects
        env = dict(
            os.environ,
            SG_WORKER_AUTHKEY=key.hex(),
            HF_HUB_ENABLE_HF_TRANSFER="0",
            MPLBACKEND="Agg",
        )
        self.proc = subprocess.Popen(
            [python, str(PILOT_DIR / "perception_worker.py"), sock],
            stdout=open(self.log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PILOT_DIR),
        )
        listener._listener._socket.settimeout(timeout_s)
        try:
            self.conn = listener.accept()
        except OSError as exc:
            self.proc.kill()
            raise RuntimeError(f"perception worker did not connect; log:\n{self.log_tail()}") from exc
        finally:
            listener.close()
        self.ready = self.call("ping")[0]
        self.roundtrip_s: list[float] = []

    def log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text().splitlines()[-n:])
        except OSError:
            return "<no log>"

    def call(self, op: str, meta: dict | None = None, arrays: dict | None = None):
        if self.proc.poll() is not None:
            raise RuntimeError(f"perception worker exited ({self.proc.returncode}); log:\n{self.log_tail()}")
        try:
            self.conn.send_bytes(pack_msg({"op": op, **(meta or {})}, arrays))
            out_meta, out_arrays = unpack_msg(self.conn.recv_bytes())
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"perception worker connection lost; log:\n{self.log_tail()}") from exc
        if "error" in out_meta:
            raise RuntimeError(f"perception worker, op {op}:\n{out_meta['error']}")
        return out_meta, out_arrays

    def configure(self, config: dict) -> "PerceptionHandle":
        meta, _ = self.call("configure", {"config": config})
        return PerceptionHandle(self, meta)

    def close(self) -> None:
        """Stop the worker (frees its ~GPU memory, e.g. before training)."""
        global _WORKER
        if self.proc.poll() is None:
            try:
                self.conn.send_bytes(pack_msg({"op": "exit"}))
                self.proc.wait(30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        if _WORKER is self:
            _WORKER = None


_WORKER: PerceptionWorker | None = None


def get_perception_worker() -> PerceptionWorker:
    global _WORKER
    if _WORKER is None or _WORKER.proc.poll() is not None:
        import atexit

        _WORKER = PerceptionWorker()
        atexit.register(_WORKER.close)
    return _WORKER


def close_perception_worker() -> bool:
    """Stop the perception worker if one is running (e.g. before training); True if one was stopped."""
    if _WORKER is None:
        return False
    _WORKER.close()
    return True


class PerceptionHandle:
    """One configured RoboMMEProcessor inside the worker, with the processor attributes the notebook
    uses (classes, num_classes, feature_dim). Models are shared across handles; a new class list
    only builds new detector/tracker wrappers."""

    def __init__(self, worker: PerceptionWorker, meta: dict):
        self.worker, self.id = worker, meta["handle"]
        self.classes, self.num_classes = meta["classes"], meta["num_classes"]
        self.feature_dim = meta["feature_dim"]

    def process_frame(
        self,
        hand_bgr_rotated,
        frame_num,
        episode_num,
        depth,
        hand_rgb_unrotated,
        intrinsics,
        extrinsics,
        return_mask: bool = False,
    ) -> FrameResult:
        """HODORProcessor.process_frame in the worker (same arguments). frame_num 0 starts an episode."""
        t0 = time.perf_counter()
        meta, arr = self.worker.call(
            "process_frame",
            dict(handle=self.id, frame_num=int(frame_num), episode_num=int(episode_num), return_mask=return_mask),
            dict(
                bgr=hand_bgr_rotated,
                depth=np.asarray(depth, np.float32),
                rgb=hand_rgb_unrotated,
                K=np.asarray(intrinsics, np.float64),
                extr=np.asarray(extrinsics, np.float64),
            ),
        )
        self.worker.roundtrip_s.append(time.perf_counter() - t0)
        return FrameResult(
            arr["representation"], arr["bboxes"], arr.get("label_mask"), meta["seed_events"], meta["compute_s"]
        )

    def obtain_features(self, image_bgr: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """HODORProcessor._obtain_features: masks [N, H, W] -> [N+1, 768] (row 0 = global CLS)."""
        _, arr = self.worker.call(
            "features",
            {"handle": self.id},
            {"bgr": image_bgr, "masks": np.asarray(masks, np.float32)},
        )
        return arr["representation"]


def make_robomme_processor(config: dict) -> PerceptionHandle:
    """The repo's HODOR pipeline as a RoboMMEProcessor (robomme_processor_class), configured in the
    persistent perception worker; see PerceptionWorker for why it runs in a second process."""
    return get_perception_worker().configure(config)


def perceived_graph(
    rep: Replay,
    processor: PerceptionHandle,
    scale: int = 2,
    episode_num: int = 0,
    keep_masks: set[int] | None = None,
):
    """Run the processor over all front frames.

    Returns graph [T, S, 8] in slot order, features [T, S, 768], every seed event (GroundingDINO+SAM
    initialisation, with its frame) and the tracker's label masks (at the detector resolution) for the
    frames in keep_masks.

    (u, v) = front projection of the perceived centroid; hidden = 1 when the slot has no mask this frame
    (process_frame resets only bbox[:4] each frame, so stale centroids/features of absent slots are
    replaced by NaN here and filled by the rule later).
    """
    import cv2

    T = len(rep.front_rgb)
    S = processor.num_classes
    label_masks, seeds = {}, []
    g = np.full((T, S, 8), np.nan, np.float32)
    feats = np.zeros((T, S, processor.feature_dim), np.float32)
    for t in range(T):
        rgb = rep.front_rgb[t]
        big = cv2.resize(
            rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_CUBIC
        )
        out = processor.process_frame(
            hand_bgr_rotated=np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_RGB2BGR)),
            frame_num=t,
            episode_num=episode_num,
            depth=rep.front_depth_m[t],
            hand_rgb_unrotated=rgb,
            intrinsics=rep.K_front,
            extrinsics=cam_to_world(rep.E_front[t]),
            return_mask=keep_masks is not None and t in keep_masks,
        )
        seeds += out.seed_events
        if out.label_mask is not None:
            label_masks[t] = out.label_mask
        for s in range(S):
            bb = out.bboxes[s + 1]
            visible = not np.allclose(bb[:4], 0.0)
            g[t, s, F_ID], g[t, s, F_LABEL] = s, (
                0 if processor.classes[s].startswith("container") else 1
            )
            g[t, s, F_HIDDEN] = 0 if visible else 1
            if visible and np.any(bb[4:7] != 0):
                g[t, s, F_X : F_Z + 1] = bb[4:7]
                g[t, s, F_U : F_V + 1] = project_points(bb[4:7], rep.K_front, rep.E_front[t])[
                    0
                ]
                feats[t, s] = out.representation[s + 1]
    return g, feats, seeds, label_masks


def assign_slots(perceived: np.ndarray, oracle: np.ndarray, t: int, radius_px: float = 25.0):
    """Hungarian assignment slot -> oracle node at frame t on (u, v) distance, same label only."""
    from scipy.optimize import linear_sum_assignment

    S, N = perceived.shape[1], oracle.shape[1]
    cost = np.full((S, N), 1e6)
    for s in range(S):
        for n in range(N):
            if (
                perceived[t, s, F_LABEL] == oracle[t, n, F_LABEL]
                and np.all(np.isfinite(perceived[t, s, F_U : F_V + 1]))
                and np.all(np.isfinite(oracle[t, n, F_U : F_V + 1]))
            ):
                cost[s, n] = np.linalg.norm(
                    perceived[t, s, F_U : F_V + 1] - oracle[t, n, F_U : F_V + 1]
                )
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < radius_px}


def assign_slots_seeded(perceived: np.ndarray, oracle: np.ndarray, radius_px: float = 25.0):
    """Slot -> oracle node, each slot matched at the first frame it has a mask (its seed frame).

    Slots seed at different frames (cubes at frame 0, containers only once they enter the view), so a
    single-frame assignment would leave some slots unmatched. Slots sharing a seed frame are matched
    jointly (assign_slots, same label, Hungarian on (u, v)); a node is used at most once.
    Returns (assignment, {slot: seed frame}).
    """
    S = perceived.shape[1]
    seed = {}
    for s in range(S):
        vis = np.nonzero(perceived[:, s, F_HIDDEN] == 0)[0]
        if len(vis):
            seed[s] = int(vis[0])
    out: dict[int, int] = {}
    for t in sorted(set(seed.values())):
        group = [s for s in seed if seed[s] == t]
        sub = perceived[:, group].copy()
        orc = oracle.copy()
        orc[:, list(out.values()), F_U : F_V + 1] = np.nan  # already taken
        for i, n in assign_slots(sub, orc, t, radius_px).items():
            out[group[i]] = n
    return out, seed


def identity_accuracy(
    perceived: np.ndarray,
    oracle: np.ndarray,
    assignment: dict,
    frames: list[int],
    radius_px: float = 25.0,
) -> dict:
    """Share of frames where each visible slot is still nearest to its assigned object; plus
    the number of frames where two slots sit on the same object."""
    ok = tot = dup = 0
    for t in frames:
        hits = {}
        for s, n in assignment.items():
            if perceived[t, s, F_HIDDEN] == 1 or oracle[t, n, F_HIDDEN] == 1:
                continue
            d = np.linalg.norm(
                oracle[t, :, F_U : F_V + 1] - perceived[t, s, F_U : F_V + 1], axis=1
            )
            d[~np.isfinite(d)] = np.inf
            nearest = int(np.argmin(d))
            tot += 1
            ok += int(nearest == n and d[nearest] < radius_px)
            hits.setdefault(nearest, []).append(s)
        dup += int(any(len(v) > 1 for v in hits.values()))
    return {
        "identity_accuracy": ok / tot if tot else float("nan"),
        "slot_frames": tot,
        "frames_with_shared_object": dup,
    }


def reorder_to_oracle(
    perceived: np.ndarray, feats: np.ndarray, assignment: dict, n_nodes: int
):
    """Perceived slots placed at their assigned oracle node index; unassigned nodes stay hidden/NaN."""
    T = perceived.shape[0]
    g = np.full((T, n_nodes, 8), np.nan, np.float32)
    f = np.zeros((T, n_nodes, feats.shape[-1]), np.float32)
    g[..., F_HIDDEN] = 1
    for s, n in assignment.items():
        g[:, n] = perceived[:, s]
        g[:, n, F_ID] = n
        f[:, n] = feats[:, s]
    return g, f


def oracle_features(
    rep: Replay, processor: PerceptionHandle, frames: list[int] | None = None
) -> np.ndarray:
    """DINOv2 masked-mean feature per node from simulator masks (A3feat training data); hidden nodes keep the
    last feature from a frame where at least 90% of their largest seen mask area was visible.
    """
    import cv2

    objs = rep.scene["objects"]
    T, N = len(rep.front_rgb), len(objs)
    frames = range(T) if frames is None else frames
    feats = np.zeros((T, N, processor.feature_dim), np.float32)
    best_area = np.zeros(N)
    last = np.zeros((N, processor.feature_dim), np.float32)
    for t in frames:
        masks = np.stack(
            [np.isin(rep.front_seg[t], rep.scene["seg_ids"][o["node"]]) for o in objs]
        )
        areas = masks.sum((1, 2))
        rep_t = processor.obtain_features(
            cv2.cvtColor(rep.front_rgb[t], cv2.COLOR_RGB2BGR), masks
        )[1:]
        for n in range(N):
            best_area[n] = max(best_area[n], areas[n])
            if areas[n] > 0 and areas[n] >= 0.9 * best_area[n]:
                last[n] = rep_t[n]
            feats[t, n] = last[n]
    return feats


# --------------------------------------------------------------------------------------------
# Step 8: the interfaces (same per-object content: id, label, (u, v), hidden)
# --------------------------------------------------------------------------------------------
def nodes_at(graph: np.ndarray, t: int, scene: dict) -> list[dict]:
    out = []
    for o in scene["objects"]:
        n = o["node"]
        u, v = graph[t, n, F_U], graph[t, n, F_V]
        out.append(
            dict(
                id=n,
                label=o["text"],
                u=None if not np.isfinite(u) else float(u),
                v=None if not np.isfinite(v) else float(v),
                hidden=bool(graph[t, n, F_HIDDEN] == 1),
            )
        )
    return out


def draw_markers(image: np.ndarray, nodes: list[dict], size: int = 6) -> np.ndarray:
    """A1: labelled markers at (u, v). Visible: filled circle; hidden: ring with an 'h'."""
    import cv2

    img = image.copy()
    for nd in nodes:
        if nd["u"] is None:
            continue
        c = (int(round(nd["u"])), int(round(nd["v"])))
        colour = (255, 255, 0) if nd["label"] == "container" else (255, 0, 255)
        if nd["hidden"]:
            cv2.circle(img, c, size, colour, max(1, size // 3))
        else:
            cv2.circle(img, c, size, colour, -1)
        tag = f"{nd['id']}{'h' if nd['hidden'] else ''}"
        cv2.putText(
            img,
            tag,
            (c[0] + size, c[1] - size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25 + size / 40,
            colour,
            1,
            cv2.LINE_AA,
        )
    return img


def serialize(nodes: list[dict], instruction: str) -> str:
    """A2: instruction + fixed-template object list (integer pixels).

    Positions use RoboMME's grounded-subgoal convention, 'at <row, col>' = <v, u> (the dataset's
    "pick up the container at <100, 139> ..."; env_record_wrapper/DemonstrationWrapper.py:570).
    """
    parts = []
    for nd in nodes:
        pos = (
            "at unknown"
            if nd["u"] is None
            else f"at <{int(round(nd['v']))}, {int(round(nd['u']))}>"
        )
        parts.append(
            f"[{nd['id']} {nd['label']}, {'hidden' if nd['hidden'] else 'visible'}, {pos}]"
        )
    return f"{instruction.rstrip('.')}. Objects: " + "; ".join(parts)


def node_tensor(
    nodes: list[dict], width: int, height: int, feats: np.ndarray | None = None, n_max: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A3/A3feat input: x [n_max, 3 (+768)], label_ids [n_max], ids [n_max], mask [n_max]."""
    d = 3 + (feats.shape[-1] if feats is not None else 0)
    x = np.zeros((n_max, d), np.float32)
    lab = np.zeros(n_max, np.int64)
    ids = np.zeros(n_max, np.int64)
    mask = np.zeros(n_max, bool)
    for i, nd in enumerate(nodes[:n_max]):
        x[i, 0] = (nd["u"] / width) if nd["u"] is not None else -1.0
        x[i, 1] = (nd["v"] / height) if nd["v"] is not None else -1.0
        x[i, 2] = float(nd["hidden"])
        if feats is not None:
            x[i, 3:] = feats[nd["id"]]
        lab[i] = 0 if nd["label"] == "container" else 1
        ids[i] = nd["id"]
        mask[i] = True
    return x, lab, ids, mask


def make_graph_tokenizer(
    in_dim: int,
    d_model: int,
    label_mode: str = "table",
    text_embed=None,
    label_words=("container", "cube"),
    max_ids: int = 16,
    hidden: int = 512,
):
    import torch
    from torch import nn

    class GraphTokenizer(nn.Module):
        """[B, N, in_dim] -> [B, N, d_model]: MLP + LayerNorm + label embedding + randomly permuted id embedding."""

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model)
            )
            self.norm = nn.LayerNorm(d_model)
            self.id_emb = nn.Embedding(max_ids, d_model)
            self.label_mode = label_mode
            if label_mode == "table":
                self.label_emb = nn.Embedding(len(label_words), d_model)
            else:  # 'text': mean of the VLM's own token embeddings of each label word (frozen)
                assert (
                    text_embed is not None
                ), "label_mode='text' needs text_embed(word) -> [d_model]"
                self.register_buffer(
                    "label_text", torch.stack([text_embed(w) for w in label_words])
                )
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.permute_ids = True

        def forward(self, x, labels, ids):
            h = self.norm(self.mlp(x))
            lab = (
                self.label_emb(labels)
                if self.label_mode == "table"
                else self.label_text[labels]
            )
            if self.training and self.permute_ids:
                perm = torch.randperm(self.id_emb.num_embeddings, device=ids.device)
                ids = perm[ids]
            return (h + lab + self.id_emb(ids)) * self.scale

    return GraphTokenizer()


class GraphPrefix:
    """Inserts graph tokens into SmolVLA's prefix: [image tokens | language tokens | GRAPH | state].

    Wraps VLAFlowMatching.embed_prefix (lerobot/policies/smolvla/modeling_smolvla.py). The graph tokens get
    att_mask 0 (bidirectional with images and language, like them) and pad mask from the node mask.
    Set .current = (x, labels, ids, mask) tensors on the policy device before each forward / select_action.
    """

    def __init__(self, flow_model, tokenizer):
        self.model, self.tok, self.current = flow_model, tokenizer, None
        self.orig = flow_model.embed_prefix
        flow_model.embed_prefix = self.embed_prefix

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        import torch

        embs, pad, att = self.orig(images, img_masks, lang_tokens, lang_masks, state=state)
        if self.current is None:
            return embs, pad, att
        x, labels, ids, mask = self.current
        g = self.tok(x.float(), labels, ids).to(embs.dtype)  # [B, N, D]
        k = 1  # state tokens are last (one token)
        embs = torch.cat([embs[:, :-k], g, embs[:, -k:]], 1)
        pad = torch.cat([pad[:, :-k], mask.bool(), pad[:, -k:]], 1)
        att = torch.cat([att[:, :-k], torch.zeros_like(mask, dtype=att.dtype), att[:, -k:]], 1)
        return embs, pad, att


# --------------------------------------------------------------------------------------------
# Step 8-9: policy, LoRA, data
# --------------------------------------------------------------------------------------------
RENAME_MAP = {
    "image": "observation.images.camera1",
    "wrist_image": "observation.images.camera2",
    "state": "observation.state",
    "actions": "action",
}  # docs/source/robomme.mdx
LORA_TARGETS = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
    r"|model\.vlm_with_expert\.vlm\..*text_model\.layers\.\d+\.self_attn\.(q|v)_proj"
    r"|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)
TOKENIZER_MAX_LENGTH = 192  # same for every arm; checked against the A2 worst case in Step 8.3
INTERFACES = ["A0", "A1", "A2", "A3", "A3feat"]


def build_policy(
    device: str = "cuda", lora_r: int = 64, tokenizer_max_length: int = TOKENIZER_MAX_LENGTH
):
    """SmolVLA from smolvla_base with RoboMME features, the same LoRA for every arm, and processors.

    Mirrors lerobot_train.py: make_policy(ds_meta, rename_map) -> wrap_with_peft -> make_pre_post_processors
    with the normalizer/rename overrides. VERIFY all names against the installed lerobot.
    """
    from peft import LoraConfig
    from lerobot.configs.policies import PreTrainedConfig  # configs/policies.py:43
    from lerobot.datasets import LeRobotDatasetMetadata  # datasets/__init__.py:25
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import (
        rename_stats,
    )  # processor/rename_processor.py:67

    meta = LeRobotDatasetMetadata(LEROBOT_REPO)
    cfg = PreTrainedConfig.from_pretrained(SMOLVLA_BASE)
    cfg.pretrained_path = SMOLVLA_BASE
    cfg.device = device
    cfg.tokenizer_max_length = tokenizer_max_length
    cfg.input_features, cfg.output_features = {}, {}
    policy = make_policy(cfg=cfg, ds_meta=meta, rename_map=RENAME_MAP)
    flow_model = policy.model
    lora = LoraConfig(r=lora_r, lora_alpha=lora_r, target_modules=LORA_TARGETS)
    policy = policy.wrap_with_peft(peft_config=lora)
    stats = rename_stats(meta.stats, RENAME_MAP)
    pre_over = {
        "device_processor": {"device": device},
        "normalizer_processor": {
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
            "stats": stats,
        },
        "rename_observations_processor": {"rename_map": RENAME_MAP},
        "tokenizer_processor": {"max_length": tokenizer_max_length},
    }
    post_over = {
        "unnormalizer_processor": {
            "features": cfg.output_features,
            "norm_map": cfg.normalization_mapping,
            "stats": stats,
        }
    }
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=SMOLVLA_BASE,
        dataset_stats=stats,
        preprocessor_overrides=pre_over,
        postprocessor_overrides=post_over,
    )
    return policy, flow_model, pre, post, cfg, meta


def prep(pre, batch: dict) -> dict:
    """Rename raw keys (image, wrist_image, state, actions) to the policy's keys, then run the SmolVLA
    preprocessor (normalise, tokenise the task text, move to device), as lerobot_train.py does.
    """
    from lerobot.processor.rename_processor import rename_batch_keys

    return pre(rename_batch_keys(batch, RENAME_MAP))


def trainable_parameters(*modules) -> int:
    return sum(
        p.numel() for m in modules if m is not None for p in m.parameters() if p.requires_grad
    )


def load_graph_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


class SGDataset:
    """One sample per policy step k of every cached episode.

    Images are the replayed frames at policy_frame[k]; state = 7 joints + first finger width (as
    lerobot/envs/robomme.py _convert_obs); action chunk = actions[k : k + chunk] with a pad mask.
    The graph is the Q0 graph (visibility-gated oracle + bound) at that frame; the interface decides
    how it reaches the policy.
    """

    def __init__(
        self,
        episodes: list[tuple[Replay, np.ndarray, np.ndarray | None]],
        interface: str,
        chunk: int = 50,
        marker_size: int = 6,
    ):
        self.ep, self.iface, self.chunk, self.marker = episodes, interface, chunk, marker_size
        self.index = [
            (e, k) for e, (rep, _, _) in enumerate(episodes) for k in range(len(rep.actions))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import torch

        e, k = self.index[i]
        rep, graph, feats = self.ep[e]
        t = int(rep.policy_frame[k])
        front, wrist = rep.front_rgb[t], rep.wrist_rgb[t]
        nodes = nodes_at(graph, t, rep.scene)
        task = rep.task_goal
        if self.iface == "A1":
            front = draw_markers(front, nodes, self.marker)
        if self.iface == "A2":
            task = serialize(nodes, task)
        acts = rep.actions[k : k + self.chunk]
        pad = np.zeros(self.chunk, bool)
        if len(acts) < self.chunk:
            pad[len(acts) :] = True
            acts = np.concatenate([acts, np.repeat(acts[-1:], self.chunk - len(acts), 0)])
        state = np.r_[rep.joint_state[t], rep.gripper_state[t][:1]].astype(np.float32)
        sample = {
            "image": torch.from_numpy(front).permute(2, 0, 1).float() / 255.0,
            "wrist_image": torch.from_numpy(wrist).permute(2, 0, 1).float() / 255.0,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(acts.astype(np.float32)),
            "action_is_pad": torch.from_numpy(pad),
            "task": task,
        }
        if self.iface in ("A3", "A3feat"):
            x, lab, ids, mask = node_tensor(
                nodes,
                front.shape[1],
                front.shape[0],
                feats[t] if (self.iface == "A3feat") else None,
            )
            sample.update(
                sg_x=torch.from_numpy(x),
                sg_label=torch.from_numpy(lab),
                sg_id=torch.from_numpy(ids),
                sg_mask=torch.from_numpy(mask),
            )
        return sample


def collate(samples: list[dict]) -> dict:
    import torch

    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        out[k] = vals if isinstance(vals[0], str) else torch.stack(vals)
    return out


def split_graph_inputs(batch: dict, device) -> tuple[dict, tuple | None]:
    g = None
    if "sg_x" in batch:
        g = tuple(batch.pop(k).to(device) for k in ["sg_x", "sg_label", "sg_id", "sg_mask"])
    pad = batch.pop("action_is_pad", None)
    return batch, (g, pad)


def train(
    interface: str,
    tasks: list[str],
    seed: int,
    steps: int,
    out_dir: Path,
    episodes_per_task: int,
    batch_size: int = 32,
    lr: float = 1e-3,
    label_mode: str = "table",
    log_every: int = 10,
    device: str = "cuda",
) -> dict:
    """Fine-tune one arm. Training graphs are always Q0. Returns the loss history and timing.

    steps_per_s counts every step; steps_per_s_steady excludes step 0 (CUDA/cuDNN warm-up and the first
    DataLoader batch), which is the rate to extrapolate from.
    """
    import torch

    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    episodes = []
    for task in tasks:
        for p in sorted((CACHE_DIR / task).glob("ep*_replay.npz"))[:episodes_per_task]:
            rep = load_replay(p)
            gc = load_graph_cache(p.with_name(p.name.replace("_replay", "_graph")))
            episodes.append((rep, gc["Q0"], gc.get("oracle_feats")))
    policy, flow, pre, post, cfg, meta = build_policy(device)
    tok = None
    if interface in ("A3", "A3feat"):
        d_model = flow.vlm_with_expert.config.text_config.hidden_size
        in_dim = 3 + (episodes[0][2].shape[-1] if interface == "A3feat" else 0)
        text_embed = None
        if label_mode == "text":
            tz = flow.vlm_with_expert.processor.tokenizer
            emb = flow.vlm_with_expert.embed_language_tokens
            text_embed = (
                lambda w: emb(torch.tensor(tz(w)["input_ids"], device=device)).mean(0).detach()
            )  # noqa: E731
        tok = make_graph_tokenizer(in_dim, d_model, label_mode, text_embed).to(device)
        gp = GraphPrefix(flow, tok)
    ds = SGDataset(episodes, interface, chunk=cfg.chunk_size)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=False,
    )
    params = [p for p in policy.parameters() if p.requires_grad] + (
        [*tok.parameters()] if tok else []
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    hist, it, t0 = [], iter(dl), time.time()
    policy.train()
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch, (g, pad) = split_graph_inputs(batch, device)
        batch = prep(pre, batch)
        if pad is not None:
            batch["action_is_pad"] = pad.to(device)
        if tok is not None:
            gp.current = g
        loss, _ = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        hist.append(float(loss.item()))  # .item() synchronises, so step times are real
        if step == 0:
            t_first = time.time()
        if step % log_every == 0:
            print(
                f"step {step:6d}  loss {hist[-1]:.4f}  {(step + 1) / (time.time() - t0):.2f} it/s"
            )
    dt = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir / "policy")
    pre.save_pretrained(out_dir / "policy")
    post.save_pretrained(out_dir / "policy")
    if tok is not None:
        torch.save(tok.state_dict(), out_dir / "graph_tokenizer.pt")
    info = dict(
        interface=interface,
        tasks=tasks,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        label_mode=label_mode,
        seconds=dt,
        steps_per_s=steps / dt,
        steps_per_s_steady=(steps - 1) / (t0 + dt - t_first) if steps > 1 else None,
        loss=hist,
        trainable_params=trainable_parameters(policy, tok),
        max_mem_GB=(
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        ),
    )
    (out_dir / "train_info.json").write_text(json.dumps(info))
    return info


# --------------------------------------------------------------------------------------------
# Step 10: closed-loop evaluation
# --------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def pilot_decision(k0: int, n0: int, k2: int, n2: int, ignored: bool | None) -> str:
    lo2, _ = wilson(k2, n2)
    _, hi0 = wilson(k0, n0)
    if k0 == 0:
        return "stop: A0 at floor (0 successes), the task cannot separate the arms"
    if ignored:
        return "stop: graph ignored (counterfactual test: policy does not follow the graph)"
    if lo2 > hi0:
        return f"continue: A2 lower bound {lo2:.3f} > A0 upper bound {hi0:.3f}"
    return f"stop: no clear gain (A2 lower bound {lo2:.3f} <= A0 upper bound {hi0:.3f})"


class OnlineGraph:
    """Builds the graph during a rollout from the env's own frames (same code path as the offline cache)."""

    def __init__(
        self,
        scene: dict,
        rule: str = "bound",
        level: str = "Q0",
        seed: int = 0,
        targets: list[int] | None = None,
        counterfactual_to: dict[int, int] | None = None,
    ):
        """targets: the episode's ordered targets; the caller sets self.active (an ActiveTarget
        result) before each step. counterfactual_to: {target: wrong container} for 10.5."""
        self.scene, self.level, self.seed = scene, level, seed
        self.targets = list(targets or [])
        self.active = self.targets[0] if self.targets else None
        self.N = len(scene["objects"])
        self.tracker = GraphTracker(self.N, rule)
        self.cf = counterfactual_to
        self.q1, self.q2 = draw_corruption(level, seed, self.targets, scene)
        self.history: list[np.ndarray] = []

    def step(
        self, rgb, depth_m, seg, wrist_depth_m, wrist_seg, K, E, Kw, Ew, gripper_width, ee_xyz
    ):
        rep_like = Replay(
            task="",
            episode=0,
            front_rgb=rgb[None],
            wrist_rgb=rgb[None],
            front_depth_m=depth_m[None],
            wrist_depth_m=wrist_depth_m[None],
            front_seg=seg[None],
            wrist_seg=wrist_seg[None],
            K_front=K,
            K_wrist=Kw,
            E_front=E[None],
            E_wrist=Ew[None],
            joint_state=np.zeros((1, 7)),
            gripper_state=np.zeros((1, 2)),
            ee_xyz=ee_xyz[None],
            is_video=np.zeros(1, bool),
            policy_frame=np.zeros(0, int),
            actions=np.zeros((0, 8)),
            task_goal="",
            scene=self.scene,
            status="",
        )
        frame, _ = oracle_graph(rep_like)
        f = self.tracker.update(frame[0], gripper_width, ee_xyz)
        hid = f[:, F_HIDDEN] == 1
        ok = hid & np.all(np.isfinite(f[:, F_X : F_Z + 1]), 1)
        if ok.any():
            f[ok, F_U : F_V + 1] = project_points(f[ok, F_X : F_Z + 1], K, E)
        tg = self.active
        if tg is not None and f[tg, F_HIDDEN] == 1:
            wrong = (self.cf or self.q2 or {}).get(tg)
            if wrong is not None:
                f[tg, F_X : F_Z + 1] = f[wrong, F_X : F_Z + 1]
                f[tg, F_U : F_V + 1] = f[wrong, F_U : F_V + 1]
            elif self.q1 is not None:
                _offset_node(f, tg, self.q1, K, E)
        self.history.append(f)
        return f


class OnlineFeatures:
    """A3feat online: DINOv2 masked-mean feature per node from the simulator masks of the current frame;
    a node keeps the last feature seen with >= 90% of its largest mask area (same rule as oracle_features).
    """

    def __init__(self, processor, scene: dict):
        self.proc, self.scene = processor, scene
        n = len(scene["objects"])
        self.best = np.zeros(n)
        self.last = np.zeros((n, processor.feature_dim), np.float32)

    def step(self, rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
        import cv2

        masks = np.stack(
            [np.isin(seg, self.scene["seg_ids"][o["node"]]) for o in self.scene["objects"]]
        )
        areas = masks.sum((1, 2))
        rep_t = self.proc.obtain_features(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), masks)[1:]
        for n, a in enumerate(areas):
            self.best[n] = max(self.best[n], a)
            if a > 0 and a >= 0.9 * self.best[n]:
                self.last[n] = rep_t[n]
        return self.last.copy()


class PerceivedOnline:
    """Q3 online: the repo's HODOR pipeline (through RoboMMEProcessor) runs on every new front frame,
    and the bound rule fills hidden slots. Nodes are the processor's slots; their label and text come
    from the class list, so the target cube's slot is found by its colour, with no simulator help.
    """

    def __init__(self, processor, scale: int = 2, rule: str = "bound"):
        self.proc, self.scale, self.t = processor, scale, 0
        S = processor.num_classes
        self.tracker = GraphTracker(S, rule)
        self.scene = {"objects": []}
        for s, c in enumerate(processor.classes):
            is_cont = c.startswith("container")
            text = "container" if is_cont else c.rstrip(".")
            self.scene["objects"].append(
                dict(
                    node=s,
                    label_id=0 if is_cont else 1,
                    text=text,
                    color=None if is_cont else text.split()[0],
                )
            )
        self.feats = np.zeros((S, processor.feature_dim), np.float32)

    def step(self, rgb, depth_m, K, E, gripper_width, ee_xyz) -> np.ndarray:
        import cv2

        big = cv2.resize(
            rgb,
            (rgb.shape[1] * self.scale, rgb.shape[0] * self.scale),
            interpolation=cv2.INTER_CUBIC,
        )
        out = self.proc.process_frame(
            hand_bgr_rotated=np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_RGB2BGR)),
            frame_num=self.t,
            episode_num=0,
            depth=depth_m,
            hand_rgb_unrotated=rgb,
            intrinsics=K,
            extrinsics=cam_to_world(E),
        )
        self.t += 1
        S = self.proc.num_classes
        f = np.full((S, 8), np.nan, np.float32)
        for s in range(S):
            bb = out.bboxes[s + 1]
            f[s, F_ID], f[s, F_LABEL] = s, self.scene["objects"][s]["label_id"]
            vis = (not np.allclose(bb[:4], 0.0)) and np.any(bb[4:7] != 0)
            f[s, F_HIDDEN] = 0 if vis else 1
            if vis:
                f[s, F_X : F_Z + 1] = bb[4:7]
                f[s, F_U : F_V + 1] = project_points(bb[4:7], K, E)[0]
                self.feats[s] = out.representation[s + 1]
        g = self.tracker.update(f, gripper_width, ee_xyz)
        hid = g[:, F_HIDDEN] == 1
        ok = hid & np.all(np.isfinite(g[:, F_X : F_Z + 1]), 1)
        if ok.any():
            g[ok, F_U : F_V + 1] = project_points(g[ok, F_X : F_Z + 1], K, E)
        return g


def evaluate(
    ckpt: Path,
    interface: str,
    task: str,
    quality: str,
    episodes: int,
    out_json: Path,
    max_steps: int = 1300,
    counterfactual: bool = False,
    device: str = "cuda",
    strip_every: int = 50,
    save_strip: Path | None = None,
    episode_offset: int = 0,
) -> dict:
    """Closed-loop success on the test split.

    The graph is rebuilt from every frame the env returns at reset (the history video included) and
    updated after every step. Q0-Q2: simulator masks + bound rule, corrupted per level. Q3: the repo's
    perception pipeline + bound rule, run online. The success flag is the env's own status.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    policy, flow, pre, post, cfg, meta = build_policy(device)
    set_peft_model_state_dict(
        policy, load_file(str(ckpt / "policy" / "adapter_model.safetensors"))
    )
    tok = gp = None
    if interface in ("A3", "A3feat"):
        state = torch.load(ckpt / "graph_tokenizer.pt", map_location=device)
        in_dim = state["mlp.0.weight"].shape[1]
        tok = make_graph_tokenizer(in_dim, flow.vlm_with_expert.config.text_config.hidden_size)
        tok = tok.to(device)
        tok.load_state_dict(state)
        tok.eval()
        gp = GraphPrefix(flow, tok)
    policy.eval()
    results, strip, t_ep = [], [], []
    for ep in range(episode_offset, episode_offset + episodes):
        t0 = time.time()
        env = make_env(task, "test", ep, max_steps=max_steps)
        obs, info = env.reset()
        scene = scene_objects(env)
        goal = info.get("task_goal", "")
        goal = goal[0] if isinstance(goal, (list, tuple)) and goal else str(goal)
        sc = {
            "objects": [
                {k: v for k, v in o.items() if k != "actor"} for o in scene["objects"]
            ],
            "cube_to_bin": scene["cube_to_bin"],
            "seg_ids": scene["seg_ids"],
            "n_bins": scene["n_bins"],
        }
        targets = target_cube_nodes(sc, goal)
        tgt = targets[0]
        cf_to = None
        if counterfactual:
            cf_to = {
                c: next(
                    o["node"]
                    for o in sc["objects"]
                    if o["label_id"] == 0 and o["node"] != scene["cube_to_bin"][c]
                )
                for c in targets
            }
        K = np.asarray(info["front_camera_intrinsic"], np.float32).reshape(3, 3)
        Kw = np.asarray(info["wrist_camera_intrinsic"], np.float32).reshape(3, 3)
        need_proc = quality == "Q3" or interface == "A3feat"
        proc = None
        if need_proc:
            colors = [o["color"] for o in sc["objects"] if o["label_id"] == 1]
            proc = make_robomme_processor(
                make_robomme_perception_config(sc["n_bins"], colors, scale=2)
            )
        if quality == "Q3":
            online = PerceivedOnline(proc)
            node_scene = online.scene
            # the perceived slot of the first target, found by its colour word
            tgt_text = next(o["text"] for o in sc["objects"] if o["node"] == tgt)
            tgt_policy = next(
                (o["node"] for o in node_scene["objects"] if o["text"] == tgt_text), None
            )
        else:
            online = OnlineGraph(
                sc, "bound", quality, seed=ep, targets=targets, counterfactual_to=cf_to
            )
            node_scene, tgt_policy = sc, tgt
        all_actors = [o["actor"] for o in scene["objects"]]
        active_t = ActiveTarget(targets, scene["cube_to_bin"])
        feat_src = (
            OnlineFeatures(proc, sc) if (interface == "A3feat" and quality != "Q3") else None
        )
        policy.reset()

        def feed(o, f_prev, feats_prev):
            f, feats = f_prev, feats_prev
            for i in range(len(o["front_rgb_list"])):
                rgb = _np(o["front_rgb_list"][i], np.uint8)
                dm = _np(o["front_depth_list"][i], np.float32).squeeze() / 1000.0
                E = _np(o["front_camera_extrinsic_list"][i], np.float32).reshape(3, 4)
                gw = float(np.sum(_np(o["gripper_state_list"][i])))
                ee = _np(o["eef_state_list"][i], np.float32).reshape(-1)[:3]
                seg = _seg_from_maniskill(o["maniskill_obs"][i], "base_camera")
                if quality == "Q3":
                    f = online.step(rgb, dm, K, E, gw, ee)
                    feats = online.feats.copy()
                else:
                    f = online.step(
                        rgb,
                        dm,
                        seg,
                        _np(o["wrist_depth_list"][i], np.float32).squeeze() / 1000.0,
                        _seg_from_maniskill(o["maniskill_obs"][i], "hand_camera"),
                        K,
                        E,
                        Kw,
                        _np(o["wrist_camera_extrinsic_list"][i], np.float32).reshape(3, 4),
                        gw,
                        ee,
                    )
                    if feat_src is not None:
                        feats = feat_src.step(rgb, seg)
            return f, feats

        def set_active():
            a = active_t.update(np.stack([actor_xyz(x) for x in all_actors]))
            if quality != "Q3":
                online.active = a

        set_active()
        f, feats = feed(obs, None, None)
        first_target_uv = None
        if f is not None and tgt_policy is not None:
            first_target_uv = f[tgt_policy, F_U : F_V + 1].tolist()
        status, steps, lifted = "ongoing", 0, None
        bin_actors = {o["node"]: o["actor"] for o in scene["objects"] if o["label_id"] == 0}
        z0 = {n: actor_xyz(a)[2] for n, a in bin_actors.items()}
        while status == "ongoing" and steps < max_steps:
            front = _np(obs["front_rgb_list"][-1], np.uint8)
            wrist = _np(obs["wrist_rgb_list"][-1], np.uint8)
            nodes = nodes_at(f[None], 0, node_scene)
            task_text = serialize(nodes, goal) if interface == "A2" else goal
            if interface == "A1":
                front = draw_markers(front, nodes)
            state = np.r_[
                _np(obs["joint_state_list"][-1]).reshape(-1)[:7],
                _np(obs["gripper_state_list"][-1]).reshape(-1)[:1],
            ].astype(np.float32)
            batch = {
                "image": torch.from_numpy(front).permute(2, 0, 1)[None].float() / 255.0,
                "wrist_image": torch.from_numpy(wrist).permute(2, 0, 1)[None].float() / 255.0,
                "state": torch.from_numpy(state)[None],
                "task": [task_text],
            }
            if tok is not None:
                x, lab, ids, mask = node_tensor(
                    nodes,
                    front.shape[1],
                    front.shape[0],
                    feats if interface == "A3feat" else None,
                )
                gp.current = tuple(
                    torch.from_numpy(a)[None].to(device) for a in (x, lab, ids, mask)
                )
            with torch.no_grad():
                act = post(policy.select_action(prep(pre, batch)))
            a = _np(act).reshape(-1)[:8]
            if save_strip is not None and ep == episode_offset and steps % strip_every == 0:
                strip.append(
                    (
                        steps,
                        draw_markers(_np(obs["front_rgb_list"][-1], np.uint8), nodes),
                        task_text,
                    )
                )
            obs, _, term, trunc, info = env.step(a)
            set_active()
            if len(obs["front_rgb_list"]):
                f, feats = feed(obs, f, feats)
            steps += 1
            if lifted is None:
                for n, actor in bin_actors.items():
                    if actor_xyz(actor)[2] - z0[n] > 0.03:
                        # judged against the first target, which is active until it is picked
                        lifted = n
                        break
            if bool(_np(term).reshape(-1)[0]) or bool(_np(trunc).reshape(-1)[0]):
                status = info.get("status", "unknown")
        last_bin_up = actor_xyz(bin_actors[active_t.bins[-1]])[2] > PICK_Z
        targets_picked = active_t.k + int(active_t.k == len(targets) - 1 and last_bin_up)
        env.close()
        t_ep.append(time.time() - t0)
        results.append(
            dict(
                episode=ep,
                status=status,
                success=status == "success",
                steps=steps,
                targets=targets,
                target=tgt,
                true_bin=scene["cube_to_bin"].get(tgt),
                targets_picked=targets_picked,
                lifted_bin=lifted,
                counterfactual_bin=None if cf_to is None else cf_to[tgt],
                first_target_uv=first_target_uv,
                seconds=t_ep[-1],
            )
        )
        print(
            f"episode {ep}: {status} in {steps} steps ({t_ep[-1]:.0f} s); lifted container "
            f"{lifted} (true {scene['cube_to_bin'].get(tgt)}, graph-named "
            f"{cf_to[tgt] if counterfactual else 'n/a'}); targets {targets}"
        )
    if gp is not None:
        flow.embed_prefix = gp.orig
    k = sum(r["success"] for r in results)
    summary = dict(
        interface=interface,
        task=task,
        quality=quality,
        episodes=episodes,
        successes=k,
        rate=k / max(1, episodes),
        wilson95=wilson(k, episodes),
        counterfactual=counterfactual,
        seconds_per_episode=float(np.mean(t_ep)),
        results=results,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, default=float))
    if save_strip is not None and strip:
        import matplotlib.pyplot as plt

        import textwrap

        cols = min(6, len(strip))
        rows = -(-len(strip) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 4.6 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (s, img, txt) in zip(axes.flat, strip):
            ax.imshow(img)
            ax.set_title(f"step {s}", fontsize=8)
            # the policy's input text: the object list (the instruction is the same every step)
            objs = txt.split("Objects: ", 1)[-1]
            ax.text(0, -0.02, textwrap.fill(objs, 52), transform=ax.transAxes, fontsize=4.5,
                    va="top", ha="left", family="monospace")
        fig.suptitle(f"{interface} rollout, {task}, {quality} (markers = graph positions)")
        fig.tight_layout()
        fig.savefig(save_strip, dpi=130)
    return summary
