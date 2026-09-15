#!/usr/bin/env python3
"""Inspect a finished Section 3.1 run, one pipeline step at a time.

    python inspect_run.py --config configs/giraffe.yaml               # inventory + storyboard ep 0
    python inspect_run.py --config configs/giraffe.yaml --episode all --video

Writes under <output_dir>/:
    step_inventory.txt           per-step file counts, sizes, step4 shape checks
    storyboard/ep<i>_<name>.png  1 GroundingDINO -> 2 SAM -> 3 Cutie -> 4 DINO, one figure
    step3_track/ep<i>_overlay.mp4  (--video) Cutie overlays as a video

Only needs numpy / omegaconf / opencv / matplotlib, so it also runs on a
laptop against a downloaded output folder. Exit code 1 if a check fails.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from omegaconf import OmegaConf


# --------------------------------------------------------------------------- #
# config / naming helpers
# --------------------------------------------------------------------------- #
def load_config(path):
    path = os.path.abspath(path)
    task_cfg = OmegaConf.load(path)
    merged = OmegaConf.create({})
    for default in task_cfg.get("defaults", []):
        merged = OmegaConf.merge(
            merged, OmegaConf.load(os.path.join(os.path.dirname(path), f"{default}.yaml"))
        )
    return OmegaConf.merge(merged, task_cfg)


def resolve_classes(cfg, out_dir):
    """Classes + slot map. LLM configs have no classes in the yaml: take them
    from the run summary (what the run actually used), else the entity cache."""
    classes = cfg.task.get("classes")
    summary_path = os.path.join(out_dir, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            classes = json.load(handle)["classes"]
    if not classes:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        cache = os.path.join(os.path.dirname(os.path.abspath(args.config)), f"{stem}.entities.json")
        if os.path.exists(cache):
            with open(cache) as handle:
                classes = json.load(handle).get("classes")
    if not classes:
        sys.exit("could not determine class list (no run_summary.json / entity cache)")
    classes = list(classes)
    slot_cfg = cfg.task.get("slot_by_class")
    if slot_cfg:
        slots = {i: int(slot_cfg.get(i, slot_cfg.get(str(i)))) for i in range(len(classes))}
    else:
        slots = {i: i + 1 for i in range(len(classes))}
    return classes, slots


PATTERNS = {
    "step1_detect": [
        ("raw detections (.npz)", re.compile(r"^(\d+)_(\d+)_(\d+)\.npz$")),
        ("annotated boxes (.png)", re.compile(r"^(\d+)_(\d+)_(\d+)_boxes\.png$")),
    ],
    "step2_segment": [
        ("SAM seed masks", re.compile(r"^(\d+)_(\d+)_(\d+)_seed\.png$")),
        ("SAM seed overlays", re.compile(r"^(\d+)_(\d+)_(\d+)_seed_overlay\.png$")),
    ],
    "step3_track": [
        ("Cutie class masks", re.compile(r"^(\d+)_(\d+)_(\d+)_mask\.png$")),
        ("Cutie overlays", re.compile(r"^(\d+)_(\d+)_overlay\.png$")),
    ],
}


def _du(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def list_step(out_dir, step, pattern, episode=None):
    """[(episode, frame, class_or_None, filename)] sorted by (episode, frame)."""
    folder = os.path.join(out_dir, step)
    if not os.path.isdir(folder):
        return []
    rows = []
    for name in os.listdir(folder):
        match = pattern.match(name)
        if not match:
            continue
        groups = [int(g) for g in match.groups()]
        e, f = groups[0], groups[1]
        c = groups[2] if len(groups) > 2 else None
        if episode is None or e == episode:
            rows.append((e, f, c, os.path.join(folder, name)))
    return sorted(rows)


# --------------------------------------------------------------------------- #
# 1. inventory
# --------------------------------------------------------------------------- #
def inventory(out_dir, classes, feature_dim, episodes):
    lines, ok = [], True
    lines.append(f"output_dir: {out_dir}")
    lines.append(f"classes   : {classes}  (N={len(classes)} -> {len(classes) + 1} slots)")
    for step, patterns in PATTERNS.items():
        folder = os.path.join(out_dir, step)
        if not os.path.isdir(folder):
            lines.append(f"[{step}] MISSING")
            ok = False
            continue
        lines.append(f"[{step}] {_human(_du(folder))}")
        for label, pattern in patterns:
            rows = list_step(out_dir, step, pattern)
            eps = sorted({r[0] for r in rows})
            lines.append(f"    {label:<24} {len(rows):>7} files  episodes {eps}")
    # step1: summarize detection quality from the npz files
    npz_rows = list_step(out_dir, "step1_detect", PATTERNS["step1_detect"][0][1])
    if npz_rows:
        per_class = {}
        for _, _, c, path in npz_rows:
            data = np.load(path)
            stats = per_class.setdefault(c, {"calls": 0, "empty": 0, "accepted": 0, "conf": []})
            stats["calls"] += 1
            stats["empty"] += int(len(data["confidence"]) == 0)
            if "accepted_confidence" in data and np.isfinite(data["accepted_confidence"]):
                stats["accepted"] += 1
                stats["conf"].append(float(data["accepted_confidence"]))
        for c, st in sorted(per_class.items()):
            name = classes[c] if c < len(classes) else c
            conf = f"mean kept conf {np.mean(st['conf']):.2f}" if st["conf"] else "no kept boxes"
            lines.append(
                f"    GDINO '{name}': {st['calls']} detect calls, {st['empty']} empty, "
                f"{st['accepted']} accepted -> SAM; {conf}"
            )
    rerun_dir = os.path.join(out_dir, "rerun")
    if os.path.isdir(rerun_dir):
        rrds = sorted(f for f in os.listdir(rerun_dir) if f.endswith(".rrd"))
        lines.append(
            f"[rerun] {len(rrds)} timeline file(s), {_human(_du(rerun_dir))}  "
            "(open: rerun <file>.rrd)"
        )
    # step4
    embed_root = os.path.join(out_dir, "step4_embed")
    if not os.path.isdir(embed_root):
        lines.append("[step4_embed] MISSING")
        return "\n".join(lines), False
    lines.append(f"[step4_embed] {_human(_du(embed_root))}")
    names = [episodes[k] for k in sorted(episodes, key=int)] if episodes else sorted(os.listdir(embed_root))
    for idx, name in enumerate(names):
        edir = os.path.join(embed_root, name)
        if not os.path.isdir(edir):
            lines.append(f"    ep{idx} {name}: MISSING")
            ok = False
            continue
        pr = np.load(os.path.join(edir, "episode_pr.npy"), mmap_mode="r")
        bb = np.load(os.path.join(edir, "episode_bboxes.npy"), mmap_mode="r")
        problems = []
        if pr.ndim != 3 or pr.shape[1] != len(classes) + 1 or pr.shape[2] != feature_dim:
            problems.append(f"pr shape != (T,{len(classes) + 1},{feature_dim})")
        if bb.shape[:2] != pr.shape[:2] or bb.shape[2] != 7:
            problems.append("bboxes shape mismatch")
        if not np.isfinite(np.asarray(pr)).all() or not np.isfinite(np.asarray(bb)).all():
            problems.append("non-finite values")
        full = os.path.join(edir, "episode_dino_full.npy")
        full_txt = f" dino_full {np.load(full, mmap_mode='r').shape}" if os.path.exists(full) else ""
        present = np.abs(np.asarray(bb)[:, 1:, :4]).sum(-1) > 0
        pres = " ".join(f"{100 * p:.0f}%" for p in present.mean(0))
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        ok &= not problems
        lines.append(
            f"    ep{idx} {name}: pr {pr.shape} bboxes {bb.shape}{full_txt} "
            f"presence[{pres}] {status}"
        )
    return "\n".join(lines), ok


# --------------------------------------------------------------------------- #
# 2. storyboard
# --------------------------------------------------------------------------- #
def _imread_rgb(path):
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _blank(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="0.4", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _show(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _pca_rgb(tokens):
    tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, tokens.shape[-1])
    side = int(round(np.sqrt(tokens.shape[0])))
    if side * side != tokens.shape[0]:  # drop CLS if present
        tokens = tokens[1:]
        side = int(round(np.sqrt(tokens.shape[0])))
    centered = tokens - tokens.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:3].T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return proj.reshape(side, side, 3)


def storyboard(out_dir, ep, ep_name, classes, slots, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["tab:red", "tab:green", "tab:blue", "tab:olive", "tab:purple", "tab:cyan"]
    n = len(classes)
    ncols = max(2 * n, 4)
    fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 11.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    # Row 1: GroundingDINO boxes -> SAM seed, per class, on the seed frame.
    seeds = list_step(out_dir, "step2_segment", PATTERNS["step2_segment"][1][1], ep)
    for c, name in enumerate(classes):
        ax_det, ax_seg = axes[0, 2 * c], axes[0, 2 * c + 1]
        ax_det.axis("on")
        ax_seg.axis("on")
        mine = [r for r in seeds if r[2] == c]
        if not mine:
            _blank(ax_det, f"1 GroundingDINO\n'{name}'\nnever detected")
            _blank(ax_seg, f"2 SAM\n'{name}'\nno seed")
            continue
        _, frame, _, seed_path = mine[0]
        box_path = os.path.join(out_dir, "step1_detect", f"{ep}_{frame}_{c}_boxes.png")
        if os.path.exists(box_path):
            _show(ax_det, _imread_rgb(box_path), f"1 GroundingDINO: {name}\nframe {frame}")
        else:
            _blank(ax_det, f"1 GroundingDINO\n{name} f{frame}\n(no boxes image)")
        _show(ax_seg, _imread_rgb(seed_path), f"2 SAM seed mask: {name}\nframe {frame}")

    # Row 2: Cutie propagation at evenly spaced frames.
    overlays = list_step(out_dir, "step3_track", PATTERNS["step3_track"][1][1], ep)
    if overlays:
        picks = np.linspace(0, len(overlays) - 1, num=min(ncols, len(overlays))).round().astype(int)
        for col, k in enumerate(picks):
            _, frame, _, path = overlays[k]
            axes[1, col].axis("on")
            _show(axes[1, col], _imread_rgb(path), f"3 Cutie track, frame {frame}/{overlays[-1][1]}")
    else:
        axes[1, 0].axis("on")
        _blank(axes[1, 0], "3 Cutie: no overlays saved")

    # Row 3: DINO outputs.
    edir = os.path.join(out_dir, "step4_embed", ep_name)
    pr_path = os.path.join(edir, "episode_pr.npy")
    if os.path.exists(pr_path):
        pr = np.load(pr_path).astype(np.float32)
        bb = np.load(os.path.join(edir, "episode_bboxes.npy")).astype(np.float32)
        present = np.abs(bb[:, 1:, :4]).sum(-1) > 0
        full_path = os.path.join(edir, "episode_dino_full.npy")

        ax = axes[2, 0]
        ax.axis("on")
        if os.path.exists(full_path):
            full = np.load(full_path, mmap_mode="r")
            _show(ax, _pca_rgb(np.asarray(full[0])), "4 DINO patch tokens, frame 0\n(PCA -> RGB)")
        else:
            _blank(ax, "4 no episode_dino_full.npy")

        ax = axes[2, 1]
        ax.axis("on")
        for c, name in enumerate(classes):
            s = slots[c]
            valid = present[:, s - 1]
            if not valid.any():
                continue
            ref = pr[np.argmax(valid), s]
            sim = pr[:, s] @ ref / (np.linalg.norm(pr[:, s], axis=1) * np.linalg.norm(ref) + 1e-8)
            sim[~valid] = np.nan
            ax.plot(sim, color=colors[c % len(colors)], label=name, lw=1.2)
        g = pr[:, 0]
        gsim = g @ g[0] / (np.linalg.norm(g, axis=1) * np.linalg.norm(g[0]) + 1e-8)
        ax.plot(gsim, color="0.5", ls="--", label="scene (slot 0)", lw=1)
        ax.set_title("4 node embedding: cos-sim to first frame", fontsize=10)
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)

        ax = axes[2, 2]
        ax.axis("on")
        for c, name in enumerate(classes):
            s = slots[c]
            valid = present[:, s - 1] & (np.abs(bb[:, s, 4:7]).sum(-1) > 0)
            if not valid.any():
                continue
            ref = bb[np.argmax(valid), s, 4:7]
            disp = np.linalg.norm(bb[:, s, 4:7] - ref, axis=1)
            disp[~valid] = np.nan
            ax.plot(disp, color=colors[c % len(colors)], label=name, lw=1.2)
        ax.set_title("3D centroid displacement (m)", fontsize=10)
        ax.set_xlabel("frame")
        ax.legend(fontsize=7)

        ax = axes[2, 3]
        ax.axis("on")
        order = [slots[c] - 1 for c in range(n)]
        ax.imshow(present[:, order].T, aspect="auto", cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(n))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("frame")
        ax.set_title(f"node present per frame  (pr {pr.shape})", fontsize=10)
    else:
        axes[2, 0].axis("on")
        _blank(axes[2, 0], f"4 no step4 output for {ep_name}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    dest_dir = os.path.join(out_dir, "storyboard")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"ep{ep}_{ep_name.replace(':', '-')}.png")
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def make_video(out_dir, ep, fps=15):
    import cv2

    overlays = list_step(out_dir, "step3_track", PATTERNS["step3_track"][1][1], ep)
    if not overlays:
        return None
    first = cv2.imread(overlays[0][3])
    h, w = first.shape[:2]
    dest = os.path.join(out_dir, "step3_track", f"ep{ep}_overlay.mp4")
    writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, frame, _, path in overlays:
        image = cv2.imread(path)
        cv2.putText(image, f"ep{ep} f{frame}", (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    parser.add_argument("--episode", nargs="+", default=["0"], help="episode indices, or 'all'")
    parser.add_argument("--video", action="store_true", help="also write step3 overlay mp4s")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output_dir or cfg.output.output_dir
    classes, slots = resolve_classes(cfg, out_dir)
    feature_dim = int(cfg.model.get("feature_dim", 768))
    episodes = {}
    if os.path.exists(os.path.join(out_dir, "episodes.json")):
        with open(os.path.join(out_dir, "episodes.json")) as handle:
            episodes = json.load(handle)
    elif os.path.isdir(os.path.join(out_dir, "step4_embed")):
        print("WARNING: no episodes.json; assuming sorted step4 folder order = episode index")
        episodes = {str(i): n for i, n in enumerate(sorted(os.listdir(os.path.join(out_dir, "step4_embed"))))}

    report, ok = inventory(out_dir, classes, feature_dim, episodes)
    with open(os.path.join(out_dir, "step_inventory.txt"), "w") as handle:
        handle.write(report + "\n")
    print(report)

    device = ""
    summary_path = os.path.join(out_dir, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            s = json.load(handle)
        device = f"  |  {s.get('device')}  {s.get('run_s', 0) / 60:.1f} min, {s.get('fps')} fps"

    wanted = sorted(episodes, key=int) if args.episode == ["all"] else args.episode
    for key in wanted:
        if key not in episodes:
            print(f"episode {key}: not in episodes.json, skipped")
            continue
        ep, name = int(key), episodes[key]
        title = f"{cfg.task.name}  |  ep{ep} {name}  |  backbone {cfg.model.vit_model}{device}"
        print("storyboard ->", storyboard(out_dir, ep, name, classes, slots, title))
        if args.video:
            print("video      ->", make_video(out_dir, ep))

    print("CHECKS", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
