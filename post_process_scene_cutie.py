import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import albumentations as A
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from .vit import get_vit
except ImportError:
    from vit import get_vit

try:
    from .extract_part_masks_cutie import build_tracker
except ImportError:
    from extract_part_masks_cutie import build_tracker

try:
    from . import stage_timer
    from .stage_timer import stage
except ImportError:
    import stage_timer
    from stage_timer import stage


@dataclass
class FrameOutput:
    representation: np.ndarray
    bboxes: np.ndarray
    dino_full: np.ndarray
    separated_representation: Optional[np.ndarray] = None
    separated_bboxes: Optional[np.ndarray] = None


def _indexed_get(mapping, index, default=None):
    if index in mapping:
        return mapping[index]
    if str(index) in mapping:
        return mapping[str(index)]
    return default


class HODORProcessor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda:0")
        self.classes = list(config.task.classes)
        self.num_classes = len(self.classes)
        self.feature_dim = int(config.model.get("feature_dim", 768))

        vit_model = str(config.model.vit_model)
        if not vit_model.startswith("dinov2"):
            print(
                f"WARNING: model.vit_model={vit_model!r} is not DINOv2; the paper "
                "(and this repo's default) uses DINOv2 ViT-B/14."
            )
        self.vit_encoder, encoder_dim = get_vit(
            config.paths.home_path,
            vit_model,
            int(config.model.vit_image_size),
            global_pool=True,
        )
        if encoder_dim != self.feature_dim:
            raise ValueError(
                f"model.feature_dim={self.feature_dim} but {vit_model} outputs "
                f"{encoder_dim}-d features; set feature_dim: {encoder_dim}"
            )
        self.vit_encoder.cuda()
        self.vit_encoder.eval()
        self.vit_encoder.freeze()
        self.image_preprocessing = A.Compose(
            [
                A.Resize(
                    height=int(config.model.vit_image_size),
                    width=int(config.model.vit_image_size),
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.slot_by_class = {
            class_index: int(_indexed_get(config.task.slot_by_class, class_index))
            for class_index in range(self.num_classes)
        }
        expected_slots = set(range(1, self.num_classes + 1))
        if set(self.slot_by_class.values()) != expected_slots:
            raise ValueError(
                "task.slot_by_class must map every class exactly once to slots "
                f"1..{self.num_classes}; got {self.slot_by_class}"
            )

        self.tracker = build_tracker(config)
        # Rerun timeline (one .rrd per episode), see rerun_logger.py.
        self.viz = None
        if bool(config.output.get("save_rerun", False)):
            try:
                from .rerun_logger import RerunLogger
            except ImportError:
                from rerun_logger import RerunLogger
            self.viz = RerunLogger(config)
        self.tracker.viz = self.viz
        self.final_rp = None
        self.final_bboxes = None
        self.final_rp_separated = None
        self.final_bboxes_separated = None

    def _reset_episode_state(self):
        self.final_rp = torch.zeros(
            (self.num_classes + 1, self.feature_dim),
            dtype=torch.float32,
        )
        self.final_bboxes = np.zeros(
            (self.num_classes + 1, 7),
            dtype=np.float32,
        )
        self.final_bboxes[0, :4] = [0.0, 1.0, 0.0, 1.0]

        separated = self.config.task.get("separated")
        if separated is not None and bool(separated.get("enabled", False)):
            num_slots = int(separated.num_slots)
            self.final_rp_separated = torch.zeros(
                (num_slots, self.feature_dim),
                dtype=torch.float32,
            )
            self.final_bboxes_separated = np.zeros((num_slots, 7), dtype=np.float32)
            self.final_bboxes_separated[0, :4] = [0.0, 1.0, 0.0, 1.0]
        else:
            self.final_rp_separated = None
            self.final_bboxes_separated = None

    def _mask_to_instances(self, label_mask):
        height, width = label_mask.shape
        labels: List[int] = []
        masks = []
        bboxes = [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]

        for class_index in range(self.num_classes):
            class_mask = label_mask == class_index
            if not class_mask.any():
                continue
            rows, columns = np.where(class_mask)
            bboxes.append(
                [
                    float(rows.min()) / height,
                    float(rows.max()) / height,
                    float(columns.min()) / width,
                    float(columns.max()) / width,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            labels.append(class_index)
            masks.append(torch.from_numpy(class_mask.astype(np.float32)))

        if masks:
            binary_masks = torch.stack(masks, dim=0).unsqueeze(0)
        else:
            binary_masks = torch.zeros((1, 0, height, width), dtype=torch.float32)
        return labels, binary_masks, np.asarray(bboxes, dtype=np.float32)

    def _obtain_features(self, image_bgr, instance_masks):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        normalized = self.image_preprocessing(image=image_rgb)["image"]
        image_tensor = (
            torch.from_numpy(normalized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.amp.autocast(device_type="cuda", enabled=True):
            with torch.no_grad():
                global_repr, patch_repr = self.vit_encoder(image_tensor)
        dino_full = patch_repr.detach().cpu().numpy()

        num_instances = instance_masks.shape[1]
        if num_instances == 0:
            return global_repr.unsqueeze(1).cpu(), dino_full

        token_count = patch_repr.shape[1]
        token_side = int(round(token_count**0.5))
        if token_side * token_side != token_count:
            raise ValueError(f"Expected square ViT token grid, got {token_count} tokens")

        token_masks = F.adaptive_max_pool2d(
            instance_masks.float(),
            output_size=(token_side, token_side),
        )
        token_masks = token_masks.reshape(1, num_instances, token_count).to(self.device)
        repeated_features = patch_repr.unsqueeze(1).expand(
            -1,
            num_instances,
            -1,
            -1,
        )
        masked = repeated_features.float().masked_fill(
            ~(token_masks.unsqueeze(-1) > 0.5),
            float("nan"),
        )
        object_repr = masked.nanmean(dim=2)
        object_repr = torch.where(
            torch.isnan(object_repr),
            torch.zeros_like(object_repr),
            object_repr,
        )
        return torch.cat([global_repr.unsqueeze(1), object_repr], dim=1).cpu(), dino_full

    def _depth_to_pointcloud(
        self,
        depth,
        rgb,
        intrinsics,
        extrinsics,
        binary_mask,
    ):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2] - 150, intrinsics[1, 2]
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth.reshape(-1)
        valid = (
            (z > float(self.config.data.depth_min_m))
            & (z < float(self.config.data.depth_max_m))
            & (binary_mask.reshape(-1) > 0)
        )
        if not valid.any():
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        u = u[valid]
        v = v[valid]
        z = z[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.column_stack([x, y, z])

        extrinsics = np.asarray(extrinsics)
        if extrinsics.shape == (4, 4):
            transform = extrinsics
        else:
            tx, ty, tz, rx, ry, rz = extrinsics.reshape(-1)[:6]
            transform = np.eye(4)
            transform[:3, :3] = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
            transform[:3, 3] = [tx, ty, tz]
        homogeneous = np.column_stack([points, np.ones(len(points))])
        points = (transform @ homogeneous.T).T[:, :3]
        return points, rgb[v, u, :]

    def _add_centroids(self, binary_masks, bboxes, depth, hand_rgb_unrotated, intrinsics, extrinsics):
        for instance_index in range(binary_masks.shape[1]):
            resized = F.interpolate(
                binary_masks[:, instance_index : instance_index + 1].float(),
                size=(800, 800),
                mode="nearest",
            )[0, 0]
            # The tracker sees a 180-degree-rotated wrist image. Rotate the mask
            # back to align it with unrotated RGB and depth.
            unrotated_mask = np.rot90(resized.numpy(), k=2)
            points, _ = self._depth_to_pointcloud(
                depth,
                hand_rgb_unrotated,
                intrinsics,
                extrinsics,
                unrotated_mask,
            )
            if len(points) > 0:
                bboxes[instance_index + 1, 4:7] = points.mean(axis=0)

    def _update_separated(
        self,
        class_index,
        feature,
        bbox,
        image_height,
    ):
        if self.final_rp_separated is None:
            return
        separated = self.config.task.separated
        if class_index == 0:
            slot = int(separated.bowl_slot)
        elif class_index == 1:
            vertical_extent = (bbox[1] - bbox[0]) * image_height
            if vertical_extent > float(separated.stuffed_extent_threshold_px):
                slot = int(separated.giraffe_slot)
            else:
                slot = int(separated.chicken_slot)
        elif class_index == 2:
            slot = int(separated.pineapple_slot)
        elif class_index == 3:
            slot = int(separated.ball_slot)
        else:
            return
        self.final_rp_separated[slot] = feature
        self.final_bboxes_separated[slot] = bbox

    def process_frame(
        self,
        hand_bgr_rotated,
        frame_num,
        episode_num,
        depth,
        hand_rgb_unrotated,
        intrinsics,
        extrinsics,
    ):
        if frame_num == 0:
            self._reset_episode_state()
        self.tracker.episode_num = episode_num
        if self.viz is not None:
            self.viz.begin_frame(frame_num, hand_bgr_rotated)

        label_mask = self.tracker.process(hand_bgr_rotated, frame_num)
        labels, binary_masks, bboxes = self._mask_to_instances(label_mask)
        with stage("centroid"):
            self._add_centroids(
                binary_masks,
                bboxes,
                depth,
                hand_rgb_unrotated,
                intrinsics,
                extrinsics,
            )
        with stage("embed"):
            frame_repr, dino_full = self._obtain_features(hand_bgr_rotated, binary_masks)

        self.final_rp[0] = frame_repr[0, 0]
        self.final_bboxes[1:, :4] = 0.0
        if self.final_rp_separated is not None:
            self.final_rp_separated[0] = frame_repr[0, 0]
            self.final_bboxes_separated[1:, :4] = 0.0

        for instance_index, class_index in enumerate(labels):
            feature = frame_repr[0, instance_index + 1]
            bbox = bboxes[instance_index + 1]
            slot = self.slot_by_class[class_index]
            self.final_rp[slot] = feature
            self.final_bboxes[slot] = bbox
            self._update_separated(
                class_index,
                feature,
                bbox,
                label_mask.shape[0],
            )

        if self.viz is not None:
            self.viz.end_frame(label_mask, self.final_rp, self.final_bboxes, dino_full)

        return FrameOutput(
            representation=self.final_rp.numpy().copy(),
            bboxes=self.final_bboxes.copy(),
            dino_full=dino_full,
            separated_representation=(
                None
                if self.final_rp_separated is None
                else self.final_rp_separated.numpy().copy()
            ),
            separated_bboxes=(
                None
                if self.final_bboxes_separated is None
                else self.final_bboxes_separated.copy()
            ),
        )


def load_config(config_path):
    config_path = os.path.abspath(config_path)
    task_config = OmegaConf.load(config_path)
    merged = OmegaConf.create({})
    for default in task_config.get("defaults", []):
        if not isinstance(default, str):
            raise ValueError(f"Only string defaults are supported, got {default!r}")
        default_path = os.path.join(os.path.dirname(config_path), f"{default}.yaml")
        if not os.path.exists(default_path):
            raise FileNotFoundError(default_path)
        merged = OmegaConf.merge(merged, OmegaConf.load(default_path))
    merged = OmegaConf.merge(merged, task_config)
    OmegaConf.resolve(merged)
    _maybe_resolve_entities(
        merged,
        os.path.dirname(config_path),
        os.path.splitext(os.path.basename(config_path))[0],
    )
    return merged


def _maybe_resolve_entities(config, config_dir, config_stem):
    """Section 3.1 LLM front-end: when task.entities.source == 'llm', derive the
    GroundingDINO class list (and slot mapping) from the language instruction via
    an LLM instead of using a hardcoded task.classes. Manual configs are left
    untouched (no entities block, or source == 'manual')."""
    task = config.get("task")
    entities_cfg = task.get("entities") if task is not None else None
    if not entities_cfg or str(entities_cfg.get("source", "manual")).lower() != "llm":
        return

    try:
        from .llm_entities import resolve_entities, build_slot_by_class
    except ImportError:
        from llm_entities import resolve_entities, build_slot_by_class

    instruction = task.get("language_instruction")
    if not instruction:
        raise ValueError(
            "task.entities.source='llm' requires task.language_instruction."
        )

    cfg = OmegaConf.to_container(entities_cfg, resolve=True)
    if not cfg.get("cache_file"):
        cfg["cache_file"] = f"{config_stem}.entities.json"

    resolved = resolve_entities(cfg, instruction, config_dir)

    OmegaConf.set_struct(config, False)
    task.classes = list(resolved["classes"])
    if task.get("slot_by_class") is None:
        task.slot_by_class = build_slot_by_class(len(resolved["classes"]))
    print(
        f"[entities] task '{task.get('name')}': {list(resolved['classes'])} "
        f"(provider={resolved.get('provider')}, model={resolved.get('model')})"
    )


def _preprocess_wrist_bgr(raw_bgr, final_size):
    if raw_bgr.shape[0] < 720 or raw_bgr.shape[1] < 1050:
        raise ValueError(f"Unexpected wrist image shape: {raw_bgr.shape}")
    image = Image.fromarray(raw_bgr).crop((150, 0, 1050, 720))
    width, height = image.size
    maximum = max(width, height)
    padding = (
        (maximum - width) // 2,
        (maximum - height) // 2,
        (maximum - width + 1) // 2,
        (maximum - height + 1) // 2,
    )
    image = ImageOps.expand(image, padding, fill=(0, 0, 0))
    resize_target = int(round(final_size * 1.125))
    crop_offset = (resize_target - final_size) // 2
    image = image.resize((resize_target, resize_target), resample=Image.BICUBIC)
    image = image.crop(
        (
            crop_offset,
            crop_offset,
            crop_offset + final_size,
            crop_offset + final_size,
        )
    )
    return np.asarray(image)


def _preprocess_depth(depth_mm):
    depth = depth_mm[:, 150:1050].astype(np.float32) / 1000.0
    depth = np.pad(
        depth,
        ((90, 90), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    return depth[50:850, 50:850]


def process_episode(processor, config, episode_path, episode_num):
    try:
        from .droid_utils import load_trajectory
    except ImportError:
        from droid_utils import load_trajectory

    trajectory_path = os.path.join(episode_path, "trajectory.h5")
    depth_path = os.path.join(
        episode_path,
        "recordings",
        "Depth",
        "wrist_camera_left_depth.h5",
    )
    mp4_path = os.path.join(episode_path, "recordings", "MP4")
    if not all(
        [
            os.path.exists(trajectory_path),
            os.path.exists(depth_path),
            os.path.isdir(mp4_path),
        ]
    ):
        print(f"Skipping incomplete episode: {episode_path}")
        return False

    with stage("load"):
        with h5py.File(depth_path, "r") as depth_file:
            depth_frames = depth_file["depth_frames"][:]
        data = load_trajectory(
            trajectory_path,
            recording_folderpath=mp4_path,
        )
    if len(data) == 0:
        print(f"Skipping empty episode: {episode_path}")
        return False
    viz = getattr(processor, "viz", None)
    if viz is not None:
        viz.start_episode(episode_num, os.path.basename(os.path.normpath(episode_path)))

    episode_pr = []
    episode_bboxes = []
    episode_pr_separated = []
    episode_bboxes_separated = []
    episode_dino_full = []
    final_size = int(config.data.preprocess_final_size)

    for frame_num, step in enumerate(data):
        if frame_num >= len(depth_frames):
            print(f"Stopping at frame {frame_num}: depth stream is shorter than trajectory")
            break
        observation = step["observation"]
        wrist_ids = [
            key
            for key, camera_type in observation["camera_type"].items()
            if camera_type == 0
        ]
        if not wrist_ids:
            raise ValueError("No wrist camera with camera_type == 0")
        wrist_key = f"{wrist_ids[0]}_left"
        raw_bgr = observation["image"][wrist_key]

        with stage("preprocess"):
            wrist_bgr_unrotated = _preprocess_wrist_bgr(raw_bgr, final_size)
            wrist_bgr_rotated = np.ascontiguousarray(
                np.rot90(wrist_bgr_unrotated, k=2)
            )
            hand_bgr_800 = np.asarray(
                Image.fromarray(wrist_bgr_unrotated).resize(
                    (800, 800),
                    resample=Image.BICUBIC,
                )
            )
            hand_rgb_800 = cv2.cvtColor(hand_bgr_800, cv2.COLOR_BGR2RGB)
            depth = _preprocess_depth(depth_frames[frame_num])

        output = processor.process_frame(
            hand_bgr_rotated=wrist_bgr_rotated,
            frame_num=frame_num,
            episode_num=episode_num,
            depth=depth,
            hand_rgb_unrotated=hand_rgb_800,
            intrinsics=observation["camera_intrinsics"][wrist_key],
            extrinsics=observation["camera_extrinsics"][wrist_key],
        )
        episode_pr.append(output.representation)
        episode_bboxes.append(output.bboxes)
        if output.separated_representation is not None:
            episode_pr_separated.append(output.separated_representation)
            episode_bboxes_separated.append(output.separated_bboxes)
        if bool(config.output.get("save_dino_full", False)):
            episode_dino_full.append(output.dino_full)

    if viz is not None:
        viz.end_episode()
    if not episode_pr:
        return False
    # Final embed-stage arrays go in step4_embed/<episode>/ under output_dir,
    # so every stage of a run lives together under one folder tree.
    embed_dir = os.path.join(
        config.output.output_dir,
        "step4_embed",
        os.path.basename(os.path.normpath(episode_path)),
    )
    os.makedirs(embed_dir, exist_ok=True)
    np.save(os.path.join(embed_dir, "episode_pr.npy"), np.asarray(episode_pr))
    np.save(
        os.path.join(embed_dir, "episode_bboxes.npy"),
        np.asarray(episode_bboxes),
    )
    if episode_pr_separated:
        np.save(
            os.path.join(embed_dir, "episode_pr_seperated.npy"),
            np.asarray(episode_pr_separated),
        )
        np.save(
            os.path.join(embed_dir, "episode_bboxes_seperated.npy"),
            np.asarray(episode_bboxes_separated),
        )
    if episode_dino_full:
        np.save(
            os.path.join(embed_dir, "episode_dino_full.npy"),
            np.asarray(episode_dino_full),
        )
    return embed_dir


def _episode_record(episode_num, episode_path, embed_dir, wall, processor):
    record = {
        "episode_index": episode_num,
        "episode": os.path.basename(os.path.normpath(episode_path)),
        "ok": bool(embed_dir),
        "wall_s": round(wall, 2),
        "stages": stage_timer.totals(),
        "seed_events": list(getattr(processor.tracker, "seed_events", [])),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
    }
    if embed_dir:
        pr = np.load(os.path.join(embed_dir, "episode_pr.npy"), mmap_mode="r")
        bb = np.load(os.path.join(embed_dir, "episode_bboxes.npy"))
        record["frames"] = int(pr.shape[0])
        viz = getattr(processor, "viz", None)
        if viz is not None and viz.paths:
            record["rerun_file"] = viz.paths[-1]
        record["pr_shape"] = list(pr.shape)
        record["bboxes_shape"] = list(bb.shape)
        # A slot is "present" in a frame when its 2D box was written this frame.
        present = np.abs(bb[:, 1:, :4]).sum(axis=-1) > 0
        record["presence"] = {
            name: round(float(present[:, processor.slot_by_class[i] - 1].mean()), 3)
            for i, name in enumerate(processor.classes)
        }
        dino_full_path = os.path.join(embed_dir, "episode_dino_full.npy")
        if os.path.exists(dino_full_path):
            record["dino_full_shape"] = list(
                np.load(dino_full_path, mmap_mode="r").shape
            )
    return record


def _format_episode(record):
    head = f"[ep {record['episode_index']} {record['episode']}]"
    if not record["ok"]:
        return f"{head} SKIPPED/FAILED after {record['wall_s']}s"
    frames = record["frames"]
    fps = frames / max(record["wall_s"], 1e-6)
    lines = [
        f"{head} T={frames} frames, {fps:.2f} fps, peak GPU {record['peak_gpu_mem_gb']} GB",
        "    time: " + stage_timer.format_line(record["stages"], record["wall_s"]),
        f"    step4 out: pr {tuple(record['pr_shape'])}  bboxes {tuple(record['bboxes_shape'])}"
        + (
            f"  dino_full {tuple(record['dino_full_shape'])}"
            if "dino_full_shape" in record
            else ""
        ),
        "    presence: "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in record["presence"].items()),
    ]
    seeded = {event["class"] for event in record["seed_events"]}
    missing = [name for name in record["presence"] if name not in seeded]
    if missing:
        lines.append(f"    WARNING never detected: {missing}")
    if record.get("rerun_file"):
        size_mb = os.path.getsize(record["rerun_file"]) / 1e6
        lines.append(f"    rerun: {record['rerun_file']} ({size_mb:.1f} MB)")
    return "\n".join(lines)


def main(config_path):
    import json
    import time

    try:
        from .droid_utils import crawler
    except ImportError:
        from droid_utils import crawler

    config = load_config(config_path)
    output_dir = config.output.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 78)
    print(f"task          : {config.task.name}")
    print(f"config        : {config_path}")
    print(f"instruction   : {config.task.get('language_instruction')}")
    print(f"classes       : {list(config.task.classes)}")
    print(f"slot_by_class : {dict(config.task.slot_by_class)}")
    print(f"backbone      : {config.model.vit_model} @ {config.model.vit_image_size}px")
    print(f"data_path     : {config.task.data_path}")
    print(f"output_dir    : {output_dir}")
    print(f"device        : {gpu}")
    print("pipeline      : 1 GroundingDINO -> 2 SAM ViT-H -> 3 Cutie -> 4 DINO embed")
    print(f"rerun viewer  : {'ON -> ' + os.path.join(output_dir, 'rerun') if config.output.get('save_rerun', False) else 'off'}")
    print("=" * 78)

    t_init = time.perf_counter()
    processor = HODORProcessor(config)
    init_s = time.perf_counter() - t_init
    print(f"models loaded in {init_s:.1f}s")

    episodes = [
        path
        for path in crawler(config.task.data_path)
        if os.path.exists(os.path.join(path, "trajectory.h5"))
    ]
    print(f"Found {len(episodes)} episodes")
    # step1-3 files are keyed by episode INDEX, step4 by episode NAME: save the map.
    with open(os.path.join(output_dir, "episodes.json"), "w") as handle:
        json.dump(
            {str(i): os.path.basename(os.path.normpath(p)) for i, p in enumerate(episodes)},
            handle,
            indent=2,
        )

    records = []
    t_run = time.perf_counter()
    for episode_num, episode_path in tqdm(
        list(enumerate(episodes)),
        desc=config.task.name,
    ):
        stage_timer.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_episode = time.perf_counter()
        embed_dir = process_episode(
            processor,
            config,
            episode_path,
            episode_num,
        )
        record = _episode_record(
            episode_num, episode_path, embed_dir, time.perf_counter() - t_episode, processor
        )
        records.append(record)
        tqdm.write(_format_episode(record))

    run_s = time.perf_counter() - t_run
    stage_sum = {}
    for record in records:
        for key, value in record["stages"].items():
            stage_sum.setdefault(key, {"seconds": 0.0, "calls": 0})
            stage_sum[key]["seconds"] = round(stage_sum[key]["seconds"] + value["seconds"], 3)
            stage_sum[key]["calls"] += value["calls"]
    total_frames = sum(r.get("frames", 0) for r in records)
    summary = {
        "task": config.task.name,
        "config": os.path.abspath(config_path),
        "device": gpu,
        "torch": torch.__version__,
        "classes": list(config.task.classes),
        "backbone": config.model.vit_model,
        "episodes_ok": sum(r["ok"] for r in records),
        "episodes_total": len(records),
        "frames_total": total_frames,
        "model_load_s": round(init_s, 2),
        "run_s": round(run_s, 2),
        "fps": round(total_frames / max(run_s, 1e-6), 3),
        "stages": stage_sum,
        "episodes": records,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 78)
    print(
        f"DONE {config.task.name}: {summary['episodes_ok']}/{len(records)} episodes, "
        f"{total_frames} frames in {run_s / 60:.1f} min ({summary['fps']:.2f} fps) on {gpu}"
    )
    print("stage totals: " + stage_timer.format_line(stage_sum, run_s))
    print(f"summary -> {os.path.join(output_dir, 'run_summary.json')}")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    main(arguments.config)
