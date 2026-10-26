"""Persistent perception worker for sg_pilot (runs in the sg-percep venv).

The repo's HODOR pipeline needs transformers 4.44 (GroundingDINO's BertModelWarper), SmolVLA needs
transformers 5, so the two cannot share a process. This worker loads GroundingDINO, SAM, Cutie and
DINOv2 once and then serves requests from sg_pilot.PerceptionWorker over a Unix socket until told to
exit or the client goes away.

    python perception_worker.py <socket path>      (authkey in $SG_WORKER_AUTHKEY; started by sg_pilot)

Ops (sg_pilot.pack_msg / unpack_msg framing):
    ping                        -> versions
    configure {config}          -> handle id, classes, num_classes, feature_dim
    process_frame {handle, ...} -> representation [S+1, D], bboxes [S+1, 7], optional label_mask,
                                   seed events of this frame, compute time
    features {handle} bgr masks -> representation [N+1, D]  (HODORProcessor._obtain_features)
    stats                       -> GPU memory
    exit
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PILOT_DIR.parents[1]
sys.path[:0] = [str(PILOT_DIR), str(REPO_ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sg_pilot as sg  # noqa: E402


def memoize_model_loaders() -> None:
    """Make every heavy model load happen once per worker, whatever the class configuration.

    The repo builds its models inside constructors (GroundedSAMDetector, CutieTracker,
    HODORProcessor), so a new class list would reload them all. These wrappers return the already
    loaded model for the same arguments; repo files are not touched.
    """
    import cutie.utils.get_default_model as cutie_default
    import groundingdino.util.inference as gdino_inference
    import post_process_scene_cutie as ppsc
    import segment_anything

    cache: dict = {}

    def memo(name, fn):
        def wrapped(*args, **kwargs):
            key = (name, args, tuple(sorted(kwargs.items())))
            if key not in cache:
                t0 = time.perf_counter()
                cache[key] = fn(*args, **kwargs)
                print(f"[worker] loaded {name} in {time.perf_counter() - t0:.1f} s", flush=True)
            return cache[key]

        return wrapped

    # Model must stay a class: GroundingDINO's own code calls Model.preprocess_image (a staticmethod)
    class CachedModel(gdino_inference.Model):
        _instances: dict = {}

        def __new__(cls, *args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
            return cls._instances[key]

        def __init__(self, *args, **kwargs):
            if getattr(self, "_loaded", False):
                return
            t0 = time.perf_counter()
            super().__init__(*args, **kwargs)
            self._loaded = True
            print(f"[worker] loaded GroundingDINO in {time.perf_counter() - t0:.1f} s", flush=True)

    gdino_inference.Model = CachedModel
    for k, builder in list(segment_anything.sam_model_registry.items()):
        segment_anything.sam_model_registry[k] = memo(f"SAM {k}", builder)
    # get_default_model also calls hydra.initialize, which may run only once per process
    cutie_default.get_default_model = memo("Cutie", cutie_default.get_default_model)
    ppsc.get_vit = memo("DINOv2", ppsc.get_vit)


class Server:
    def __init__(self):
        from omegaconf import OmegaConf

        self.OmegaConf = OmegaConf
        self.Processor = sg.robomme_processor_class()
        self.handles: dict[int, object] = {}
        self.by_config: dict[str, int] = {}
        self.last_mask: dict[int, np.ndarray] = {}

    def configure(self, meta, arrays):
        key = json.dumps(meta["config"], sort_keys=True)
        if key not in self.by_config:
            proc = self.Processor(self.OmegaConf.create(meta["config"]))
            hid = len(self.handles)
            orig = proc.tracker.process

            def recording_process(image_bgr, frame_index, _hid=hid, _orig=orig):
                out = _orig(image_bgr, frame_index)
                self.last_mask[_hid] = out
                return out

            proc.tracker.process = recording_process
            self.handles[hid], self.by_config[key] = proc, hid
        hid = self.by_config[key]
        proc = self.handles[hid]
        return dict(
            handle=hid,
            classes=list(proc.classes),
            num_classes=proc.num_classes,
            feature_dim=proc.feature_dim,
        ), {}

    def process_frame(self, meta, arrays):
        proc = self.handles[meta["handle"]]
        t0 = time.perf_counter()
        n_seed = 0 if meta["frame_num"] == 0 else len(proc.tracker.seed_events)
        out = proc.process_frame(
            hand_bgr_rotated=arrays["bgr"],
            frame_num=meta["frame_num"],
            episode_num=meta["episode_num"],
            depth=arrays["depth"],
            hand_rgb_unrotated=arrays["rgb"],
            intrinsics=arrays["K"],
            extrinsics=arrays["extr"],
        )
        res = dict(representation=out.representation, bboxes=out.bboxes)
        if meta.get("return_mask"):
            res["label_mask"] = self.last_mask[meta["handle"]].astype(np.int32)
        return dict(
            seed_events=proc.tracker.seed_events[n_seed:],
            compute_s=time.perf_counter() - t0,
        ), res

    def features(self, meta, arrays):
        proc = self.handles[meta["handle"]]
        masks = torch.from_numpy(arrays["masks"]).float()[None]
        rep, _ = proc._obtain_features(arrays["bgr"], masks)
        return {}, dict(representation=rep[0].numpy())

    def ping(self, meta, arrays):
        import transformers

        return dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
        ), {}

    def stats(self, meta, arrays):
        return dict(
            gpu_allocated_GB=torch.cuda.memory_allocated() / 1e9,
            gpu_max_allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
            gpu_reserved_GB=torch.cuda.memory_reserved() / 1e9,
            handles=len(self.handles),
        ), {}


def main() -> None:
    sock = sys.argv[1]
    conn = Client(sock, family="AF_UNIX", authkey=bytes.fromhex(os.environ["SG_WORKER_AUTHKEY"]))
    memoize_model_loaders()
    server = Server()
    print(f"[worker] ready, pid {os.getpid()}", flush=True)
    while True:
        try:
            meta, arrays = sg.unpack_msg(conn.recv_bytes())
        except (EOFError, OSError):
            break  # client gone
        op = meta.pop("op")
        if op == "exit":
            break
        try:
            out_meta, out_arrays = getattr(server, op)(meta, arrays)
        except Exception:  # noqa: BLE001
            out_meta, out_arrays = {"error": traceback.format_exc()}, {}
        conn.send_bytes(sg.pack_msg(out_meta, out_arrays))
    print("[worker] exit", flush=True)


if __name__ == "__main__":
    main()
