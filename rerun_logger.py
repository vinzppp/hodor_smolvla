"""Rerun timeline for the Section 3.1 pipeline: one .rrd file per episode.

Enable with ``output.save_rerun: true``. Files land in
``<output_dir>/rerun/ep<i>_<episode>.rrd``. Open one on your laptop with

    pip install rerun-sdk==0.37.2
    rerun ep0_<episode>.rrd

Everything is logged against the ``frame`` timeline, so dragging the time
slider shows every stage of the pipeline for that frame:

    camera/image                       wrist frame the models see (448 px, rotated)
    camera/step1_gdino/candidates/<c>  all GroundingDINO boxes    (only on frames it ran)
    camera/step1_gdino/accepted/<c>    box that passed the filters (only on frames it ran)
    camera/step2_sam/<c>               SAM mask that seeded Cutie   (only on its seed frame)
    camera/step3_cutie/mask            Cutie tracked masks          (every frame)
    camera/step3_cutie/boxes           2D node boxes from those masks
    dino/pca                           step 4: DINO patch features, PCA -> RGB
    world/centroids, world/trails/<c>  step 3b: 3D object centroids (depth back-projection)
    plots/embedding_similarity/<c>     step 4: node embedding vs its first appearance
    plots/presence/<c>                 1 if the node is in the scene graph this frame
    plots/stage_ms/<stage>             per-frame time per model
    events                             log of detections / seeds (click to jump there)
    info                               run description

Steps 1-2 only run until an object is found, so their layers appear on a few
frames only; click an entry in the events panel to jump to them.
"""
import os

import numpy as np

try:
    from . import stage_timer
except ImportError:
    import stage_timer

RERUN_VERSION = "0.37.2"

# RGB, same order as the BGR overlay colors in extract_part_masks_cutie.py
CLASS_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]
STAGE_ORDER = [key for key, _ in stage_timer.STAGES]


def _slug(name):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return cleaned or "obj"


class RerunLogger:
    def __init__(self, config):
        import rerun as rr

        self.rr = rr
        self.config = config
        self.classes = list(config.task.classes)
        self.names = [_slug(c) for c in self.classes]
        self.num_classes = len(self.classes)
        self.slot_by_class = {
            i: int(
                config.task.slot_by_class.get(i, config.task.slot_by_class.get(str(i)))
            )
            for i in range(self.num_classes)
        }
        self.out_dir = os.path.join(config.output.output_dir, "rerun")
        os.makedirs(self.out_dir, exist_ok=True)
        self.rec = None
        self.path = None
        self.paths = []

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    def _blueprint(self):
        import rerun.blueprint as rrb

        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin="camera",
                        name="Camera: 1 GroundingDINO / 2 SAM / 3 Cutie",
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="dino", name="4 DINO features (PCA)"),
                        rrb.Spatial3DView(origin="world", name="3D node centroids"),
                    ),
                    row_shares=[3, 2],
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin="plots/embedding_similarity",
                        name="4 node embedding: cos-sim to first appearance",
                    ),
                    rrb.TimeSeriesView(origin="plots/presence", name="node present"),
                    rrb.TimeSeriesView(origin="plots/stage_ms", name="time per stage (ms)"),
                    rrb.Tabs(
                        rrb.TextLogView(origin="events", name="events"),
                        rrb.TextDocumentView(origin="info", name="run info"),
                    ),
                    row_shares=[2, 1, 2, 2],
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=False,
        )

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def start_episode(self, episode_num, episode_name):
        rr = self.rr
        self.end_episode()
        self.episode_num = episode_num
        self.path = os.path.join(
            self.out_dir, f"ep{episode_num}_{episode_name.replace(':', '-')}.rrd"
        )
        rec = rr.RecordingStream(
            "hodor_section3_1",
            recording_id=f"{self.config.task.name}/ep{episode_num}",
        )
        rec.save(self.path, default_blueprint=self._blueprint())
        self.rec = rec

        # static: class names + colors for all segmentation images / boxes
        rec.log(
            "/",
            rr.AnnotationContext(
                [(0, "background", (0, 0, 0, 0))]
                + [
                    (i + 1, name, CLASS_COLORS[i % len(CLASS_COLORS)])
                    for i, name in enumerate(self.classes)
                ]
            ),
            static=True,
        )
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for i, name in enumerate(self.names):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            for plot in ("embedding_similarity", "presence"):
                rec.log(
                    f"plots/{plot}/{name}",
                    rr.SeriesLines(colors=[color], names=[self.classes[i]]),
                    static=True,
                )
        rec.log(
            "plots/embedding_similarity/scene",
            rr.SeriesLines(colors=[(160, 160, 160)], names=["scene (slot 0)"]),
            static=True,
        )
        cfg = self.config
        rec.log(
            "info",
            rr.TextDocument(
                "\n".join(
                    [
                        f"# {cfg.task.name} - episode {episode_num}",
                        f"**episode**: `{episode_name}`",
                        f"**instruction**: {cfg.task.get('language_instruction')}",
                        f"**classes (GroundingDINO prompts)**: {self.classes}",
                        f"**slot_by_class**: {self.slot_by_class}",
                        f"**backbone**: {cfg.model.vit_model} @ {cfg.model.vit_image_size}px",
                        "",
                        "Pipeline: 1 GroundingDINO -> 2 SAM ViT-H -> 3 Cutie (+3D centroid) -> 4 DINO embedding",
                        "",
                        "Steps 1-2 run only until an object is found; use the *events* tab to jump to those frames.",
                    ]
                ),
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )
        self._transient = set()
        self._prev_transient = set()
        self._refs = {}
        self._scene_ref = None
        self._trails = {i: [] for i in range(self.num_classes)}
        self._pca = None
        self._stage_snapshot = {k: v["seconds"] for k, v in stage_timer.totals().items()}
        self.frame = 0

    def end_episode(self):
        if self.rec is None:
            return
        self.rec.flush()
        self.rec.disconnect()
        self.paths.append(self.path)
        self.rec = None

    # ------------------------------------------------------------------ #
    # per frame
    # ------------------------------------------------------------------ #
    def _log_transient(self, path, archetype):
        self.rec.log(path, archetype)
        self._transient.add(path)

    def begin_frame(self, frame_num, image_bgr):
        rr = self.rr
        self.frame = int(frame_num)
        self.rec.set_time("frame", sequence=self.frame)
        # steps 1-2 layers are only valid on the frame they ran: clear them.
        for path in self._prev_transient:
            self.rec.log(path, rr.Clear(recursive=False))
        self._prev_transient = set()
        self._transient = set()
        self.height, self.width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        self.rec.log("camera/image", rr.Image(rgb).compress(jpeg_quality=85))

    def log_detection(self, class_index, raw, accepted):
        rr = self.rr
        name = self.names[class_index]
        color = CLASS_COLORS[class_index % len(CLASS_COLORS)]
        if raw is not None and len(raw["xyxy"]):
            self._log_transient(
                f"camera/step1_gdino/candidates/{name}",
                rr.Boxes2D(
                    array=np.asarray(raw["xyxy"], dtype=np.float32),
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"{self.classes[class_index]} {c:.2f}" for c in raw["confidence"]],
                    colors=[(255, 165, 0)],
                    radii=0.5,
                ),
            )
        if accepted is not None:
            box, conf = accepted
            self._log_transient(
                f"camera/step1_gdino/accepted/{name}",
                rr.Boxes2D(
                    array=np.asarray([box], dtype=np.float32),
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"GDINO kept {self.classes[class_index]} {conf:.2f}"],
                    colors=[color],
                    radii=2.0,
                ),
            )
        num = 0 if raw is None else len(raw["xyxy"])
        if num:
            self.rec.log(
                "events",
                rr.TextLog(
                    f"f{self.frame} step1 GroundingDINO '{self.classes[class_index]}': "
                    f"{num} candidate(s), "
                    + ("accepted" if accepted is not None else "all rejected by filters"),
                    level="INFO" if accepted is not None else "DEBUG",
                ),
            )

    def log_sam_seed(self, class_index, mask, event):
        rr = self.rr
        seg = mask.astype(np.uint8) * np.uint8(class_index + 1)
        self._log_transient(
            f"camera/step2_sam/{self.names[class_index]}",
            rr.SegmentationImage(seg, opacity=0.6, draw_order=20.0),
        )
        self.rec.log(
            "events",
            rr.TextLog(
                f"f{self.frame} step2 SAM '{self.classes[class_index]}': "
                f"{event['sam_mask_px']} px mask (GDINO conf {event['gdino_conf']:.2f}) "
                "-> step3 Cutie seeded",
                level="INFO",
            ),
        )

    def _pca_image(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, tokens.shape[-1])
        side = int(round(np.sqrt(tokens.shape[0])))
        if side * side != tokens.shape[0]:
            tokens = tokens[1:]
            side = int(round(np.sqrt(tokens.shape[0])))
        if self._pca is None:  # fixed basis per episode -> stable colors
            mean = tokens.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(tokens - mean, full_matrices=False)
            proj = (tokens - mean) @ vt[:3].T
            lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
            self._pca = (mean, vt[:3], lo, np.maximum(hi - lo, 1e-6))
        mean, basis, lo, span = self._pca
        proj = np.clip(((tokens - mean) @ basis.T - lo) / span, 0, 1)
        image = (proj.reshape(side, side, 3) * 255).astype(np.uint8)
        scale = max(1, self.height // side)
        return np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))

    def end_frame(self, label_mask, final_rp, final_bboxes, dino_full):
        rr = self.rr
        rec = self.rec
        h, w = label_mask.shape

        # step 3: Cutie masks + node boxes
        seg = (label_mask + 1).astype(np.uint8)
        seg[label_mask >= self.num_classes] = 0
        rec.log("camera/step3_cutie/mask", rr.SegmentationImage(seg, opacity=0.45, draw_order=10.0))
        rp = final_rp.numpy() if hasattr(final_rp, "numpy") else np.asarray(final_rp)
        bb = np.asarray(final_bboxes)
        boxes, ids, labels, points, point_ids, point_labels = [], [], [], [], [], []
        for c in range(self.num_classes):
            slot = self.slot_by_class[c]
            r0, r1, c0, c1 = bb[slot, :4]
            present = float(abs(r0) + abs(r1) + abs(c0) + abs(c1)) > 0
            rec.log(f"plots/presence/{self.names[c]}", rr.Scalars(float(present)))
            if not present:
                continue
            boxes.append([c0 * w, r0 * h, c1 * w, r1 * h])
            ids.append(c + 1)
            labels.append(f"node {slot}: {self.classes[c]}")
            # step 4: node embedding drift
            feat = rp[slot]
            if c not in self._refs:
                self._refs[c] = feat.copy()
            ref = self._refs[c]
            sim = float(feat @ ref / (np.linalg.norm(feat) * np.linalg.norm(ref) + 1e-8))
            rec.log(f"plots/embedding_similarity/{self.names[c]}", rr.Scalars(sim))
            # step 3b: 3D centroid
            xyz = bb[slot, 4:7]
            if np.abs(xyz).sum() > 0:
                points.append(xyz)
                point_ids.append(c + 1)
                point_labels.append(self.classes[c])
                self._trails[c].append(xyz.tolist())
                if len(self._trails[c]) > 1:
                    rec.log(
                        f"world/trails/{self.names[c]}",
                        rr.LineStrips3D([self._trails[c]], class_ids=[c + 1], radii=0.002),
                    )
        if boxes:
            rec.log(
                "camera/step3_cutie/boxes",
                rr.Boxes2D(array=np.asarray(boxes, np.float32), array_format=rr.Box2DFormat.XYXY,
                           class_ids=ids, labels=labels, draw_order=30.0),
            )
            rec.log(
                "dino/node_boxes",
                rr.Boxes2D(array=np.asarray(boxes, np.float32), array_format=rr.Box2DFormat.XYXY,
                           class_ids=ids, labels=labels),
            )
        else:
            rec.log("camera/step3_cutie/boxes", rr.Clear(recursive=False))
            rec.log("dino/node_boxes", rr.Clear(recursive=False))
        if points:
            rec.log(
                "world/centroids",
                rr.Points3D(np.asarray(points, np.float32), class_ids=point_ids,
                            labels=point_labels, radii=0.01, show_labels=True),
            )
        else:
            rec.log("world/centroids", rr.Clear(recursive=False))

        # scene node
        scene = rp[0]
        if self._scene_ref is None:
            self._scene_ref = scene.copy()
        rec.log(
            "plots/embedding_similarity/scene",
            rr.Scalars(float(scene @ self._scene_ref / (np.linalg.norm(scene) * np.linalg.norm(self._scene_ref) + 1e-8))),
        )

        # step 4: DINO feature map
        if dino_full is not None:
            rec.log("dino/pca", rr.Image(self._pca_image(dino_full[0] if dino_full.ndim == 3 else dino_full)))

        # per-frame stage times
        now = {k: v["seconds"] for k, v in stage_timer.totals().items()}
        for key in STAGE_ORDER:
            if key in now or key in self._stage_snapshot:
                delta = now.get(key, 0.0) - self._stage_snapshot.get(key, 0.0)
                rec.log(f"plots/stage_ms/{key}", rr.Scalars(1000.0 * delta))
        self._stage_snapshot = now
        self._prev_transient = set(self._transient)
