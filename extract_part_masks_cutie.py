import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor


def _indexed_get(mapping, index, default=None):
    if mapping is None:
        return default
    if index in mapping:
        return mapping[index]
    if str(index) in mapping:
        return mapping[str(index)]
    return default


def _max_prior_mask_coverage(
    candidate_box_mask: np.ndarray,
    occupied_masks,
) -> float:
    """Largest fraction of any prior class covered by a candidate box.

    The legacy trackers compare each discovered category independently. A
    single union denominator weakens the filter as more objects are found.
    """
    if isinstance(occupied_masks, np.ndarray):
        occupied_masks = [occupied_masks]

    maximum = 0.0
    for occupied_mask in occupied_masks:
        occupied_mask = np.asarray(occupied_mask, dtype=bool)
        occupied_area = int(occupied_mask.sum())
        if occupied_area == 0:
            continue
        covered_area = int(np.logical_and(candidate_box_mask, occupied_mask).sum())
        maximum = max(maximum, covered_area / (occupied_area + 1e-6))
    return maximum


class GroundedSAMDetector:
    """GroundingDINO box detection followed by one SAM mask per class."""

    def __init__(
        self,
        classes,
        confidence_threshold,
        class_area_ranges,
        global_area_min,
        global_area_max,
        overlap_threshold,
        box_threshold,
        text_threshold,
        nms_threshold,
        home_path,
        groundingdino_config,
        groundingdino_checkpoint,
        sam_checkpoint,
        sam_model_type,
    ):
        # groundingdino and segment_anything are expected to be installed as
        # packages (see README), not vendored under this directory.
        import groundingdino
        from groundingdino.util.inference import Model
        from segment_anything import SamPredictor, sam_model_registry

        self.classes = list(classes)
        self.confidence_threshold = float(confidence_threshold)
        self.class_area_ranges = class_area_ranges
        self.global_area_min = global_area_min
        self.global_area_max = global_area_max
        self.overlap_threshold = float(overlap_threshold)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_threshold = float(nms_threshold)

        if groundingdino_config:
            model_config_path = os.path.join(home_path, groundingdino_config)
        else:
            model_config_path = os.path.join(
                os.path.dirname(groundingdino.__file__),
                "config",
                "GroundingDINO_SwinT_OGC.py",
            )
        self.grounding_dino = Model(
            model_config_path=model_config_path,
            model_checkpoint_path=os.path.join(home_path, groundingdino_checkpoint),
        )
        sam = sam_model_registry[sam_model_type](
            checkpoint=os.path.join(home_path, sam_checkpoint)
        )
        sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.sam_predictor = SamPredictor(sam)
        # Holds the most recent RAW GroundingDINO output (all candidate boxes +
        # confidences, before NMS / area / overlap / confidence filtering). The
        # tracker dumps this per (episode, frame, class) when saving is on, so
        # the detect-stage output is inspectable, not just the final boxes.
        self.last_raw = None

    def detect_class(
        self,
        image_bgr: np.ndarray,
        class_index: int,
        occupied_masks,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        with torch.inference_mode():
            detections = self.grounding_dino.predict_with_classes(
                image=image_bgr,
                classes=[self.classes[class_index]],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        # Snapshot the raw detect-stage output before any filtering.
        self.last_raw = {
            "class_index": int(class_index),
            "xyxy": np.asarray(detections.xyxy, dtype=np.float32),
            "confidence": np.asarray(detections.confidence, dtype=np.float32),
        }
        if len(detections.xyxy) == 0:
            return None, None, None

        keep = torchvision.ops.nms(
            torch.as_tensor(detections.xyxy),
            torch.as_tensor(detections.confidence),
            self.nms_threshold,
        ).cpu().numpy()
        boxes = detections.xyxy[keep]
        confidences = detections.confidence[keep]

        height, width = image_bgr.shape[:2]
        accepted = []
        for index, raw_box in enumerate(boxes):
            box = np.asarray(raw_box, dtype=np.int64).copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            area = int((box[2] - box[0]) * (box[3] - box[1]))
            area_range = _indexed_get(self.class_area_ranges, class_index, (None, None))
            min_area, max_area = area_range
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            if self.global_area_min is not None and area < self.global_area_min:
                continue
            if self.global_area_max is not None and area > self.global_area_max:
                continue
            if float(confidences[index]) < self.confidence_threshold:
                continue

            box_mask = np.zeros((height, width), dtype=bool)
            box_mask[box[1] : box[3], box[0] : box[2]] = True
            overlap = _max_prior_mask_coverage(box_mask, occupied_masks)
            if overlap > self.overlap_threshold:
                continue
            accepted.append((box, float(confidences[index])))

        if not accepted:
            return None, None, None

        box, confidence = accepted[0]
        self.sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
        mask = masks[int(np.argmax(scores))].astype(bool)
        if not mask.any():
            return None, None, None
        return mask, box, confidence


class TrackerBase:
    def __init__(self, classes, detector, output_dir, save_annotations, save_masks):
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.background_label = self.num_classes
        self.detector = detector
        self.output_dir = output_dir
        self.save_annotations = bool(save_annotations)
        self.save_masks = bool(save_masks)
        self.episode_num = 0
        # Per-stage output folders, one per model in the pipeline. Every
        # artifact lands in the folder for the stage that produced it, so the
        # run is organized by step instead of by encoded filename:
        #   step1_detect   raw GroundingDINO candidate boxes  (.npz)
        #   step2_segment  SAM masks that seed tracking        (.png)
        #   step3_track    Cutie-propagated masks + overlays   (.png)
        #   step4_embed    final DINO arrays (written by post_process)
        self.dir_detect = os.path.join(self.output_dir, "step1_detect")
        self.dir_segment = os.path.join(self.output_dir, "step2_segment")
        self.dir_track = os.path.join(self.output_dir, "step3_track")
        for _d in (self.output_dir, self.dir_detect, self.dir_segment, self.dir_track):
            os.makedirs(_d, exist_ok=True)

    def _empty_output(self, shape):
        return np.full(shape, self.background_label, dtype=np.int32)

    def _save_mask(self, class_mask, frame_index, class_index, suffix):
        # SAM's seed mask (init frame) is the segment stage -> step2_segment;
        # Cutie's propagated masks are the track stage -> step3_track.
        if not self.save_masks:
            return
        is_seed = "seed" in suffix
        stage_dir = self.dir_segment if is_seed else self.dir_track
        kind = "seed" if is_seed else "mask"
        path = os.path.join(
            stage_dir,
            f"{self.episode_num}_{frame_index}_{class_index}_{kind}.png",
        )
        Image.fromarray((class_mask.astype(np.uint8) * 255)).save(path)

    def _save_detections(self, raw, frame_index):
        # Persist the raw GroundingDINO detect-stage output (all candidate
        # boxes + confidences before filtering) as a small .npz in step1_detect,
        # keyed by episode/frame/class. Gated by save_masks so it rides along
        # with the other per-frame intermediates. `raw` is detector.last_raw.
        if not self.save_masks or not raw:
            return
        np.savez(
            os.path.join(
                self.dir_detect,
                f"{self.episode_num}_{frame_index}_{raw['class_index']}.npz",
            ),
            xyxy=raw["xyxy"],
            confidence=raw["confidence"],
        )

    def _save_overlay(self, image_bgr, label_mask, frame_index):
        if not self.save_annotations:
            return
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        overlay = image_bgr.copy()
        for class_index in range(self.num_classes):
            pixels = label_mask == class_index
            if not pixels.any():
                continue
            color = np.asarray(colors[class_index % len(colors)], dtype=np.float32)
            overlay[pixels] = (
                0.5 * overlay[pixels].astype(np.float32) + 0.5 * color
            ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(
                self.dir_track,
                f"{self.episode_num}_{frame_index}_overlay.png",
            ),
            overlay,
        )


class CutieTracker(TrackerBase):
    """One independent CUTIE inference core per semantic class."""

    def __init__(
        self,
        classes,
        detector,
        output_dir,
        save_annotations,
        save_masks,
        max_internal_size,
        skip_non_first_frame,
    ):
        super().__init__(classes, detector, output_dir, save_annotations, save_masks)
        # cutie (https://github.com/hkchengrex/Cutie) is expected to be
        # installed as a package (see README), not vendored under this
        # directory. Its own weights auto-download on first use.
        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model

        self.InferenceCore = InferenceCore
        self.cutie = get_default_model()
        self.max_internal_size = int(max_internal_size)
        self.skip_non_first_frame = skip_non_first_frame
        self.detected = [False] * self.num_classes
        self.processors = self._new_processors()

    def _new_processor(self):
        processor = self.InferenceCore(self.cutie, cfg=self.cutie.cfg)
        processor.max_internal_size = self.max_internal_size
        return processor

    def _new_processors(self):
        return [self._new_processor() for _ in range(self.num_classes)]

    def _skip_late_detection(self, class_index):
        setting = self.skip_non_first_frame
        if isinstance(setting, bool):
            return setting and class_index > 0
        return bool(_indexed_get(setting, class_index, False))

    @staticmethod
    def _rgb_tensor(image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(image_rgb).cuda().float()

    def reset(self):
        self.detected = [False] * self.num_classes
        self.processors = self._new_processors()

    def process(self, image_bgr, frame_index):
        if frame_index == 0:
            self.reset()

        output = self._empty_output(image_bgr.shape[:2])
        occupied_masks = []
        rgb = self._rgb_tensor(image_bgr)

        for class_index in range(self.num_classes):
            if self.detected[class_index]:
                # Match the original CUTIE implementation: after a class has
                # been initialized once, keep propagating with the same
                # processor. An empty mask means that the object is currently
                # absent; it must not reset the processor or trigger a new
                # GroundingDINO/SAM initialization on the same frame.
                with torch.inference_mode():
                    probability = self.processors[class_index].step(rgb)
                    tracked = (
                        self.processors[class_index]
                        .output_prob_to_mask(probability)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                        > 0
                    )
                if tracked.any():
                    output[tracked] = class_index
                    occupied_masks.append(tracked.copy())
                    self._save_mask(tracked, frame_index, class_index, "cutie_mask")
                continue

            if frame_index > 0 and self._skip_late_detection(class_index):
                continue

            detected_mask, _, _ = self.detector.detect_class(
                image_bgr,
                class_index,
                occupied_masks,
            )
            self._save_detections(self.detector.last_raw, frame_index)
            if detected_mask is None:
                continue

            # Initialize CUTIE on whichever frame the class is first found.
            seed = torch.from_numpy(detected_mask.astype(np.uint8)).cuda().long()
            with torch.inference_mode():
                probability = self.processors[class_index].step(
                    rgb,
                    seed,
                    objects=[1],
                )
                initialized = (
                    self.processors[class_index]
                    .output_prob_to_mask(probability)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    > 0
                )
            if not initialized.any():
                initialized = detected_mask
            self.detected[class_index] = True
            output[initialized] = class_index
            occupied_masks.append(initialized.copy())
            self._save_mask(initialized, frame_index, class_index, "cutie_seed")

        self._save_overlay(image_bgr, output, frame_index)
        return output


def build_tracker(config):
    detector = GroundedSAMDetector(
        classes=config.task.classes,
        confidence_threshold=config.task.confidence_threshold,
        class_area_ranges=config.perception.get("class_area_ranges"),
        global_area_min=config.perception.get("global_area_min"),
        global_area_max=config.perception.get("global_area_max"),
        overlap_threshold=config.perception.get("overlap_threshold", 0.7),
        box_threshold=config.perception.get("box_threshold", 0.25),
        text_threshold=config.perception.get("text_threshold", 0.25),
        nms_threshold=config.perception.get("nms_threshold", 0.8),
        home_path=config.paths.home_path,
        groundingdino_config=config.paths.get("groundingdino_config"),
        groundingdino_checkpoint=config.paths.groundingdino_checkpoint,
        sam_checkpoint=config.paths.sam_checkpoint,
        sam_model_type=config.paths.sam_model_type,
    )
    common = {
        "classes": config.task.classes,
        "detector": detector,
        "output_dir": config.output.output_dir,
        "save_annotations": config.output.get("save_annotations", True),
        "save_masks": config.output.get("save_masks", True),
    }
    tracker_name = str(config.task.tracker).lower()
    if tracker_name != "cutie":
        raise ValueError(
            "This refactor is CUTIE-only; task.tracker must be 'cutie', "
            f"got {config.task.tracker!r}"
        )
    return CutieTracker(
        **common,
        max_internal_size=config.perception.get("cutie_max_internal_size", 480),
        skip_non_first_frame=config.perception.get(
            "skip_non_first_frame",
            False,
        ),
    )
