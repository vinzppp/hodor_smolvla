# Expanding Spatial and Temporal Context for Robotic Imitation Learning with HODOR

**CVPR 2026**

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Qian_Expanding_Spatial_and_Temporal_Context_for_Robotic_Imitation_Learning_With_CVPR_2026_paper.html)

---

## Installation

This folder is self-contained: all first-party code lives here, and the
`vit.py` ViT encoder is a local copy (not imported from the wider repo). Two
third-party perception models are required but not vendored — install them
from source:

```bash
pip install -r requirements.txt

# GroundingDINO (https://github.com/IDEA-Research/GroundingDINO)
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO

# segment_anything (https://github.com/facebookresearch/segment-anything)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Cutie (https://github.com/hkchengrex/Cutie)
# Do NOT `pip install cutie` from PyPI — that is an unrelated package with
# the same name. Install from the Cutie source instead:
git clone https://github.com/hkchengrex/Cutie.git
pip install -e Cutie
```

Cutie downloads its own tracking weights automatically on first run. The
GroundingDINO and DINO ViT-B/16 checkpoints are bundled in `pretrained/`
here. The SAM checkpoint (`sam_vit_h_4b8939.pth`, ~2.6GB) exceeds GitHub's
2GB Git LFS file limit, so it isn't committed — download it into
`pretrained/` yourself:

```bash
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

`data/<task>/` includes 10 sample DROID episodes per task (trajectory,
wrist depth, and MP4 recordings only — raw `.svo2` stereo files and
precomputed outputs are not included) so the four configs run out of the
box. Point `task.data_path` in `configs/<task>.yaml` at your own captured
episodes to run on more data.

## HODOR preprocessing

This version keeps task configuration in YAML while preserving the important
behavioral differences in the original scripts:

- All four tasks use CUTIE for mask propagation.
- Every tracker returns direct semantic class IDs; background is
  `len(task.classes)`.
- Mask channels, boxes, DINO features, and point-cloud centroids share the same
  compact instance order.
- Slot positions are explicit in each task YAML.
- RGB preprocessing and the unrotated 800×800 image use the same crop, padding,
  and field of view as depth.
- Depth limits and input resolution are task-specific.
- Cross-class duplicate filtering compares each candidate box against every
  previously accepted class mask independently. The old chicken and pineapple
  detector/filter behavior is retained, but their propagation backend is CUTIE.

The chicken detection values are copied from the legacy
`extract_part_masks.py`: box threshold `0.25`, text threshold `0.25`, NMS
threshold `0.8`, confidence threshold `0.2`, box area `1000–150000`, and
per-class occupied-mask coverage threshold `0.7`.

Run from this directory:

```bash
python droid_hodor_test_cutie.py --config configs/giraffe.yaml
python droid_hodor_test_cutie.py --config configs/cabinet.yaml
python droid_hodor_test_cutie.py --config configs/pineapple.yaml
python droid_hodor_test_cutie.py --config configs/chicken.yaml
```

Generated episode files:

- `episode_pr.npy`
- `episode_bboxes.npy`
- `episode_pr_seperated.npy` and `episode_bboxes_seperated.npy` for chicken
- `episode_dino_full.npy` when `output.save_dino_full` is enabled

Feature slot semantics:

- giraffe: `[global, pineapple, giraffe]`
- cabinet: `[global, pineapple, purple_ball, red_ball]`
- pineapple: `[global, bowl, pineapple]`
- chicken: `[global, bowl, stuffed_animal, pineapple, ball]`
- chicken separated: `[global, bowl, chicken, giraffe, pineapple, ball]`
