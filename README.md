# hodor_smolvla

This repository reproduces the HODOR scene-graph perception pipeline and tests ways to pass object memory into SmolVLA and π0.5. The default branch contains the earlier DROID implementation. For the RoboMME notebooks, checkpoint instructions, and measured pilot results, start at the [pi05-pilot branch](https://github.com/vinzppp/hodor_smolvla/tree/pi05-pilot).

## What is available

- The DROID pipeline detects and tracks objects with GroundingDINO, SAM, and Cutie. It combines appearance features from DINOv2 with depth-derived centroids. The scripts and sample recordings are on this branch.
- The RoboMME SmolVLA pilot compares a baseline without the graph (A0) and a textual graph (A2). Both scored 2 out of 50 on held-out VideoUnmaskSwap episodes. See the [pilot results](https://github.com/vinzppp/hodor_smolvla/tree/pi05-pilot/notebooks/pilot/results).
- The π0.5 port was exercised with short smoke runs. See the [executed notebook](https://github.com/vinzppp/hodor_smolvla/blob/pi05-pilot/notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb) for saved outputs and the [research status](https://github.com/vinzppp/hodor_smolvla/blob/pi05-pilot/docs/RESEARCH_STATUS.md) for the scope of the results.

## DROID entry points

| Component | File |
| --- | --- |
| Run the scene-graph pipeline on recorded DROID episodes | [DROID driver](droid_hodor_test_cutie.py) |
| Maintain the graph and compute object features | [Perception pipeline](post_process_scene_cutie.py) |
| Configure scenes and tracked entities | [Task configurations](configs/) |

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
