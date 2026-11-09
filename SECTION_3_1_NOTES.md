> Historical development record: some commands, paths, pending checks and setup assumptions below describe an earlier state. See the repository README and docs/RESEARCH_STATUS.md for the current status. Preserve this record as provenance.

# Section 3.1 replication notes

Target: reproduce Section 3.1, "Task-Driven Scene Graph," of *Expanding Spatial
and Temporal Context for Robotic Imitation Learning With Scene Graphs* (Qian et
al., CVPR 2026). This file records what the upstream release already covers,
what was added on the `feature/section3.1-faithful` branch, and how to run it.

## What 3.1 specifies vs. what the code does

| 3.1 component | Upstream release | This branch |
| --- | --- | --- |
| Grounding DINO detection | present (`GroundedSAMDetector`) | unchanged |
| SAM segmentation (Grounded-SAM) | present (needs weights) | unchanged |
| Object embedding = masked avg-pool of ViT patch tokens | present (`_obtain_features`) | unchanged |
| Global scene node = ViT CLS/global token | present | DINOv2 path returns the true CLS token |
| 2D bounding box per node | present (`_mask_to_instances`) | unchanged |
| 3D centroid via depth backprojection | present (`_depth_to_pointcloud`) | unchanged |
| Out-of-view: zero 2D box, keep 3D centroid | present | unchanged |
| Temporal tracker for identity | Cutie (XMem's successor; upstream's deliberate swap) | unchanged |
| **Visual backbone = DINOv2** | DINO **v1** ViT-B/16 only | **DINOv2 ViT-B/14 option added** |
| **Task-relevant objects from an LLM** | missing (hardcoded `task.classes`) | **LLM front-end added** |
| Dynamic new-node insertion (Algorithm 1, tau_m) | not implemented (fixed per-class slots) | still not implemented (out of scope for this pass) |

Section 3.2 (the diffusion policy) is a separate stage and is not in this repo.

## What was added on this branch

`llm_entities.py` — provider-agnostic (Anthropic or OpenAI) extraction of the
task-relevant noun entities from `task.language_instruction`, converted to the
GroundingDINO caption tokens the pipeline expects (e.g. "yellow pineapple" ->
"yellow.pineapple."). Results are cached to `configs/<task>.entities.json` so
repeat runs are deterministic and offline. Runnable as a library, from the CLI,
or automatically at config-load time. Depends only on the standard library.

`post_process_scene_cutie.py` — `load_config` now calls `_maybe_resolve_entities`.
When a task sets `entities.source: llm`, `task.classes` (and `slot_by_class` if
absent) are filled from the instruction. Configs without an `entities` block, or
with `source: manual`, behave exactly as before.

`dinov2_encoder.py` + `vit.py` — `get_vit` returns a `DINOv2Encoder` for any
`model.vit_model` starting with `dinov2` (`dinov2-vit-s14/b14/l14/g14`), keeping
the existing `(global, patch)` interface. **DINOv2 ViT-B/14 is now the default
backbone for all configs** (`configs/base.yaml`); DINO-v1 (`dino-vit-b16`) is only
reachable by overriding `model.vit_model` explicitly, for baseline comparisons.

`configs/pineapple_llm.yaml` — pineapple task with LLM-derived classes.
`configs/giraffe.yaml` — giraffe task, manual classes, DINOv2 (no API key needed).
(`configs/giraffe_dinov2.yaml` was removed once DINOv2 became the default; the
v1-vs-v2 run of 2026-09-15 is kept in `results/run_20260915_111519_4090/`.)

`setup_gpu.sh` — installs GroundingDINO / segment-anything / Cutie, downloads the
SAM ViT-H checkpoint, and prefetches DINOv2.

## Running (on a Linux + NVIDIA GPU box; ~12 GB VRAM is comfortable, inference only)

    bash setup_gpu.sh

    # Everything below uses DINOv2. One-shot on an RTX 4090: bash run_all_4090.sh

    # 1) Manual classes - validates the perception core
    python droid_hodor_test_cutie.py --config configs/giraffe.yaml

    # 2) LLM entity front-end
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
    python llm_entities.py --config configs/pineapple_llm.yaml --write   # inspect/cache
    python droid_hodor_test_cutie.py --config configs/pineapple_llm.yaml


Outputs per episode: `episode_pr.npy` (features, [T, N+1, 768]) and
`episode_bboxes.npy` ([T, N+1, 7] = 4 box + 3 centroid).

## Verified vs. still to verify on the GPU box

Unit-tested on CPU: all of `llm_entities.py` (parsing, GroundingDINO-token
formatting, provider selection, caching, slot mapping). All edited/added Python
files pass `python -m py_compile`.

Not yet exercised here (no GPU / heavy deps in the dev environment), to confirm
on the box: (a) the OmegaConf assignment inside `_maybe_resolve_entities`
(mirrors patterns the existing manual configs already rely on); (b) DINOv2
end-to-end - token grid is square at 224 (16x16) and dim 768 matches, but the
forward pass hasn't been run; (c) a real LLM call and the full pipeline on
sample episodes.

## Deliberately out of scope for this pass

Dynamic mid-episode node insertion (Algorithm 1's tau_m branch) and reverting
the tracker from Cutie to XMem. Both are fidelity choices to revisit later.
