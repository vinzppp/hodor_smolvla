# hodor_smolvla — scene-graph memory for robotic policies

Research code for reproducing HODOR's task-driven scene-graph perception pipeline and studying how object memory can be supplied to vision-language-action policies on RoboMME.

## What is completed

- **HODOR perception pipeline replication:** GroundingDINO → SAM → Cutie tracking, DINOv2 features, and depth-derived 3D centroids. Earlier DROID runs and evidence are preserved in the archive. This is not a full validated replication of HODOR's policy results: dynamic node insertion is absent, Cutie replaces XMem, and the original diffusion-policy stage is not reproduced here.
- **SmolVLA pilot:** trained on 100 VideoUnmaskSwap demonstrations; overfit and achieved approximately **4% test success**. The single-task recipe, high learning rate and execution of all 50 chunk actions open-loop limit interpretation. This is a useful negative pilot, not evidence of robust generalization.
- **π0.5 port:** implemented and smoke-tested, including short training and evaluation checks. **Full training and controlled interface comparisons remain pending.** Smoke outputs do not establish policy performance.

The perceived graph is unusually close to the oracle on VideoUnmaskSwap because Cutie can track the covering container in the hidden cube's slot. Results on this task alone cannot establish general object-memory robustness. See [research status](docs/RESEARCH_STATUS.md) for the detailed results and caveats; planned paper tables are not completed experiments.

## Start reading here

1. [π0.5 notebook](notebooks/pilot/sg_interfaces_pi05.ipynb): current policy direction and shared data-to-graph walkthrough.
2. [Executed π0.5 smoke notebook](notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb): recorded checks and outputs.
3. [SmolVLA notebook](notebooks/pilot/sg_interfaces_pilot.ipynb): original pilot and teaching narrative.
4. [Build notes](notebooks/pilot/BUILD_NOTES.md): environment versions, debugging history and measured diagnostics. Earlier sections are historical; later entries supersede them.

## Repository map

| Location | Contents |
| --- | --- |
| Root Python modules and `configs/base.yaml` | Shared perception pipeline; `post_process_scene_cutie.py` is the main implementation |
| `notebooks/pilot/sg_pilot.py` | Replay, graph rules, interfaces, training/evaluation utilities |
| `notebooks/pilot/pi05_policy.py` | π0.5 adaptation |
| `notebooks/pilot/train.py`, `eval.py`, `tools/` | Experiment entry points and diagnostics; inspect arguments before running |
| `notebooks/pilot/executed/`, `logs/`, `figures/` | Executed evidence, pilot/smoke logs and figures (under `notebooks/pilot/`) |
| `notebooks/pilot/_build/` | Historical notebook plans, draft HTML and notebook generator; retained for provenance |
| `archive/droid-section3.1/` | DROID scripts, task configs, recordings and earlier results |
| `docs/` | Research notes, guides and PDFs |
| `pretrained/` | LFS-managed perception weights; SAM must be acquired separately |
| `archive/documentation/` | Superseded README and archive index |

## Minimal setup and verified offline check

From the repository root, Python 3 is sufficient for this check:

```bash
python3 tools/check_repository.py
```

It parses Python source, checks notebook JSON/basic structure and verifies key cited paths. It does **not** load models, execute notebooks, access APIs or require a GPU.

For perception development, use a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` is an unpinned perception dependency list, not a complete policy environment or reproducibility lockfile. GroundingDINO, Meta's segment-anything and hkchengrex/Cutie also require source installs and model weights. Do not install the unrelated PyPI package named `cutie`. Inspect `setup_gpu.sh` before using it; it installs/downloads dependencies and pins an older transformers version incompatible with the policy environment.

Full notebook execution requires **Linux, an NVIDIA GPU, Vulkan, RoboMME data, and separate policy/perception environments**. Use the recorded versions and two-environment arrangement in [BUILD_NOTES.md](notebooks/pilot/BUILD_NOTES.md), with [research status](docs/RESEARCH_STATUS.md) for the latest π0.5 state. The [pod runbook](notebooks/pilot/POD_RUNBOOK.md) records historical setup, including machine-specific paths and upload assets that are not bundled. A fresh full installation was not verified during cleanup.

Large recordings and weights use Git LFS. Check availability with `git lfs ls-files`; a clone may contain pointers rather than usable data. Full RoboMME caches and trained checkpoints are not guaranteed to be present locally; Contact the maintainer for access to remote caches and checkpoints. No automatic download or training is part of the offline check.

## Results, history and reproduction

Read `notebooks/pilot/logs/pilot/` for the SmolVLA pilot and `notebooks/pilot/logs/pi05_smoke/` alongside the executed notebook for the π0.5 smoke evidence. Earlier perception results are in `archive/droid-section3.1/results/`. Keep logs, executed notebooks and their settings together when interpreting results.

The DROID scripts retain historical root-relative path assumptions; consult the [archive notes](archive/droid-section3.1/README.md) before reproducing them. They are not the default quick-start. [Section 3.1 notes](SECTION_3_1_NOTES.md) preserve the original implementation scope, not current pending-run status.

See the [cleanup manifest](docs/CLEANUP_MANIFEST.md) for preserved artifacts, removed generated files and recovery information. Report-cited code and notebook paths remain unchanged.
