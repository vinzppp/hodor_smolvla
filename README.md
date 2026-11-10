# hodor_smolvla

HODOR is a scene-graph approach to giving a robot policy access to objects outside its current view. This project reproduces its perception pipeline and tests ways of passing the resulting object memory to SmolVLA and π0.5 on RoboMME.

## Start here

| If you want to… | Open |
| --- | --- |
| See the original SmolVLA pilot setup | [SmolVLA notebook](notebooks/pilot/sg_interfaces_pilot.ipynb) |
| See how the π0.5 port is wired | [π0.5 notebook](notebooks/pilot/sg_interfaces_pi05.ipynb) |
| Inspect a run with saved outputs | [Executed π0.5 smoke notebook](notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb) |
| Read the measured results and their limits | [Research status](docs/RESEARCH_STATUS.md) |
| Inspect the earlier DROID replication | [DROID walkthrough and artifacts](archive/droid-section3.1/README.md) |

## How the pipeline works

1. **Find and track objects.** GroundingDINO detects task objects, SAM segments them, and Cutie tracks the masks through frames.
2. **Build the scene graph.** DINOv2 provides appearance features, and depth gives 3D positions. Graph update rules preserve information about objects while they are occluded.
3. **Give memory to the policy.** The comparison includes a baseline without graph input (A0), visual markers (A1), a textual graph (A2), and graph tokens (A3).
4. **Train and evaluate.** The training entry point fits the policy adapter. The evaluation entry point loads the adapter and runs it on the specified RoboMME task.

The implementation reproduces the perception stage with Cutie in place of XMem. Its current scope is the tracked-object graph and the downstream policy interfaces. The original HODOR diffusion policy and dynamic node insertion are outside the completed replication.

| Component | Where to look |
| --- | --- |
| Scene-graph perception | [Pipeline](post_process_scene_cutie.py), [mask extraction](extract_part_masks_cutie.py), [DINOv2 encoder](dinov2_encoder.py) |
| Replay and graph interfaces | [Pilot implementation](notebooks/pilot/sg_pilot.py), [perception worker](notebooks/pilot/perception_worker.py) |
| π0.5 adapter | [Policy implementation](notebooks/pilot/pi05_policy.py) |
| Training and closed-loop evaluation | [Train](notebooks/pilot/train.py), [evaluate](notebooks/pilot/eval.py) |
| Setup and recorded versions | [Environment](docs/ENVIRONMENT.md), [perception setup script](setup_gpu.sh) |

## Results

SmolVLA was trained on 100 demonstrations of RoboMME’s `VideoUnmaskSwap` task. The A0 and A2 pilots each ran for 20,000 updates with batch size 32.

| SmolVLA interface | Held-out success | Saved result |
| --- | ---: | --- |
| A0, no scene graph | 2 / 50 (4%) | [A0 result](notebooks/pilot/results/pilot_A0_Q0.json) |
| A2, scene graph as text | 2 / 50 (4%) | [A2 result](notebooks/pilot/results/pilot_A2_Q0.json) |

Both variants performed poorly on the held-out episodes. The single-task training recipe, high learning rate, and execution of long action chunks limit what this pilot can say about the value of the graph interface.

The π0.5 port passed 35 smoke checks. A0–A3 each trained for ten steps, and A2/A3 exercised closed-loop evaluation. Full training and a controlled π0.5 performance comparison remain pending. The [executed notebook](notebooks/pilot/executed/sg_interfaces_pi05_smoke.ipynb) records those checks.

## Inspect the repo on a CPU

```sh
git clone --branch pi05-pilot https://github.com/vinzppp/hodor_smolvla.git
cd hodor_smolvla
python3 tools/check_repository.py
```

The check parses Python, validates notebook structure, and checks README links. The executed notebooks and result JSON files can be inspected without loading a model.

## Evaluate a saved checkpoint

Full evaluation requires Linux, an NVIDIA GPU, Vulkan, RoboMME replay data, and the model files described in the [recorded environment](docs/ENVIRONMENT.md). Use separate policy and perception environments to match their respective Transformers versions. The perception dependencies in `requirements.txt` cover one part of that setup.

The two SmolVLA pilot checkpoints are retained by the maintainer. Each run folder contains the trained LoRA adapter, its configuration, and the training and processor settings. After obtaining the checkpoint and preparing the recorded environment, run from `notebooks/pilot/`:

```sh
python eval.py --ckpt runs/pilot_A2_s0 --interface A2 \
  --task VideoUnmaskSwap --quality Q0 --episodes 50 \
  --out results/pilot_A2_Q0_recheck.json
```

Use `runs/pilot_A0_s0` and `--interface A0` for the baseline. The [research status](docs/RESEARCH_STATUS.md) records limitations of the data and perception conditions. The [DROID archive](archive/droid-section3.1/README.md) provides the earlier perception implementation and example results.
