# Build notes (built 2026-10-26; all steps run in smoke mode on the pod)

Date: 2026-10-26. Built without a GPU, so nothing GPU-dependent has run. All results in the notebook are `= ?` until it is executed.

## Done
- `sg_pilot.py`, `train.py`, `eval.py` and `run_grid.sh` are written. They pass pyflakes, and `run_grid.sh` passes `bash -n`.
- The GPU-free logic was tested on a synthetic swap scene (geometry round trip, frozen vs bound rule, online = offline tracker, corruption levels, interfaces, Wilson interval, pilot decision rule). All checks passed.
- `sg_interfaces_pilot.ipynb` has 112 cells (62 code). It is valid nbformat, every code cell parses, and it is formatted to 95 columns.
- Checker on the first draft gave 1 ERROR and 12 CHECK:
  - the ERROR (`exec-order`) is expected until the notebook is executed;
  - 10 `code-width` CHECKs are long string literals;
  - 2 `unbacked-number` CHECKs (50, 1300) are printed by Steps 8.1 and 10.1 once the notebook runs.

## Design decisions made while reading the sources
These come from RoboMME @53673b3 and LeRobot @5831498.
- **Training data** comes from replaying the raw HDF5 demos (`Yinpei/robomme_data_h5`), not from `lerobot/robomme`.
  - The replay gives segmentation, depth and cameras, aligned frame by frame with images, states and actions.
  - `lerobot/robomme` is used only for feature metadata and normalisation statistics.
- **Hidden cubes move with their containers.** `VideoUnmaskSwap.step` carries each cube with its container (`lift_and_drop_objectA_onto_objectB`). So the full oracle gives a hidden cube its container's position, and the `bound` rule matches the simulator's physics.
- **LoRA targets** (identical in every arm) are SmolVLA's default PEFT targets (action expert q/v plus the projections) plus the VLM text layers' q/v. Without the VLM layers, A2 text and A3 tokens would pass through frozen VLM layers only.
- **A3 tokens** sit in the prefix between the language tokens and the state token. They use attention mask 0, like the image and text tokens (`GraphPrefix` wraps `VLAFlowMatching.embed_prefix`).
- **Evaluation** uses `BenchmarkEnvBuilder` directly, not the LeRobot wrapper. This gives access to the history video, segmentation and actor poses, with max_steps 1300.

## To verify on the GPU machine (each is printed by the notebook)
- `BenchmarkEnvBuilder` metadata paths and `get_episode_num()` for each split (2.1).
- The H5 file name pattern in `Yinpei/robomme_data_h5` (`download_h5`).
- The maniskill segmentation key path, `segmentation_id_map` and the actor attribute names (3.1, 3.5).
- That the replay length equals the H5 length (3.3).
- The SmolVLA config fields and the prefix builder location (8.1).
- `make_policy` with `ds_meta` + `rename_map`, `wrap_with_peft(peft_config=...)`, and the processor override names (8.1).
- LoRA target regex matches (trainable count in 8.5).
- The perception thresholds at 512 px (5.3).

## Added on the second pass (2026-10-26)
- Teaching prose for reader level a, following `_build/explain-plan.md`:
  - a running example (the first training episode) carried through the steps;
  - a four-rung ladder for camera projection;
  - a stage table for the perception chain;
  - a frame-by-frame walkthrough of the bound rule;
  - a ladder for SmolVLA (tokens, prefix, action expert, where each interface enters, LoRA);
  - one traced training iteration;
  - the Wilson interval with example counts.
- "Check yourself" prompts at 2.7, 6, and 10.
- "How to read the output above" placeholders (= ?) for every result reading.
- At a Glance: data-flow structure drafted from the code. The opening paragraph, results and takeaways stay = ? until the notebook is run.
- Q3 online evaluation (`PerceivedOnline`): the perception stack runs every step, and the target slot is found by its colour class. `run_grid.sh` now includes Q3.
- A3feat online features (`OnlineFeatures`). This fixes a bug where A3feat evaluation passed no DINOv2 vectors.

## Not implemented
- At a Glance: written last, after execution.

## Pod run, Steps 1-7 (2026-10-26, RunPod RTX 4090)
Executed in smoke mode on a fresh `sg-main` kernel through the end of Step 7 (cells 0-82): every check prints PASS (27), no errors. Steps 8-10 not yet run. The notebook file itself is not yet executed in place; runs went to a scratch copy.

### Environment (all under /workspace; setup_gpu.sh not run)
- **`/workspace/envs/sg-main`** (notebook kernel `sg-main`): Python 3.12, torch 2.9.1+cu128, `lerobot[smolvla,peft,dataset] @ 5831498` (0.6.2, transformers 5.5.4, peft 0.21.2), `robomme @ 53673b3` (mani_skill 3.0.0b21 fork).
  - Overrides in `/workspace/envs/sg-main-overrides.txt`: numpy 1.26.4 and gymnasium 0.29.1 (per the LeRobot RoboMME docs), plus packages whose new releases need numpy 2: scipy 1.15.3, contourpy 1.3.2, opencv(-headless) 4.11.0.86, pyarrow 21.0.0; torchcodec 0.9.1 (0.11 is built for torch 2.11).
  - Kernelspec at `/workspace/envs/jupyter/share/jupyter/kernels/sg-main` (set `JUPYTER_PATH=/workspace/envs/jupyter/share/jupyter`); its `env` sets `VK_ICD_FILENAMES`, `XDG_RUNTIME_DIR`, `HF_HOME`, `TORCH_HOME`.
- **`/workspace/envs/sg-percep`** (perception worker): Python 3.12, torch 2.9.1, transformers 4.44.2 / tokenizers 0.19.1 / huggingface_hub 0.25.2 / timm 0.9.16 (setup_gpu.sh pins), opencv < 5, numpy 2.2. Requirements in `/workspace/envs/sg-percep-requirements.txt`.
  - GroundingDINO @ 856dde2, built with `TORCH_CUDA_ARCH_LIST=8.9` and setup_gpu.sh's three kernel patches; Cutie @ ec5cdd4 installed `--no-deps` (GUI/training extras skipped; `cchardet` has no 3.12 wheel); segment-anything @ dca509f. Clones in `/workspace/src`.
  - Weights in `/workspace/models` (repo `pretrained/` was not in the upload): `sam_vit_h_4b8939.pth` (md5 4b8939…), `groundingdino_swint_ogc.pth` (v0.1.0-alpha release, 693,997,677 bytes). Cutie weights download into the clone; DINOv2 and bert-base-uncased into the /workspace caches.
- Data: `cache/h5/record_dataset_VideoUnmaskSwap.h5` (23 GB, extracted from the repo's `.h5.tar.xz`).

### Code changes in this run
- **Perception in a second process.** `perception_worker.py` (sg-percep) loads GroundingDINO, SAM, Cutie and DINOv2 once and serves `configure`, `process_frame`, `features` over a Unix socket; messages are a JSON header plus raw array bytes (no cross-numpy pickles). Model loaders are memoized in the worker, so a new class list (e.g. 4 containers) reuses the loaded models. `sg_pilot.PerceptionWorker` / `PerceptionHandle` are the client; `make_robomme_processor` returns a handle with the processor's `process_frame` arguments. The worker starts with `MPLBACKEND=Agg` and `HF_HUB_ENABLE_HF_TRANSFER=0`. `PerceptionWorker.close()` frees its GPU memory (to call before training).
- **Latency (episode 0, 321 frames):** start + model load 30 s once; configure with models loaded 0.0 s; frame 0 (detect + SAM + seed) 5.7 s; frames 1+ round trip median 66.5 ms, p90 92.6 ms (compute 65.2 ms, socket 1.3 ms); worker GPU peak 7.3 GB allocated. Q3 evaluation adds about 90 s of perception per 1300-step episode.
- **Detection config fixes (5.1):** cube phrases are `"<colour> cube."` (the old `"<colour>.cube."` is two GroundingDINO phrases and matched any cube); container boxes need area >= 1200 px² at 512 px (`CONTAINER_MIN_AREA_512`; measured cube boxes <= 728 from masks, 930 from GroundingDINO; container 5th percentile >= 1800), else container slots seed on cubes at frame 0; container candidates touching the top image edge are dropped (`TopEdgeFilter`: the robot enters there and is container-sized).
- **Slot matching at the seed frame** (`assign_slots_seeded`): cubes seed at frame 0, containers at frame 32 when they enter the view, so each slot is matched to an oracle object at its own first visible frame.
- **Two-target episodes** (54 of 100 training episodes): `target_cube_nodes` returns the ordered targets; `ActiveTarget` switches when the current target's container pose z > 0.15 m (RoboMME's `is_bin_pickup`). Step 6 errors, Q1/Q2 corruption (`corrupt`, `OnlineGraph`, shared draws in `draw_corruption`) and Step 10 metrics use the active target. Training graphs hold every cube.
- **Replay** records actor positions per frame (`obj_xyz`, NaN for video frames before the last reset frame). It stops at env success, like RoboMME's `dataset_replay.py` (episode 0: 152 of 158 recorded actions); `SGDataset` reads everything from the replay, so checks 3.2/3.3 now require "ends in success" and "prefix of the recording".
- **A2 text** uses RoboMME's grounded-subgoal convention `at <row, col>` (`DemonstrationWrapper.py:570` formats `<center_y, center_x>`); At a Glance updated.
- `download_h5` extracts the `.tar.xz` archive; `h5_task_goals` reads instructions without frames; `pass_fail` returns None; Setup filters three known warnings.

### Results so far (episode 0; written into the notebook's readings at the final run)
- Replay vs recording: mean |RGB diff| 0.00 on video frames, 0.78/255 at the last frame.
- Oracle projection: median 0.19 px from mask centroids.
- Perception: all six slots seed on the right objects. Identity accuracy 0.983 over 719 post-mid-swap slot-frames; 8 frames with two slots on one object; median centroid error 0.020 m. The green and blue cube slots are never hidden: Cutie keeps them on the containers that cover them. The container-2 slot error is 4-10 cm because the green-cube slot claims part of its mask (later class wins).
- Rules: frozen vs bound at t* 2.363 vs 2.362 px (the swap returns container 2 to its start, so t* does not separate them); over the 281 hidden frames frozen 9.4 px mean (max 39.1), bound 2.5 px (max 3.2).
- Levels at t*: Q0 2.4 px, Q1_20 2.3 (draw (2.5, -2.6)), Q1_50 7.4 (draw (6.3, -6.6)), Q2_25 / Q2_50 2.4 (Bernoulli did not fire for seed 0), Q3 0.3 px (the green-cube slot sits on the covering container).

### Q3 is close to the oracle on this task (read before comparing levels)
On VideoUnmaskSwap, Q3 (perceived graph + bound) is nearly as good as the oracle, and at t* of episode 0 even better than Q0 (0.3 px vs 2.4 px). The cause is a Cutie behaviour, not good memory: when a container drops over a cube, Cutie does not lose the cube's slot but keeps its mask on the container that now covers it. The cube slot then reports the covering container's position, which is exactly where the hidden cube is, because in this task the cube rides with its container. So Q3 is not a hard "perceived graph" condition here; it would degrade on tasks where the covering object and the hidden object move apart. Q0's 2.4 px is the bound rule's binding offset (cube centre vs container visible-surface centroid). The full oracle gives the hidden cube its container's position, so Q3 matches it almost exactly.

## Pod run, Steps 8-10 (2026-10-26, smoke mode)
Full notebook (122 cells) executed on a fresh `sg-main` kernel in a detached tmux window: all cells run, 33 checks PASS, none FAIL; wall time 290 s (caches warm). Executed copy in the session scratchpad; the notebook file in this folder is not yet executed in place, and the "= ?" readings are not yet written.

- **Q2 check (new cell 7.3b):** the first seed whose Q2_50 draw fires is 2; the hidden green cube then reports container 0's position (78.8, 91.1) instead of container 2's (139.1, 100.5): error at t* 61.0 px vs 2.4 px at Q0. PASS.
- **8.1 facts, from installed source (file:line printed):** SmolVLA defaults `tokenizer_max_length` 48 (configuration_smolvla.py:60), `train_expert_only` True (:70), `freeze_vision_encoder` True (:69), padding to (512, 512) (:45); wrapper keeps `front_rgb_list[-1]` (envs/robomme.py:159); `episode_length` 300 (envs/configs.py:932). Prefix width d = 960; prefix of one sample 321 tokens (15 text tokens); prefix builder `VLAFlowMatching.embed_prefix` (modeling_smolvla.py:605).
- **8.2-8.5:** marker size 6 (size 3's ring is hidden by its label after preprocessing); A2 worst case 127 tokens over the 2 cached episodes, limit 192 (re-checked over all cached episodes in pilot mode); A3 prefix 321 -> 329 (6 nodes + 2 masked pads), action shape unchanged; trainable parameters A0/A1/A2 6,247,424, A3 +513,729, A3feat +906,945.
- **9.5 smoke training (batch 8, 10 steps):** perception worker stopped first (`close_perception_worker`); loss finite (A0 3.64 -> 0.57, A2 4.17 -> 0.56); checkpoint reloads (138 adapter tensors).
- **9.6 speed at the pilot batch (32), excluding step 0:** A0 2.34 steps/s, A2 2.32 steps/s; peak GPU memory 16.9 GB. 20,000 steps per arm ~ 2.4 h on the RTX 4090; the 18-run grid ~ 43 GPU h of training. (The first run measured at batch 8 including warm-up; `train` now reports `steps_per_s_steady` and resets the peak-memory counter.)
- **10.1:** max_steps 1300; 115 frames returned at reset reach the graph.
- **10.3 smoke rollout (A2, Q0, test episode 0, two targets):** not solved, 1300 steps in 28 s, no container lifted; the robot hovers near container 0 and nudges it, and the online graph follows. Expected after 10 training steps; the run checks the loop, not the policy. Rollout strip is now a 6-column grid with the A2 object list under each frame.
- **10.5 counterfactual (1 episode):** no lift, so 0 of 0. **10.4, 10.6:** = ? (run in pilot mode).

## Moving to a multi-GPU pod (same /workspace volume, RTX 4090s)
Everything the run needs is on the volume: venvs (`/workspace/envs`), sources (`/workspace/src`), weights (`/workspace/models`), HF/torch/uv caches (`/workspace/.cache`), the H5 data and replay/graph caches (`notebooks/pilot/cache`), the Vulkan ICD and env file (`/workspace/env`). What lives outside the volume (tmux, Vulkan tools, Claude Code and its state) is handled by two scripts at the volume root.

**On the old pod, last thing before stopping it**
1. Stop running work (`tmux ls`; nothing in `sg-pilot` or `sg-train-*` should be running).
2. `/workspace/save_claude_home.sh`: snapshots `~/.claude` (settings, skills, project memory, sessions, `.credentials.json` = your Claude login, mode 600) and `~/.claude.json` into `/workspace/claude_home/` (directory mode 700), and every session scratchpad into `/workspace/claude_home/scratchpads/<session>/` (this build's runner scripts and the executed smoke notebook `full_run.ipynb` are in session `d7dd337e-...`). Writes `SAVED_AT`.
3. Stop the pod.

**Create the new pod**
- Same template (RunPod PyTorch: the venvs use its `/usr/bin/python3.12`), RTX 4090s, the same network volume at `/workspace`, and Environment Variable `NVIDIA_DRIVER_CAPABILITIES=all` (POD_RUNBOOK.md step 1; without it ManiSkill cannot render).
- GroundingDINO's CUDA extension is built for sm_89 (4090). A different GPU type needs it rebuilt (`TORCH_CUDA_ARCH_LIST` for that GPU; see "Environment" above).

**On the new pod**
1. `bash /workspace/setup_new_pod.sh`: installs tmux, libvulkan1 and vulkan-tools; restores `~/.claude` and `~/.claude.json` (any existing copy is kept as `*.pod-<time>`); installs Claude Code if missing; checks GPUs, Vulkan and both venvs. Safe to re-run.
2. `source /workspace/env/pod_env.sh`, `export JUPYTER_PATH=/workspace/envs/jupyter/share/jupyter`, then `cd /workspace/sg && claude`. Project memory is keyed by the path `/workspace/sg`, so it carries over. If the login was invalidated, run `/login`.

**Pilot training on several GPUs**
1. Build the 100-episode caches once (single process): set `RUN_MODE = "pilot"` in the Setup cell, then run the notebook only through the Step 7 cache cell:
   `python tools/run_upto.py sg_interfaces_pilot.ipynb /workspace/logs/pilot_cache.ipynb 84` (in a tmux window; 100 replays plus their graph caches).
2. `./run_arms_parallel.sh --interfaces A0,A2 --label pilot`: one tmux window per GPU (session `sg-train-pilot`), `CUDA_VISIBLE_DEVICES` set per window, 20,000 steps at batch 32 (~2.4 h per arm on a 4090, measured in 9.6). Checkpoints go to `runs/pilot_A0_s0`, `runs/pilot_A2_s0`, the paths notebook 9.7/10.4/10.5 use; logs, per-GPU runner scripts and exit codes to `/workspace/logs/train/pilot/`. Progress: `./run_arms_parallel.sh --label pilot --status`. More arms than GPUs are dealt round-robin and run one after another per window; finished arms (with `train_info.json`) are skipped on re-run.
3. Run the whole notebook in pilot mode: 9.7 loads the trained arms instead of retraining, 10.4/10.5 evaluate them (evaluation is still sequential in the notebook).
- Grid seeds use the same script, e.g. `--interfaces A0,A1,A2,A3,A3feat --tasks VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap --seed 1 --label grid` (needs the cache for all four tasks).
- Tested on the 1-GPU pod (2 episodes, 5 steps): A0 and A3 trained through `train.py` in the tmux window (exit 0; A3 saved `graph_tokenizer.pt`, 6,761,153 trainable parameters as in 8.5), and a re-launch skipped both.

### Open
- Write the "= ?" readings and At a Glance from the executed outputs, then the build-ipynb checker, self-check and HTML render.
- Node count varies per episode (episode 3 has 4 containers, 7 nodes). `node_tensor` pads to 8 slots with a mask, so A3 handles it, but no 7-node episode has been through training yet (smoke uses episodes 0-1); pilot mode exercises it.
