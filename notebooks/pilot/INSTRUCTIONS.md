# Notebook instructions: scene-graph interfaces for SmolVLA on RoboMME (pilot, data to training)

## Goal
Build one Jupyter notebook that walks, step by step and runnable, from raw benchmark data to a trained and evaluated pilot policy for this research question:

> With the same per-object content and the same trainable parameters, which interface for giving an explicit object scene graph to a VLA (image markers, text, or projected prefix tokens, with or without appearance features) degrades least as the scene graph gets worse?

The notebook teaches and pilots. It must:
- reuse the existing scene-graph pipeline in this repo;
- split the work into small steps;
- after every step, show the intermediate result: a printed shape and sample with its meaning, and an image or plot wherever the result is visual.

It ends with a printed pilot decision. All reusable code lives in a module that command-line scripts also import, so the full experiment grid later runs exactly the same code outside the notebook.

## Run modes
Define `RUN_MODE` in the first code cell. Every heavy cell reads it.
- `"smoke"`: VideoUnmaskSwap, 2 training episodes, 10 training steps, 1 evaluation episode. Must finish in under 30 minutes on one 24 GB GPU. This is the mode the notebook is executed in when it is built.
- `"pilot"`: VideoUnmaskSwap, all 100 training demos, the step count chosen in Step 9.6, 50 test episodes. Run by the user later.

## Environment
- How to set up the pod, check Vulkan, upload the code and hand off to a Claude Code session there: `POD_RUNBOOK.md`.
- Linux with an NVIDIA GPU and working Vulkan. ManiSkill renders with Vulkan, and `HODORProcessor.__init__` hard-codes `torch.device("cuda:0")`. The notebook will not run on macOS or CPU-only machines.
- If the build machine has no GPU or no Vulkan, stop after Step 1 and report it as a Gap. Never fake an output.
- Install LeRobot with the RoboMME and SmolVLA extras, following https://huggingface.co/docs/lerobot/robomme. Note the numpy/gymnasium pin conflict described there.
- Install this repo's perception dependencies as in `README.md` (GroundingDINO, segment-anything, Cutie from source; SAM ViT-H checkpoint in `pretrained/`).

## Materials (read all before writing code)
1. **This repo.** Read in full: `post_process_scene_cutie.py`, `archive/droid-section3.1/droid_hodor_test_cutie.py`, `dinov2_encoder.py`, `vit.py`, `llm_entities.py`, `rerun_logger.py`, `configs/base.yaml`, `archive/droid-section3.1/configs/giraffe.yaml`, `README.md`, `SECTION_3_1_NOTES.md`.
   - Reuse: `HODORProcessor.process_frame(hand_bgr_rotated, frame_num, episode_num, depth, hand_rgb_unrotated, intrinsics, extrinsics)`.
   - It returns `FrameOutput.representation`, shape `[N+1, 768]` (slot 0 = global CLS feature, slots 1..N = object features), and `FrameOutput.bboxes`, shape `[N+1, 7]` (4 box values + 3D centroid).
   - Slots come from `task.classes` and `task.slot_by_class`.
   - Internal steps to expose one at a time: `tracker.process`, `_mask_to_instances`, `_add_centroids` (via `_depth_to_pointcloud`), `_obtain_features`.
2. **The research plan:** `output/research-map-scene-graph-memory-2026-10-24.md`, sections 1, 4 and 5.
3. **The benchmark:**
   - RoboMME paper: https://arxiv.org/abs/2603.04639
   - dataset `lerobot/robomme` on Hugging Face
   - LeRobot RoboMME docs (link above)
   - env repo: https://github.com/RoboMME/robomme_benchmark (`doc/env_format.md`, `scripts/dataset_replay.py`)
4. **The policy:** SmolVLA in the installed LeRobot (`configuration_smolvla.py`, `modeling_smolvla.py`).

**Facts to verify, not assume.** These come from reading LeRobot main-branch source on 2026-10-26. Print the file and line you checked for each:
- SmolVLA defaults are `tokenizer_max_length = 48`, `train_expert_only = True` and `freeze_vision_encoder = True`. Images are padded to 512 and reduced to 64 visual tokens per frame.
- The RoboMME LeRobot wrapper (`RoboMMEGymEnv._convert_obs`) keeps only `front_rgb_list[-1]`, so the step-0 history video of Video* tasks never reaches the policy.
- `env.episode_length` defaults to 300. VideoUnmaskSwap averages about 348 steps; the paper caps evaluation at 1300.
- The dataset holds front and wrist RGB, state, actions and a per-frame `grounded_subgoal` string, but no depth. Depth, camera parameters, segmentation and object poses must come from replaying in the simulator.

## Code layout (create these files)
```
notebooks/pilot/
  sg_pilot.py                  all reusable functions and classes (see the list in each step)
  sg_interfaces_pilot.ipynb    the notebook; imports sg_pilot and explains each function where first used
  train.py                     CLI: python train.py --interface {A0,A1,A2,A3,A3feat} --tasks VideoUnmaskSwap[,ButtonUnmask,...] --seed 0 --steps N --label-mode {table,text} --out runs/...
  eval.py                      CLI: python eval.py --ckpt ... --interface ... --task ... --quality {Q0,Q1_20,Q1_50,Q2_25,Q2_50,Q3} --episodes 50 --out results/...json
  run_grid.sh                  loops over interfaces x seeds x tasks x qualities by calling train.py and eval.py; written, not run
  figures/                     every plot and image the notebook saves
```
The notebook calls the same functions that `train.py` and `eval.py` call. Show each function's source once in the notebook, at the step where it is first explained (e.g. with `inspect.getsource`).

## Steps
Each step:
- opens with one Markdown line on what goes in and what comes out (shapes and meaning);
- runs;
- shows the listed intermediate output;
- ends with a one-line check that prints PASS or FAIL with the measured value.

### Step 1. Environment
1.1. Print the Python, torch, CUDA, LeRobot, mani_skill and robomme versions, plus the GPU name and memory. Check: GPU visible.
1.2. Run `vulkaninfo --summary`, or a ManiSkill smoke environment for one step, and show its rendered frame. Check: the frame is not all black.
1.3. Set seeds (Python, numpy, torch) to 0. Print `RUN_MODE` and the episode, step and evaluation counts it implies.

### Step 2. Look at the data
2.1. Load `lerobot/robomme`. Print the number of episodes per task for the 4 permanence tasks and the split sizes.
2.2. Select one VideoUnmaskSwap training episode. Print every feature key with its shape, dtype and meaning, and the fps.
2.3. Print the action vector's dimension and what each dimension means, the state vector likewise, and the instruction text.
2.4. Plot the action dimensions over time, one panel per dimension.
2.5. Show a grid of front and wrist frames at 6 moments: start, objects visible, covers lowered, mid-swap, swap done, end. Save to `figures/02_frames.png`.
2.6. Print the `grounded_subgoal` strings with the frame ranges where each one holds.
2.7. Write one Markdown sentence saying what this episode requires the policy to remember, and at which frame the remembered fact is needed.

### Step 3. Replay in the simulator to get privileged data
3.1. Create the RoboMME env for the same task and seed and reset it. Print the observation keys and every flag that enables depth, camera parameters, segmentation or poses.
3.2. Replay the episode's actions (via `scripts/dataset_replay.py` or the env API, whichever exists). Collect per frame: front and wrist RGB, depth, intrinsics, extrinsics, segmentation map, segmentation id-to-object table (ManiSkill `segmentation_id_map`), and every object's world pose. Print the shape and meaning of each collected array.
3.3. Check that the replay matches the dataset: mean absolute pixel difference on front RGB at 5 frames, printed, plus a side-by-side image of dataset and replay at the mid-swap frame. Check: mean difference below 5 out of 255.
3.4. Show depth as a colormap next to RGB at 3 frames. Print the depth units and range.
3.5. Show the segmentation map in colours with a legend of object names at 3 frames.
3.6. If any field in 3.2 is not reachable, write exactly what is missing and continue with what exists.

### Step 4. Oracle scene graph
4.1. List the task objects (cubes, containers, any buttons) from the id table and assign stable node ids. Print the id-to-name table.
4.2. Project each object's world position into the front camera with the intrinsics and extrinsics. Implement `project_points(xyz, K, T_world_cam) -> uv` in `sg_pilot.py`.
   - Visual check: draw the projected points on the front image at 3 frames. They must sit on the objects.
4.3. Compute `hidden` per object per frame: 1 if the object has zero segmentation pixels in both cameras.
   - Visual check: a timeline strip, one row per object, dark where hidden.
4.4. Assemble `oracle_graph` with shape `[T, N, F]`, where F = `[id, label_id, u, v, x, y, z, hidden]`. Print the layout as a table, and print the full table at 4 key frames.
4.5. Plot the target cube's true (u, v) over time with hidden frames shaded, and overlay the containers' (u, v). The swap must show as the hidden cube's position moving with its container. Save to `figures/04_oracle_track.png`.

### Step 5. Perceived scene graph with the existing pipeline
Do not edit any repo file. Write the adapter in `sg_pilot.py`.

5.1. Decide which camera feeds the processor, and justify it in one line; the repo was built for a wrist camera. Build a task config for VideoUnmaskSwap (classes, `slot_by_class`, area ranges). Get the classes either from `llm_entities.py` with cached entities or manually, and state which.
5.2. Convert one replayed frame into the `process_frame` arguments: image orientation, BGR order, depth units as `_preprocess_depth` expects, intrinsics and extrinsics in the format `_depth_to_pointcloud` expects. Print each argument's shape, dtype and value range.
5.3. Run detection only on frame 0 (GroundingDINO). Draw the boxes with class names and scores on the image. Save to `figures/05_detect.png`.
5.4. Show the SAM masks for frame 0 as coloured overlays with their slot numbers.
5.5. Run Cutie tracking across the episode. Show the tracked-mask overlays at the same 6 moments as Step 2.5. Mark frames where a slot has no mask.
5.6. Show the 3D centroids:
   - a top-down scatter (x, y) of the point cloud with each object's centroid, at 2 frames;
   - the centroids projected back onto the image.
5.7. Show the feature step: for one object, the mask over the DINOv2 patch grid (which patches are averaged), and the resulting 768-d vector summarised as norm and first 8 values.
5.8. Assemble `perceived_graph` in the same `[T, N, F]` layout as Step 4.4. Print it at the same 4 key frames next to the oracle table.
5.9. Report three things:
   - identity accuracy after the swap (does each slot still follow the same physical object, judged against the oracle id table?);
   - per-frame centroid error to the oracle, as a plot;
   - the list of frames where tracking was lost.

   Check: print the identity accuracy. It is a measurement, not a pass condition.

### Step 6. Hidden-object update rules
6.1. Implement `apply_rule(graph, rule, gripper_state, ee_pos) -> graph` in `sg_pilot.py`, with two rules:
   - `frozen`: keep the last seen position.
   - `bound`: when an object becomes hidden while its last position lies inside another tracked object's footprint, it follows that object (keeping the offset at binding time). When the gripper is closed on it, it follows the end effector.
6.2. Build the visibility-gated oracle: oracle values while visible, the rule while hidden.
6.3. Print the binding events: frame, hidden object, bound to.
6.4. Plot the hidden target's position error against the oracle over time for `frozen` and `bound`, with the swap interval shaded. Save to `figures/06_rules.png`.
6.5. Show the front image at the decision frame with three markers: the true position, `frozen`'s position and `bound`'s position. Check: `bound` error < `frozen` error at the decision frame.

### Step 7. Graph-quality levels
7.1. Implement `corrupt(graph, level, seed) -> graph` with these levels:
   - `Q0`: oracle with the `bound` rule.
   - `Q1_20`, `Q1_50`: Gaussian position offset on (u, v) and a consistent 3D offset, sigma 20 px or 50 px.
   - `Q2_25`, `Q2_50`: with probability 0.25 or 0.5 per episode, the hidden target takes the other container's position.
   - `Q3`: the perceived graph with the `bound` rule.
7.2. Show one image per level at the decision frame, with the target marker as each level reports it. Save to `figures/07_levels.png`.
7.3. Print the target's position error per level at the decision frame.

### Step 8. Three interfaces with identical content
Each interface receives exactly the same fields per object: id, label, (u, v), hidden. A0 means no graph. A3feat (8.4b) is A3 plus the DINOv2 node feature; it is the only arm whose content differs, by design, and is compared with A3 alone.

8.1. **Inspect SmolVLA first.** Print:
   - its config values (the verify list above);
   - the input feature names;
   - image preprocessing (pad and resize sizes);
   - the name and location of the function that builds the prefix;
   - the prefix length for one real sample: image tokens per camera, language tokens, state token.
8.2. **A1 `draw_markers(image, nodes) -> image`.** Labelled markers at (u, v), with a distinct style for hidden objects. Show three images:
   - the raw marked image;
   - the image after SmolVLA's own preprocessing;
   - a crop zoomed on the marker after preprocessing.

   Try 3 marker sizes and choose the smallest that stays clearly visible after preprocessing; justify it in one line. Save to `figures/08_markers.png`.
8.3. **A2 `serialize(nodes) -> str`.** Use a fixed template, e.g. `"objects: [1 red cube, hidden, (520, 290)]; [3 container, visible, (520, 300)] ..."`.
   - Print the string and its token count under the SmolVLA tokenizer.
   - Raise `tokenizer_max_length` so the instruction plus the graph is never truncated, and print the new value.
   - Show the token ids decoded back to text to prove nothing is cut.
8.4. **A3 `GraphTokenizer(nn.Module)`.** An MLP over `[u/W, v/H, hidden, label_emb]` plus a learned id embedding (ids randomly permuted per sample during training), producing `[N, d]`, with d equal to SmolVLA's prefix width. Insert it into the prefix through a subclass or wrapper of the function found in 8.1.
   - Print the prefix shape and the attention mask with and without A3.
   - Check: a forward pass runs and the output action shape is unchanged.
8.4b. **A3feat.** Same as A3, but the node vector also carries the 768-d DINOv2 feature e_i: `[e_i, u/W, v/H, hidden]` → MLP → `[N, d]`.
   - For hidden nodes, e_i is the last feature from a frame where the object was fully visible.
   - For training data, compute e_i from the simulator's segmentation masks with the repo's `_obtain_features` logic (DINOv2 forward plus masked mean), skipping detection and tracking.
   - Print one node vector's layout, `[N, 771]`, and the prefix shape.
8.4c. **Label encoding switch.** Implement `label_mode` with two values:
   - `table`: a learned embedding per label id (closed set). This is the default.
   - `text`: the mean of SmolVLA's own token embeddings of the label word, which works for unseen object names.

   Show both for "cube" and "container". Only `table` is used in the pilot.
8.5. **Equal capacity.** Apply the same LoRA configuration to every arm (same rank, same target modules).
   - Print the trainable parameter count per arm.
   - A3 adds the projector's parameters. Print the difference and state it explicitly in the results.

### Step 9. Training data and training
9.1. Implement `SGDataset`. It wraps the LeRobot dataset and, per frame, adds the Q0 graph from the replay, applying the chosen interface. Print one A1 sample's image, one A2 sample's text and one A3 sample's node tensor, all at the same frame.
9.2. Cache the replayed oracle graphs to disk, so training does not re-simulate. Print the cache size.
9.3. Show a batch: shapes of every tensor with their meaning.
9.4. Implement `train(interface, task, seed, steps, out_dir)` in `sg_pilot.py`. `train.py` is a thin CLI over it.
9.5. **Smoke training.** Run A0 and A2 for 10 steps each. Plot the loss. Check: the loss is finite and the checkpoint saves and reloads.
9.6. Measure steps per second and GPU memory at the pilot batch size, and print both. From them, choose the pilot step count. LeRobot docs report about 4 h for 20k steps on one A100; use that only as a reference. Print the estimated pilot training time per arm.
9.7. **Pilot mode only:** train A0 and A2 on VideoUnmaskSwap with all 100 demos. Plot both loss curves on one axis and save to `figures/09_loss.png`.

### Step 10. Evaluation and pilot decision
10.1. Fix the wrapper issues:
   - set `episode_length >= 1300`;
   - build the graph from the raw observation list at reset, so the history video is used.

   Print both settings.
10.2. Implement `evaluate(ckpt, interface, quality, episodes, out_json)` in `sg_pilot.py`, and make `eval.py` a thin CLI over it. During the rollout, the graph is updated online each step from the simulator: the oracle with the rule, corrupted per quality level.
10.3. **Smoke:** run 1 episode for A2 at Q0. Save a rollout strip with front frames every 50 steps, A2's input text shown under each frame, and the graph markers drawn. Save to `figures/10_rollout.png`. Print success or failure and the step count.
10.4. **Pilot mode only:** A0 and A2 at Q0, 50 test episodes each. Print the success rate with 95% Wilson intervals.
10.5. **Counterfactual test for A2:** put the hidden target under the wrong container in the graph. Record how often the policy lifts the container the graph names rather than the true one. Smoke: 1 episode. Pilot: 20.
10.6. **Pilot decision, printed as one line:**
   - **continue** if A0 success > 0 and A2's interval lies above A0's;
   - otherwise **stop**, with the reason (A0 at floor, no gain from the graph, or graph ignored per 10.5).

   In smoke mode, print `= ? (run in pilot mode)`.
10.7. Write `run_grid.sh` with two parts.
   - **Main grid:** one model per (interface, seed), trained **jointly on all 4 permanence tasks**, giving one set of weights per arm with the instruction distinguishing the task. Interfaces A0, A1, A2, A3, A3feat; seeds {0, 1, 2}. Evaluate every model on each of the 4 tasks × qualities {Q0, Q1_20, Q1_50, Q2_25, Q2_50, Q3}.
   - **Held-out task:** for A0, A2, A3feat, seed 0, train on 3 tasks and evaluate on the 4th, which is never seen in training. Make the held-out task a parameter, defaulting to ButtonUnmaskSwap. Print the total number of training runs and evaluation runs, and the estimated GPU hours from 9.6. Do not run it.

## Constraints
- Never modify or delete existing files in this repo. All new code goes in `notebooks/pilot/`.
- Every number shown comes from a run in this notebook or a cited source. Results not produced in smoke mode appear as `= ? (run in pilot mode)`, never estimated as if measured.
- Verify every external API name against the installed source before use, and print the file and line used.
- One idea per cell, and keep cells short. Every figure gets a title, axis labels with units, and a one-line caption saying what to look at.
- API keys are read from the environment, never written into the notebook.

## Output
- `notebooks/pilot/sg_interfaces_pilot.ipynb`, executed in smoke mode, plus its HTML render
- `notebooks/pilot/sg_pilot.py`, `train.py`, `eval.py`, `run_grid.sh`
- `notebooks/pilot/figures/` with every saved figure
