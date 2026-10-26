# Build notes (build-ipynb Step 2 done, Steps 3-5 pending on a GPU machine)

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
