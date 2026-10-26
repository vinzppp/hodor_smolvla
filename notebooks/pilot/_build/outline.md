# Build outline: sg_interfaces_pilot.ipynb (build-ipynb v0.28/spec v0.29, Steps 0-1)

## Observe
1. Tasks (source: notebooks/pilot/INSTRUCTIONS.md, "Steps"; IDs keep the source numbering; text abbreviated to its first clause, full wording in the source line cited)
   S1.1 | "Print Python, torch, CUDA, LeRobot, mani_skill and robomme versions, GPU name and memory. Check: GPU visible." | §Step 1
   S1.2 | "Run vulkaninfo --summary or a ManiSkill smoke env for one step and show its rendered frame. Check: frame is not all black." | §Step 1
   S1.3 | "Set seeds ... Print RUN_MODE and the episode/step/eval counts it implies." | §Step 1
   S2.1 | "Load lerobot/robomme. Print the number of episodes per task for the 4 permanence tasks and the split sizes." | §Step 2
   S2.2 | "Pick one VideoUnmaskSwap training episode. Print every feature key with shape, dtype and meaning, and the fps." | §Step 2
   S2.3 | "Print the action vector dimension and what each dimension means, the state vector likewise, and the instruction text." | §Step 2
   S2.4 | "Plot the action dimensions over time (one panel per dimension)." | §Step 2
   S2.5 | "Show a grid of front and wrist frames at 6 moments ... Save to figures/02_frames.png." | §Step 2
   S2.6 | "Print the grounded_subgoal strings with the frame ranges where each holds." | §Step 2
   S2.7 | "Write one Markdown sentence: what this episode requires the policy to remember, and at which frame it is needed." | §Step 2
   S3.1 | "Create the RoboMME env ... reset it. Print observation keys and every flag that enables depth, camera parameters, segmentation or poses." | §Step 3
   S3.2 | "Replay the episode's actions ... Collect per frame: ... Print shape and meaning of each collected array." | §Step 3
   S3.3 | "Check the replay matches the dataset: mean absolute pixel difference on front RGB at 5 frames ... Check: mean difference below 5/255." | §Step 3
   S3.4 | "Show depth as a colormap next to RGB at 3 frames. Print depth units and range." | §Step 3
   S3.5 | "Show the segmentation map coloured with a legend of object names at 3 frames." | §Step 3
   S3.6 | "If any field in 3.2 is not reachable, write exactly what is missing and continue with what exists." | §Step 3
   S4.1 | "List the task objects ... assign stable node ids. Print the id-to-name table." | §Step 4
   S4.2 | "Project each object's world position into the front camera ... project_points ... Visual check: ... must sit on the objects." | §Step 4
   S4.3 | "Compute hidden per object per frame ... Visual check: a timeline strip ..." | §Step 4
   S4.4 | "Assemble oracle_graph of shape [T, N, F] ... Print the layout as a table and the full table at 4 key frames." | §Step 4
   S4.5 | "Plot the target cube's true (u, v) over time with hidden frames shaded and the containers' (u, v) overlaid ... Save to figures/04_oracle_track.png." | §Step 4
   S5.1 | "Decide which camera feeds the processor ... Build a task config ... state which [class source]." | §Step 5
   S5.2 | "Convert one replayed frame to the process_frame arguments ... Print each argument's shape, dtype and value range." | §Step 5
   S5.3 | "Run detection only on frame 0 (GroundingDINO); draw boxes with class names and scores ... Save to figures/05_detect.png." | §Step 5
   S5.4 | "Show the SAM masks for frame 0 as coloured overlays with their slot numbers." | §Step 5
   S5.5 | "Run Cutie tracking across the episode; show tracked-mask overlays at the same 6 moments ... Mark frames where a slot has no mask." | §Step 5
   S5.6 | "Show the 3D centroids: a top-down scatter ... and the centroids projected back onto the image." | §Step 5
   S5.7 | "Show the feature step: for one object, the mask over the DINOv2 patch grid ... and the resulting 768-d vector summarised." | §Step 5
   S5.8 | "Assemble perceived_graph in the same [T, N, F] layout ... Print it at the same 4 key frames next to the oracle table." | §Step 5
   S5.9 | "Report: identity accuracy after the swap ..., per-frame centroid error to oracle (plot), and the list of frames where tracking was lost." | §Step 5
   S6.1 | "Implement apply_rule(graph, rule, gripper_state, ee_pos) -> graph ... frozen ... bound ..." | §Step 6
   S6.2 | "Build the visibility-gated oracle ..." | §Step 6
   S6.3 | "Print the binding events (frame, hidden object, bound to)." | §Step 6
   S6.4 | "Plot the hidden target's position error to the oracle over time for frozen and bound ... Save to figures/06_rules.png." | §Step 6
   S6.5 | "Show the front image at the decision frame with three markers ... Check: bound error < frozen error at the decision frame." | §Step 6
   S7.1 | "Implement corrupt(graph, level, seed) -> graph with levels Q0, Q1_20, Q1_50, Q2_25, Q2_50, Q3 ..." | §Step 7
   S7.2 | "Show one image per level at the decision frame ... Save to figures/07_levels.png." | §Step 7
   S7.3 | "Print the target's position error per level at the decision frame." | §Step 7
   S8.1 | "Inspect SmolVLA first. Print its config values ..., input feature names, image preprocessing, the name and location of the function that builds the prefix, and the prefix length for one real sample ..." | §Step 8
   S8.2 | "A1 draw_markers(image, nodes) -> image ... show raw, after SmolVLA preprocessing, zoomed crop. Try 3 marker sizes ... Save to figures/08_markers.png." | §Step 8
   S8.3 | "A2 serialize(nodes) -> str ... token count ... Raise tokenizer_max_length ... decoded back to text ..." | §Step 8
   S8.4 | "A3 GraphTokenizer(nn.Module) ... Print the prefix shape and attention mask with and without A3. Check: forward pass runs and output action shape unchanged." | §Step 8
   S8.5 | "Equal capacity: the same LoRA configuration for every arm ... print trainable parameter count per arm ... state the difference explicitly." | §Step 8
   S9.1 | "Implement SGDataset ... Print one A1 sample's image, one A2 sample's text, one A3 sample's node tensor, at the same frame." | §Step 9
   S9.2 | "Cache the replayed oracle graphs to disk ... Print the cache size." | §Step 9
   S9.3 | "Show a batch: shapes of every tensor with their meaning." | §Step 9
   S9.4 | "Implement train(interface, task, seed, steps, out_dir) in sg_pilot.py; train.py is a thin CLI over it." | §Step 9
   S9.5 | "Smoke training: A0 and A2 for 10 steps each. Plot the loss. Check: loss is finite and the checkpoint saves and reloads." | §Step 9
   S9.6 | "Measure steps/second and GPU memory at the pilot batch size ... choose the pilot step count ... Print the estimated pilot training time per arm." | §Step 9
   S9.7 | "Pilot mode only: train A0 and A2 ... Plot both loss curves ... figures/09_loss.png." | §Step 9
   S10.1 | "Fix the wrapper issues: episode_length >= 1300; build the graph from the raw observation list at reset ... Print both settings." | §Step 10
   S10.2 | "Implement evaluate(ckpt, interface, quality, episodes, out_json) ... eval.py is a thin CLI ... graph updated online each step ..." | §Step 10
   S10.3 | "Smoke: 1 episode A2 at Q0. Save a rollout strip ... figures/10_rollout.png. Print success and step count." | §Step 10
   S10.4 | "Pilot mode only: A0 and A2 at Q0, 50 test episodes each. Print success rate with 95% Wilson intervals." | §Step 10
   S10.5 | "Counterfactual test for A2 ... Smoke: 1 episode; pilot: 20." | §Step 10
   S10.6 | "Pilot decision, printed as one line ... In smoke mode print = ? (run in pilot mode)." | §Step 10
   S10.7 | "Write run_grid.sh ... Print the total number of training and evaluation runs and the estimated GPU hours from 9.6. Do not run it." | §Step 10
   Every step | "opens with one Markdown line ..., runs, shows the listed intermediate output, and ends with a one-line check that prints PASS or FAIL with the measured value." | §Steps intro
2. Materials
   notebooks/pilot/INSTRUCTIONS.md | the specification | useful | all
   post_process_scene_cutie.py | HODOR pipeline | useful | HODORProcessor (init, _mask_to_instances, _obtain_features, _depth_to_pointcloud, _add_centroids, process_frame); _preprocess_wrist_bgr, _preprocess_depth (DROID-only, not reused)
   extract_part_masks_cutie.py | GroundedSAMDetector + CutieTracker | useful | detect_class (GroundingDINO -> NMS -> filters -> first accepted box -> SAM), CutieTracker.process (one Cutie processor per class)
   dinov2_encoder.py, vit.py | DINOv2 ViT-B/14 encoder via get_vit | useful | DINOv2Encoder.forward -> (global [1,768], patches [1,256,768])
   llm_entities.py | LLM entity extraction | useful (optional) | class list from instruction, cached
   configs/base.yaml, configs/giraffe.yaml | config template | useful | perception thresholds, slot_by_class pattern
   README.md, SECTION_3_1_NOTES.md | install + fidelity notes | useful | install steps; tau_m insertion not implemented
   rerun_logger.py, stage_timer.py, inspect_run.py | viz/timing utilities | not useful (not required by the spec) | -
   droid_utils.py, droid_hodor_test_cutie.py, data/, results/, Claude outputs/ | DROID loading, sample DROID episodes and earlier outputs | not useful (DROID-only; spec uses RoboMME) | -
   pretrained/groundingdino_swint_ogc.pth | GroundingDINO weights | useful | -
   pretrained/dino_vitbase16_pretrain.pth | DINO v1 weights | not useful (base.yaml defaults to DINOv2) | -
   output/research-map-scene-graph-memory-2026-10-24.md | research plan | useful | §1, §4, §5
   output/*.pdf, output/documents/*, Scene_Graph_Experiment_Requirements.csv | earlier reports | not useful (superseded by the research map) | -
   lerobot/robomme (HF), RoboMME paper, robomme_benchmark repo, LeRobot SmolVLA source | external | useful | not readable from this build machine; read on the GPU machine at Step 0 of the run
   Code used by tasks:
   post_process_scene_cutie.py | HODORProcessor.* used by S5.2-S5.8
   extract_part_masks_cutie.py | GroundedSAMDetector.detect_class, CutieTracker.process used by S5.3-S5.5
   dinov2_encoder.py / vit.py | get_vit used by S5.7
3. Constraints
   Never modify or delete existing repo files; new code only in notebooks/pilot/ | INSTRUCTIONS §Constraints
   Numbers come from a run or a cited source; unrun results shown as "= ? (run in pilot mode)" | §Constraints
   Verify every external API name against installed source; print file and line | §Constraints, §Materials
   One idea per cell; every figure titled, labelled with units, captioned | §Constraints
   API keys from environment only | §Constraints
   Executed in RUN_MODE="smoke"; smoke under 30 min on one 24 GB GPU | §Run modes
   Linux + NVIDIA GPU + Vulkan; if absent stop after Step 1 and report a Gap | §Environment
   Outputs: ipynb + HTML, sg_pilot.py, train.py, eval.py, run_grid.sh, figures/ | §Output
4. Gaps
   G1 No NVIDIA GPU / Vulkan on any machine reachable now (user's Mac is arm64 without CUDA; build container has no GPU) | environment | INSTRUCTIONS §Environment | run Steps 2-5 of build-ipynb on a RunPod 24 GB pod after the Vulkan check passes
   G2 HODOR tracks one object per class (detect_class keeps the first accepted box; one Cutie processor per class), but Swap tasks have two identical containers | conflict (code vs task) | extract_part_masks_cutie.py detect_class / process; _mask_to_instances | list the phrase twice ("container." "container."): the second pass skips boxes overlapping the first (overlap_threshold), giving two slots; verify on frame 0 in S5.3
   G3 HODOR geometry is DROID-specific: _depth_to_pointcloud subtracts 150 from cx (DROID crop), _add_centroids resizes masks to 800x800 and rotates them 180 deg | conflict | post_process_scene_cutie.py _depth_to_pointcloud, _add_centroids | subclass HODORProcessor in sg_pilot.py and override these two methods for RoboMME camera geometry; mark # changed
   G4 RoboMME replay exposure of segmentation, object poses and depth unverified | unclear | INSTRUCTIONS §Materials (facts to verify) | check in S3.1; S3.6 handles absence
   G5 SmolVLA prefix-building function, tokenizer limit and LeRobot wrapper behaviour come from main-branch reading | unclear | INSTRUCTIONS §Materials | verify in S8.1 and S10.1 against installed source
   G7 The repo copy of INSTRUCTIONS.md was stale (older T1-T10 version) | conflict | file md5 | replaced on 2026-10-26; the 17,796-byte version governs
   G6 Bounding-box field order in HODOR is [row_min, row_max, col_min, col_max] normalised, unlike (u, v) pixel order used by the oracle | conflict (convention) | _mask_to_instances | convert explicitly in the S5.8 assembler and say so

## Tree (think-logical), revised after independent P6 (p6.md: 18 REQUIRED, 8 OPTIONAL, all addressed)
Pilot notebook plan
Goal: a runnable data-to-training pipeline that builds oracle and perceived object scene graphs for RoboMME VideoUnmaskSwap, builds three content-matched interfaces (A1 markers, A2 text, A3 tokens) plus A3feat (A3 + DINOv2 feature), with matched LoRA, trains and evaluates A0 and A2 only (pilot), and prints the pilot decision   [all S]   ← sequence
  kind: artifact | conditions: spec = notebooks/pilot/INSTRUCTIONS.md (17,796 bytes, md5 1902a325...; the earlier repo copy was stale and is replaced); RUN_MODE smoke for the delivered run; Linux GPU with Vulkan; repo files unchanged | acceptance: notebook executes top to bottom; each sub-step prints PASS/FAIL with its value; decision line printed (= ? in smoke)
Start from: given: INSTRUCTIONS.md (above), repo code, lerobot/robomme, RoboMME env, installed SmolVLA; rules: pinhole projection, Hungarian assignment, Wilson interval, LoRA; sources: RoboMME paper, LeRobot docs, installed source files (file:line printed)
  Step 1 GPU visible, Vulkan renders, a CUDA forward/backward runs, seeds set   [S1.1-S1.3]   ← parts
    kind: constraint | conditions: none | acceptance: GPU name; rendered frame not all black; one tiny forward/backward on cuda; seeds printed
  Step 2 the selected VideoUnmaskSwap episode is described, and the decision frame t* (where the remembered fact is first needed) = ? is fixed here   [S2.1-S2.7]   ← sequence
    kind: explanation | conditions: one training episode; t* chosen from subgoal ranges (2.6) before any graph is built, and reused by Steps 6, 7, 10 | acceptance: keys/shapes, action plot, 6-moment grid, subgoal ranges, t* printed
  Step 3 the replay reproduces the episode and adds whichever of depth / camera parameters / segmentation / poses RoboMME exposes (= ?)   [S3.1-S3.6]   ← sequence
    kind: artifact | conditions: same task seed; replay length = dataset length (needed for per-frame alignment in 9.1) | acceptance: mean |RGB diff| < 5/255 at 5 frames of which at least one is before the covers lower; target cube's replay pose projects inside the cube's pixels in the dataset image at that frame; lengths equal; list of exposed fields
  Step 4 oracle_graph [T, N, 8] = [id, label_id, u, v, x, y, z, hidden] follows every object, including the hidden cube moving with its container   [S4.1-S4.5]   ← cases
    kind: artifact | conditions: needs segmentation and poses from Step 3 | acceptance: projected points on objects; hidden timeline; target (u,v) moves during the swap
    case A: segmentation and poses available → build as specified
    case B: either missing → stop after Step 3, report the Gap, Steps 4-10 not run (all later results = ?)
    4.2 project_points(xyz, K, T) → (u, v) in front pixels   ← rule: pinhole projection, from Step 3
      kind: artifact | conditions: K and extrinsic convention as printed in S3.1, converted to a 4x4 camera-to-world OpenCV matrix | acceptance: points inside object masks
  Step 5 the existing HODOR pipeline, run through an adapter, yields perceived_graph in the same layout; its slot identity through the swap and its centroid error are measured, not assumed (= ?)   [S5.1-S5.9]   ← sequence
    kind: artifact | conditions: repo files unchanged; manual class config (the LLM list would give one "container"), ordered containers first then cubes, "container." listed twice; per-class independent Cutie processors with no mutual exclusion, overlap resolved by class order (later class overwrites); option noted: one InferenceCore with objects=[1,2] in sg_pilot.py | acceptance: per-stage figures; slots seeded on distinct containers at frame 0 (PASS condition); identity accuracy and error plot
    5.1 adapter config and camera choice   ← given: S5.1; source: configs/giraffe.yaml pattern
    5.2 frame → process_frame arguments; RoboMMEProcessor(HODORProcessor) overrides _depth_to_pointcloud and _add_centroids (# changed): no cx-150 shift, no 800x800 resize, no 180-degree rotation; extrinsics converted to 4x4 cam-to-world OpenCV; depth in metres inside [depth_min_m, depth_max_m] (window reset if the scene needs it); mask, depth, RGB and K at one resolution; _preprocess_wrist_bgr/_preprocess_depth not called   ← source: post_process_scene_cutie.py l.221-280
      kind: artifact | conditions: as stated | acceptance: one visible object's perceived centroid within a stated tolerance of its oracle xyz
    5.3 detection: GroundingDINO per class → NMS (0.8) → area filters (class and global ranges), confidence ≥ threshold, reject a box covering > overlap_threshold (0.7) of any earlier mask → first surviving box → SAM   ← source: extract_part_masks_cutie.py detect_class l.115-191
      kind: artifact | conditions: two "container" slots are distinct only if slot 2's accepted box lies on the other container | acceptance: slot→oracle assignment at the seeding frame by Hungarian on projected centroids, one-to-one within radius r (PASS/FAIL); seed_events printed
    5.4-5.5 SAM masks; Cutie propagation; frames with no mask marked   ← source: extract_part_masks_cutie.py CutieTracker.process l.380-480
    5.6 centroid = mean of visible-surface points of the mask   ← source: post_process_scene_cutie.py _add_centroids (overridden)
    5.7 feature = mean of normalized DINOv2 patch tokens over the 16x16 patches containing any mask pixel (224x224 resize, aspect not kept); empty mask → zeros   ← source: post_process_scene_cutie.py _obtain_features l.173-219; dinov2_encoder.py DINOv2Encoder.forward
    5.8 perceived_graph: (u,v) = project_points(perceived centroid) into the front camera; hidden = 1 when bbox[:4] == 0 this frame; stale xyz/features of absent slots flagged (process_frame resets only bbox[:4], l.340), not treated as measurements   ← rule: field conversion incl. bbox order [rmin, rmax, cmin, cmax] (G6), from 4.2, 5.3-5.7
    5.9 identity accuracy = fraction of post-swap frames (and at t*) where the frame-0 assignment still holds; frames where both slots map to one object counted; centroid error labelled "visible-surface centroid vs pose origin" (pixels and metres) = ?   ← source: run output, from 4, 5.3, 5.8
      kind: quantity | conditions: as stated | acceptance: printed values and plot
  Step 6 on oracle input, bound tracks the hidden target through the swap (implementation check: bound < frozen is expected by construction when the binding names the right container); the informative test of bound is Q3 in Step 7   [S6.1-S6.5]   ← compare
    kind: comparison | conditions: visibility-gated oracle (oracle while visible); apply_rule causal (frames ≤ t only); footprint = radius r in world xy around the container position, tie-break nearest; gripper rule: gripper closed below threshold g and ee within d; error = Euclidean in front pixels and in metres; evaluated at t* from Step 2 | acceptance: binding event names the oracle's true container; error curves; errors at t* printed
    options: frozen, bound; dimensions: error over time, error at t*
  Step 7 six designed corruption levels; target error at t* per level = ? (expected order only within families: Q1_20 < Q1_50, Q2_25 < Q2_50 in expectation)   [S7.1-S7.3]   ← sequence
    kind: artifact | conditions: Q0 = visibility-gated oracle + bound (error = Step 6 bound error, not 0); Q1 offset drawn once per episode, Gaussian in pixels, 3D offset obtained by back-projecting at the object's depth; Q2 one Bernoulli draw per episode; Q3 = perceived + bound | acceptance: per-level image and printed errors
  Step 8 the same four fields per object (id, label, (u,v), hidden) are given to A1-A3, and A3feat adds e_i (the only content difference, compared with A3); label_mode table|text implemented; fidelity after preprocessing is measured; LoRA identical across arms   [S8.1-S8.5]   ← sequence
    kind: artifact | conditions: A1 and A3 built and forward-checked only, A0 and A2 trained in this notebook; A3 projector parameters are the only capacity difference, reported | acceptance: SmolVLA facts with file:line; marker visible after preprocessing at the chosen size; worst-case A2 token count over all training frames ≤ new tokenizer_max_length (argmax sample printed); A3 forward runs with unchanged action shape; parameter table
  Step 9 SmolVLA trains on per-frame Q0 graphs (visibility-gated oracle + bound, never the raw oracle); smoke loss is finite and checkpoints reload; pilot step count set from measured speed   [S9.1-S9.7]   ← sequence
    kind: artifact | conditions: graphs aligned per frame (replay length = dataset length); speed measured for A0 and A2 on the build GPU, A1/A3 assumed equal to A0 (unmeasured); estimates valid for the same GPU type; pilot results = ? | acceptance: batch shapes; loss plot; reload check; steps/s and memory printed
  Step 10 closed-loop evaluation with the wrapper fixed; decision = ? in smoke   [S10.1-S10.7]   ← sequence
    kind: decision | conditions: episode_length ≥ 1300; graph built causally from the raw history at reset and updated online; eval env exposes segmentation and poses; n = 50 per arm only if the test split has ≥ 50 episodes (from S2.1); rule: continue iff A0 > 0 and A2's 95% Wilson lower bound > A0's upper bound (e.g. A0 = 5/50 → A2 needs ≥ 17/50); stop reasons: A0 at floor, no gain, graph ignored per 10.5 | acceptance: at the first policy step the online Q0 target position equals the offline one for the same seed; rollout strip; counterfactual = lifted container (first whose z rises > h cm) vs graph-named container; decision line; run_grid.sh (joint 4-task training per interface x seed; held-out-task runs) with run counts
Open questions: G1 no GPU now | default: build on a RunPod 24 GB pod; G2 container instancing | default: duplicated phrase with per-class Cutie, measured in 5.3/5.9; switch to one InferenceCore with objects=[1,2] if 5.3 fails; Step 4 case B | default: stop and report

## Architecture contract
Reader and outcome: checkpoint answer 2026-10-26: little background assumed (a); comfortable with framework code; goal = able to modify the implementation; the reader can run each step, read every intermediate output, and modify sg_pilot.py functions (interfaces, rules, corruptions) for the full grid
Top-level units: Step 1 Environment, Step 2 Look at the data, Step 3 Replay, Step 4 Oracle scene graph, Step 5 Perceived scene graph, Step 6 Update rules, Step 7 Graph-quality levels, Step 8 Interfaces, Step 9 Training, Step 10 Evaluation and pilot decision (source labels and order kept)
Index and root relation: Step (source-defined); sequence, each step uses earlier outputs
Supporting sections: At a Glance; Setup (imports, RUN_MODE, paths, sg_pilot import); References (RoboMME paper, LeRobot docs, repo files); no AI Use Disclosure (not coursework, not required)
Code: functions live in sg_pilot.py, shown once with inspect.getsource beside first use; reading depth: internals for Steps 4-8 (graph building, rules, corruptions, interfaces), interface for Step 9-10 training/eval loops, purpose only for env setup and caching
Run model: one linear run per RUN_MODE; Steps 9.7 and 10.4 run only in pilot mode

## Requirement contract
S1.1-S1.3 | Step 1 | ## Step 1 cells 1-3
S2.1-S2.7 | Step 2 | ## Step 2, one ### per sub-step; S2.7 as Markdown result sentence
S3.1-S3.6 | Step 3 | ## Step 3; S3.3 PASS/FAIL line
S4.1-S4.5 | Step 4 | ## Step 4; figures 04_*
S5.1-S5.9 | Step 5 | ## Step 5; figures 05_*; S5.9 printed metrics
S6.1-S6.5 | Step 6 | ## Step 6; comparison table frozen vs bound + figures 06_*
S7.1-S7.3 | Step 7 | ## Step 7; figure 07_levels
S8.1-S8.5 | Step 8 | ## Step 8; figure 08_markers; parameter table
S9.1-S9.7 | Step 9 | ## Step 9; figure 09_loss (pilot)
S10.1-S10.7 | Step 10 | ## Step 10; figure 10_rollout; decision line; run_grid.sh
Every-step PASS/FAIL | all | last cell of each ### sub-step
Constraints (repo unchanged, = ? rule, API verification, figure labels, keys) | output-wide | checker + final validity review
