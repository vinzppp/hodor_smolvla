# Research map: scene-graph memory for imitation policies

Seed: Qian et al., *Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs*, CVPR 2026 ([arXiv 2606.01072](https://arxiv.org/abs/2606.01072)).
Searched 2026-10-24 (about 35 queries by the builder plus 23 query rounds by an independent critic). Stop condition: budget, not saturation. Scope from checkpoint: both within-replication and new directions; no access to the authors' data; under $500.
Draft produced by the research-academics skill v0.1 (first hand run).

## 1. Summary

| # | Opening | Type | Verdict | Confidence |
|---|---|---|---|---|
| A | Rebuild the seed's graph diffusion policy and test H1/H2 on the 4 public RoboMME permanence tasks, front+wrist and wrist-only cameras | improve (replication on public data) | reframe of I7 | medium |
| B | Containment binding: a hidden node follows the node that covers it (or the gripper when grasped), compared with the seed's frozen centroid, on the Swap tasks | improve | reframe of I1 (+ I5's gripper rule) | medium |
| C | Three graph sources: perceived, visibility-gated oracle, full oracle, to split perception error from memory error | improve (diagnostic) | reframe of I2 | medium-high |
| D | DROID measurement: how often manipulated objects are missing from the instruction, and how many the LLM entity extractor recovers | disrupt (questions the instruction-driven object list) | reframe of I6 | low |

A + B + C form one project. The critic estimates about 250-350 GPU-hours, roughly $125-175 on an RTX 3090 at $0.50/h, under $400 even if debugging doubles it. D is a cheap side section.

The main change since your 2026-09-12 map: two September preprints now sit right on top of this space. [Ledger](https://arxiv.org/abs/2609.34554) and [ECoMEM](https://arxiv.org/abs/2610.00801) both give a pi0.5 VLA explicit per-object records on RoboMME and reach 91-97% on VideoUnmaskSwap. Both use 2D image coordinates and a large pretrained VLA. What is left is a controlled study of a small 3D object-graph policy: which part fails (perception, memory content, update rule), at a cost a student can afford. It is not a new memory mechanism.

## 2. Problem tree (to level 2)

```text
Goal: a dynamically maintained, task-driven object scene graph fed to a diffusion policy improves long-horizon success
      under partial observability over RGB, visible-objects and PTP baselines            ← support
  conditions: few task objects named in the instruction; objects static while unseen; accurate identification;
              per-task policies from 300-400 demos
  1 Setting: needed objects leave view in long tasks (single wrist camera real; 2 cameras sim)
  2 Graph keeps one node per task object over time                                     ← sequence
    2.1 LLM lists task nouns from the instruction        2.2 Grounding DINO + SAM first-frame masks
    2.3 XMem tracking (repo: Cutie)                      2.4 node = [DINOv2 masked-mean, 2D box, 3D centroid]
    2.5 out of view: box zeroed, centroid frozen         2.6 new node if mask overlap < tau_m (tau_m unspecified)
    2.7 no edges; relations implicit in shared coordinates
  3 Policy reads all nodes as tokens (no retrieval)                                    ← sequence
    3.1 E_i = [e_i, MLP(b_i), MLP(c_i)] + scene root     3.2 transformer encoder → conditioning
    3.3 DDPM diffusion policy, DDIM sampling, 3.5-3.8 Hz on a 3090
  4 Evidence                                                                           ← compare
    4.1 sim Spot (3 tasks, 400 demos)   4.2 real Franka wrist-only (4 tasks, 300 demos)
    4.3 baselines RGB / Visible-Objects / PTP          4.4 ablations w/o 3D, w/o locations
    4.5 stated limits: identification accuracy, clutter, dynamic scenes
```
Node 2.5 is where the openings concentrate: the seed keeps a hidden object's position "assuming no movement unless re-observed".

## 3. Literature map

### Node coverage
| Node | Papers | Usual approach | What no checked paper does |
|---|---|---|---|
| 1 setting | RoboMME, LIBERO-Mem (Embodied-SlotSSM), SOMA, See2Act | memory benchmarks are now standard; active looking is the alternative | - |
| 2 graph | Ledger, ECoMEM, Embodied-SlotSSM, RoboEXP, Out of Sight Still in Mind, Lost & Found | explicit external object records, 2D | 3D object centroids with a containment update, tested on a policy at small scale |
| 2.1 entity list | seed only; MOO, GROOT assume the object is given | closed list from instruction | measure how often needed objects are unnamed (3 critic rounds, none found) |
| 2.5 frozen centroid | Ledger (cover-following), ECoMEM (INSIDE relation, stale flag, age), Object Permanence Filter | follow the cover, or mark stale | - (taken, in 2D) |
| 3 policy | LTM, EventVLA, MemBodied, Chameleon, ECoMEM, GeoMem-VLA, HALO, IMPRINT | VLA memory on pi0.5 is crowded | seed-style object-graph diffusion policy on a public memory benchmark |
| 4 evidence | RoboMME, MIKASA-Robo, RMBench, LIBERO-Mem, DROID | public sim benchmarks with demos | three-way perception / memory / policy split for an object-graph policy |
| 3.x retrieval under budget | not searched in this run (your 09-12 map covers it) | | |

### Reading list (in order)
1. **Ledger**, [2609.34554](https://arxiv.org/abs/2609.34554): method and per-task table. Its tracker-only variant already gets 92.0 / 76.0 on VideoUnmaskSwap / ButtonUnmaskSwap. This is the bar for opening B.
2. **ECoMEM**, [2610.00801](https://arxiv.org/abs/2610.00801): §4.3 and Table 5 (11-field records, 32×32 grid positions, INSIDE relation, one token per record). This is the closest work to anything token-based.
3. **RoboMME**, [2603.04639](https://arxiv.org/abs/2603.04639) plus [env code](https://github.com/RoboMME/robomme_benchmark): permanence task definitions, cameras, replay. Your testbed.
4. **The seed's §3.2** plus Diffusion Policy (Chi et al. 2023): what you must rebuild (gap G1).
5. **More Structure, Not More Capacity**, [2607.09825](https://arxiv.org/abs/2607.09825): oracle-vs-perceived object representations for imitation learning without memory. A template for opening C.
6. **Embodied-SlotSSM / LIBERO-Mem**, [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37337): object slots over time; a second public benchmark if RoboMME saturates.
7. **LTM**, [2610.21362](https://arxiv.org/abs/2610.21362), and **EventVLA**, [2606.20092](https://arxiv.org/abs/2606.20092): implicit memory baselines you will be compared against.

## 4. Openings

### A. Public replication of H1/H2 (from I7)
**Claim.** Rebuild the seed's §3.2 graph diffusion policy from Diffusion Policy code. Test H1 (graph beats visible-only) and H2 (removing 3D hurts later subtasks) on VideoUnmask, ButtonUnmask, VideoUnmaskSwap and ButtonUnmaskSwap, under front+wrist and wrist-only cameras, with 3 seeds each.

**Why it could work.** RoboMME has 100 demos per task in LeRobot format. Its environment exposes depth and camera intrinsics/extrinsics, and demos can be replayed, so your perception pipeline can produce 3D centroids. The seed trained 200 epochs on one 3090.

**Holes.**
- With a fixed front camera, the 2D box already locates objects, so H2 may not show up. Wrist-only input recreates the seed's condition.
- 100 demos is a third or less of the seed's 300-400.
- Implementation fidelity cannot be checked without the seed's code. Report this.

**Smallest test.** A 1-task pilot on VideoUnmask, about $25. It kills the plan if the visible-only policy is not clearly below the full-oracle graph policy.

### B. Containment binding vs frozen centroid (from I1, I5's gripper rule folded in)
**Claim.** Add a rule-based update: when a node is lost while overlapping another tracked node, its centroid follows that node; while the gripper is closed on it, it follows the end effector. Hypothesis: this matches the frozen graph on no-swap tasks and closes most of the gap to the full-oracle graph on Swap tasks, with no LLM planner.

**Closest work.**
- Ledger: same cover-following mechanism, but in 2D, read by an LLM planner, on pi0.5.
- ECoMEM: an INSIDE relation fed as tokens to pi0.5.

So this is a small-policy, 3D re-test of a known mechanism, and it should be framed that way.

**Holes.**
- The two papers above score 91-97 on VideoUnmaskSwap, a high bar for a 100-demo policy.
- Binding fails when containers overlap during the swap.
- Identical containers stress Cutie's identity tracking.

**Smallest test.** Before any training: replay one Swap episode and plot the cube centroid under the frozen and bound rules against ground truth.

### C. Perception vs memory split (from I2)
**Claim.** Train the same policy on three graph sources and report results per task, with Swap tasks separated:
- perceived: Grounding DINO + SAM + Cutie + depth;
- visibility-gated oracle: true pose only while the segmentation mask is visible, frozen or bound otherwise;
- full oracle: true pose always.

Perceived vs gated isolates perception; gated vs full isolates memory.

**Closest work.**
- RoboMME's oracle-subgoal split: on language subgoals, not object graphs.
- 2607.09825's 3D oracle: no memory.

**Holes.**
- The gated oracle needs B's update rule, so C depends on B.
- 3 seeds × 50 episodes may not separate a 5-10 point gap. Use 100 episodes on the two key tasks.
- It is unverified whether ManiSkill's `segmentation_id_map` is reachable through RoboMME's wrapper.

**Smallest test.** Confirm segmentation masks and object poses come out of a RoboMME replay.

### D. Unnamed objects in instructions (from I6)
**Claim.** On 200 DROID episodes, label the objects the gripper contacts or moves. Report the share absent from the instruction, and the recall of your existing `llm_entities.py`.

**Holes.**
- It is a dataset statistic, not a policy result.
- It needs a labelling rule.
- It takes about 6-10 hours of labelling. API cost is under $5.

**Smallest test.** Label 30 episodes first; stop if omissions are under 5%.

### Dropped
- **I3: graph tokens into SmolVLA.** ECoMEM already puts explicit object records as VLA prefix tokens on RoboMME; a smaller backbone adds no testable claim.
- **I4: position covariance growing while unseen.** With a fixed motion prior it is a deterministic function of time since seen, and the Swap failure is a wrong position, not missing uncertainty.
- **I5 as its own idea.** The trigger (a held object out of view of both cameras) is probably rare in RoboMME, and in the Swap tasks the environment moves the objects, not the gripper. Its gripper rule lives on in B.

## 5. Data plan and budget
- **Instead of the authors' data:** RoboMME, with 1,600 demos in LeRobot format ([dataset](https://huggingface.co/datasets/lerobot/robomme)). It runs on ManiSkill, Linux only, and Docker avoids the numpy pin conflict. DROID (CC-BY 4.0, [project](https://droid-dataset.github.io)) is only for opening D. Its standard release has RGB and raw stereo but no ready depth, per the critic's check of the Daft docs. LIBERO-Mem is the fallback benchmark.
- **Compute:** RunPod RTX 3090 $0.50/h, RTX 4090 $0.89/h (pricing page, 2026-10-24). The critic's estimate (±50%, its own arithmetic):
  - graph policy with precomputed DINOv2 features: 1-1.5 GPU-h per training run;
  - RGB diffusion policy: 3-5 GPU-h;
  - closed-loop evaluation: about 1.4 GPU-h per (task, variant, seed).
- **Order:**
  1. Replay check (C's smallest test, then B's).
  2. 1-task pilot (A), about $25.
  3. Full grid only if the pilot separates the variants.

## 6. Open questions
- G1: the seed's policy code is not released; it has to be rebuilt from the paper.
- G2: the seed's success numbers exist only as bar charts; read them from the PDF figures if you need exact baselines.
- G4: the repo's task names (cabinet, chicken) differ from the paper's (Clean-Table, Feed-Animals). Follow the paper for any claim.
- Ledger has no code link; ECoMEM's code status was not checked.
- From your 09-12 map, SuperMap and DynaMem were not re-verified in this run. HitMem was opened by the critic ([2609.00950](https://arxiv.org/abs/2609.00950)).

## Appendix A. Observe, full tree, evidence table, coverage, all ideas (builder)

```text
## Observe
1. Tasks
   T1 | "帮忙做 academic research，然后想想有什么新的研究突破口" | chat 2026-10-24
   T2 | scope: both within-replication and new directions ("都要") | chat checkpoint
   T3 | "我没有办法获得他的原始数据，可能需要个方案用其他数据" | chat checkpoint
   T4 | budget under $500 | chat checkpoint
2. Materials
   arXiv 2606.01072v2 (seed) | the paper | seed | §3.1, §3.2, §4.1, §4.2, §5 (HTML read; figure numbers not readable)
   CVPR 2026 supplement PDF | seed supplement | seed | GPT-4 entity selection 100%, 200 epochs on one 3090, 300 demos, no tau_m, no data release stated
   scene-graph-robot/README.md | user's fork of the official repo (jianingq/scene-graph-robot) | seed code | perception pipeline only; 10 sample DROID-format episodes per task (giraffe, cabinet, pineapple, chicken)
   SECTION_3_1_NOTES.md | user's replication notes | project note | 3.1 coverage table; tau_m insertion not implemented; §3.2 policy not in repo
   Research Proposal.pdf | user's proposal | project note | H1 memory > visible-only; H2 no-3D hurts memory tasks
   Robotic_AI_Memory_Literature_Map_2026-09-12.pdf | user's earlier literature map, 43 refs, cutoff 2026-09-12 | project note | gap list, 8 innovation points, benchmark map
3. Constraints
   no access to authors' full demonstration data | T3
   total compute + API spend < $500 | T4
4. Gaps
   G1 §3.2 policy code not released in repo | missing file | SECTION_3_1_NOTES | implement from Diffusion Policy (Chi et al. 2023) public code
   G2 seed success numbers only in bar charts | missing knowledge | HTML | read figures from PDF if exact numbers are needed
   G3 seed's Spot MuJoCo sim not stated as released | missing file | paper + supplement | use a public memory benchmark instead
   G4 repo task names (cabinet, chicken) differ from paper tasks (Clean-Table, Feed-Animals) | conflict | README vs paper §4.2 | follow the paper for claims

## Tree (think-logical, to level 2)
Goal: feeding a dynamically maintained, task-driven object scene graph to a diffusion policy improves long-horizon success under partial observability over RGB, visible-objects and long-context (PTP) baselines   ← support
  kind: claim | conditions: few task-relevant objects named in the instruction; objects static while unseen; accurate identification; per-task policies from 300-400 demos | acceptance: subtask/full success bars in sim (3 tasks) and real (4 tasks)
  1 Setting: needed objects leave the camera view during long tasks   ← given: §1, §4
    kind: constraint | conditions: single wrist camera (real) / two cameras (sim); episodes 100-1000 steps
  2 Graph construction keeps one node per task object across time   ← sequence
    kind: artifact | conditions: object set from instruction; static unseen objects | acceptance: tracking/segmentation failure counts (4/64 seg failures real)
    2.1 LLM names task-relevant nouns from the instruction (GPT-4/5)          ← source: §3.1, supp §6
    2.2 Grounding DINO + SAM give first-frame masks                          ← source: §3.1
    2.3 XMem tracks identity (repo: Cutie)                                    ← source: §3.1; README
    2.4 node = [DINOv2 masked-mean feature, 2D box, 3D centroid from depth + odometry]   ← source: §3.1
    2.5 out of view: box zeroed, centroid frozen at last seen value          ← source: Alg. 1
    2.6 new detection added if mask overlap < tau_m (tau_m unspecified)       ← source: Alg. 1; supp
    2.7 no edges; relations implicit in shared coordinates                   ← source: §3.1
  3 Policy reads all nodes as tokens   ← sequence
    kind: artifact | conditions: all nodes fit in context (no retrieval); task-specific training
    3.1 E_i = [e_i, MLP(b_i), MLP(c_i)] + scene root (CLS, full box, origin)  ← source: §3.2
    3.2 transformer encoder over all tokens → conditioning vector            ← source: §3.2
    3.3 DDPM-trained diffusion policy, DDIM sampling, action chunks, 3.5-3.8 Hz on a 3090   ← source: §3.2, §4.2
  4 Evidence   ← compare
    kind: comparison | conditions: same demos per method; success read from stacked bars
    4.1 sim Spot MuJoCo: Throw Trash, Heat-up Tea, Heat-up Tea Long; 400 demos   ← source: §4.1
    4.2 real Franka, wrist ZED Mini only: Pineapple-Bowl, Feed-Animals, Feed-Giraffe, Clean-Table; 300 demos   ← source: §4.2
    4.3 baselines RGB DP, Visible-Objects, PTP (PTP dropped in real)          ← source: §4.1-4.2
    4.4 ablations w/o 3D, w/o locations: equal on first subtask, drop on later   ← source: §4.1
    4.5 limits stated: needs accurate identification; clutter and dynamic scenes   ← source: §5

## Evidence table (map-literature)   searched 2026-10-24; stop: budget (~35 queries) for 1-4; saturation not reached for 3
L-qian-26 | Qian et al. 2026, Expanding Spatial and Temporal Context ... With Scene Graphs | https://arxiv.org/abs/2606.01072 ; CVPR 2026 https://openaccess.thecvf.com/content/CVPR2026/html/Qian_Expanding_Spatial_and_Temporal_Context_for_Robotic_Imitation_Learning_With_CVPR_2026_paper.html | published CVPR 2026 | read: sections (3,4,5, supp) | nodes: all
L-ledger-26 | Dieudonné, Jedlicki, Yang 2026, Where Memory Belongs: Ledger, an Object Ledger for Memory-Augmented VLAs | https://arxiv.org/abs/2609.34554 | preprint | read: sections (method, results) | nodes 2, 2.5, 3
   does: external object ledger (description, 2D image coord or lost, stationary intervals, covered timestamp) from SAM3 tracking + VLM transcript; LLM planner (Claude Sonnet 5) reads it at step boundaries and sends subgoals with a coordinate slot to fine-tuned pi0.5; RoboMME 4-suite avg 64.3 vs 45.9; permanence 86.7 vs 56.2
   limit: observed: 2D image coords only, object set fixed from first frame, relies on LLM planner; trails frame-sampling baseline on StopCube, PatternLock, RouteStick | strength: supports 2, 3
L-ecomem-26 | Liu, Wang, Schwager, Xu, Wu 2026, ECoMEM: Explicit Concept Memory for Memory-Dependent Robot Control | https://arxiv.org/abs/2610.00801 | preprint | read: abstract | nodes 2, 3
   does: library of grounded concept records written by evidence-based Writer, learned Reader turns them into memory tokens for a VLA; leads 15/16 RoboMME tasks; real 86.1 vs 8.6 | strength: discovery
L-robomme-26 | Dai et al. 2026, RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies | https://arxiv.org/abs/2603.04639 ; dataset https://huggingface.co/datasets/lerobot/robomme | published ICML 2026 (oral page, PMLR v306) | read: abstract + LeRobot docs | nodes 1, 4
   does: 16 ManiSkill tasks in 4 suites (counting, permanence incl. VideoUnmaskSwap / ButtonUnmaskSwap, reference, imitation); 100 demos/task, 1,600 total, LeRobot format; 14 memory variants on pi0.5; finding: memory representation effectiveness is task-dependent | strength: supports 4 (data plan)
L-ltm-26 | Sun et al. 2026, LTM: Latent Transition Based In-Context Memory for VLAs | https://arxiv.org/abs/2610.21362 | preprint | read: abstract | nodes 3 | does: latent transition memory with KV cache on pi0.5; RoboMME 58.54, LIBERO-Mem 80.5 | strength: discovery
L-slotssm-26 | Chung et al. 2026, Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective (Embodied-SlotSSM, LIBERO-Mem) | https://ojs.aaai.org/index.php/AAAI/article/view/37337 | published AAAI 2026 | read: abstract | nodes 2, 3, 4 | does: slot identities kept over time + slot SSM + relational encoder; introduces LIBERO-Mem (object-level partial observability) | strength: discovery
L-eventvla-26 | Yang et al. 2026, EventVLA: Event-Driven Visual Evidence Memory | https://arxiv.org/abs/2606.20092 | preprint | read: abstract | nodes 3 | does: keyframe evidence memory chosen by predicted keyframe probability; RoboTwin-MeM benchmark | strength: discovery
L-membodied-26 | Pala et al. 2026, MemBodied: Recurrent Associative Memory for VLAs | https://arxiv.org/abs/2609.28256 | preprint | read: abstract | nodes 3 | does: fixed-size associative state + episode anchor; RMBench | strength: discovery
L-rmbench-26 | Chen et al. 2026, RMBench: Memory-Dependent Robotic Manipulation Benchmark | https://arxiv.org/abs/2603.01229 | preprint | read: abstract | nodes 4 | does: 9 memory tasks, Mem-0 modular policy for ablations | strength: discovery
L-mikasa-25 | Cherepanov et al. 2025, Memory, Benchmark & Robots (MIKASA-Robo) | https://arxiv.org/abs/2502.10550 | published ICLR 2026 (proceedings PDF) | read: abstract | nodes 4 | does: 32 memory-intensive tabletop tasks, pip-installable; RL framing | strength: discovery
L-chameleon-26 | Guo et al. 2026, Chameleon: Control-Indexed Prospective Memory | https://arxiv.org/abs/2603.24576 | preprint | read: abstract | nodes 3, 4 | does: ~60M event-memory policy; LIBERO-10 87.1, MemoryBench 97.3, MIKASA-Robo 75.1 | strength: discovery
L-soma-26 | Li et al. 2026, Spatial Memory for Out-of-Vision Manipulation in VLA (SOMA) | https://arxiv.org/abs/2605.22283 | preprint | read: abstract | nodes 1, 2 | does: spatial-semantic memory from head-camera scans, refinement, instruction retrieval; 5 real out-of-vision tasks | strength: discovery
L-geomem-26 | Zhang et al. 2026, GeoMem-VLA | https://arxiv.org/abs/2610.29866 | preprint | read: abstract | nodes 2.4, 3 | does: learned geometry tokens distilled from a geometry model, stored in temporal memory; scene-level not object-level (abstract) | strength: discovery
L-oc-il-26 | Li, Chapin, Chen, Peters, Kshirsagar 2026, More Structure, Not More Capacity: Object-Centric Representations for Visuomotor IL | https://arxiv.org/abs/2607.09825 | preprint | read: abstract | nodes 2.4, 3.1 | does: frozen slot encoder beats dense DINO by ~22 pts on ManiSkill3 PickCube; 16x more patch tokens no better; occlusion main bottleneck on StackCube | strength: discovery
L-see2act-26 | Wang et al. 2026, See2Act: Diffusion Models for Active Perception in Robot Imitation | https://arxiv.org/abs/2606.23625 | preprint | read: abstract | nodes 1 | does: diffusion policy also predicts viewpoints under occlusion; RLBench +33 pts | strength: discovery
L-roboexp-24 | Jiang et al. 2024, RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration | https://proceedings.mlr.press/v270/jiang25c.html | published CoRL 2024 (PMLR v270) | read: title | nodes 2 | strength: discovery
L-oosm-23 | Huang et al. 2023, Out of Sight, Still in Mind (DOOM/LOOM) | https://arxiv.org/abs/2309.15278 | preprint | read: abstract | nodes 2.5 | does: transformer relational dynamics over object trajectory history from partial point clouds; handles occlusion and reappearance; beats implicit memory baseline | strength: discovery
L-egg-26 | Nguyen, Verdoja, Kyrki 2026, Event-Grounding Graph | https://arxiv.org/abs/2510.18697 | published RA-L (DOI 10.1109/LRA.2026.3669042) | read: abstract | nodes 2.7 | does: scene graph linking objects to events for spatio-temporal QA | strength: discovery
L-droid-24 | DROID dataset (Khazatsky et al. 2024) | https://droid-dataset.github.io (via roboticscenter.ai dataset page) | dataset | read: dataset page | nodes 4 | does: 76k Franka trajectories, 564 scenes, wrist ZED Mini RGB-D + 2 external, language, CC-BY 4.0, DROID-100 sample on HF | strength: supports data plan
Prices: RunPod pods page 2026-10-24: RTX 4090 $0.89/h, RTX 3090 $0.50/h, A100 80GB $1.79/h (https://www.runpod.io/gpu-cloud/pricing)
Project note (user's map, 2026-09-12): HitMem (temporal decay, displacement search), SuperMap (confidence, stale pruning), HALO (supervised retrieval), MemoryVLA, SAM2Act/MemoryBench, DynaMem — leads only, not re-verified except titles seen in search results (SAM2Act 2501.18564, MemoryVLA 2508.19236).

## Node coverage
1 setting | L-robomme-26, L-slotssm-26, L-soma-26, L-see2act-26 | memory benchmarks now standard | active looking (See2Act) as alternative to memory | -
2 graph | L-ledger-26, L-ecomem-26, L-slotssm-26, L-roboexp-24, L-oosm-23 | explicit external object records are now common (Ledger, ECoMEM after Sep 2026) | 2D ledger + LLM planner; concept records; slots | no kept paper feeds 3D object nodes with explicit staleness/uncertainty to a policy and tests moved-while-unseen; queries: "object permanence moved while unseen robot memory policy", "diffusion policy object memory uncertainty belief ..." (0 direct)
2.1 entity selection | L-qian-26 only | closed set from instruction | - | no paper measures how often needed objects are unnamed in instructions (query: "task-relevant object selection ... LLM grounding failure", 0 direct)
2.5 frozen centroid | L-ledger-26 (covered timestamp), L-oosm-23 (relational dynamics) | freeze or mark lost | - | none propagates a hidden object's position from robot action
3 policy | L-ltm-26, L-eventvla-26, L-membodied-26, L-chameleon-26, L-ecomem-26, L-geomem-26 | VLA memory is crowded; most built on pi0.5 and RoboMME/LIBERO-Mem/RMBench | latent, keyframe, recurrent, concept | object-graph tokens inside a small open policy on RoboMME: not found
4 evidence | L-robomme-26, L-mikasa-25, L-rmbench-26, L-slotssm-26 (LIBERO-Mem), L-droid-24 | public sim benchmarks with demos | - | seed evaluated only on own tasks; no oracle-vs-perceived graph split (query "perception vs memory attribution", not run separately → partial)
3.2 retrieval under budget | not searched this run (user's map covers it)

## Ideas (think-creative), kept 7 of 14 raw
I1 Test the frozen-centroid assumption: run the scene-graph policy on RoboMME permanence tasks with and without swaps, then add a staleness field (time since seen) and a re-observation reconciliation step
   generator: invert (2.5 objects static while unseen) | changes: 2.5, 3.1 | type: improve
   why it might work: RoboMME has VideoUnmaskSwap/ButtonUnmaskSwap (L-robomme-26 docs); Ledger permanence gains come from explicit object records (L-ledger-26) | closest known work: L-ledger-26 (2D, LLM planner) | smallest test: oracle-pose graph tokens + DP on VideoUnmask vs VideoUnmaskSwap, 100 demos each | kills it if: swap and no-swap success equal for frozen graph (assumption never bites)
I2 Separate perception from memory: train the same policy with an oracle graph (simulator poses and IDs) and a perceived graph (Grounding DINO + SAM + Cutie on rendered RGB-D), report the gap per subtask
   generator: re-measure | changes: 4, 2 | type: improve (diagnostic)
   why it might work: seed attributes most failures to the policy and reports 4/64 seg failures (L-qian-26 §4.2); occlusion is the main bottleneck for object-centric IL on StackCube (L-oc-il-26 abstract) | closest known work: not checked | smallest test: 2 RoboMME tasks × {oracle, perceived} × 3 seeds | kills it if: gap < 5 pts on both tasks
I3 Make a generalist policy read the scene graph: feed the graph tokens as extra prefix tokens to an open small VLA (e.g. SmolVLA in LeRobot) and compare with its frame-history memory on RoboMME
   generator: combine (graph tokens × generalist VLA) | changes: 3 | type: disrupt (replaces per-task DP)
   why it might work: speculative; Ledger/ECoMEM show explicit records help pi0.5 (L-ledger-26, L-ecomem-26) | closest known work: L-ecomem-26, L-ledger-26, L-slotssm-26 | smallest test: SmolVLA fine-tune on 4 permanence tasks ± graph tokens | kills it if: no gain over frame sampling
I4 Give each node an uncertainty that grows while unseen (position covariance from a motion prior, reset on re-detection), as a token feature, borrowed from multi-object tracking
   generator: borrow (MOT / Kalman track management) | changes: 2.5, 3.1 | type: improve
   why it might work: speculative; user's map lists temporal decay in HitMem and confidence in SuperMap (project note) | closest known work: HitMem, SuperMap (project note, not re-verified) | smallest test: same as I1 with uncertainty feature vs staleness scalar | kills it if: no gain over the staleness scalar
I5 Kinematic propagation: while an object is grasped (gripper closed on it), move its node with the end effector instead of freezing it when it leaves view
   generator: limit-to-target (seed §5 dynamic scenes) + extreme | changes: 2.5 | type: improve
   why it might work: speculative; RoboEXP builds action-conditioned graphs (title only) | closest known work: L-roboexp-24 (title only), L-oosm-23 | smallest test: RoboMME reference/imitation tasks where held cube leaves wrist view; compare frozen vs propagated | kills it if: held objects rarely leave view in the data (count first)
I6 Drop the instruction-driven object list: admit every detected object and let attention choose, or measure how often instructions omit objects the robot must use
   generator: invert (2.1 needed objects are named) | changes: 2.1 | type: disrupt
   why it might work: seed's Clean-Table drawer is not a fruit named target (L-qian-26 §4.2 task list); DROID gives 76k instructions with wrist RGB-D (L-droid-24) | closest known work: not checked | smallest test: on DROID-100, label which manipulated objects are absent from the instruction; run the user's existing LLM entity extractor | kills it if: < 5% of episodes use an unnamed object
I7 Public replication testbed: reproduce H1/H2 (memory vs visible-only, 3D vs no-3D) on RoboMME + DROID instead of the authors' data
   generator: limit-to-target (no data access) | changes: 4 | type: improve
   why it might work: RoboMME 100 demos/task in LeRobot (L-robomme-26); seed trains 200 epochs on one 3090 with 300 demos (supp) | closest known work: RoboMME's own memory variants on pi0.5 | smallest test: 1 permanence task, DP with {RGB, visible-objects, graph, graph w/o 3D} | kills it if: RGB DP already near ceiling on chosen task
Raw not kept: dual-address retrieval (dup of user's map pt 5, out of scope for few nodes); KV materialization (budget); edges via relation tokens (weaker than I2 as first step); active looking + graph (dup of See2Act direction); multi-session lifelong (budget); LLM-free entity selection by VLM pointing (merged into I6); counting suite with graph (graph has no event memory, weak fit)
```

## Appendix B. Independent critique with query log

```text
Critique   (independent)

Facts checked once and used by several ideas (all opened 2026-10-24):
  F1 RoboMME (Dai et al., arXiv 2603.04639v3, ICML 2026) https://arxiv.org/abs/2603.04639 : 16 tasks, 100 demos/task, front + wrist RGB 256x256 + proprio; Video* tasks get a history video at step 0; Permanence = VideoUnmask / ButtonUnmask (uncover the named-colour cube hidden under a container) and *Swap variants (containers swap positions while cubes are hidden); avg steps 217-400 for permanence; eval cap 1300 steps; pi0.5 no-memory 18.7 / 6.7 on VideoUnmaskSwap / ButtonUnmaskSwap. Policies in the paper are pi0.5 only (80k steps, bs 64); no DP baseline.
  F2 RoboMME env code https://github.com/RoboMME/robomme_benchmark (doc/env_format.md): built on ManiSkill; exposes front/wrist depth, camera intrinsics/extrinsics, raw "maniskill_obs" via include_* flags; demos replayable with scripts/dataset_replay.py. Segmentation and object poses are not listed as fields; ManiSkill itself gives rgb+depth+segmentation obs and env.unwrapped.segmentation_id_map (https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html), so oracle poses/masks need a small wrapper over the RoboMME env (not verified to work through RoboMME's wrapper).
  F3 Ledger (Dieudonne, Jedlicki, Yang, arXiv 2609.34554, 2026-09-28) https://arxiv.org/abs/2609.34554 : per-object record {description, 2D image coord or lost, stationary intervals, covered timestamp c_i}; when an object is covered, its coordinate follows the covering entity's track. RoboMME permanence: Ledger 92.7 / 90.0 / 91.3 / 72.7; a "Tracker (reproduced)" variant without event history already gets 92.7 / 88.0 / 92.0 / 76.0 (VideoUnmask / ButtonUnmask / VideoUnmaskSwap / ButtonUnmaskSwap).
  F4 ECoMEM (Liu, Wang, Schwager, Xu, Wu, arXiv 2610.00801, 2026-09-30) https://arxiv.org/abs/2610.00801 : pi0.5 backbone; 11-field concept records incl. entity IDs, 32x32 image-grid coords (no 3D), INSIDE(subject, container) relations, status tentative/confirmed/stale, confidence, age; one 2048-d memory token per record placed in the policy prefix; VideoUnmaskSwap 96.7, ButtonUnmaskSwap 84.0; trained 4xH200 x 24 h.
  F5 Seed code/text (https://www.seas.upenn.edu/~dineshj/publication/qian-2026-scaling/qian-2026-scaling.pdf, same title and author list as arXiv 2606.01072): out-of-view centroid kept "assuming no movement unless re-observed"; no oracle graph; project site sites.google.com/view/objgraph.
  F6 Compute: LeRobot hardware guide https://huggingface.co/docs/lerobot/en/hardware_guide : image DP on RTX 4090/3090, bs 8, 28k steps at 640x480 = ~2-4 h; SmolVLA ~3-6 h for 56k steps on a 24 GB card. Seed: 200 epochs, 300 demos, one 3090 (supp).
     Estimate for this project (mine, +-50%): RoboMME task ~100 demos x ~300 steps = ~30-40k frames; graph-token DP with precomputed DINOv2 features, 200 epochs at bs 64 = ~100k steps, ~1-1.5 GPU-h; RGB DP baseline at 256x256 ~3-5 GPU-h; closed-loop eval with the perception stack at the seed's ~3.5-4 Hz, 50 episodes x ~400 steps = ~1.4 GPU-h per (task, variant, seed). One task x 5 variants x 3 seeds = ~40-50 GPU-h = ~$20-25 on a 3090, ~$40 on a 4090.

  I1 Restated: add to the seed's object graph a per-node staleness field (time since last seen) and a reconciliation step on re-observation, and show on RoboMME VideoUnmask(Swap) / ButtonUnmask(Swap) that the frozen-centroid graph fails under swaps and the staleness version recovers.
     prior work: F3 Ledger partial (object record + covered status + cover-following on the same 4 tasks; differs: 2D coords, LLM planner, pi0.5, no staleness scalar) | F4 ECoMEM partial (age + stale status + INSIDE relation as policy tokens on RoboMME; differs: 2D grid, pi0.5, learned Writer) | Embodied-SlotSSM / LIBERO-Mem (Chung et al., AAAI 2026) https://ojs.aaai.org/index.php/AAAI/article/view/37337 partial (slot identity over time under object-level partial observability; no staleness/containment test) | HALO (Shah et al., arXiv 2606.25136) https://arxiv.org/abs/2606.25136 different (attention retrieval over raw history) | rounds: 3
     holes: the fix does not address the failure: in the Swap tasks the cube moves because its container moves; a staleness scalar says the centroid is old but not where the cube went, and reconciliation happens only after a container is lifted, i.e. after the decisive pick (fatal) | the "kills it if swap = no-swap" test is near-certain not to fire: per F5 a frozen cube centroid points at the old location, so the frozen graph fails swaps by construction; the result is low-information (risk) | the working mechanism (covered object follows its cover) is already shown by F3, and F3's tracker-only variant gets 92/76 on the Swap tasks, so a DP version must be framed as "does a small 3D-graph DP need anything beyond cover-binding", not as a new mechanism (risk) | 100 demos/task vs the seed's 300-400; a from-scratch DP may sit near floor on ButtonUnmaskSwap (pi0.5 gets 6.7) (risk) | identical-looking containers make the Cutie track itself the likely failure point during a swap (risk, links to I2)
     verdict: reframe → "On the 4 RoboMME permanence tasks, a seed-style 3D object-graph DP with a containment-binding rule (when a node is lost while overlapping another tracked node, its centroid follows that node; when the gripper is closed on it, it follows the end effector) matches the frozen graph on no-swap tasks and closes most of the gap to a full-oracle graph on Swap tasks, without an LLM planner." Re-checked: prior work same 4 searches; closest F3 (cover-following, 2D, LLM planner, pi0.5) and F4 (INSIDE relation, pi0.5); the reframe differs in 3D centroids, rule-based binding, and a ~50M-parameter DP trainable for ~$25/task; it is a small-policy re-test of F3's mechanism, not a new mechanism. Holes left: F3/F4 numbers (91-97 on VideoUnmaskSwap) set a high bar a 100-demo DP may not reach (risk); binding fails when two containers overlap during the swap motion (risk); the evaluation does distinguish success from failure because frozen vs bound differ only on Swap tasks | confidence: medium (raise by: one replayed Swap episode showing the cube centroid under frozen vs bound rules against ground truth, before any training)

  I2 Restated: train the same graph DP with an oracle graph (simulator poses and IDs) and a perceived graph (Grounding DINO + SAM + Cutie on rendered RGB-D), and report the success gap per subtask to split perception error from policy/memory error.
     prior work: RoboMME F1 partial (Oracle vs Gemini vs fine-tuned QwenVL subgoals; splits subgoal-generation error for symbolic memory, not object-graph perception) | "More Structure, Not More Capacity" (Li et al., arXiv 2607.09825) https://arxiv.org/abs/2607.09825 partial (privileged 3D-oracle upper bound vs perceived slots on ManiSkill3 PickCube/StackCube; no memory, no partial observability) | F3 Ledger partial (Tracker variant vs full Ledger; no oracle-tracker split) | RMBench (Chen et al., arXiv 2603.01229) https://arxiv.org/abs/2603.01229 partial (GT subtask-termination classifier ablation, 28.5 → 45.3; isolates termination, not memory content) | FOCI (Fu et al., arXiv 2609.08743) https://arxiv.org/abs/2609.08743 different | rounds: 4
     holes: "oracle graph" as stated gives true poses of hidden objects, which is oracle memory, not oracle perception; the gap then mixes perception and memory and cannot attribute either (fatal as stated) | RoboMME cubes are saturated colours on a plain table; perception gap may be small on no-swap tasks, triggering the 5-pt kill for a trivial reason (risk) | Grounding DINO on 256x256 renders of small cubes may fail outright; render at higher resolution through the env (F2) instead of using the dataset frames (risk) | oracle masks require reaching ManiSkill segmentation through RoboMME's wrapper (F2, unverified) (risk)
     verdict: reframe → "Train the graph DP on three graph sources: perceived (GDINO+SAM+Cutie+depth), visibility-gated oracle (true pose only while the segmentation mask is visible, frozen or bound otherwise), and full oracle (true pose always); perceived vs gated isolates perception, gated vs full isolates memory, reported per task with Swap tasks separated because identical containers stress tracker identity." Re-checked: same prior work; none gives a three-way perception / memory / policy split for an object-graph policy on a memory benchmark; closest RoboMME Oracle-subgoal split, which differs in operating on language subgoals. Holes left: the gated oracle needs the same freeze/bind rule as I1, so I2 depends on I1 (risk); three seeds may not separate a 5-10 pt gap at 50 episodes (risk; use 100 episodes on the 2 key tasks) | confidence: medium-high (raise by: confirming segmentation_id_map is reachable in RoboMME replay)

  I3 Restated: feed the scene-graph node tokens as extra prefix tokens to SmolVLA (LeRobot) and compare with its frame-history memory on 4 RoboMME permanence tasks.
     prior work: F4 ECoMEM same (explicit per-object records turned into prefix tokens of a VLA on RoboMME, 15/16 tasks; only the backbone differs: pi0.5 vs SmolVLA) | F3 Ledger partial (records read by an LLM planner, not as tokens) | Simple Agentic Memory / SimpleARM (Zhang et al., arXiv 2609.36595) https://arxiv.org/abs/2609.36595 partial (training-free typed state for frozen policies on RoboMME, 67.2 mean) | MIKASA-Robo-VLA (Cherepanov et al., arXiv 2610.00604) https://arxiv.org/abs/2610.00604 different (benchmark, pi0.5 baseline) | SmolVLA (Shukor et al., arXiv 2506.01844) https://arxiv.org/abs/2506.01844 (backbone, single-GPU trainable) | rounds: 3
     holes: the combination is done by F4 on the same benchmark; swapping the backbone to a smaller VLA is an engineering variation (fatal for a research claim) | a gain could come from the tracker alone, as F3's Tracker variant shows, not from token form (risk) | SmolVLA on 4 tasks x 100 demos ~3-6 GPU-h per run (F6) is affordable, so cost is not the issue
     verdict: drop | one line: ECoMEM already places explicit object records as VLA prefix tokens on RoboMME; a smaller backbone adds no testable claim. confidence: medium (would change if ECoMEM's records turn out to omit positions in a way that 3D tokens fix; check its Table 5 against a 3D-needing task)

  I4 Restated: give each node a position covariance that grows while unseen under a motion prior and resets on re-detection, fed as a token feature, and show it beats a staleness scalar.
     prior work: F4 ECoMEM partial (confidence + age + stale status per record as policy input) | BEACON (Lee, Bhattacharya, Kantor, Kroemer, arXiv 2609.22730) https://arxiv.org/abs/2609.22730 partial (Bayesian belief mean + uncertainty conditions a diffusion policy; tactile/latch tasks, not object positions) | Object Permanence Filter (Peng et al., arXiv 2403.08231) https://arxiv.org/abs/2403.08231 partial (particle-filter tracks through prolonged occlusion; no policy) | HitMem (Tang et al., arXiv 2609.00950) https://arxiv.org/abs/2609.00950 different (temporal decay for retrieval/relocation, not a policy feature) | SuperMap: not found in 1 targeted query | rounds: 4
     holes: with a fixed motion prior, covariance = q x (time since seen); it is a deterministic function of the staleness scalar, so "covariance vs staleness" has no information difference to test (fatal) | in the only public tasks where hidden objects move (Swap), the mean is wrong; a wider covariance around the wrong mean does not say which container (fatal) | an interaction-dependent covariance (grows only when something touches the cover) collapses into I1's binding rule
     verdict: drop | one line: a motion-prior covariance carries no information beyond the staleness scalar, and the benchmark failure is a wrong mean, not missing variance. confidence: high (would change only with a task where hidden objects drift randomly)

  I5 Restated: while the gripper is closed on an object, move its node with the end effector instead of freezing it when it leaves view.
     prior work: RoboStream (Huang et al., arXiv 2603.12939) https://arxiv.org/abs/2603.12939 partial (causal spatio-temporal graph of action-triggered state transitions, read by a VLM planner) | Lost & Found (Behrens et al., arXiv 2411.19162, RA-L 2025) https://arxiv.org/abs/2411.19162 partial (hand-pose-driven object pose updates in a 3D scene graph during interaction; no policy) | Mem-World (Zheng et al., arXiv 2606.18960) https://arxiv.org/abs/2606.18960 different (action-conditioned world model) | OpenSPM (Lei et al., arXiv 2606.29936) https://arxiv.org/abs/2606.29936 different (Kalman-tracked key poses for retrieval) | Object Permanence Filter partial (above) | rounds: 3
     holes: RoboMME has a fixed front camera over the table and a wrist camera that looks between the fingers, so a held cube is very likely visible in at least one view for most of the grasp; the trigger condition is rare (fatal as a standalone idea; my inference from F1's camera setup, not yet counted) | in the Swap tasks the environment, not the gripper, moves the hidden object, so I5 does not touch the tasks where freezing fails (fatal) | the seed's real setup is wrist-only, where the held object sits in view
     verdict: drop as a standalone idea; keep the end-effector rule as one clause of I1's binding rule | one line: the trigger rarely occurs in RoboMME and the failing tasks move objects by the environment. confidence: medium (raise by: counting, in 100 replayed demos with segmentation, frames where a grasped object is invisible in both cameras; under ~5% confirms the drop)

  I6 Restated: replace the instruction-derived object list with all detected objects, or measure how often instructions omit objects the robot must use.
     prior work: GROOT (Zhu, Jiang, Stone, Zhu, arXiv 2310.14386, CoRL 2023) https://arxiv.org/abs/2310.14386 partial (object-centric 3D policy compared with object-proposal baselines) | MOO (Stone et al., arXiv 2303.00905) https://arxiv.org/abs/2303.00905 partial (VLM extracts object identity from the command for the policy; no omission rate) | SlotFlow (Li et al., arXiv 2609.24155) https://arxiv.org/abs/2609.24155 different (distractor robustness via what/where split; selection rule not stated) | no paper found that measures omission rate of manipulated objects in robot instructions | rounds: 3
     holes: RoboMME instructions name every object the policy needs (cubes, containers, buttons), so the "all objects" arm has nothing to win there (fatal for the policy half on the chosen benchmark) | DROID allows only offline measurement: no closed-loop evaluation without the robot, and depth is not in the standard release (raw ZED SVO only, needs stereo processing), so a DROID policy result is out of budget (risk) | "object the robot must use" needs a labelling rule (touched? moved? occluder?) or the rate is arbitrary (risk) | the seed's own Clean-Table drawer example is one anecdote (risk)
     verdict: reframe → "On DROID-100 (or 200 sampled DROID episodes), label objects the gripper contacts or moves, and report (a) the share absent from the instruction and (b) the recall of the seed's GPT entity extractor against those labels." Re-checked: no measurement of this found in 3 rounds; closest MOO/GROOT, which assume the object is given. Holes left: a dataset statistic, useful as one section or workshop note, not a policy result; labelling ~200 episodes by hand is ~6-10 h of the student's time; API cost < $5 | confidence: low (raise by: labelling 30 episodes before the rest; if omissions < 5% stop)

  I7 Restated: reproduce the seed's H1 (memory graph beats visible-only) and H2 (removing 3D hurts on later subtasks) on RoboMME + DROID instead of the authors' data, with a DP over {RGB, visible-objects, graph, graph w/o 3D}.
     prior work: RoboMME F1 partial (14 memory variants on pi0.5, no object-graph DP) | F3, F4 partial (object records on RoboMME, pi0.5) | Embodied-SlotSSM / LIBERO-Mem partial (object-slot policy on its own memory benchmark) | IMPRINT (Wu et al., arXiv 2610.14406) https://arxiv.org/abs/2610.14406 different (latent multi-stream memory) | seed F5 (own tasks only) | no seed-style graph DP evaluated on a public memory benchmark found | rounds: 3
     holes: DROID cannot test H1/H2 because success needs closed-loop rollouts on a Franka (fatal for the DROID half) | RoboMME permanence tasks are tabletop with a fixed front camera; "w/o 3D" keeps the 2D box in a static camera frame, which already locates objects, so H2 may not bite (the seed's 3D gain came from a moving wrist / mobile base) (risk; use wrist-only input to recreate the seed's condition) | 100 demos vs 300-400: an RGB DP may sit at floor, not ceiling, on Swap tasks, and both can fail together (risk) | policy architecture rebuilt from the paper (G1), so a gap vs the seed may be implementation, not data (risk)
     verdict: reframe → "Reimplement the seed's §3.2 graph DP from Diffusion Policy code and test H1/H2 on the 4 RoboMME permanence tasks under two camera settings (front+wrist, wrist-only), 3 seeds each; DROID dropped." Re-checked: prior work as above; closest RoboMME's own variants, which differ in backbone and memory form. Holes left: implementation fidelity unverifiable without seed code (risk, report it); this is the base for I1/I2, not a contribution alone | confidence: medium (raise by: a 1-task pilot where the visible-only DP is clearly below the graph DP on VideoUnmask)

  Query log: (index: WebSearch = web search; WebFetch = page opened)
  shared | WebFetch arxiv.org/abs + html + pdf 2603.04639v3 | WebFetch | RoboMME task table, cameras, demos, pi0.5 numbers (F1)
  shared | WebFetch robomme.github.io; github.com/RoboMME/robomme_benchmark; doc/env_format.md | WebFetch | depth/extrinsics fields, replay script (F2)
  I1 | object-centric memory policy objects moved while occluded swap containers robot manipulation 2026 (extended) | WebSearch | Embodied-SlotSSM (opened), RoboMME, FOCI
  I1 | "time since last seen" object memory tokens visuomotor policy occlusion | WebSearch | HALO 2606.25136 (opened), PRISM (title only, not cited)
  I1 | WebFetch arxiv 2609.34554 abs + html | WebFetch | Ledger fields, cover-following, per-task table (F3)
  I1,I3 | explicit object memory tokens small VLA RoboMME 2026 open-source policy LeRobot memory (extended) | WebSearch | ECoMEM (opened abs + html, F4), SimpleARM (opened), MIKASA-Robo-VLA (opened), Ledger
  I2 | oracle ground-truth object state vs perceived segmentation tracking gap imitation learning object-centric policy ablation | WebSearch | 2607.09825 (opened), FOCI 2609.08743 (opened), GROOT
  I2 | memory-dependent manipulation failure analysis perception error versus policy error oracle tracking ablation 2026 (extended) | WebSearch | RMBench (opened html), 2608.17247 (opened, irrelevant: LLM agents)
  I2 | ManiSkill3 obs_mode state_dict actor pose ground truth segmentation observation documentation | WebSearch | ManiSkill observation docs (opened)
  I2,I7 | diffusion policy object scene graph tokens memory benchmark RoboMME MIKASA-Robo LIBERO-Mem baseline 3D object centroid (extended) | WebSearch | IMPRINT 2610.14406 (opened), MemoryVLA, SlotSSM
  I3 | SmolVLA object tokens scene graph memory prefix tokens RoboMME fine-tune | WebSearch | SmolVLA 2506.01844 (opened), RoboMME
  I4 | HitMem robot memory temporal decay displacement search | WebSearch | HitMem 2609.00950 (opened)
  I4 | SuperMap robot object memory confidence stale pruning | WebSearch | no SuperMap hit; KRVF, VLM-DEWM (titles only)
  I4 | imitation learning policy object belief uncertainty occluded objects Kalman covariance tokens manipulation (extended) | WebSearch | BEACON 2609.22730 (opened), BEAC 2503.16803 (opened, different)
  I4,I5 | object permanence belief tracking occluded object position uncertainty robot manipulation policy 2026 arXiv (extended) | WebSearch | Object Permanence Filter 2403.08231 (opened), RoboStream 2603.12939 (opened)
  I5 | scene graph update grasped object attached to end-effector pose propagation out of view manipulation memory | WebSearch | Lost & Found 2411.19162 (opened), OpenSPM 2606.29936 (opened)
  I5 | action-conditioned update of occluded object position carried by robot gripper object memory imitation policy 2026 (extended) | WebSearch | Mem-World 2606.18960 (opened), MemER (title only)
  I6 | instruction omits task-relevant objects LLM object extraction from instruction grounding failure robot manipulation dataset analysis | WebSearch | none relevant (InstructMove, WALL-E titles)
  I6 | task-relevant object selection versus all detected objects object-centric policy ablation distractor tokens attention manipulation (extended) | WebSearch | SlotFlow 2609.24155 (opened), GROOT 2310.14386 (opened)
  I6 | implicit objects not mentioned in language instruction robot manipulation open-vocabulary detection task-relevant object discovery VLM (extended) | WebSearch | MOO 2303.00905 (opened)
  I6,I7 | DROID dataset depth images RLDS stereo SVO ZED SDK depth not included droid_100 | WebSearch | Daft DROID docs (opened: RGB MP4, raw SVO, no depth listed)
  I7 | "Expanding Spatial and Temporal Context" scene graphs imitation learning Qian | WebSearch | seed PDF on UPenn site (opened, F5), CVPR poster "Scaling Spatial and Temporal Context..." (title only)
  cost | diffusion policy training time hours single GPU RTX 4090 100 demonstrations image-based | WebSearch | LeRobot hardware guide (opened, F6)

Cross-idea notes
  - One coherent project = I7 (reframed) + I1 (reframed) + I2 (reframed): rebuild the seed's graph DP, run it on the 4 RoboMME permanence tasks with graph sources {perceived, visibility-gated oracle, full oracle} x update rules {frozen, containment-bound} plus RGB and visible-only baselines.
  - Budget: ~4 tasks x ~8 variants x 3 seeds at ~1.5-5 GPU-h train + ~1.4 GPU-h eval = roughly 250-350 GPU-h; on a 3090 at $0.50/h that is ~$125-175, x2 for debugging still under $400. Run a 1-task pilot (~$25) before the full grid.
  - I6 (reframed) is a cheap side measurement (<$5 API, ~8 h labelling); I3, I4, I5 are dropped, with I5's gripper rule folded into I1.
  - Biggest risk: Ledger and ECoMEM already reach 91-97 on VideoUnmaskSwap with the same cover-following / INSIDE mechanism on pi0.5, so the contribution must be stated as a controlled decomposition (perception vs memory vs update rule) for a small 3D-graph policy; if a 100-demo DP stays near floor on the Swap tasks, every arm fails together and the decomposition reads nothing. The pilot must show the visible-only DP clearly below the full-oracle graph DP before scaling up.
```
