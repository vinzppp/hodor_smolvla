"""Generate sg_interfaces_pi05.ipynb from sg_interfaces_pilot.ipynb (v1, SmolVLA).

Keeps the title, At a Glance, Setup and Steps 1-7 (shared through sg_pilot.py), replaces Steps 8-10 with
their π0.5 versions (pi05_policy.py), and appends π0.5 references. Re-run after editing this file:
    python _build/make_pi05_notebook.py
"""

import json
import uuid
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
SRC, DST = PILOT / "sg_interfaces_pilot.ipynb", PILOT / "sg_interfaces_pi05.ipynb"


def cell(kind: str, text: str) -> dict:
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def md(t):
    return cell("markdown", t)


def code(t):
    return cell("code", t)


nb = json.loads(SRC.read_text())
cells = nb["cells"]
src = lambda c: "".join(c["source"])  # noqa: E731
i8 = next(i for i, c in enumerate(cells) if src(c).startswith("## Step 8"))
iref = next(i for i, c in enumerate(cells) if src(c).startswith("## References"))
head, refs = cells[:i8], cells[iref]

# title and setup
head[0] = md("""
# Scene-Graph Interfaces for π0.5 on RoboMME: Data to Training (Pilot, π0.5 version)

This is the π0.5 version of `sg_interfaces_pilot.ipynb` (v1, SmolVLA, kept as the record of the first pilot). Steps 1-7 (data, replay, oracle graph, perception, update rules, quality levels) are the same code through `sg_pilot.py`; Steps 8-10 use π0.5 through `pi05_policy.py`, with the RoboMME training recipe adapted to one 24 GB GPU (LoRA instead of full fine-tuning).

Run on a Linux machine with an NVIDIA GPU and working Vulkan: **Kernel → Restart & Run All**. `RUN_MODE` in the Setup cell: `"smoke"` (one task, a few training steps, 1-2 evaluation episodes) or `"pilot"` (the four permanence tasks, trained outside the notebook with `run_arms_parallel.sh --policy pi05`).
""")
setup_i = next(i for i, c in enumerate(head) if src(c).startswith("# Setup"))
s = src(head[setup_i])
s = s.replace('RUN_MODE = "pilot"  # "smoke" or "pilot"', 'RUN_MODE = "smoke"  # "smoke" or "pilot"')
s = s.replace("import sg_pilot as sg\n\nimportlib.reload(sg)\n", "import sg_pilot as sg\nimport pi05_policy as p5\n\nimportlib.reload(sg)\nimportlib.reload(p5)\n")
s = s.replace('TASK = CFG["task"]\n', 'TASK = CFG["task"]\n# one task for smoke, one multi-task model over the four permanence tasks for the pilot (RoboMME §5.1)\nTRAIN_TASKS = [TASK] if RUN_MODE == "smoke" else sg.PERMANENCE_TASKS\n')
assert "pi05_policy" in s and "TRAIN_TASKS" in s and 'RUN_MODE = "smoke"' in s
head[setup_i]["source"] = s.splitlines(keepends=True)

new = [
    md("""
## Step 8. π0.5 and the three interfaces

**In:** the graph from Steps 4-7. **Out:** the same four fields per object (id, label, (u, v), hidden) delivered to π0.5 three ways, with the RoboMME recipe's settings.

π0.5, in four rungs:

1. **Tokens.** Each camera image becomes 256 tokens (SigLIP, 224 x 224 input, 14-pixel patches); the prompt becomes PaliGemma tokens. Every token is a vector of width d = 2048 (printed in 8.1).
2. **Prefix.** Image tokens of both cameras, then the prompt tokens `Task: ...;\\nAction: `. Stock π0.5 also writes the robot state into the prompt as 256-bin numbers; RoboMME leaves proprioception out, and so does this notebook.
3. **Action expert.** A smaller transformer (300M parameters) attends to the prefix and turns noise into a chunk of 20 future joint commands by flow matching; 16 of them are executed before the next prediction.
4. **Where each interface enters.** A1 draws markers on the front image before SigLIP sees it; A2 appends the object list to the task text inside the prompt; A3 appends one learned token per object to the end of the prefix.

**What changes from the SmolVLA version** (details in `BUILD_NOTES.md`, "π0.5 version"):

| | SmolVLA (v1) | π0.5 (this notebook) |
|---|---|---|
| model | smolvla_base, ~0.45B | pi05_base, 3.2B (4.2B with SigLIP in LeRobot's count) |
| trained | LoRA, VLM text layers + action expert | LoRA, same scope; SigLIP frozen |
| action chunk | 50 predicted, 50 executed | 20 predicted, 16 executed |
| state | input token | not used (RoboMME: no proprioception) |
| normalisation | mean/std of lerobot/robomme (16 tasks) | q01/q99 of our training data |
| optimiser | AdamW 1e-3, clip 10 | AdamW 5e-5 (0.9, 0.95), warm-up, clip 1.0, EMA 0.999 |
| A3 tokens | before the state token | after the prompt tokens (no state token) |
| padded actions in the loss | masked | not masked (π0.5's loss has no pad mask) |
"""),
    md("### 8.1 Inside π0.5"),
    code("""
# Step 8.1
print("perception worker stopped:", sg.close_perception_worker())  # frees ~8 GB before π0.5 loads
from lerobot.policies.pi05 import modeling_pi05, processor_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def field_line(obj, prefix):
    \"\"\"file:line of the first source line of obj starting with prefix.\"\"\"
    src, start = inspect.getsourcelines(obj)
    i = next(i for i, l in enumerate(src) if l.strip().startswith(prefix))
    return f"{inspect.getsourcefile(obj)}:{start + i}  {src[i].strip()}"


print("π0.5 defaults (installed source):")
for f in ["chunk_size:", "n_action_steps:", "tokenizer_max_length:", "dtype:", "freeze_vision_encoder:", '"STATE":', '"ACTION":']:
    print("  ", field_line(PI05Config, f))
step = processor_pi05.Pi05PrepareStateTokenizerProcessorStep
print("   prompt with state:   ", field_line(step, 'full_prompt = f"Task: {cleaned_text}, State'))
print("   prompt without state:", field_line(step, 'full_prompt = f"Task: {cleaned_text};'))

stats = p5.quantile_stats(TRAIN_TASKS, CFG["train_episodes"])
policy, flow, pre, post, pcfg_policy, lmeta = p5.build_policy("cuda", stats=stats)
c = pcfg_policy
print("   prefix builder:      ", field_line(type(flow), "def embed_prefix"))
print("   training loss:       ", field_line(modeling_pi05.PI05Policy, "losses = losses[:, :, :original_action_dim]"))
print("\\nvalues used here (pi05_policy.build_policy):")
for k in ["chunk_size", "n_action_steps", "tokenizer_max_length", "dtype", "freeze_vision_encoder", "gradient_checkpointing", "image_resolution"]:
    print(f"  {k} = {getattr(c, k)}")
print("  normalization_mapping =", {str(k): str(v) for k, v in c.normalization_mapping.items()})
print("  state in the prompt: no (Pi05PrepareStateTokenizerProcessorStep(include_state_in_prompt=False))")
print("input features:", list(c.input_features), "| output:", list(c.output_features))
D_MODEL = p5.d_model(flow)
print("prefix token width d =", D_MODEL, "| trainable LoRA parameters:", f"{sg.trainable_parameters(policy):,}")
"""),
    code("""
# Step 8.1 (prefix of one real sample)
import torch
from transformers import AutoTokenizer

tz = AutoTokenizer.from_pretrained(c.text_tokenizer_name)
smp = {
    "image": torch.from_numpy(rep.front_rgb[t_star_h5]).permute(2, 0, 1)[None].float() / 255,
    "wrist_image": torch.from_numpy(rep.wrist_rgb[t_star_h5]).permute(2, 0, 1)[None].float() / 255,
    "state": torch.from_numpy(
        np.r_[rep.joint_state[t_star_h5], rep.gripper_state[t_star_h5][:1]].astype(np.float32)
    )[None],
    "task": [rep.task_goal],
}
b = sg.prep(pre, smp)
inner = policy.get_base_model()
images, img_masks = inner._preprocess_images(b)
tokens, masks = b["observation.language.tokens"], b["observation.language.attention_mask"]
with torch.no_grad():
    embs, pad, att = flow.embed_prefix(images, img_masks, tokens, masks)
n_lang = tokens.shape[1]
print("prefix embeddings", tuple(embs.shape), "= [batch, tokens, d]")
print(f"  image tokens: {embs.shape[1] - n_lang} ({len(images)} cameras x {(embs.shape[1] - n_lang) // len(images)})")
print(f"  prompt tokens: {n_lang} slots, {int(masks.sum())} used")
print("  prompt as the model reads it:", repr(tz.decode(tokens[0][masks[0]])))
sg.pass_fail("no state in the prompt", "State" not in tz.decode(tokens[0][masks[0]]), "Task: ...;\\\\nAction:")
"""),
    md("### 8.2 A1: markers drawn on the image"),
    code("""
# Step 8.2
show(sg.draw_markers)
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

nodes = sg.nodes_at(bound_g, t_star_h5, rep.scene)
R = c.image_resolution[0]
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, size in enumerate([3, 6, 10]):
    marked = sg.draw_markers(rep.front_rgb[t_star_h5], nodes, size)
    x = torch.from_numpy(marked).float()[None] / 255  # [B, H, W, C], as π0.5's _preprocess_images
    pre_img = resize_with_pad_torch(x, R, R)[0].numpy()
    axes[i, 0].imshow(marked)
    axes[i, 1].imshow(pre_img)
    u, v = nodes[tgt]["u"], nodes[tgt]["v"]
    s2 = R / 256
    axes[i, 2].imshow(
        pre_img[max(0, int(v * s2) - 28) : int(v * s2) + 28, max(0, int(u * s2) - 28) : int(u * s2) + 28]
    )
    axes[i, 0].set_ylabel(f"size {size}")
for a, ttl in zip(axes[0], ["raw marked (256 px)", f"after π0.5 resize/pad ({R} px)", "zoom on target (4 x 4 SigLIP patches)"]):
    a.set_title(ttl, fontsize=8)
for a in axes.flat:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(sg.FIG_DIR / "08_markers_pi05.png", dpi=120)
plt.show()
MARKER_SIZE = 6
print("chosen marker size:", MARKER_SIZE, "= ? (confirm from the figure: smallest size whose ring and label stay distinct)")
"""),
    md("### 8.3 A2: the graph as text inside π0.5's prompt"),
    code("""
# Step 8.3
show(sg.serialize)


def prompt_of(text):
    # the prompt Pi05PrepareStateTokenizerProcessorStep builds without state (see 8.1)
    return f"Task: {text.strip().replace('_', ' ').replace(chr(10), ' ')};\\nAction: "


worst = (0, None)
for task_ in TRAIN_TASKS:
    for p in sorted((sg.CACHE_DIR / task_).glob("ep*_replay.npz"))[: CFG["train_episodes"]]:
        r_ = sg.load_replay(p, ["policy_frame"])
        g_ = sg.load_graph_cache(p.with_name(p.name.replace("_replay", "_graph")))["Q0"]
        for t_ in r_.policy_frame:
            s_ = prompt_of(sg.serialize(sg.nodes_at(g_, int(t_), r_.scene), r_.task_goal))
            worst = max(worst, (len(tz(s_)["input_ids"]), s_))
print("worst case prompt tokens:", worst[0], "| tokenizer_max_length:", c.tokenizer_max_length)
print(worst[1])
ids = tz(worst[1], truncation=True, max_length=c.tokenizer_max_length)["input_ids"]
print("decoded back (end):", repr(tz.decode(ids)[-100:]))
sg.pass_fail("no truncation at tokenizer_max_length", worst[0] <= c.tokenizer_max_length, (worst[0], c.tokenizer_max_length))
"""),
    md("### 8.4 A3: graph tokens in π0.5's prefix"),
    code("""
# Step 8.4
show(sg.make_graph_tokenizer)
show(p5.GraphPrefix)
tok3 = sg.make_graph_tokenizer(3, D_MODEL).to("cuda")
gp = p5.GraphPrefix(flow, tok3)
x, lab, ids_, mask = sg.node_tensor(nodes, 256, 256, None)
gp.current = tuple(torch.from_numpy(a)[None].to("cuda") for a in (x, lab, ids_, mask))
with torch.no_grad():
    e2, p2, a2 = flow.embed_prefix(images, img_masks, tokens, masks)
print("prefix without graph", tuple(embs.shape), "| with graph", tuple(e2.shape),
      "| graph tokens", int(mask.sum()), "(+ padding", int((~mask).sum()), "masked)")
print("pad mask of the appended tokens:", p2[0, -len(mask):].int().tolist(), "| attention flags:", a2[0, -len(mask):].int().tolist())
policy.reset()
with torch.no_grad():
    act = policy.select_action(sg.prep(pre, smp))
print("action shape", tuple(act.shape), "| actions left in the queue:", len(inner._action_queue))
gp.current = None
flow.embed_prefix = gp.orig
sg.pass_fail("A3 forward runs, action shape unchanged", act.shape[-1] == 8 and len(inner._action_queue) == c.n_action_steps - 1, tuple(act.shape))
"""),
    md("### 8.4c Label encoding"),
    code("""
# Step 8.4c
te = p5.text_embed_fn(flow, c)
for w in ["cube", "container"]:
    v = te(w)
    print(f"text label '{w}': vector {tuple(v.shape)}, norm {float(v.norm()):.2f}")
print("pilot uses label_mode='table'")
"""),
    md("### 8.5 Equal capacity"),
    code("""
# Step 8.5
base = sg.trainable_parameters(policy)
rows = {"A0": base, "A1": base, "A2": base, "A3": base + sg.trainable_parameters(sg.make_graph_tokenizer(3, D_MODEL))}
for k, v in rows.items():
    print(f"{k:3s} trainable parameters {v:,}  (+{v - base:,} vs A0)")
"""),
    md("A3 carries the graph tokenizer's extra parameters; this difference is reported with every result."),
    md("""
## Step 9. Training

**In:** cached replays and Q0 graphs of `TRAIN_TASKS`. **Out:** π0.5 checkpoints (LoRA adapters, EMA weights) for each interface.

One iteration as `sg.train(policy_name="pi05")` runs it: sample a policy step, its two images and the next 20 recorded actions; the interface changes the image (A1), the text (A2) or adds node tensors (A3); the preprocessor normalises actions with the quantiles of 9.1 and tokenises the prompt; π0.5's flow-matching loss is computed over the chunk; AdamW updates the LoRA adapters (and the A3 tokenizer); the EMA copy follows. Gradient accumulation (`accum_steps=2`) reaches RoboMME's batch 64 from batches of 32.
"""),
    md("### 9.1 Normalisation statistics (q01/q99)"),
    code("""
# Step 9.1
show(p5.quantile_stats)
np.set_printoptions(precision=3, suppress=True, linewidth=140)
for k in ("action", "observation.state"):
    print(k)
    for s_ in ("q01", "q99", "mean", "std"):
        print(f"  {s_:4s}", np.array(stats[k][s_]))
print("stats file:", p5.stats_path(TRAIN_TASKS, CFG["train_episodes"]))
sg.pass_fail("q01/q99 for action and state", all(s_ in stats[k] for k in ("action", "observation.state") for s_ in ("q01", "q99")), list(stats))
"""),
    md("### 9.2 A batch at chunk 20"),
    code("""
# Step 9.2
eps = [
    sg.load_training_episode(p)
    for task_ in TRAIN_TASKS
    for p in sorted((sg.CACHE_DIR / task_).glob("ep*_replay.npz"))[: CFG["train_episodes"]]
]
batch = sg.collate([sg.SGDataset(eps, "A3", chunk=p5.CHUNK_SIZE)[i] for i in range(4)])
for k_, v_ in batch.items():
    print(f"{k_:14s}", tuple(v_.shape) if hasattr(v_, "shape") else f"list of {len(v_)} strings")
print(f"{len(eps)} episodes, {sum(len(e[0].actions) for e in eps)} samples")
"""),
    md("### 9.3 Smoke training: every interface, a few steps"),
    code("""
# Step 9.3
show(p5.make_optimizer)
import gc

for n_ in ["policy", "flow", "inner", "gp", "tok3", "te", "pre", "post", "b", "images", "img_masks",
           "tokens", "masks", "embs", "pad", "att", "e2", "p2", "a2", "act"]:
    globals().pop(n_, None)  # the Step 8 inspection model and every tensor or closure holding it
gc.collect()
torch.cuda.empty_cache()
smoke_info = {}
for iface in ["A0", "A1", "A2", "A3"]:
    smoke_info[iface] = sg.train(
        iface, TRAIN_TASKS, 0, CFG["steps"] or 10, sg.RUNS_DIR / f"pi05_smoke_{iface}",
        CFG["train_episodes"], batch_size=8, policy_name="pi05", log_every=5,
    )
    gc.collect()
    torch.cuda.empty_cache()
plt.figure(figsize=(5, 2.5))
for k_, v_ in smoke_info.items():
    plt.plot(v_["loss"], label=k_)
plt.xlabel("step")
plt.ylabel("flow-matching loss")
plt.legend()
plt.title("π0.5 smoke training (batch 8)")
plt.show()
from safetensors.torch import load_file

ok = all(np.isfinite(v_["loss"]).all() for v_ in smoke_info.values())
reloaded = load_file(str(sg.RUNS_DIR / "pi05_smoke_A2" / "policy" / "adapter_model.safetensors"))
sg.pass_fail("loss finite, checkpoint (EMA) reloads, stats saved",
             ok and len(reloaded) > 0 and (sg.RUNS_DIR / "pi05_smoke_A2" / "norm_stats.json").exists(),
             f"{len(reloaded)} adapter tensors")
"""),
    md("### 9.4 Speed and memory at the pilot batch"),
    code("""
# Step 9.4: 25 steps at batch 32 (gradient checkpointing on), A0 and A2
PILOT_BATCH, PILOT_ACCUM, PILOT_STEPS = 32, 2, 80_000  # RoboMME: 80k steps at batch 64
speed = {}
for iface in ["A0", "A2"]:
    speed[iface] = sg.train(iface, TRAIN_TASKS, 0, 25, sg.RUNS_DIR / f"pi05_speed_{iface}",
                            CFG["train_episodes"], batch_size=PILOT_BATCH, policy_name="pi05", log_every=100)
    gc.collect()
    torch.cuda.empty_cache()
for k_, v_ in speed.items():
    print(f"{k_}: {v_['steps_per_s_steady']:.3f} steps/s after step 0, peak GPU memory {v_['max_mem_GB']:.1f} GB at batch {PILOT_BATCH}")
sps = min(v_["steps_per_s_steady"] for v_ in speed.values())
hours = p5.estimate_recipe_hours(1 / sps, PILOT_BATCH, PILOT_STEPS, PILOT_BATCH * PILOT_ACCUM)
print(f"RoboMME recipe ({PILOT_STEPS} steps, batch {PILOT_BATCH} x {PILOT_ACCUM}): ~{hours:.0f} h per arm on one GPU of this type")
"""),
    md("**How to read the output above.** = ? (after the run: steps per second, peak memory, and the time per arm for the recipe.)"),
    code("""
# Step 9.5: pilot training (outside the notebook, one GPU per arm)
if RUN_MODE == "pilot":
    pilot_info = {}
    for iface in ["A0", "A1", "A2", "A3"]:
        p = sg.RUNS_DIR / f"pi05_{iface}_s0" / "train_info.json"
        if p.exists():
            pilot_info[iface] = json.loads(p.read_text())
            print(f"{iface}: {pilot_info[iface]['steps']} steps, {pilot_info[iface]['seconds'] / 3600:.1f} h")
        else:
            print(f"{iface}: not trained; run ./run_arms_parallel.sh --policy pi05 --interfaces A0,A1,A2,A3 "
                  f"--tasks permanence --steps {PILOT_STEPS} --batch {PILOT_BATCH} --accum {PILOT_ACCUM} --label pi05")
else:
    print("= ? (run in pilot mode)")
"""),
    md("""
## Step 10. Evaluation

**In:** π0.5 checkpoints and the test split. **Out:** success rates and a rollout strip. Same loop as v1 (`sg.evaluate`): episodes capped at 1300 steps, the graph built from every frame returned at reset and updated online, 16 of every 20 predicted actions executed.
"""),
    code("""
# Step 10.1
info_ = json.loads((sg.RUNS_DIR / "pi05_smoke_A2" / "train_info.json").read_text())
print({k: info_[k] for k in ("policy", "chunk_size", "n_action_steps", "ema", "lr", "accum_steps")})
env = sg.make_env(TASK, "test", 0, max_steps=1300)
obs, info = env.reset()
env.close()
print("max_steps = 1300 | frames returned at reset (history video included):", len(obs["front_rgb_list"]))
sg.pass_fail("executes 16 of 20", (info_["chunk_size"], info_["n_action_steps"]) == (20, 16), (info_["chunk_size"], info_["n_action_steps"]))
"""),
    code("""
# Step 10.2: smoke evaluation, A2 on 2 episodes (rollout strip) and A3 on 1 (graph tokens online)
s_a2 = sg.evaluate(sg.RUNS_DIR / "pi05_smoke_A2", "A2", TASK, "Q0", 2, sg.RESULTS_DIR / "pi05_smoke_A2_Q0.json",
                   save_strip=sg.FIG_DIR / "10_rollout_pi05.png")
s_a3 = sg.evaluate(sg.RUNS_DIR / "pi05_smoke_A3", "A3", TASK, "Q0", 1, sg.RESULTS_DIR / "pi05_smoke_A3_Q0.json")
from IPython.display import Image

display(Image(str(sg.FIG_DIR / "10_rollout_pi05.png")))
for s_ in (s_a2, s_a3):
    print(s_["interface"], [(r_["episode"], r_["status"], r_["steps"]) for r_ in s_["results"]], f"{s_['seconds_per_episode']:.0f} s/episode")
sg.pass_fail("evaluation runs for A2 and A3", len(s_a2["results"]) == 2 and len(s_a3["results"]) == 1, "3 episodes")
"""),
    code("""
# Step 10.3: pilot evaluation, every arm on 50 test episodes of each permanence task at Q0
if RUN_MODE == "pilot":
    motion = sg.load_motion("VideoUnmaskSwap")
    for iface in ["A0", "A1", "A2", "A3"]:
        for task_ in TRAIN_TASKS:
            out_ = sg.RESULTS_DIR / f"pi05_{iface}_{task_}_Q0.json"
            r_ = json.loads(out_.read_text()) if out_.exists() else sg.evaluate(
                sg.RUNS_DIR / f"pi05_{iface}_s0", iface, task_, "Q0", CFG["eval_episodes"], out_)
            for name, sp in sg.success_by_split(r_, motion if task_ == "VideoUnmaskSwap" else None).items():
                print(f"{iface} {task_:16s} {name:14s} {sp['k']}/{sp['n']} = {sp['rate']:.3f}, 95% Wilson {np.round(sp['wilson95'], 3)}")
else:
    print("= ? (run in pilot mode)")
"""),
    md("**How to read the output above.** = ? (after the pilot run.)"),
    code("""
# Step 10.4: grid size and cost (not run)
n_arms, n_seeds = 4, 3
print(f"π0.5 grid: {n_arms} interfaces x {n_seeds} seeds = {n_arms * n_seeds} training runs of ~{hours:.0f} h "
      f"= ~{n_arms * n_seeds * hours:.0f} GPU h at the speed measured in 9.4")
print("launch per seed: ./run_arms_parallel.sh --policy pi05 --interfaces A0,A1,A2,A3 --tasks permanence "
      f"--steps {PILOT_STEPS} --batch {PILOT_BATCH} --accum {PILOT_ACCUM} --seed <s> --label pi05")
"""),
]

ref_text = src(refs).rstrip("\n") + """
- π0.5: Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025; LeRobot port `lerobot/policies/pi05` (read at commit 5831498: `configuration_pi05.py`, `modeling_pi05.py`, `processor_pi05.py`), weights `lerobot/pi05_base`, tokenizer `google/paligemma-3b-pt-224`.
- RoboMME training recipe: arXiv 2603.04639 v3, Appendix B.2.3 and Table 6; §5.1 (multi-task training, chunk 20 / execute 16).
"""
refs["source"] = ref_text.splitlines(keepends=True)
nb["cells"] = head + new + [refs]
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {DST} ({len(nb['cells'])} cells; Steps 8-10: {len(new)} new cells)")
