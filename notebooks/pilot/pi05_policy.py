"""pi05_policy: π0.5 (LeRobot pi05, lerobot/pi05_base) for the scene-graph interface study.

sg_pilot.py stays the model-agnostic core (data, replay, graphs, perception, rules, quality levels,
SGDataset, train/evaluate loops). This module is the π0.5 backend those loops call through
sg_pilot.get_backend("pi05"); the SmolVLA backend is sg_pilot's own build_policy/GraphPrefix (v1).

Recipe, following RoboMME (arXiv 2603.04639 v3, Appendix B.2.3, Table 6; §5.1), adapted to one 24 GB GPU:
  - LoRA (rank 64) instead of full fine-tuning of the VLM and action experts (full fine-tuning of 3.2B
    parameters does not fit in 24 GB); SigLIP frozen, as in the paper.
  - action chunk 20, first 16 executed; joint-space actions.
  - no proprioception: the state is not put into the prompt (stock π0.5 writes it there as 256-bin tokens).
  - quantile normalisation (q01/q99) with statistics computed from our training data
    (lerobot/robomme has no quantiles).
  - AdamW (0.9, 0.95), weight decay 0, lr 5e-5 constant after a linear warm-up (10k steps), gradient
    clip 1.0, EMA 0.999 over the trained parameters (the EMA weights are what is saved and evaluated).
  - optional SigLIP token caching (SiglipCache), off by default: at ~1 MiB per image (256 tokens x 2048,
    bf16) it needs ~2 MiB per training sample, far beyond the free disk of the pilot pod.

Interfaces (same per-object content as SmolVLA: id, label, (u, v), hidden):
  A0 no graph; A1 markers on the front image (π0.5 resizes 256 -> 224 with padding, so the marker size is
  re-checked); A2 text appended to the instruction inside π0.5's prompt "Task: ...;\\nAction: "; A3 graph
  tokens appended to π0.5's prefix after the language tokens (π0.5 has no state token to insert before).
Differences from SmolVLA are listed in BUILD_NOTES, "π0.5 version".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import sg_pilot as sg

PI05_BASE = "lerobot/pi05_base"
CHUNK_SIZE = 20  # RoboMME Table 6: action horizon 20
N_ACTION_STEPS = 16  # §5.1: "execute the first 16 steps during evaluation"
TOKENIZER_MAX_LENGTH = 256  # π0.5 default 200; raised for A2's object list (checked in notebook 8.3)
# LoRA scope as in the SmolVLA pilot: q/v of the language model and of the action expert, plus the
# action/time projections (π0.5 has no state projection).
LORA_TARGETS = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj"
    r"|.*\.language_model\..*\.self_attn\.(q|v)_proj"
    r"|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)
RECIPE = dict(lr=5e-5, warmup_steps=10_000, betas=(0.9, 0.95), weight_decay=0.0, clip=1.0, ema=0.999)


# --------------------------------------------------------------------------------------------
# Normalisation statistics (q01/q99) from our own training data
# --------------------------------------------------------------------------------------------
def stats_path(tasks: list[str], episodes_per_task: int) -> Path:
    return sg.CACHE_DIR / "norm_stats" / f"pi05_{'+'.join(tasks)}_{episodes_per_task}.json"


def quantile_stats(tasks: list[str], episodes_per_task: int, recompute: bool = False) -> dict:
    """{"observation.state": {...}, "action": {...}} with q01, q99, mean, std, min, max.

    Actions: every recorded action of the training episodes; state: the state at every policy frame
    (7 joints + first finger width, as SGDataset builds it). Cached as JSON (stats_path).
    """
    path = stats_path(tasks, episodes_per_task)
    if path.exists() and not recompute:
        return json.loads(path.read_text())
    acts, states = [], []
    for task in tasks:
        for p in sorted((sg.CACHE_DIR / task).glob("ep*_replay.npz"))[:episodes_per_task]:
            r = sg.load_replay(p, ["actions", "joint_state", "gripper_state", "policy_frame"])
            pf = r.policy_frame.astype(int)
            acts.append(r.actions)
            states.append(np.c_[r.joint_state[pf], r.gripper_state[pf][:, :1]])
    out = {}
    for key, x in (("action", np.concatenate(acts)), ("observation.state", np.concatenate(states))):
        x = x.astype(np.float64)
        out[key] = dict(
            q01=np.quantile(x, 0.01, axis=0).tolist(),
            q99=np.quantile(x, 0.99, axis=0).tolist(),
            mean=x.mean(0).tolist(),
            std=x.std(0).tolist(),
            min=x.min(0).tolist(),
            max=x.max(0).tolist(),
            count=[int(len(x))],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def _check_stats(stats: dict) -> None:
    # The normaliser silently passes a feature through when its stats are missing
    # (lerobot/processor/normalize_processor.py, _apply_transform), so fail loudly here instead.
    for key in ("action", "observation.state"):
        for s in ("q01", "q99"):
            if s not in stats.get(key, {}):
                raise ValueError(f"π0.5 quantile normalisation needs stats[{key!r}][{s!r}]")


# --------------------------------------------------------------------------------------------
# Policy, processors
# --------------------------------------------------------------------------------------------
def make_processors(cfg, stats: dict):
    """π0.5's pre/post pipelines (lerobot/policies/pi05/processor_pi05.py make_pi05_pre_post_processors)
    with two changes: the state is not written into the prompt (no proprioception), and the
    normaliser uses our q01/q99 statistics."""
    import torch
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep
    from lerobot.processor.factory import (
        make_default_policy_processor_steps,
        make_policy_processor_pipelines,
    )

    tstats = {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}
    steps = make_default_policy_processor_steps(cfg, tstats)
    pre, post = make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            steps.normalize,
            Pi05PrepareStateTokenizerProcessorStep(
                max_state_dim=cfg.max_state_dim, include_state_in_prompt=False
            ),
            TokenizerProcessorStep(
                tokenizer_name=cfg.text_tokenizer_name,
                max_length=cfg.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            steps.to_device,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
    return pre, post


def build_policy(
    device: str = "cuda",
    stats: dict | None = None,
    lora_r: int = 64,
    tokenizer_max_length: int = TOKENIZER_MAX_LENGTH,
    gradient_checkpointing: bool = True,
    chunk_size: int = CHUNK_SIZE,
    n_action_steps: int = N_ACTION_STEPS,
):
    """π0.5 from lerobot/pi05_base with the RoboMME feature names, LoRA and our processors.

    Same return signature as sg_pilot.build_policy: (policy, flow_model, pre, post, cfg, meta).
    flow_model is PI05Pytorch (policy.model), whose embed_prefix A3 wraps.
    """
    import torch
    from peft import LoraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    if stats is None:
        raise ValueError("π0.5 needs q01/q99 statistics: pass quantile_stats(tasks, episodes)")
    _check_stats(stats)
    meta = LeRobotDatasetMetadata(sg.LEROBOT_REPO)  # feature names and shapes only
    cfg = PreTrainedConfig.from_pretrained(PI05_BASE)
    cfg.pretrained_path = PI05_BASE
    cfg.device = device
    cfg.dtype = torch.bfloat16  # pi05 keeps the vision tower in float32
    cfg.freeze_vision_encoder = True  # RoboMME: "we freeze the SigLIP2 vision backbone"
    cfg.gradient_checkpointing = gradient_checkpointing  # batch 32 needs it on 24 GB (tools/pi05_fit.py)
    cfg.chunk_size, cfg.n_action_steps = chunk_size, n_action_steps
    cfg.tokenizer_max_length = tokenizer_max_length
    cfg.input_features, cfg.output_features = {}, {}
    policy = make_policy(cfg=cfg, ds_meta=meta, rename_map=sg.RENAME_MAP)
    flow = policy.model
    policy = policy.wrap_with_peft(
        peft_config=LoraConfig(r=lora_r, lora_alpha=lora_r, target_modules=LORA_TARGETS)
    )
    pre, post = make_processors(cfg, stats)
    return policy, flow, pre, post, cfg, meta


def d_model(flow) -> int:
    """Width of π0.5's prefix tokens (PaliGemma's Gemma 2B: 2048)."""
    return int(flow.paligemma_with_expert.paligemma.model.language_model.config.hidden_size)


def text_embed_fn(flow, cfg, device: str = "cuda"):
    """label_mode 'text': mean of PaliGemma's own token embeddings of a label word."""
    import torch
    from transformers import AutoTokenizer

    tz = AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
    emb = flow.paligemma_with_expert.embed_language_tokens
    return lambda w: emb(torch.tensor(tz(w, add_special_tokens=False)["input_ids"], device=device)).mean(0).detach()


class GraphPrefix:
    """A3 for π0.5: graph tokens appended to the prefix, [image tokens | language tokens | GRAPH].

    Wraps PI05Pytorch.embed_prefix (lerobot/policies/pi05/modeling_pi05.py). Graph tokens get attention
    flag 0 (bidirectional within the prefix, like the image and language tokens) and the node mask as pad
    mask. Set .current = (x, labels, ids, mask) on the policy device before each forward / select_action.
    """

    def __init__(self, flow_model, tokenizer):
        self.model, self.tok, self.current = flow_model, tokenizer, None
        self.orig = flow_model.embed_prefix
        flow_model.embed_prefix = self.embed_prefix

    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        import torch

        embs, pad, att = self.orig(images, img_masks, tokens, masks, states=states, state_masks=state_masks)
        if self.current is None:
            return embs, pad, att
        x, labels, ids, mask = self.current
        g = self.tok(x.float(), labels, ids).to(embs.dtype)  # [B, N, D]
        embs = torch.cat([embs, g], 1)
        pad = torch.cat([pad, mask.bool()], 1)
        att = torch.cat([att, torch.zeros_like(mask, dtype=att.dtype)], 1)
        return embs, pad, att


# --------------------------------------------------------------------------------------------
# Optimiser, schedule, EMA
# --------------------------------------------------------------------------------------------
class EMA:
    """Exponential moving average of the trained parameters (fp32 shadow copies)."""

    def __init__(self, params, decay: float):
        self.params, self.decay = list(params), decay
        self.shadow = [p.detach().float().clone() for p in self.params]

    def update(self) -> None:
        import torch

        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                s.mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    def swap(self) -> None:
        """Exchange live and EMA weights (call twice to restore)."""
        import torch

        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                tmp = p.detach().clone()
                p.copy_(s.to(p.dtype))
                s.copy_(tmp.float())


def make_optimizer(params, steps: int, lr: float | None = None, warmup_steps: int | None = None):
    """(optimizer, scheduler, clip norm, EMA) for the RoboMME recipe. Warm-up: 10k steps, but at most
    an eighth of the run (the paper's 10k of 80k), so shorter runs keep the same proportion."""
    import torch

    lr = RECIPE["lr"] if lr is None else lr
    warm = RECIPE["warmup_steps"] if warmup_steps is None else warmup_steps
    warm = max(1, min(warm, steps // 8))
    opt = torch.optim.AdamW(params, lr=lr, betas=RECIPE["betas"], weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))  # then constant
    return opt, sched, RECIPE["clip"], EMA(params, RECIPE["ema"])


# --------------------------------------------------------------------------------------------
# Optional SigLIP token cache (training only)
# --------------------------------------------------------------------------------------------
class SiglipCache:
    """Caches π0.5's per-camera image embeddings on disk, keyed by sample, so SigLIP (frozen) runs once
    per training image. Wraps PI05Pytorch._embed_images; set .keys = [(sample_key, variant), ...] for the
    batch before each forward (variant 'A1' for marked front images, 'raw' otherwise).

    Cost: 256 tokens x 2048 x 2 bytes = 1 MiB per camera image, 2 MiB per sample. The VideoUnmaskSwap
    training set alone (18,667 samples) is ~37 GiB; the four permanence tasks ~150 GiB. Off by default.
    """

    def __init__(self, flow_model, root: Path):
        self.flow, self.root, self.keys = flow_model, Path(root), None
        self.orig = flow_model._embed_images
        flow_model._embed_images = self.embed_images
        self.hits = self.misses = 0

    @staticmethod
    def gib_per_sample(n_cameras: int = 2) -> float:
        return n_cameras * 256 * 2048 * 2 / 2**30

    def _path(self, key: str, variant: str, cam: int) -> Path:
        return self.root / variant / f"{key.replace('/', '__')}_cam{cam}.npy"

    def embed_images(self, images, img_masks):
        import torch

        if self.keys is None:
            return self.orig(images, img_masks)
        out = []
        for cam, img in enumerate(images):
            variant = lambda v: v if cam == 0 else "raw"  # noqa: E731  markers only on the front image
            paths = [self._path(k, variant(v), cam) for k, v in self.keys]
            if all(p.exists() for p in paths):
                self.hits += len(paths)
                arr = np.stack([np.load(p) for p in paths])
                out.append(torch.from_numpy(arr).to(img.device).view(torch.bfloat16))
                continue
            self.misses += len(paths)
            with torch.no_grad():
                e = self.flow.paligemma_with_expert.embed_image(img)
            for p, row in zip(paths, e):
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, row.to(torch.bfloat16).view(torch.int16).cpu().numpy())
            out.append(e)
        return out


def estimate_recipe_hours(s_per_step: float, batch: int, steps: int = 80_000, target_batch: int = 64) -> float:
    """Hours for `steps` steps at `target_batch`, reached by gradient accumulation of `batch`."""
    return steps * math.ceil(target_batch / batch) * s_per_step / 3600
