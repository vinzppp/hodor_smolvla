"""Does pi0.5 (LeRobot pi05, lerobot/pi05_base) fine-tune with LoRA on one 24 GB RTX 4090?

    python tools/pi05_fit.py [--steps 100] [--batches 32,16,8,4] [--episodes 10]

Measures speed and peak GPU memory only; this is not a training run and nothing is saved.
- Model: lerobot/pi05_base through LeRobot's make_policy (same path as sg.build_policy for SmolVLA), bfloat16
  (pi05 keeps its vision tower in float32), vision encoder frozen (as in the RoboMME recipe), chunk 20
  (RoboMME's action horizon).
- LoRA: rank 64 on the q/v projections of the PaliGemma language model and of the action expert, plus the
  action/time projections; the same scope as the SmolVLA pilot's LORA_TARGETS.
- Data: real VideoUnmaskSwap samples (sg.SGDataset, interface A0: two cameras, state, 20-step action chunk),
  actions normalised with mean/std. Language tokens are random ids at pi05's tokenizer_max_length (200),
  48 of them unmasked: the PaliGemma tokenizer is gated on the Hub, and token values do not change the
  cost of a step.
- Tries (batch, gradient checkpointing) from the largest batch without checkpointing down, and runs --steps
  steps with the first setting that fits. Step time excludes the first 5 steps (warm-up).
Output: /workspace/logs/pi05_fit/result.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT))
import sg_pilot as sg  # noqa: E402

OUT = Path("/workspace/logs/pi05_fit")
TASK = "VideoUnmaskSwap"
LORA_TARGETS = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj"
    r"|.*\.language_model\..*\.self_attn\.(q|v)_proj"
    r"|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)


def build(gradient_checkpointing: bool):
    from peft import LoraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    meta = LeRobotDatasetMetadata(sg.LEROBOT_REPO)
    cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_base")
    cfg.pretrained_path = "lerobot/pi05_base"
    cfg.device = "cuda"
    cfg.dtype = torch.bfloat16
    cfg.freeze_vision_encoder = True
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.chunk_size, cfg.n_action_steps = 20, 16
    cfg.input_features, cfg.output_features = {}, {}
    policy = make_policy(cfg=cfg, ds_meta=meta, rename_map=sg.RENAME_MAP)
    policy = policy.wrap_with_peft(peft_config=LoraConfig(r=64, lora_alpha=64, target_modules=LORA_TARGETS))
    n_lora = sum(1 for n, _ in policy.named_modules() if n.endswith("lora_A"))
    return policy, cfg, meta, n_lora


def batches(meta, cfg, bs: int, n_episodes: int):
    from lerobot.processor.rename_processor import rename_stats

    st = rename_stats(meta.stats, sg.RENAME_MAP)
    am, asd = (torch.tensor(np.asarray(st["action"][k], np.float32)) for k in ("mean", "std"))
    sm, ssd = (torch.tensor(np.asarray(st["observation.state"][k], np.float32)) for k in ("mean", "std"))
    paths = sorted((sg.CACHE_DIR / TASK).glob("ep*_replay.npz"))[:n_episodes]
    ds = sg.SGDataset([sg.load_training_episode(p) for p in paths], "A0", chunk=cfg.chunk_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=sg.collate,
                                     num_workers=2, drop_last=True)
    L = cfg.tokenizer_max_length
    while True:
        for b in dl:
            mask = torch.zeros(bs, L, dtype=torch.bool)
            mask[:, :48] = True
            yield {
                "observation.images.camera1": b["image"].cuda(),
                "observation.images.camera2": b["wrist_image"].cuda(),
                "observation.state": ((b["state"] - sm) / ssd).cuda(),
                "action": ((b["actions"] - am) / asd).cuda(),
                "action_is_pad": b["action_is_pad"].cuda(),
                "observation.language.tokens": torch.randint(1, 257152, (bs, L)).cuda(),
                "observation.language.attention_mask": mask.cuda(),
            }


def attempt(bs: int, ckpt: bool, steps: int, n_episodes: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy, cfg, meta, n_lora = build(ckpt)
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.AdamW(params, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0)
    it = batches(meta, cfg, bs, n_episodes)
    policy.train()
    times = []
    for step in range(steps):
        b = next(it)
        torch.cuda.synchronize()
        t0 = time.time()
        loss, _ = policy.forward(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d} loss {loss.item():.4f} {times[-1]:.2f} s/step "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    steady = float(np.mean(times[5:]))
    res = dict(batch=bs, gradient_checkpointing=ckpt, steps=steps, s_per_step=steady,
               steps_per_s=1 / steady, samples_per_s=bs / steady,
               peak_alloc_GB=torch.cuda.max_memory_allocated() / 1e9,
               peak_reserved_GB=torch.cuda.max_memory_reserved() / 1e9,
               trainable_params=n_train, total_params=n_all, lora_modules=n_lora,
               chunk_size=cfg.chunk_size)
    del policy, opt, it
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batches", default="32,16,8,4")
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tried = []
    result = None
    for bs in [int(x) for x in a.batches.split(",")]:
        for ckpt in (False, True):
            print(f"try batch {bs}, gradient checkpointing {ckpt}", flush=True)
            try:
                result = attempt(bs, ckpt, a.steps, a.episodes)
                tried.append(dict(batch=bs, gradient_checkpointing=ckpt, fits=True))
                break
            except torch.cuda.OutOfMemoryError as e:
                print(f"  out of memory: {str(e).splitlines()[0][:160]}", flush=True)
                tried.append(dict(batch=bs, gradient_checkpointing=ckpt, fits=False))
                gc.collect()
                torch.cuda.empty_cache()
        if result:
            break
    if result is None:
        print("pi0.5 with LoRA does not fit on this GPU at any tried setting")
    else:
        # RoboMME recipe: 80k steps at batch 64 (B.2.3, Table 6); a smaller batch reaches 64 by accumulation
        accum = max(1, 64 // result["batch"])
        h = 80_000 * accum * result["s_per_step"] / 3600
        result.update(recipe_accumulation=accum, recipe_hours_one_4090=h, recipe_days_one_4090=h / 24)
        print(json.dumps(result, indent=1))
    (OUT / "result.json").write_text(json.dumps(dict(result=result, tried=tried), indent=1))


if __name__ == "__main__":
    main()
