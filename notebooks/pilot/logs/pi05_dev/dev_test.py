"""Quick checks of pi05_policy before the notebook run (development only)."""
import gc
import sys
import time

sys.path.insert(0, "/workspace/sg/notebooks/pilot")
import torch

import pi05_policy as p5
import sg_pilot as sg

T = "VideoUnmaskSwap"
t0 = time.time()
stats = p5.quantile_stats([T], 2, recompute=True)
print("stats keys", {k: sorted(v) for k, v in stats.items()}, f"{time.time() - t0:.0f}s", flush=True)
policy, flow, pre, post, cfg, meta = p5.build_policy("cuda", stats=stats)
print("built", type(flow).__name__, "d_model", p5.d_model(flow), "trainable", sg.trainable_parameters(policy),
      f"{time.time() - t0:.0f}s", flush=True)
eps = [sg.load_training_episode(p) for p in sorted((sg.CACHE_DIR / T).glob("ep*_replay.npz"))[:2]]
from transformers import AutoTokenizer

tz = AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
for iface in ["A0", "A2", "A3"]:
    batch = sg.collate([sg.SGDataset(eps, iface, chunk=cfg.chunk_size)[i] for i in range(4)])
    batch.pop("sample_key")
    batch, (g, pad) = sg.split_graph_inputs(batch, "cuda")
    b = sg.prep(pre, dict(batch))
    gp = None
    if iface == "A3":
        tok = sg.make_graph_tokenizer(3, p5.d_model(flow)).cuda()
        gp = p5.GraphPrefix(flow, tok)
        gp.current = g
    loss, _ = policy.forward(b)
    m = b["observation.language.attention_mask"][0]
    print(iface, "loss", float(loss), "| prompt:", repr(tz.decode(b["observation.language.tokens"][0][m])[:160]),
          "| action norm range", float(b["action"].min()), float(b["action"].max()), flush=True)
    if gp is not None:
        policy.reset()
        smp = {k: v[:1] for k, v in batch.items() if k not in ("actions",)}
        gp.current = tuple(t[:1] for t in g)
        with torch.no_grad():
            a = policy.select_action(sg.prep(pre, smp))
        print("A3 select_action", tuple(a.shape), "queue", len(policy.get_base_model()._action_queue), flush=True)
        flow.embed_prefix = gp.orig
del policy, flow
gc.collect()
torch.cuda.empty_cache()
for iface in ["A0", "A3"]:
    info = sg.train(iface, [T], 0, 3, sg.RUNS_DIR / f"pi05_dev_{iface}", 2, batch_size=4, policy_name="pi05", log_every=1)
    print(iface, "train ok", {k: info[k] for k in ("steps", "lr", "ema", "chunk_size", "n_action_steps", "max_mem_GB")}, flush=True)
    gc.collect()
    torch.cuda.empty_cache()
s = sg.evaluate(sg.RUNS_DIR / "pi05_dev_A3", "A3", T, "Q0", 1, sg.RESULTS_DIR / "pi05_dev_A3.json", max_steps=100)
print("eval ok", s["policy"], [(r["status"], r["steps"]) for r in s["results"]], f"{s['seconds_per_episode']:.0f}s", flush=True)
print(f"total {time.time() - t0:.0f}s")
