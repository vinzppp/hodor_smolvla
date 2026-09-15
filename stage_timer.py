"""Per-stage wall-clock timers for the run log.

Each pipeline stage is wrapped in ``with stage("<name>"):``. Timers call
``torch.cuda.synchronize()`` on entry/exit so GPU time is attributed to the
stage that queued it (otherwise async CUDA kernels get billed to whatever
stage happens to block next).
"""
import time
from collections import defaultdict
from contextlib import contextmanager

try:
    import torch

    _CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - CPU-only tooling
    torch = None
    _CUDA = False

# Display order + human labels for the log. Model stages first.
STAGES = [
    ("load", "load h5/mp4"),
    ("preprocess", "preprocess"),
    ("detect", "1 GroundingDINO"),
    ("segment", "2 SAM"),
    ("track", "3 Cutie"),
    ("centroid", "3b depth->3D"),
    ("embed", "4 DINO embed"),
]

_totals = defaultdict(float)
_counts = defaultdict(int)


def _sync():
    if _CUDA:
        torch.cuda.synchronize()


@contextmanager
def stage(name):
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _totals[name] += time.perf_counter() - start
        _counts[name] += 1


def reset():
    _totals.clear()
    _counts.clear()


def totals():
    """{stage: {"seconds": float, "calls": int}} for the current window."""
    return {
        name: {"seconds": round(_totals[name], 3), "calls": _counts[name]}
        for name in _totals
    }


def format_line(stage_totals, wall_seconds=None):
    parts = []
    accounted = 0.0
    for key, label in STAGES:
        if key in stage_totals:
            seconds = stage_totals[key]["seconds"]
            accounted += seconds
            parts.append(f"{label} {seconds:.1f}s")
    if wall_seconds is not None:
        parts.append(f"other {max(wall_seconds - accounted, 0.0):.1f}s")
        parts.append(f"TOTAL {wall_seconds:.1f}s")
    return " | ".join(parts)
