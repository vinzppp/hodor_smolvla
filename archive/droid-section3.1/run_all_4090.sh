#!/usr/bin/env bash
# =============================================================================
# run_all_4090.sh — one-shot HODOR Section 3.1 re-run on an RTX 4090 pod.
#
#   fresh pod -> code+data -> setup (sm_89 kernel) -> 3 configs -> per-step
#   outputs, storyboards, videos, timings -> logs + a downloadable bundle.
#
# Bootstrap on a fresh RunPod pod (torch 2.8 / CUDA 12.8 / Py 3.12 image):
#   export GH_PAT=...                  # repo-scoped token for vinzppp/scene-graph
#   export OPENROUTER_API_KEY=...      # only if configs/pineapple_llm.entities.json is absent
#   cd /workspace
#   apt-get update && apt-get install -y git-lfs && git lfs install
#   git clone https://github.com/jianingq/scene-graph-robot.git && cd scene-graph-robot
#   git remote add fork "https://${GH_PAT}@github.com/vinzppp/scene-graph.git" && git fetch fork
#   git checkout fork/feature/section3.1-faithful -- run_all_4090.sh
#   bash run_all_4090.sh 2>&1 | tee /workspace/run_all_4090.console.log   # or inside tmux
#
# Knobs (env vars):
#   CONFIGS="giraffe pineapple_llm"                  which configs, in order (both DINOv2)
#   BRANCH=feature/section3.1-faithful               fork branch to overlay
#   ALLOW_ANY_GPU=1                                  skip the "is this a 4090" check
#   SKIP_SETUP=1                                     env already built on this pod
#   STORYBOARD_EPISODES=all                          or e.g. "0 3"   (default: all)
#
# Everything is logged under /workspace/run_logs/<RUN_ID>/ :
#   00_master.log          this script's full output (all steps, timestamps)
#   01_env.txt             GPU, driver, torch/CUDA, git revs, pip freeze
#   02_setup.log           setup_gpu.sh
#   03_kernel_test.log     GroundingDINO CUDA-kernel smoke test on this GPU
#   1x_<config>.log        pipeline log: per-episode detections + per-stage timings
#   1x_<config>_inspect.log step inventory + shape checks
#   <config>_run_summary.json, <config>_step_inventory.txt, <config>_storyboard/
#   <config>_rerun/        Rerun timeline of episode 0 (all episodes: <output_dir>/rerun/)
#   gpu_usage.csv          nvidia-smi every 5 s (util, memory, power, temp)
#   SUMMARY.md             one table for the whole run
# =============================================================================
set -uo pipefail

WS="${WS:-/workspace}"
REPO="${REPO:-$WS/scene-graph-robot}"
BRANCH="${BRANCH:-feature/section3.1-faithful}"
CONFIGS="${CONFIGS:-giraffe pineapple_llm}"
STORYBOARD_EPISODES="${STORYBOARD_EPISODES:-all}"
export TORCH_CUDA_ARCH_LIST=8.9            # RTX 4090 = Ada = sm_89
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=5                  # fewer progress-bar lines in log files

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_4090"
LOG_DIR="$WS/run_logs/$RUN_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/00_master.log") 2>&1

ts()      { date '+%F %T'; }
log()     { echo "[$(ts)] $*"; }
section() { echo; echo "================================================================"; log "$*"; echo "================================================================"; }
die()     { log "FATAL: $*"; exit 1; }
RUN_START=$(date +%s)

# -----------------------------------------------------------------------------
section "STEP 0/6  Preflight  (run id $RUN_ID)"
# -----------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this must run on a GPU pod"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *4090* && "${ALLOW_ANY_GPU:-0}" != 1 ]]; then
  die "expected an RTX 4090, got '$GPU_NAME' (set ALLOW_ANY_GPU=1 and TORCH_CUDA_ARCH_LIST to override)"
fi
FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
log "free disk on $WS: ${FREE_GB} GB"
[ "${FREE_GB:-0}" -ge 40 ] || log "WARNING: <40 GB free; episode_dino_full.npy + PNGs for 3 configs need ~25-35 GB"
[ -n "${GH_PAT:-}" ] || log "WARNING: GH_PAT not set — OK only if the fork remote is already fetched"

# -----------------------------------------------------------------------------
section "STEP 1/6  Code + data"
# -----------------------------------------------------------------------------
command -v git-lfs >/dev/null || { apt-get update -qq && apt-get install -y -qq git-lfs; }
git lfs install >/dev/null
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/jianingq/scene-graph-robot.git "$REPO" || die "upstream clone failed"
fi
cd "$REPO" || die "cannot cd to $REPO"
git lfs pull || die "git lfs pull failed"
if ! git remote | grep -qx fork; then
  [ -n "${GH_PAT:-}" ] || die "GH_PAT needed to add the fork remote"
  git remote add fork "https://${GH_PAT}@github.com/vinzppp/scene-graph.git"
fi
git fetch -q fork || die "git fetch fork failed (token expired/rotated?)"
# Overlay only code/config from the fork; upstream keeps the real LFS data/weights.
# (run_all_4090.sh itself is not re-checked-out here: never rewrite a running script.)
git checkout "fork/$BRANCH" -- '*.py' configs setup_gpu.sh run_section3.1.sh \
    SECTION_3_1_NOTES.md requirements.txt || die "overlay checkout failed"
log "upstream rev: $(git rev-parse --short HEAD)   fork/$BRANCH rev: $(git rev-parse --short "fork/$BRANCH")"

# Real data, not LFS pointers?
SAMPLE="$(find data -name trajectory.h5 | head -1)"
SIZE=$(stat -c %s "$SAMPLE" 2>/dev/null || echo 0)
[ "$SIZE" -gt 10000 ] || die "$SAMPLE is ${SIZE} bytes — looks like an LFS pointer, data missing"
# (DINOv2 weights come from torch.hub via setup_gpu.sh, not from pretrained/.)
W=pretrained/groundingdino_swint_ogc.pth
[ "$(stat -c %s "$W" 2>/dev/null || echo 0)" -gt 1000000 ] || die "$W missing or LFS pointer"
log "data + weights look real (e.g. $SAMPLE = $((SIZE/1024)) KB)"

# -----------------------------------------------------------------------------
section "STEP 2/6  Environment (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
# -----------------------------------------------------------------------------
kernel_test() {
python - <<'PY'
import torch
from groundingdino import _C
dev = "cuda"
print("torch", torch.__version__, "cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0), "| arch list", torch.cuda.get_arch_list())
value = torch.rand(1, 4, 8, 16, device=dev)                 # (bs, keys, heads, dim/head)
shapes = torch.tensor([[2, 2]], device=dev, dtype=torch.long)
start = torch.tensor([0], device=dev, dtype=torch.long)
loc = torch.rand(1, 3, 8, 1, 4, 2, device=dev)              # (bs, queries, heads, levels, points, 2)
attn = torch.rand(1, 3, 8, 1, 4, device=dev)
out = _C.ms_deform_attn_forward(value, shapes, start, loc, attn, 64)
torch.cuda.synchronize()
print("MSDeformAttn CUDA kernel OK on this GPU, out", tuple(out.shape))
PY
}

if [ "${SKIP_SETUP:-0}" != 1 ]; then
  # A GroundingDINO build left on a reused /workspace volume from an A40 pod
  # (sm_86) still imports fine but has no sm_89 kernel -> rebuild it.
  if [ -f "$WS/hodor_env.sh" ] && ! grep -q "TORCH_CUDA_ARCH_LIST=\"8.9\"" "$WS/hodor_env.sh"; then
    log "previous build was for another GPU arch -> forcing GroundingDINO rebuild"
    pip uninstall -y groundingdino >/dev/null 2>&1 || true
    rm -rf GroundingDINO/build GroundingDINO/*.egg-info
    find GroundingDINO -name '_C*.so' -delete 2>/dev/null || true
  fi
  bash setup_gpu.sh 2>&1 | tee "$LOG_DIR/02_setup.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "setup_gpu.sh failed (see 02_setup.log)"
fi
# shellcheck disable=SC1091
source "$WS/hodor_env.sh"
export TORCH_CUDA_ARCH_LIST=8.9
python -c "import matplotlib" 2>/dev/null || pip install -q matplotlib
# RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1 without shipping hf_transfer,
# which makes every Hugging Face download (GroundingDINO's BERT) crash.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" = 1 ] && ! python -c "import hf_transfer" 2>/dev/null; then
  pip install -q hf_transfer 2>/dev/null || { log "hf_transfer unavailable -> disabling fast HF downloads"; export HF_HUB_ENABLE_HF_TRANSFER=0; }
fi
# Fail fast if model downloads are broken, instead of failing every config.
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased'); print('HF download OK (bert-base-uncased)')" \
  || die "cannot load bert-base-uncased from Hugging Face (network / HF cache problem)"

python -c 'import groundingdino, segment_anything, cutie; print("perception deps OK")' \
  || die "perception imports failed"
if ! kernel_test > "$LOG_DIR/03_kernel_test.log" 2>&1; then
  cat "$LOG_DIR/03_kernel_test.log"
  log "kernel test failed -> rebuilding GroundingDINO for sm_89 once"
  pip uninstall -y groundingdino >/dev/null 2>&1 || true
  rm -rf GroundingDINO/build GroundingDINO/*.egg-info
  find GroundingDINO -name '_C*.so' -delete 2>/dev/null || true
  pip install -q --no-build-isolation -e GroundingDINO 2>&1 | tee -a "$LOG_DIR/02_setup.log"
  kernel_test > "$LOG_DIR/03_kernel_test.log" 2>&1 || { cat "$LOG_DIR/03_kernel_test.log"; die "GroundingDINO kernel still broken"; }
fi
cat "$LOG_DIR/03_kernel_test.log"

{
  echo "run_id: $RUN_ID"; echo "date: $(date -u)"; echo "host: $(hostname)"
  echo; nvidia-smi
  echo; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  echo; python -c "import sys,torch,torchvision;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print('torchvision',torchvision.__version__)"
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST HF_HOME=${HF_HOME:-} TORCH_HOME=${TORCH_HOME:-}"
  echo; echo "upstream: $(git rev-parse HEAD)"; echo "fork/$BRANCH: $(git rev-parse "fork/$BRANCH")"
  echo; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
  echo; pip freeze
} > "$LOG_DIR/01_env.txt" 2>&1
log "environment snapshot -> 01_env.txt"

# -----------------------------------------------------------------------------
section "STEP 3/6  GPU monitor"
# -----------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 5 > "$LOG_DIR/gpu_usage.csv" 2>/dev/null &
GPU_MON=$!
trap 'kill $GPU_MON 2>/dev/null' EXIT
log "sampling GPU every 5 s -> gpu_usage.csv (pid $GPU_MON)"

# -----------------------------------------------------------------------------
section "STEP 4/6  Pipelines: $CONFIGS"
# -----------------------------------------------------------------------------
out_dir_of() {
  python -c "from omegaconf import OmegaConf as O;import sys;print(O.load(sys.argv[1]).output.output_dir)" "configs/$1.yaml"
}
declare -A STATUS DURATION
i=10
for cfg in $CONFIGS; do
  i=$((i + 1))
  [ -f "configs/$cfg.yaml" ] || { log "no configs/$cfg.yaml — skipped"; STATUS[$cfg]="missing"; continue; }
  OUT="$(out_dir_of "$cfg")"
  section "[$cfg] start   (output -> $OUT)"

  # Don't mix a previous run's (old flat-layout) files into this one.
  if [ -e "$OUT" ]; then
    mv "$OUT" "${OUT}_prev_${RUN_ID}"
    log "moved previous output aside -> ${OUT}_prev_${RUN_ID}"
  fi

  if [ "$cfg" = pineapple_llm ]; then
    log "[$cfg] LLM entity front-end (task string -> GroundingDINO classes)"
    if [ ! -f configs/pineapple_llm.entities.json ] && [ -z "${OPENROUTER_API_KEY:-}${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
      log "[$cfg] no entity cache and no LLM API key — skipped"; STATUS[$cfg]="skipped (no API key)"; continue
    fi
    python -u llm_entities.py --config "configs/$cfg.yaml" --write 2>&1 | tee "$LOG_DIR/${i}_${cfg}_entities.log"
  fi

  T0=$(date +%s)
  python -u droid_hodor_test_cutie.py --config "configs/$cfg.yaml" 2>&1 | tee "$LOG_DIR/${i}_${cfg}.log"
  RC=${PIPESTATUS[0]}
  DURATION[$cfg]=$(( $(date +%s) - T0 ))
  if [ "$RC" -ne 0 ]; then
    log "[$cfg] FAILED rc=$RC after ${DURATION[$cfg]}s — continuing with next config"
    STATUS[$cfg]="FAILED rc=$RC"
    continue
  fi
  log "[$cfg] pipeline finished in $((DURATION[$cfg] / 60)) min $((DURATION[$cfg] % 60)) s"

  log "[$cfg] inspecting per-step outputs + storyboards + videos"
  # shellcheck disable=SC2086
  python -u inspect_run.py --config "configs/$cfg.yaml" --episode $STORYBOARD_EPISODES --video \
    2>&1 | tee "$LOG_DIR/${i}_${cfg}_inspect.log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then STATUS[$cfg]="OK"; else STATUS[$cfg]="checks FAILED"; fi

  cp "$OUT/run_summary.json"     "$LOG_DIR/${cfg}_run_summary.json" 2>/dev/null
  cp "$OUT/step_inventory.txt"   "$LOG_DIR/${cfg}_step_inventory.txt" 2>/dev/null
  cp -r "$OUT/storyboard"        "$LOG_DIR/${cfg}_storyboard" 2>/dev/null
  mkdir -p "$LOG_DIR/${cfg}_videos" && cp "$OUT"/step3_track/*.mp4 "$LOG_DIR/${cfg}_videos/" 2>/dev/null
  # Rerun timeline: episode 0 goes in the bundle; all episodes stay in $OUT/rerun/.
  mkdir -p "$LOG_DIR/${cfg}_rerun" && cp "$OUT"/rerun/ep0_*.rrd "$LOG_DIR/${cfg}_rerun/" 2>/dev/null
  # Small sample of every step (episode 0) for quick browsing / git.
  S="$LOG_DIR/${cfg}_samples"; mkdir -p "$S"/step1_detect "$S"/step2_segment "$S"/step3_track
  cp "$OUT"/step1_detect/0_*_boxes.png          "$S/step1_detect/"  2>/dev/null
  cp "$OUT"/step2_segment/0_*                   "$S/step2_segment/" 2>/dev/null
  (cd "$OUT/step3_track" && ls 0_*_overlay.png 2>/dev/null | sort -t_ -k2 -n | awk 'NR%10==1' \
    | xargs -r cp -t "$S/step3_track/")
  log "[$cfg] ${STATUS[$cfg]}"
done

# -----------------------------------------------------------------------------
section "STEP 5/6  Summary"
# -----------------------------------------------------------------------------
kill $GPU_MON 2>/dev/null
python - "$LOG_DIR" "$GPU_NAME" $CONFIGS <<'PY' | tee "$LOG_DIR/SUMMARY.md"
import csv, json, os, sys
log_dir, gpu, cfgs = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"# Section 3.1 re-run on {gpu}\n\nrun: `{os.path.basename(log_dir)}`\n")
print("| config | episodes | frames | run min | fps | GDINO s | SAM s | Cutie s | 3D s | DINO s | load+prep s | peak GPU GB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in cfgs:
    p = os.path.join(log_dir, f"{c}_run_summary.json")
    if not os.path.exists(p):
        print(f"| {c} | — | — | — | — | — | — | — | — | — | — | — |"); continue
    s = json.load(open(p)); st = s["stages"]
    g = lambda k: f"{st.get(k, {}).get('seconds', 0):.0f}"
    lp = st.get("load", {}).get("seconds", 0) + st.get("preprocess", {}).get("seconds", 0)
    peak = max([e.get("peak_gpu_mem_gb") or 0 for e in s["episodes"]] or [0])
    print(f"| {c} | {s['episodes_ok']}/{s['episodes_total']} | {s['frames_total']} | {s['run_s']/60:.1f} | {s['fps']:.2f} "
          f"| {g('detect')} | {g('segment')} | {g('track')} | {g('centroid')} | {g('embed')} | {lp:.0f} | {peak} |")
    shapes = {tuple(e['pr_shape']) for e in s['episodes'] if e.get('pr_shape')}
    print(f"\n<!-- {c}: classes {s['classes']}, backbone {s['backbone']}, pr shapes {sorted(shapes)[:3]}... -->")
gpu_csv = os.path.join(log_dir, "gpu_usage.csv")
if os.path.exists(gpu_csv):
    rows = list(csv.reader(open(gpu_csv)))[1:]
    try:
        util = [float(r[1].strip().split()[0]) for r in rows if len(r) > 4]
        mem = [float(r[2].strip().split()[0]) for r in rows if len(r) > 4]
        pw = [float(r[4].strip().split()[0]) for r in rows if len(r) > 4]
        print(f"\nGPU over run: mean util {sum(util)/len(util):.0f}%, peak mem {max(mem)/1024:.1f} GB, "
              f"mean power {sum(pw)/len(pw):.0f} W ({len(rows)} samples)")
    except Exception as exc:
        print(f"\n(gpu_usage.csv unparsed: {exc})")
PY
echo
for cfg in $CONFIGS; do
  log "$(printf '%-16s %-22s %s' "$cfg" "${STATUS[$cfg]:-not run}" "${DURATION[$cfg]:+$((DURATION[$cfg]/60)) min}")"
done
TOTAL=$(( $(date +%s) - RUN_START ))
log "total wall time: $((TOTAL / 60)) min $((TOTAL % 60)) s"

# -----------------------------------------------------------------------------
section "STEP 6/6  Bundle"
# -----------------------------------------------------------------------------
BUNDLE="$WS/run_bundle_${RUN_ID}.tar.gz"
tar -czf "$BUNDLE" -C "$WS/run_logs" "$RUN_ID"
log "logs + summaries + storyboards + videos + samples -> $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"
log "full outputs (all step folders) stay in $REPO/hodor_output/"
log "Rerun timelines (one per episode): $REPO/hodor_output/<config>/rerun/*.rrd"
log "  view on your laptop:  pip install rerun-sdk==0.37.2 && rerun <file>.rrd"
log "to keep them in git:  cp -r $LOG_DIR $REPO/results/run_logs/  (then commit from the pod)"
