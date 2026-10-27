#!/usr/bin/env bash
# Train several arms in parallel, one GPU each, every GPU in its own tmux window.
# Same code path as the notebook and run_grid.sh: each arm is `python train.py ...` (sg_pilot.train).
#
#   ./run_arms_parallel.sh --interfaces A0,A2 --label pilot              # pilot arms, one GPU each
#   ./run_arms_parallel.sh --interfaces A0,A1,A2,A3,A3feat --seed 1 \
#       --tasks VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap --label grid_s1
#   ./run_arms_parallel.sh --policy pi05 --interfaces A0,A1,A2,A3 --tasks permanence \
#       --steps 80000 --batch 32 --accum 2 --label pi05_s0                 # pi0.5, RoboMME recipe
#   ./run_arms_parallel.sh --label pilot --status                         # progress of a running set
#
# More arms than GPUs: arms are dealt round-robin and each window trains its arms one after another.
# An arm whose output already has train_info.json is skipped, so a crashed set can simply be re-run.
# Logs, per-GPU runner scripts and exit codes: /workspace/logs/train/<label>/
# Checkpoints: notebooks/pilot/runs/<label>_<interface>_s<seed>/ (label "pilot" = the notebook's 9.7 paths)
# Needs the replay + graph cache (notebook Steps 3.6 and 7) for --episodes episodes of every task.
set -euo pipefail

PILOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=${SG_MAIN_PYTHON:-/workspace/envs/sg-main/bin/python}
POLICY=smolvla
INTERFACES=A0,A2
TASKS=VideoUnmaskSwap
SEED=0
STEPS=20000          # notebook Step 9.6: ~2.4 h per arm at batch 32 on one RTX 4090
EPISODES=100
BATCH=32
ACCUM=1
LABEL=pilot
GPUS=""
STATUS=0
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --policy) POLICY=$2; shift 2 ;;
    --interfaces) INTERFACES=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --episodes) EPISODES=$2; shift 2 ;;
    --batch) BATCH=$2; shift 2 ;;
    --accum) ACCUM=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --status) STATUS=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$TASKS" = permanence ] && TASKS=VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap
LOGDIR=/workspace/logs/train/$LABEL
RUNDIR=$PILOT_DIR/runs
SESSION=sg-train-$LABEL

if [ "$STATUS" = 1 ]; then
  for f in "$LOGDIR"/*.log; do
    [ -e "$f" ] || { echo "no logs in $LOGDIR"; exit 0; }
    arm=$(basename "$f" .log)
    st=$(cat "$LOGDIR/$arm.status" 2>/dev/null || echo running)
    last=$(grep -E "^step " "$f" | tail -1 || true)
    echo "$arm  [$st]  ${last:-no step logged yet}"
  done
  exit 0
fi

# GPUs to use
if [ -z "$GPUS" ]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
IFS=, read -r -a GPU_LIST <<< "$GPUS"
IFS=, read -r -a ARM_LIST <<< "$INTERFACES"
[ ${#GPU_LIST[@]} -gt 0 ] || { echo "no GPUs found" >&2; exit 1; }

# The cache must hold enough episodes for every task, or train() would quietly use fewer
for t in ${TASKS//,/ }; do
  n=$(ls "$PILOT_DIR/cache/$t"/ep*_graph.npz 2>/dev/null | wc -l)
  if [ "$n" -lt "$EPISODES" ]; then
    echo "cache for $t has $n graph files, need $EPISODES: run the notebook (Steps 3.6 and 7) in pilot mode," >&2
    echo "or tools/build_cache.py --task $t --episodes $EPISODES, first," >&2
    echo "or pass --episodes $n" >&2
    exit 1
  fi
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists (a set with this label is running?)." >&2
  echo "Check it with: $0 --label $LABEL --status   or remove it: tmux kill-session -t $SESSION" >&2
  exit 1
fi

mkdir -p "$LOGDIR" "$RUNDIR"
echo "label $LABEL | policy $POLICY | tasks $TASKS | seed $SEED | steps $STEPS | episodes/task $EPISODES | batch $BATCH x $ACCUM"

# Deal arms round-robin over the GPUs and write one runner script per GPU
declare -A ASSIGN
for i in "${!ARM_LIST[@]}"; do
  g=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  ASSIGN[$g]="${ASSIGN[$g]:-} ${ARM_LIST[$i]}"
done

first=1
for g in "${GPU_LIST[@]}"; do
  arms=${ASSIGN[$g]:-}
  [ -n "$arms" ] || continue
  runner=$LOGDIR/gpu$g.sh
  {
    echo "#!/usr/bin/env bash"
    echo "# GPU $g, arms:$arms (written by run_arms_parallel.sh)"
    echo "set -uo pipefail"
    echo "source /workspace/env/pod_env.sh"
    echo "export CUDA_VISIBLE_DEVICES=$g"
    echo "cd '$PILOT_DIR'"
    for arm in $arms; do
      out=$RUNDIR/${LABEL}_${arm}_s$SEED
      log=$LOGDIR/${arm}_s$SEED.log
      echo "if [ -f '$out/train_info.json' ]; then echo 'skip $arm: $out/train_info.json exists'; echo skipped > '$LOGDIR/${arm}_s$SEED.status'; else"
      echo "  echo \"[\$(date '+%F %T')] GPU $g: $arm start\" | tee -a '$log'"
      echo "  '$PY' train.py --policy $POLICY --interface $arm --tasks $TASKS --seed $SEED --steps $STEPS \\"
      echo "    --episodes-per-task $EPISODES --batch-size $BATCH --accum $ACCUM --out '$out' 2>&1 | tee -a '$log'"
      echo "  rc=\$?; echo \"exit \$rc\" > '$LOGDIR/${arm}_s$SEED.status'"
      echo "  echo \"[\$(date '+%F %T')] GPU $g: $arm exit \$rc\" | tee -a '$log'"
      echo "fi"
    done
    echo "touch '$LOGDIR/gpu$g.done'"
  } > "$runner"
  chmod +x "$runner"
  echo "GPU $g:$arms   (runner $runner)"
  [ "$DRY" = 1 ] && continue
  rm -f "$LOGDIR/gpu$g.done"
  if [ $first = 1 ]; then
    tmux new-session -d -s "$SESSION" -n "gpu$g" "bash '$runner'"
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
    first=0
  else
    tmux new-window -d -t "$SESSION" -n "gpu$g" "bash '$runner'"
  fi
done

if [ "$DRY" = 1 ]; then
  echo "dry run: runner scripts written, nothing started"
  exit 0
fi
echo
echo "started tmux session $SESSION (one window per GPU)"
echo "  watch:    tmux attach -t $SESSION        (Ctrl-b n / p switches GPU windows)"
echo "  progress: $0 --label $LABEL --status"
echo "  logs:     $LOGDIR"
