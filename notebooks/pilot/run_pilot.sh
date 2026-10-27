#!/usr/bin/env bash
# Reduced pilot on one GPU: train A0 then A2 (sg_pilot.train via train.py), then evaluate both at Q0 on the
# test split (sg_pilot.evaluate via eval.py). Same code as notebook 9.7 / 10.4; checkpoints at the
# notebook's paths, so the notebook in pilot mode loads them instead of retraining.
#   tmux new-window -d -n pilot "bash run_pilot.sh"
# Logs: /workspace/logs/pilot/. Finished steps (train_info.json / result JSON present) are skipped on re-run.
set -uo pipefail
cd "$(dirname "$0")"
source /workspace/env/pod_env.sh
PY=${SG_MAIN_PYTHON:-/workspace/envs/sg-main/bin/python}
TASK=VideoUnmaskSwap
STEPS=${STEPS:-20000}
EPISODES=${EPISODES:-50}
GPU=${GPU:-0}
LOG=/workspace/logs/pilot
mkdir -p "$LOG" results
export CUDA_VISIBLE_DEVICES=$GPU

step() {  # name, done-marker, command...
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[$(date '+%F %T')] skip $name ($marker exists)" | tee -a "$LOG/pilot.log"; return 0; fi
  echo "[$(date '+%F %T')] start $name" | tee -a "$LOG/pilot.log"
  "$@" 2>&1 | tee -a "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] end $name exit $rc" | tee -a "$LOG/pilot.log"
  return $rc
}

for A in A0 A2; do
  step "train_$A" "runs/pilot_${A}_s0/train_info.json" \
    "$PY" train.py --interface $A --tasks $TASK --seed 0 --steps $STEPS --episodes-per-task 100 \
      --batch-size 32 --out "runs/pilot_${A}_s0" || exit 1
done
for A in A0 A2; do
  step "eval_${A}_Q0" "results/pilot_${A}_Q0.json" \
    "$PY" eval.py --ckpt "runs/pilot_${A}_s0" --interface $A --task $TASK --quality Q0 \
      --episodes $EPISODES --out "results/pilot_${A}_Q0.json" || exit 1
done
echo "[$(date '+%F %T')] pilot done" | tee -a "$LOG/pilot.log"
touch "$LOG/pilot.done"
