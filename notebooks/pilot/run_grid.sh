#!/usr/bin/env bash
# Full experiment grid. Written by the notebook build; NOT run by it.
# Main grid: one model per (interface, seed) trained jointly on all 4 permanence tasks,
#            evaluated on each task x quality level.
# Held-out:  A0, A2, A3feat, seed 0, trained on 3 tasks, evaluated on the held-out task.
# Q3 runs the perception stack online in every step (slower; time per episode is printed by eval.py).
set -euo pipefail
cd "$(dirname "$0")"

STEPS=${STEPS:-20000}           # set from the pilot's measured speed (notebook Step 9.6)
EPISODES=${EPISODES:-50}
HELDOUT=${HELDOUT:-ButtonUnmaskSwap}
TASKS="VideoUnmask VideoUnmaskSwap ButtonUnmask ButtonUnmaskSwap"
ALL=$(echo $TASKS | tr ' ' ',')
QUALITIES="Q0 Q1_20 Q1_50 Q2_25 Q2_50 Q3"

for IFACE in A0 A1 A2 A3 A3feat; do
  for SEED in 0 1 2; do
    RUN=runs/grid/${IFACE}_all_s${SEED}
    [ -f "$RUN/train_info.json" ] || python train.py --interface $IFACE --tasks $ALL --seed $SEED \
        --steps $STEPS --out $RUN
    for TASK in $TASKS; do
      for Q in $QUALITIES; do
        OUT=results/grid/${IFACE}_s${SEED}_${TASK}_${Q}.json
        [ -f "$OUT" ] || python eval.py --ckpt $RUN --interface $IFACE --task $TASK --quality $Q \
            --episodes $EPISODES --out $OUT
      done
    done
  done
done

TRAIN3=$(echo $TASKS | tr ' ' '\n' | grep -v "^${HELDOUT}$" | paste -sd, -)
for IFACE in A0 A2 A3feat; do
  RUN=runs/heldout/${IFACE}_no${HELDOUT}_s0
  [ -f "$RUN/train_info.json" ] || python train.py --interface $IFACE --tasks $TRAIN3 --seed 0 \
      --steps $STEPS --out $RUN
  OUT=results/heldout/${IFACE}_s0_${HELDOUT}_Q0.json
  [ -f "$OUT" ] || python eval.py --ckpt $RUN --interface $IFACE --task $HELDOUT --quality Q0 \
      --episodes $EPISODES --out $OUT
done
