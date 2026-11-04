#!/usr/bin/env bash
# Sequential formal A2 then A3 (single GPU). Do not edit model code mid-run.
set -euo pipefail
ROOT=/root/projects/ASRT
LOGDIR=/root/data/experiments/ASRT/vnext_v1/logs
mkdir -p "$LOGDIR"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttsr
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

COMMIT=$(git rev-parse HEAD)
echo "FORMAL START commit=$COMMIT $(date -Is)" | tee -a "$LOGDIR/formal_orchestrator.log"

echo "=== A2 START $(date -Is) ===" | tee -a "$LOGDIR/formal_orchestrator.log"
python scripts/train_irp.py \
  --config configs/asrt_irp_v1_a2.yaml \
  --max_steps 80000 \
  2>&1 | tee "$LOGDIR/a2_train_stdout.log"
echo "=== A2 DONE $(date -Is) exit=${PIPESTATUS[0]} ===" | tee -a "$LOGDIR/formal_orchestrator.log"

echo "=== A3 START $(date -Is) ===" | tee -a "$LOGDIR/formal_orchestrator.log"
python scripts/train_irp.py \
  --config configs/asrt_irp_v1_a3.yaml \
  --max_steps 80000 \
  2>&1 | tee "$LOGDIR/a3_train_stdout.log"
echo "=== A3 DONE $(date -Is) exit=${PIPESTATUS[0]} ===" | tee -a "$LOGDIR/formal_orchestrator.log"

echo "FORMAL ALL DONE $(date -Is)" | tee -a "$LOGDIR/formal_orchestrator.log"
