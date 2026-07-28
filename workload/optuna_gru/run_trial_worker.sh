#!/bin/bash
set -euo pipefail

TRIALS="$1"
STORAGE="$2"
STUDY_NAME="$3"

apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload \
  /apps/containers/optuna_gru.sif \
  python -m src.models.sequence_tune \
    --model gru \
    --trials "$TRIALS" \
    --cpu-threads "${SLURM_CPUS_PER_TASK:-1}" \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME"
