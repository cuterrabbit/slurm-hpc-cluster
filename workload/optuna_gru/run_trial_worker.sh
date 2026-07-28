#!/bin/bash
# Usage: sbatch run_trial_worker.sh <trials> <backend: sqlite|mysql> <study-name>
set -euo pipefail

TRIALS="$1"
BACKEND="$2"
STUDY_NAME="$3"

case "$BACKEND" in
  mysql)
    DB_PASSWORD="$(cat /apps/containers/optuna_db_password)"
    STORAGE="mysql+pymysql://optuna:${DB_PASSWORD}@head:3306/optuna_db"
    ;;
  sqlite)
    STORAGE="sqlite:////apps/workload/optuna_gru/data/study.db"
    ;;
  *)
    echo "Unknown backend: $BACKEND (expected sqlite or mysql)" >&2
    exit 1
    ;;
esac

apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload \
  /apps/containers/optuna_gru.sif \
  python -m src.models.sequence_tune \
    --model gru \
    --trials "$TRIALS" \
    --cpu-threads "${SLURM_CPUS_PER_TASK:-1}" \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME"
