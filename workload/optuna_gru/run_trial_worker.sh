#!/bin/bash
# Usage: sbatch run_trial_worker.sh <trials> <backend: sqlite|mysql> <study-name>
#SBATCH --output=/apps/workload/optuna_gru/data/slurm-%j.out
#SBATCH --error=/apps/workload/optuna_gru/data/slurm-%j.err
set -euo pipefail

TRIALS="$1"
BACKEND="$2"
STUDY_NAME="$3"

case "$BACKEND" in
  mysql)
    DB_PASSWORD="$(cat /apps/containers/optuna_db_password)"
    # base64 비밀번호의 +, /, = 는 URL에서 특수 의미가 있어 SQLAlchemy 파싱이 깨진다 — percent-encode 필요
    DB_PASSWORD_ENCODED="$(echo -n "$DB_PASSWORD" | sed 's/+/%2B/g; s/\//%2F/g; s/=/%3D/g')"
    STORAGE="mysql+pymysql://optuna:${DB_PASSWORD_ENCODED}@head:3306/optuna_db"
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
