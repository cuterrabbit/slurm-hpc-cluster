# optuna_gru — 부하 생성용 GRU/Optuna 튜닝 워크로드

이 클러스터의 스케줄러·스토리지·정책이 실제 부하 아래서 동작하는지 검증하기 위한 부하 생성기다.

## 출처

[FISA-Agri-Pay/ai-prediction-model](https://github.com/FISA-Agri-Pay/ai-prediction-model)의 GRU 시퀀스 모델 + Optuna 튜닝 경로만 가져왔다(Prophet/SARIMA/서비스 오토스케일링 실험 등은 제외). 

원본 대비 수정한 부분은 `src/models/sequence_tune.py`의 `--storage`/`--study-name` 옵션뿐이다 — 여러 프로세스가 같은 Optuna study를 공유 storage로 나눠 처리할 수 있게 하기 위함(원본은 프로세스 하나 안에서만 동작하는 구조였음).

## 실행 환경

Apptainer 이미지로 실행한다. 이미지에는 Python/torch/optuna 등 실행 환경만 들어있고, 코드는 이미지에 넣지 않고 실행 시점에 바인드 마운트한다.

```bash
# 이미지 빌드 (최초 1회, 또는 optuna_gru.def가 바뀌었을 때)
apptainer build --fakeroot /apps/containers/optuna_gru.sif optuna_gru.def

# 이후 모든 실행은 이 형태 — --bind로 이 디렉터리를 컨테이너 /workload에 연결
apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload \
  /apps/containers/optuna_gru.sif python -m <모듈경로> <인자...>
```

**주의**: `/apps/workload/optuna_gru/` 최상위 디렉터리 자체는 root 소유(0755)라 컨테이너 프로세스(비-root)가 그 안에 새 파일을 못 만든다 — study DB나 임시 산출물은 반드시 `data/`(0777) 밑에 둘 것.

## 실행 전 준비

```bash
apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload /apps/containers/optuna_gru.sif \
  python -m src.data.generate_dummy_data   # data/processed/traffic.csv 생성 (합성 데이터, 실제 데이터셋 불필요)
```

## 단일 프로세스로 실행 (기본, storage 미지정 시 인메모리)

```bash
apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload /apps/containers/optuna_gru.sif \
  python -m src.models.sequence_tune --model gru --trials 10
```

## 분산 실행 (여러 프로세스가 같은 study 공유)

```bash
# 프로세스마다 --trials는 "이 프로세스가 처리할 트라이얼 수" — 전체는 프로세스 수 × trials
apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload /apps/containers/optuna_gru.sif \
  python -m src.models.sequence_tune --model gru --trials 10 \
  --storage "sqlite:////workload/data/study.db" \
  --study-name gru-loadtest

# MariaDB(ADR-0011) storage로 — SQLite-on-NFS 동시성 문제를 피하려면 이쪽.
# 워커 노드에서 돌 수도 있으니 127.0.0.1이 아니라 head 호스트명 사용.
apptainer exec --bind /apps/workload/optuna_gru:/workload --pwd /workload /apps/containers/optuna_gru.sif \
  python -m src.models.sequence_tune --model gru --trials 10 \
  --storage "mysql+pymysql://optuna:<password>@head:3306/optuna_db" \
  --study-name gru-loadtest
```

여러 노드에 걸쳐 Slurm으로 분산 실행할 때는 `run_trial_worker.sh`를 `sbatch`로 제출한다(직접 `apptainer exec`를 셸에서 돌리면 Slurm의 cgroup 메모리 격리를 못 받는다).

## 트라이얼당 자원 고정

`--cpu-threads`/`--interop-threads`로 프로세스당 PyTorch CPU 스레드 수를 고정할 수 있다 — 병렬도(동시 프로세스 수)를 늘려가며 처리량을 측정할 때, 트라이얼 하나가 쓰는 자원 자체는 일정하게 유지하기 위함.

## 메모리 주의사항

`suggest_params`가 뽑는 `batch_size=0`은 미니배치가 아니라 풀배치(전체 학습 데이터를 한 번에)를 의미한다. 5년치 전체 데이터셋(약 4.4만 행)에서 이 값이 나오면 메모리 부족으로 프로세스가 죽을 수 있다. 스모크 테스트는 `--start`/`--end`로 기간을 좁힌 작은 데이터를 권장.
