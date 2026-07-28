# optuna_gru — 부하 생성용 GRU/Optuna 튜닝 워크로드

이 클러스터의 스케줄러·스토리지·정책이 실제 부하 아래서 동작하는지 검증하기 위한 부하 생성기다. 튜닝 결과 자체(모델 정확도)는 이 프로젝트의 관심사가 아니다 — `ARCHITECTURE.md` 참고.

## 출처

[FISA-Agri-Pay/ai-prediction-model](https://github.com/FISA-Agri-Pay/ai-prediction-model)의 GRU 시퀀스 모델 + Optuna 튜닝 경로만 가져왔다(Prophet/SARIMA/서비스 오토스케일링 실험 등은 제외). 가져온 범위와 이유: [ADR-0012](../../docs/decisions/ADR-0012-optuna-workload-source-and-distribution-design.md).

원본 대비 수정한 부분은 `src/models/sequence_tune.py`의 `--storage`/`--study-name` 옵션뿐이다 — 여러 프로세스가 같은 Optuna study를 공유 storage로 나눠 처리할 수 있게 하기 위함(원본은 프로세스 하나 안에서만 동작하는 구조였음).

## 실행 전 준비

```bash
pip install -r requirements.txt
python -m src.data.generate_dummy_data   # data/processed/traffic.csv 생성 (합성 데이터, 실제 데이터셋 불필요)
```

## 단일 프로세스로 실행 (기본, storage 미지정 시 인메모리)

```bash
python -m src.models.sequence_tune --model gru --trials 10
```

## 분산 실행 (여러 프로세스가 같은 study 공유)

```bash
# 프로세스마다 --trials는 "이 프로세스가 처리할 트라이얼 수" — 전체는 프로세스 수 × trials
python -m src.models.sequence_tune --model gru --trials 10 \
  --storage "sqlite:////apps/workload/optuna_gru/study.db" \
  --study-name gru-loadtest

# MariaDB(ADR-0011) storage로 — SQLite-on-NFS 동시성 문제를 피하려면 이쪽
python -m src.models.sequence_tune --model gru --trials 10 \
  --storage "mysql+pymysql://optuna:<password>@127.0.0.1:3306/optuna_db" \
  --study-name gru-loadtest
```

## 트라이얼당 자원 고정

`--cpu-threads`/`--interop-threads`로 프로세스당 PyTorch CPU 스레드 수를 고정할 수 있다 — 병렬도(동시 프로세스 수)를 늘려가며 처리량을 측정할 때, 트라이얼 하나가 쓰는 자원 자체는 일정하게 유지하기 위함.
