import argparse

import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--storage", required=True)
parser.add_argument("--study-name", required=True)
args = parser.parse_args()

optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    direction="minimize",
    load_if_exists=True,
)
print(f"Study ready: {args.study_name}")
