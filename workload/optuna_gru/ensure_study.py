"""Create (or confirm) an Optuna study once before launching concurrent workers.

Multiple worker processes calling optuna.create_study(..., load_if_exists=True)
at the same time can race to create the study schema tables, causing
"table already exists" errors. Run this once before submitting concurrent
workers so the schema and study row already exist.
"""
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
