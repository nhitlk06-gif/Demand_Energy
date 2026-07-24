#!/usr/bin/env python3
"""Run only the training stage: electricity_features.csv -> models/*.pkl

By default this also fits the random-search-tuned XGBoost/LightGBM variants
(``--no-tuned`` to skip them and train only the four base models).
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from demandforecast.train import run_training_pipeline 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Use lighter hyperparameters for a quick run.")
    parser.add_argument("--no-tuned", action="store_true", help="Skip the tuned XGBoost/LightGBM variants.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_training_pipeline(fast_mode=args.fast, include_tuned=not args.no_tuned)
    print(summary.to_string())
