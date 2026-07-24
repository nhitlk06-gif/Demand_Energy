#!/usr/bin/env python3
"""Run only the feature-engineering stage:
electricity_cleaned.csv -> electricity_features.csv

Note: this variant uses a minimum lag of 12 settlement periods (6 hours) —
see src/demandforecast/config.py -> SHORT_TERM_LAGS.
"""
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from demandforecast.features import run_feature_pipeline  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_feature_pipeline()
