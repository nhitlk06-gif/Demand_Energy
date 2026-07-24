#!/usr/bin/env python3
"""Run only the cleaning stage: raw demanddata_*.csv -> electricity_cleaned.csv"""
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from demandforecast.cleaning import run_cleaning_pipeline  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cleaning_pipeline()
