"""One-call orchestrator that runs cleaning -> features -> training in
sequence, mirroring running notebooks 01, 03 and 04 back to back.
"""
from __future__ import annotations

import logging

from . import config
from .cleaning import run_cleaning_pipeline
from .features import run_feature_pipeline
from .train import run_training_pipeline

logger = logging.getLogger(__name__)


def run_full_pipeline(fast_mode: bool = False, include_tuned: bool = True) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/3: cleaning raw data")
    run_cleaning_pipeline()

    logger.info("Step 2/3: building feature matrix")
    run_feature_pipeline()

    logger.info("Step 3/3: training models")
    summary = run_training_pipeline(fast_mode=fast_mode, include_tuned=include_tuned)
    logger.info("Pipeline complete.\n%s", summary.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_full_pipeline()
