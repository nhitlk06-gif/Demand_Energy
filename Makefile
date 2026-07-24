.PHONY: install clean features train pipeline pipeline-fast api dashboard test lint

install:
	pip install -e ".[dev,app,viz]"

clean:
	python scripts/run_cleaning.py

features:
	python scripts/run_features.py

train:
	python scripts/run_training.py

pipeline:
	python scripts/run_pipeline.py

pipeline-fast:
	python scripts/run_pipeline.py --fast

api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check src app tests
