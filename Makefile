.PHONY: install install-dev lint format test test-cov clean run-pipeline dashboard help

PYTHON := python
PIP := pip

help:
	@echo "Comandos disponiveis:"
	@echo "  make install       - Instala dependencias de producao"
	@echo "  make install-dev   - Instala dependencias de desenvolvimento"
	@echo "  make lint          - Executa linting (ruff check + format check)"
	@echo "  make format        - Formata codigo (ruff fix + format)"
	@echo "  make test          - Executa testes"
	@echo "  make test-cov      - Executa testes com cobertura"
	@echo "  make clean         - Remove arquivos temporarios"
	@echo "  make run-pipeline  - Executa pipeline completo (dvc repro)"
	@echo "  make dashboard     - Inicia o dashboard Streamlit"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

format:
	ruff check --fix src/ scripts/
	ruff format src/ scripts/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]"

run-pipeline:
	dvc repro

dashboard:
	streamlit run app/dashboard.py

# Targets individuais do pipeline
build-features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.modeling.train

train-regional:
	$(PYTHON) -m src.modeling.train_regional

train-conformal:
	$(PYTHON) -m src.modeling.train_conformal

evaluate:
	$(PYTHON) -m src.modeling.baselines
	$(PYTHON) -m src.evaluation.evaluate

temporal-cv:
	$(PYTHON) -m scripts.temporal_cv

predict:
	$(PYTHON) -m src.inference.predict
	$(PYTHON) -m src.business.risk_translator
