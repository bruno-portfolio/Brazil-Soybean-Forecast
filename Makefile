.PHONY: install install-dev lint format test test-cov clean run-pipeline help demo

PYTHON := python
PIP := pip

help:
	@echo "Comandos disponiveis:"
	@echo "  make install       - Instala dependencias de producao"
	@echo "  make install-dev   - Instala dependencias de desenvolvimento"
	@echo "  make demo          - Modo demonstracao rapida (sem DVC)"
	@echo "  make lint          - Executa linting (ruff + black check)"
	@echo "  make format        - Formata codigo (ruff fix + black)"
	@echo "  make test          - Executa testes"
	@echo "  make test-cov      - Executa testes com cobertura"
	@echo "  make clean         - Remove arquivos temporarios"
	@echo "  make run-pipeline  - Executa pipeline completo"
	@echo "  make dvc-pull      - Baixa dados versionados"
	@echo "  make dvc-push      - Envia dados versionados"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name "htmlcov" -delete
	find . -type f -name ".coverage" -delete

run-pipeline:
	@echo "Executando pipeline..."
	$(PYTHON) -m src.ingest.municipalities
	$(PYTHON) -m src.ingest.pam
	$(PYTHON) -m src.ingest.climate_power
	$(PYTHON) -m src.features.build_features
	$(PYTHON) -m src.modeling.train
	$(PYTHON) -m src.evaluation.evaluate
	@echo "Pipeline concluido!"

dvc-pull:
	dvc pull

dvc-push:
	dvc push

# Targets individuais do pipeline
ingest-geo:
	$(PYTHON) -m src.ingest.municipalities

ingest-target:
	$(PYTHON) -m src.ingest.pam

ingest-climate:
	$(PYTHON) -m src.ingest.climate_power

build-features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.modeling.train

evaluate:
	$(PYTHON) -m src.evaluation.evaluate

predict:
	$(PYTHON) -m src.inference.predict

# Demo mode - quick validation without full data
demo:
	@echo "Preparando modo demonstracao..."
	$(PYTHON) scripts/prepare_demo.py
	@echo "Iniciando dashboard com dados de exemplo..."
	streamlit run app/dashboard.py -- --demo
