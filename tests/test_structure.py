"""Testes para verificar a estrutura do projeto."""

from pathlib import Path

import pytest


def get_project_root() -> Path:
    """Retorna o diretorio raiz do projeto."""
    return Path(__file__).parent.parent


class TestProjectStructure:
    """Testes da estrutura de diretorios."""

    def test_src_directories_exist(self):
        """Verifica se os diretorios src existem."""
        root = get_project_root()
        expected_dirs = [
            "src",
            "src/ingest",
            "src/validation",
            "src/features",
            "src/modeling",
            "src/evaluation",
            "src/inference",
        ]
        for dir_name in expected_dirs:
            assert (root / dir_name).exists(), f"Diretorio {dir_name} nao encontrado"

    def test_config_files_exist(self):
        """Verifica se os arquivos de configuracao existem."""
        root = get_project_root()
        expected_configs = [
            "configs/geo.yaml",
            "configs/target.yaml",
            "configs/climate.yaml",
            "configs/features.yaml",
            "configs/split.yaml",
            "configs/model.yaml",
        ]
        for config in expected_configs:
            assert (root / config).exists(), f"Config {config} nao encontrado"

    def test_project_files_exist(self):
        """Verifica se os arquivos principais do projeto existem."""
        root = get_project_root()
        expected_files = [
            "pyproject.toml",
            "Makefile",
            "README.md",
            ".gitignore",
            "dvc.yaml",
            ".pre-commit-config.yaml",
        ]
        for file_name in expected_files:
            assert (root / file_name).exists(), f"Arquivo {file_name} nao encontrado"

    def test_data_directories_structure(self):
        """Verifica se os diretorios de dados podem ser criados."""
        root = get_project_root()
        expected_dirs = [
            "data/raw",
            "data/processed",
            "models",
            "results",
        ]
        for dir_name in expected_dirs:
            dir_path = root / dir_name
            # Cria o diretório se não existir (são ignorados pelo git)
            dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists(), f"Diretorio {dir_name} nao pode ser criado"


class TestImports:
    """Testes de importacao dos modulos."""

    def test_import_src(self):
        """Verifica se o modulo src pode ser importado."""
        import src

        assert src.__version__ == "0.1.0"

    def test_import_submodules(self):
        """Verifica se os submodulos podem ser importados."""
        import src.ingest
        import src.validation
        import src.features
        import src.modeling
        import src.evaluation
        import src.inference

        assert True  # Se chegou aqui, imports funcionaram
