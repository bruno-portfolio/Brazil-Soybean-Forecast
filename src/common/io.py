from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

_MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
_TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"


def load_config(config_name: str, section: str | None = None) -> dict[str, Any]:
    """Carrega configuracao YAML do diretorio configs/.

    Args:
        config_name: Nome do arquivo sem extensao (ex: "climate", "target").
        section: Chave de primeiro nivel a retornar. Se None, retorna o dict inteiro.
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if section is not None:
        return config[section]
    return config


def load_municipalities(columns: list[str] | None = None) -> pd.DataFrame:
    """Carrega tabela de municipios com coordenadas.

    Args:
        columns: Colunas a retornar. Se None, retorna todas.

    Raises:
        FileNotFoundError: Se o arquivo nao existe.
    """
    if not _MUNICIPALITIES_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de municipios nao encontrado: {_MUNICIPALITIES_PATH}\n"
            "Execute primeiro: python -m src.ingest.municipalities"
        )
    df = pd.read_parquet(_MUNICIPALITIES_PATH)
    if columns is not None:
        df = df[columns]
    return df


def load_target_municipalities() -> set[int]:
    """Carrega conjunto de cod_ibge dos municipios produtores de soja.

    Raises:
        FileNotFoundError: Se o arquivo nao existe.
    """
    if not _TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de target nao encontrado: {_TARGET_PATH}\n"
            "Execute primeiro: python -m src.ingest.pam"
        )
    df = pd.read_parquet(_TARGET_PATH)
    return set(df["cod_ibge"].unique())
