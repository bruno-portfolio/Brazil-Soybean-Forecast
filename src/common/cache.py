from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_cached_codes(cache_dir: Path) -> set[int]:
    """Retorna conjunto de cod_ibge ja presentes no cache.

    Args:
        cache_dir: Diretorio de cache com arquivos {cod_ibge}.parquet.
    """
    if not cache_dir.exists():
        return set()

    cached = set()
    for f in cache_dir.glob("*.parquet"):
        try:
            cached.add(int(f.stem))
        except ValueError:
            continue

    return cached


def save_to_cache(data: pd.DataFrame | dict, cod_ibge: int, cache_dir: Path) -> None:
    """Salva dados de um municipio no cache como parquet.

    Args:
        data: DataFrame ou dict (convertido para DataFrame automaticamente).
        cod_ibge: Codigo IBGE do municipio.
        cache_dir: Diretorio de cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cod_ibge}.parquet"
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    data.to_parquet(cache_path, index=False)


def consolidate_cache(cache_dir: Path) -> pd.DataFrame:
    """Consolida todos os arquivos parquet de cache em um unico DataFrame.

    Args:
        cache_dir: Diretorio de cache.

    Raises:
        FileNotFoundError: Se o diretorio nao existe.
        ValueError: Se nenhum arquivo de cache foi encontrado.
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"Diretorio de cache nao encontrado: {cache_dir}")

    all_dfs = []
    for f in cache_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Erro ao ler {f}: {e}")

    if not all_dfs:
        raise ValueError("Nenhum arquivo de cache encontrado")

    return pd.concat(all_dfs, ignore_index=True)
