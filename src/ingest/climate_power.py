"""Ingestao de dados climaticos diarios via agrobr (NASA POWER)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.common.cache import consolidate_cache, get_cached_codes, save_to_cache
from src.common.io import (
    PROJECT_ROOT,
    load_config,
    load_municipalities,
    load_target_municipalities,
)

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "climate"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"

COLUMN_RENAME = {
    "data": "date",
    "temp_min": "tmin",
    "temp_media": "tmean",
    "temp_max": "tmax",
    "precip_mm": "precip",
    "umidade_rel": "rh",
    "radiacao_mj": "radiation",
    "vento_ms": "wind_speed",
}


async def download_climate_for_municipality(
    cod_ibge: int, lat: float, lon: float, config: dict
) -> pd.DataFrame | None:
    """Baixa dados climaticos para um municipio via agrobr."""
    from agrobr.nasa_power import clima_ponto

    inicio = f"{config['year_start']}-01-01"
    fim = f"{config['year_end']}-12-31"

    try:
        df = await clima_ponto(lat, lon, inicio, fim, agregacao="diario")
        df = df.rename(columns=COLUMN_RENAME)
        df["cod_ibge"] = cod_ibge
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning(f"cod_ibge={cod_ibge}: erro - {e}")
        return None


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset climatico."""
    return {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "periodo": {
            "min": str(df["date"].min().date()),
            "max": str(df["date"].max().date()),
        },
    }


async def _fetch_one(row: dict, config: dict, semaphore: asyncio.Semaphore, progress: dict):
    """Baixa clima para um municipio com semaphore para rate limiting."""
    async with semaphore:
        cod_ibge = row["cod_ibge"]
        df = await download_climate_for_municipality(cod_ibge, row["lat"], row["lon"], config)
        progress["done"] += 1
        if df is not None:
            save_to_cache(df, cod_ibge, CACHE_DIR)
            progress["success"] += 1
            logger.info(
                f"  [{progress['done']}/{progress['total']}] {cod_ibge}: OK ({len(df)} dias)"
            )
        else:
            progress["failed"] += 1
            logger.warning(f"  [{progress['done']}/{progress['total']}] {cod_ibge}: FALHOU")


async def _fetch_all(pending_rows: list[dict], config: dict, semaphore: asyncio.Semaphore):
    """Baixa clima para todos os municipios pendentes com concorrencia real."""
    progress = {"done": 0, "success": 0, "failed": 0, "total": len(pending_rows)}
    tasks = [_fetch_one(row, config, semaphore, progress) for row in pending_rows]
    await asyncio.gather(*tasks)
    return progress["success"], progress["failed"]


def fetch_climate_for_municipalities(
    only_soy_producers: bool = True, max_municipalities: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Pipeline principal de ingestao de clima."""
    logger.info("=" * 60)
    logger.info("INGESTAO CLIMA - NASA POWER (via agrobr)")
    logger.info("=" * 60)

    config = load_config("climate", section="climate")
    logger.info(f"Periodo: {config['year_start']} - {config['year_end']}")

    df_mun = load_municipalities()
    logger.info(f"Municipios disponiveis: {len(df_mun):,}")

    if only_soy_producers:
        soy_producers = load_target_municipalities()
        df_mun = df_mun[df_mun["cod_ibge"].isin(soy_producers)]
        logger.info(f"Municipios produtores de soja: {len(df_mun):,}")

    if max_municipalities is not None:
        df_mun = df_mun.head(max_municipalities)

    cached = get_cached_codes(CACHE_DIR)
    logger.info(f"Municipios ja em cache: {len(cached):,}")

    pending = df_mun[~df_mun["cod_ibge"].isin(cached)]
    logger.info(f"Municipios pendentes: {len(pending):,}")

    if len(pending) > 0:
        requests_per_minute = config.get("rate_limit", {}).get("requests_per_minute", 30)
        semaphore = asyncio.Semaphore(requests_per_minute)
        pending_rows = pending[["cod_ibge", "lat", "lon"]].to_dict("records")
        success, failed = asyncio.run(_fetch_all(pending_rows, config, semaphore))
        logger.info(f"Download: {success} sucesso, {failed} falha")

    logger.info("Consolidando cache...")
    df_consolidated = consolidate_cache(CACHE_DIR)
    df_consolidated = df_consolidated.sort_values(["cod_ibge", "date"]).reset_index(drop=True)

    output_cols = [
        "cod_ibge",
        "date",
        "tmin",
        "tmean",
        "tmax",
        "precip",
        "rh",
        "radiation",
        "wind_speed",
    ]
    output_cols = [c for c in output_cols if c in df_consolidated.columns]
    df_consolidated = df_consolidated[output_cols]

    stats = calculate_statistics(df_consolidated)
    logger.info(
        f"Total: {stats['total_registros']:,} registros, {stats['municipios_unicos']:,} municipios"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_consolidated.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df_consolidated, stats


def main(only_soy_producers: bool = True, max_municipalities: int | None = None):
    """Pipeline principal de ingestao de clima."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return fetch_climate_for_municipalities(only_soy_producers, max_municipalities)


if __name__ == "__main__":
    main()
