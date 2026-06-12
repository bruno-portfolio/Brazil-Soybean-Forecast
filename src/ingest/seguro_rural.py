"""Ingestao de dados de seguro rural (MAPA PSR) via agrobr."""

import asyncio
import logging

import pandas as pd

from src.common.io import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "seguro_rural.parquet"


async def fetch_seguro_rural(year_start: int, year_end: int) -> pd.DataFrame:
    """Busca dados de sinistros e apolices de soja via agrobr."""
    from agrobr.alt.mapa_psr.api import apolices, sinistros

    logger.info(f"Buscando PSR soja {year_start}-{year_end}...")

    df_sin = await sinistros(cultura="soja", ano_inicio=year_start, ano_fim=year_end)
    logger.info(f"  Sinistros: {len(df_sin):,}")

    df_apol = await apolices(cultura="soja", ano_inicio=year_start, ano_fim=year_end)
    logger.info(f"  Apolices: {len(df_apol):,}")

    sin_agg = (
        df_sin.groupby(["cd_ibge", "ano_apolice"])
        .agg(n_sinistros=("nr_apolice", "count"))
        .reset_index()
    )

    apol_agg = (
        df_apol.groupby(["cd_ibge", "ano_apolice"])
        .agg(n_apolices=("nr_apolice", "count"))
        .reset_index()
    )

    merged = apol_agg.merge(sin_agg, on=["cd_ibge", "ano_apolice"], how="left")
    merged["sinistro_rate"] = merged["n_sinistros"].fillna(0) / merged["n_apolices"]

    merged = merged.sort_values(["cd_ibge", "ano_apolice"])
    merged["sinistro_rate_3yr"] = merged.groupby("cd_ibge")["sinistro_rate"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    merged = merged.rename(columns={"cd_ibge": "cod_ibge", "ano_apolice": "ano"})
    merged["cod_ibge"] = pd.to_numeric(merged["cod_ibge"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["cod_ibge"])
    merged["cod_ibge"] = merged["cod_ibge"].astype("int64")
    merged["ano"] = merged["ano"].astype(int)

    logger.info(f"  Registros finais: {len(merged):,}")
    logger.info(f"  Municipios: {merged['cod_ibge'].nunique():,}")

    return merged[["cod_ibge", "ano", "sinistro_rate", "sinistro_rate_3yr"]]


def main():
    """Pipeline de ingestao de seguro rural."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO SEGURO RURAL (MAPA PSR via agrobr)")
    logger.info("=" * 60)

    config = load_config("target", section="target")
    # PSR disponivel a partir de 2006
    year_start = max(config["year_start"], 2006)
    df = asyncio.run(fetch_seguro_rural(year_start, config["year_end"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
