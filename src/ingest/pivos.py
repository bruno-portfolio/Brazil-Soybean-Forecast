"""Ingestao de dados de pivos de irrigacao (ANA) via agrobr."""

import asyncio
import logging

import pandas as pd

from src.common.io import PROJECT_ROOT

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pivos_irrigacao.parquet"


async def fetch_pivos() -> pd.DataFrame:
    """Busca dados de pivos de irrigacao via agrobr."""
    from agrobr.ana import pivos_irrigacao

    logger.info("Buscando pivos de irrigacao via agrobr (ANA)...")
    df = await pivos_irrigacao()
    logger.info(f"  Registros brutos: {len(df):,}")

    df_agg = (
        df.groupby("codigo_municipio")
        .agg(area_irrigada_ha=("area_ha", "sum"), n_pivos=("area_ha", "count"))
        .reset_index()
    )
    df_agg = df_agg.rename(columns={"codigo_municipio": "cod_ibge"})
    df_agg["cod_ibge"] = pd.to_numeric(df_agg["cod_ibge"], errors="coerce").astype("Int64")
    df_agg = df_agg.dropna(subset=["cod_ibge"])
    df_agg["cod_ibge"] = df_agg["cod_ibge"].astype("int64")

    logger.info(f"  Municipios com pivos: {len(df_agg):,}")
    logger.info(f"  Area irrigada total: {df_agg['area_irrigada_ha'].sum():,.0f} ha")

    return df_agg


def main():
    """Pipeline de ingestao de pivos."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO PIVOS DE IRRIGACAO (ANA via agrobr)")
    logger.info("=" * 60)

    df = asyncio.run(fetch_pivos())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
