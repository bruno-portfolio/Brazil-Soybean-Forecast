"""Ingestao de dados de entregas de fertilizante (ANDA) via agrobr."""

import asyncio
import logging

import pandas as pd

from src.common.io import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fertilizante_uf.parquet"

# Mapeamento UF sigla -> codigo IBGE (2 digitos)
UF_TO_CODE = {
    "AC": 12,
    "AL": 27,
    "AM": 13,
    "AP": 16,
    "BA": 29,
    "CE": 23,
    "DF": 53,
    "ES": 32,
    "GO": 52,
    "MA": 21,
    "MG": 31,
    "MS": 50,
    "MT": 51,
    "PA": 15,
    "PB": 25,
    "PE": 26,
    "PI": 22,
    "PR": 41,
    "RJ": 33,
    "RN": 24,
    "RO": 11,
    "RR": 14,
    "RS": 43,
    "SC": 42,
    "SE": 28,
    "SP": 35,
    "TO": 17,
}


async def fetch_fertilizante(year_start: int, year_end: int) -> pd.DataFrame:
    """Busca dados de entregas de fertilizante via agrobr."""
    from agrobr.anda import entregas

    logger.info(f"Buscando entregas ANDA {year_start}-{year_end}...")
    all_dfs = []

    for ano in range(year_start, year_end + 1):
        try:
            df = await entregas(ano, produto="total")
            all_dfs.append(df)
            logger.info(f"  {ano}: {len(df):,} registros")
        except Exception as e:
            logger.warning(f"  {ano}: falhou - {e}")

    if not all_dfs:
        logger.warning("Nenhum dado ANDA obtido")
        return pd.DataFrame(columns=["ano", "fert_total_br_ton"])

    df = pd.concat(all_dfs, ignore_index=True)

    # ANDA PDFs so tem dados nivel Brasil, entao agregamos por ano
    df_agg = df.groupby("ano").agg(fert_total_br_ton=("volume_ton", "sum")).reset_index()
    df_agg["ano"] = df_agg["ano"].astype(int)

    logger.info(f"  Total: {len(df_agg):,} anos")
    logger.info(f"  Volume medio: {df_agg['fert_total_br_ton'].mean():,.0f} ton/ano")

    return df_agg[["ano", "fert_total_br_ton"]]


def main():
    """Pipeline de ingestao de fertilizante."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO FERTILIZANTE (ANDA via agrobr)")
    logger.info("=" * 60)

    config = load_config("target", section="target")
    df = asyncio.run(fetch_fertilizante(config["year_start"], config["year_end"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
