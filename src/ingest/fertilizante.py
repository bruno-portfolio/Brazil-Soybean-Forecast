"""Ingestao de dados de fertilizante.

Fontes (em ordem de preferencia):
1. ComexStat (importacao por UF, mensal, livre) — via agrobr.comexstat
2. ANDA (entregas Brasil, mensal, zona_cinza) — via agrobr.anda

ComexStat capta ~85% do consumo BR (importacao). UF = destino na declaracao.
"""

import asyncio
import logging

import pandas as pd

from src.common.io import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fertilizante_uf.parquet"

# NCMs cap. 31 — fertilizantes
NCMS_FERTILIZANTE = [
    "31042090",  # KCl
    "31021010",  # Ureia
    "31022100",  # Sulfato de amonio
    "31054000",  # MAP
    "31055900",  # NPK complexos
]

# Mapeamento UF sigla -> codigo IBGE
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


async def fetch_comexstat(year_start: int, year_end: int) -> pd.DataFrame | None:
    """Busca importacao de fertilizantes via ComexStat (preferencial)."""
    try:
        from agrobr.comexstat import importacao

        logger.info(f"Buscando ComexStat fertilizantes {year_start}-{year_end}...")
        all_dfs = []

        for ano in range(year_start, year_end + 1):
            try:
                df = await importacao(ano=ano, ncm=NCMS_FERTILIZANTE, agregacao="uf")
                all_dfs.append(df)
                logger.info(f"  {ano}: {len(df):,} registros")
            except Exception as e:
                logger.warning(f"  {ano}: falhou - {e}")

        if not all_dfs:
            return None

        df = pd.concat(all_dfs, ignore_index=True)

        # Agregar: total anual por UF (peso em ton)
        peso_col = "peso_kg" if "peso_kg" in df.columns else "volume_ton"
        divisor = 1000 if peso_col == "peso_kg" else 1

        uf_col = "uf" if "uf" in df.columns else "estado"

        df_agg = (
            df.groupby([uf_col, "ano"])
            .agg(fert_import_ton=(peso_col, lambda x: x.sum() / divisor))
            .reset_index()
        )

        df_agg["uf_cod"] = df_agg[uf_col].map(UF_TO_CODE)
        df_agg = df_agg.dropna(subset=["uf_cod"])
        df_agg["uf_cod"] = df_agg["uf_cod"].astype(int)
        df_agg["ano"] = df_agg["ano"].astype(int)

        logger.info(f"  ComexStat: {len(df_agg):,} registros UF-ano")
        return df_agg[["uf_cod", "ano", "fert_import_ton"]]

    except ImportError:
        logger.info("  agrobr.comexstat nao disponivel, tentando ANDA...")
        return None


async def fetch_anda(year_start: int, year_end: int) -> pd.DataFrame | None:
    """Fallback: entregas ANDA (nivel Brasil apenas)."""
    try:
        from agrobr.anda import entregas

        logger.info(f"Buscando ANDA {year_start}-{year_end} (nivel Brasil)...")
        all_dfs = []

        for ano in range(year_start, year_end + 1):
            try:
                df = await entregas(ano, produto="total")
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"  {ano}: falhou - {e}")

        if not all_dfs:
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        df_agg = df.groupby("ano").agg(fert_total_br_ton=("volume_ton", "sum")).reset_index()
        df_agg["ano"] = df_agg["ano"].astype(int)

        logger.info(f"  ANDA: {len(df_agg):,} anos (nivel Brasil)")
        return df_agg[["ano", "fert_total_br_ton"]]

    except ImportError:
        logger.warning("  agrobr.anda nao disponivel")
        return None


def main():
    """Pipeline de ingestao de fertilizante."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO FERTILIZANTE")
    logger.info("=" * 60)

    config = load_config("target", section="target")
    year_start = config["year_start"]
    year_end = config["year_end"]

    # Tentar ComexStat primeiro (por UF), depois ANDA (Brasil)
    df = asyncio.run(fetch_comexstat(year_start, year_end))

    if df is None:
        df = asyncio.run(fetch_anda(year_start, year_end))

    if df is None:
        logger.warning("Nenhuma fonte de fertilizante disponivel")
        df = pd.DataFrame()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
