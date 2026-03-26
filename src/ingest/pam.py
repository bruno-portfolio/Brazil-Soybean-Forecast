"""Ingestao PAM (Producao Agricola Municipal) via agrobr."""

import asyncio
import logging

import pandas as pd

from src.common.io import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"

COLUMN_RENAME = {
    "localidade_cod": "cod_ibge",
    "rendimento": "produtividade_kg_ha",
    "area_colhida": "area_colhida_ha",
    "producao": "producao_ton",
}

OUTLIER_THRESHOLD = 10000  # kg/ha


async def fetch_pam_data(config: dict) -> pd.DataFrame:
    """Busca dados PAM de soja via agrobr."""
    from agrobr.ibge import pam as agrobr_pam

    logger.info(f"Buscando PAM soja {config['year_start']}-{config['year_end']} via agrobr...")
    df = await agrobr_pam(
        "soja",
        ano=list(range(config["year_start"], config["year_end"] + 1)),
        nivel="municipio",
        variaveis=["rendimento", "area_colhida", "producao"],
    )
    logger.info(f"  Registros brutos: {len(df):,}")
    return df


def process_pam_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processa dados brutos da PAM para formato padronizado."""
    logger.info("Processando dados da PAM...")

    df = df.rename(columns=COLUMN_RENAME)
    df["cod_ibge"] = pd.to_numeric(df["cod_ibge"], errors="coerce").astype("Int64")
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int32")

    df = df.dropna(subset=["cod_ibge", "ano"])
    df["cod_ibge"] = df["cod_ibge"].astype("int64")
    df["ano"] = df["ano"].astype("int32")

    if "area_colhida_ha" in df.columns:
        n_before = len(df)
        df = df[df["area_colhida_ha"] > 0].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"  Removidos {n_removed} registros com area_colhida = 0")

    n_before = len(df)
    df = df.dropna(subset=["produtividade_kg_ha"])
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"  Removidos {n_removed} registros com produtividade nula")

    n_before = len(df)
    outliers = df[df["produtividade_kg_ha"] > OUTLIER_THRESHOLD]
    if len(outliers) > 0:
        logger.warning(f"  Outliers extremos (> {OUTLIER_THRESHOLD} kg/ha): {len(outliers)}")
        df = df[df["produtividade_kg_ha"] <= OUTLIER_THRESHOLD].copy()
        logger.info(f"  Removidos {n_before - len(df)} outliers")

    cols = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]
    cols = [c for c in cols if c in df.columns]

    df = df[cols].sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    logger.info(f"  Registros processados: {len(df):,}")
    logger.info(f"  Municipios unicos: {df['cod_ibge'].nunique():,}")
    logger.info(f"  Anos: {df['ano'].min()} - {df['ano'].max()}")

    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset para documentacao."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "anos": {
            "min": int(df["ano"].min()),
            "max": int(df["ano"].max()),
            "total": df["ano"].nunique(),
        },
        "produtividade_kg_ha": {
            "min": float(df["produtividade_kg_ha"].min()),
            "max": float(df["produtividade_kg_ha"].max()),
            "mean": float(df["produtividade_kg_ha"].mean()),
            "median": float(df["produtividade_kg_ha"].median()),
        },
    }
    return stats


def main():
    """Pipeline principal de ingestao da PAM."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO PAM - PRODUTIVIDADE DE SOJA (via agrobr)")
    logger.info("=" * 60)

    config = load_config("target", section="target")

    df_raw = asyncio.run(fetch_pam_data(config))
    df = process_pam_data(df_raw)

    stats = calculate_statistics(df)
    logger.info(f"\nTotal: {stats['total_registros']:,} registros")
    logger.info(f"Produtividade media: {stats['produtividade_kg_ha']['mean']:.1f} kg/ha")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
