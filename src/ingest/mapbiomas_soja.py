"""Ingestao de dados MapBiomas uso do solo (soja) via agrobr."""

import asyncio
import logging
import unicodedata

import pandas as pd

from src.common.io import PROJECT_ROOT, load_municipalities

logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mapbiomas_soja.parquet"

CLASSE_SOJA = 39


def _normalize_name(name: str) -> str:
    """Normaliza nome de municipio para matching (lowercase, sem acentos)."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def _build_name_lookup(df_mun: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Constroi lookup (uf, nome_normalizado) -> cod_ibge."""
    lookup = {}
    for _, row in df_mun.iterrows():
        key = (str(row.get("uf", "")).upper(), _normalize_name(row["nome"]))
        lookup[key] = row["cod_ibge"]
    return lookup


async def fetch_mapbiomas_soja() -> pd.DataFrame:
    """Busca dados de cobertura de soja do MapBiomas."""
    from agrobr.mapbiomas import cobertura

    logger.info("Buscando MapBiomas soja (classe 39) nivel municipal...")
    logger.info("  ATENCAO: download pode ser grande (~660MB)")

    df = await cobertura(nivel="municipio", classe_id=CLASSE_SOJA)
    logger.info(f"  Registros brutos: {len(df):,}")

    return df


def match_municipalities(df_mapbiomas: pd.DataFrame, df_mun: pd.DataFrame) -> pd.DataFrame:
    """Faz matching de nomes de municipios MapBiomas -> cod_ibge."""
    logger.info("Fazendo matching de municipios...")

    lookup = _build_name_lookup(df_mun)

    df = df_mapbiomas.copy()
    df["_uf"] = df["estado"].str.upper()
    df["_nome_norm"] = df["municipio"].apply(_normalize_name)

    df["cod_ibge"] = df.apply(lambda row: lookup.get((row["_uf"], row["_nome_norm"])), axis=1)

    n_matched = df["cod_ibge"].notna().sum()
    n_total = len(df)
    logger.info(f"  Matched: {n_matched:,}/{n_total:,} ({100 * n_matched / n_total:.1f}%)")

    unmatched = df[df["cod_ibge"].isna()]["municipio"].unique()
    if len(unmatched) > 0:
        logger.warning(f"  Municipios nao encontrados: {len(unmatched)}")
        for name in unmatched[:10]:
            logger.warning(f"    - {name}")

    df = df.dropna(subset=["cod_ibge"])
    df["cod_ibge"] = df["cod_ibge"].astype("int64")
    df["ano"] = df["ano"].astype(int)

    df_result = df.groupby(["cod_ibge", "ano"]).agg(area_soja_ha=("area_ha", "sum")).reset_index()

    return df_result


def main():
    """Pipeline de ingestao MapBiomas soja."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("INGESTAO MAPBIOMAS - USO DO SOLO SOJA (via agrobr)")
    logger.info("=" * 60)

    df_raw = asyncio.run(fetch_mapbiomas_soja())

    df_mun = load_municipalities(columns=["cod_ibge", "nome", "uf"])
    df = match_municipalities(df_raw, df_mun)

    logger.info(f"Registros finais: {len(df):,}")
    logger.info(f"Municipios: {df['cod_ibge'].nunique():,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Arquivo salvo: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()
