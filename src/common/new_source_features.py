"""Feature engineering para fontes de dados novas (Phase 2+).

Cada funcao carrega o parquet correspondente e faz merge no dataset.
Se o arquivo nao existir, retorna o df inalterado (graceful degradation).
"""

import logging

import pandas as pd

from src.common.io import PROJECT_ROOT

logger = logging.getLogger(__name__)

IRRIGACAO_PATH = PROJECT_ROOT / "data" / "processed" / "pivos_irrigacao.parquet"
FERT_PATH = PROJECT_ROOT / "data" / "processed" / "fertilizante_uf.parquet"
SINISTRO_PATH = PROJECT_ROOT / "data" / "processed" / "seguro_rural.parquet"
MAPBIOMAS_PATH = PROJECT_ROOT / "data" / "processed" / "mapbiomas_soja.parquet"


def add_irrigacao_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona pct_irrigado (fracao de area com pivos no municipio)."""
    if not IRRIGACAO_PATH.exists():
        logger.info("pivos_irrigacao.parquet nao encontrado, pulando irrigacao")
        return df

    logger.info("Adicionando features de irrigacao...")
    df_irr = pd.read_parquet(IRRIGACAO_PATH)

    df = df.merge(df_irr[["cod_ibge", "area_irrigada_ha"]], on="cod_ibge", how="left")

    if "area_colhida_ha" in df.columns:
        df["pct_irrigado"] = (
            df["area_irrigada_ha"].fillna(0) / df["area_colhida_ha"].clip(lower=1)
        ).clip(0, 1)
    else:
        df["pct_irrigado"] = 0.0

    df = df.drop(columns=["area_irrigada_ha"], errors="ignore")

    n_with = (df["pct_irrigado"] > 0).sum()
    logger.info(f"  Municipios com irrigacao: {n_with:,}")

    return df


def add_fertilizante_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona feature de fertilizante (ComexStat por UF ou ANDA Brasil)."""
    if not FERT_PATH.exists():
        logger.info("fertilizante_uf.parquet nao encontrado, pulando fertilizante")
        return df

    logger.info("Adicionando features de fertilizante...")
    df_fert = pd.read_parquet(FERT_PATH)

    if "fert_import_ton" in df_fert.columns and "uf_cod" in df_fert.columns:
        # ComexStat: importacao por UF (melhor granularidade)
        df["_uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)
        df = df.merge(df_fert, left_on=["_uf_cod", "ano"], right_on=["uf_cod", "ano"], how="left")
        df = df.drop(columns=["_uf_cod", "uf_cod"], errors="ignore")
        # Normalizar para milhoes de ton
        df["fert_import_ton"] = df["fert_import_ton"].fillna(0) / 1e6
        n_with = (df["fert_import_ton"] > 0).sum()
        logger.info(f"  ComexStat: {n_with:,} registros com dados")
    elif "fert_total_br_ton" in df_fert.columns:
        # ANDA: nivel Brasil (fallback)
        df = df.merge(df_fert[["ano", "fert_total_br_ton"]], on="ano", how="left")
        df["fert_total_br_ton"] = df["fert_total_br_ton"].fillna(0) / 1e6
        n_with = (df["fert_total_br_ton"] > 0).sum()
        logger.info(f"  ANDA (Brasil): {n_with:,} registros com dados")

    return df


def add_sinistro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona sinistro_rate_3yr (taxa de sinistro media 3 anos)."""
    if not SINISTRO_PATH.exists():
        logger.info("seguro_rural.parquet nao encontrado, pulando sinistro")
        return df

    logger.info("Adicionando features de sinistro...")
    df_sin = pd.read_parquet(SINISTRO_PATH)

    df = df.merge(
        df_sin[["cod_ibge", "ano", "sinistro_rate_3yr"]], on=["cod_ibge", "ano"], how="left"
    )
    df["sinistro_rate_3yr"] = df["sinistro_rate_3yr"].fillna(0.0)

    n_with = (df["sinistro_rate_3yr"] > 0).sum()
    logger.info(f"  Registros com sinistro: {n_with:,}")

    return df


def add_uso_solo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona pct_soja (fracao da area municipal com soja, MapBiomas)."""
    if not MAPBIOMAS_PATH.exists():
        logger.info("mapbiomas_soja.parquet nao encontrado, pulando uso do solo")
        return df

    logger.info("Adicionando features MapBiomas...")
    df_mb = pd.read_parquet(MAPBIOMAS_PATH)

    df = df.merge(df_mb[["cod_ibge", "ano", "area_soja_ha"]], on=["cod_ibge", "ano"], how="left")

    if "area_colhida_ha" in df.columns:
        # Ratio > 1 e possivel: MapBiomas mede area total (incl. safrinha),
        # IBGE mede area colhida da safra principal. Clip em 10 para outliers.
        df["pct_soja"] = (df["area_soja_ha"].fillna(0) / df["area_colhida_ha"].clip(lower=1)).clip(
            0, 10
        )
    else:
        df["pct_soja"] = df["area_soja_ha"]

    df = df.drop(columns=["area_soja_ha"], errors="ignore")

    n_with = df["pct_soja"].notna().sum()
    logger.info(f"  Registros com MapBiomas: {n_with:,}")

    return df


def add_new_source_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes entre fontes novas e features existentes."""
    logger.info("Adicionando interacoes de fontes novas...")
    n_added = 0

    if "pct_irrigado" in df.columns and "water_deficit_mm" in df.columns:
        deficit_norm = df["water_deficit_mm"] / (df["water_deficit_mm"].std() + 1e-8)
        df["irrigacao_x_deficit"] = df["pct_irrigado"] * deficit_norm
        n_added += 1

    fert_col = "fert_import_ton" if "fert_import_ton" in df.columns else "fert_total_br_ton"
    if fert_col in df.columns and "precip_anomaly" in df.columns:
        fert_norm = df[fert_col].fillna(0) / (df[fert_col].std() + 1e-8)
        df["fert_x_precip"] = fert_norm * df["precip_anomaly"].fillna(0)
        n_added += 1

    if "sinistro_rate_3yr" in df.columns and "is_la_nina" in df.columns:
        df["sinistro_x_la_nina"] = df["sinistro_rate_3yr"] * df["is_la_nina"]
        n_added += 1

    logger.info(f"  Interacoes adicionadas: {n_added}")

    return df
