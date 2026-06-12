"""Gera previsoes para a safra alvo a partir dos dados e modelos ja existentes.

Nao faz ingest nem retrain: constroi as features de inferencia e aplica os
modelos regionais (fallback: modelo global). Safras em andamento recebem flag
de parcialidade (clima_completo / enso_disponivel) no parquet de saida.

Uso:
    python -m scripts.update_pipeline                    # safra corrente
    python -m scripts.update_pipeline --safra 2025/26    # safra especifica
"""

from __future__ import annotations

import argparse
import calendar
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.common.io import PROJECT_ROOT, load_config, load_municipalities
from src.common.phenology import get_default_phenology, get_regional_phenology
from src.features.build_features import load_climate_data, load_ndvi_data
from src.inference.predict import (
    build_inference_features,
    generate_predictions_for,
    load_inference_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_area(df_target: pd.DataFrame, year: int) -> pd.DataFrame:
    """Estima area colhida para anos futuros usando tendencia linear por municipio."""
    max_year = df_target["ano"].max()
    recent = df_target[df_target["ano"] >= max_year - 4].copy()

    results = []
    for cod_ibge, group in recent.groupby("cod_ibge"):
        y = group["area_colhida_ha"].dropna().values
        x = group.loc[group["area_colhida_ha"].notna(), "ano"].values.astype(float)
        if len(x) < 2:
            area = y[-1] if len(y) > 0 else 0
        else:
            coef = np.polyfit(x, y, 1)
            area = max(0, np.polyval(coef, year))
        results.append({"cod_ibge": cod_ibge, "area_estimada_ha": area})

    return pd.DataFrame(results)


def add_completeness_flags(
    dp: pd.DataFrame,
    df_climate: pd.DataFrame,
    features_config: dict,
    years_to_predict: list[int],
) -> pd.DataFrame:
    """Marca previsoes de safras com janela climatica incompleta ou ENSO ausente.

    Fases fenologicas ausentes entram como 0 mm na agregacao (nao NaN), entao
    uma safra em andamento gera anomalias de seca ficticias — a flag e a unica
    forma de distinguir previsao validavel de extrapolacao parcial.
    """
    dp = dp.copy()

    end_months = [get_default_phenology(features_config)["end_month"]] + [
        cal.get("end_month", 4) for cal in get_regional_phenology(features_config).values()
    ]
    end_month = max(end_months)

    clima_max = df_climate["date"].max()

    dp["clima_completo"] = True
    dp["enso_disponivel"] = dp["oni_avg"].notna() if "oni_avg" in dp.columns else False

    for year in years_to_predict:
        window_end = pd.Timestamp(year, end_month, calendar.monthrange(year, end_month)[1])
        if clima_max < window_end:
            dp.loc[dp["ano"] == year, "clima_completo"] = False
            logger.warning(
                f"  [!] Safra {year - 1}/{str(year)[2:]}: clima disponivel ate "
                f"{clima_max.date()}, janela vai ate {window_end.date()} — "
                f"PREVISAO PARCIAL, fases ausentes tratadas como sem chuva/calor"
            )

        n_sem_enso = int((~dp.loc[dp["ano"] == year, "enso_disponivel"]).sum())
        if n_sem_enso > 0:
            logger.warning(
                f"  [!] Safra {year - 1}/{str(year)[2:]}: {n_sem_enso:,} linhas sem ONI "
                f"(ENSO nao cobre o ano) — features ENSO entram como NaN"
            )

    return dp


def print_report(dp: pd.DataFrame, df_target: pd.DataFrame, years: list[int]):
    """Imprime relatorio de previsao com estimativa de area."""
    df_mun = load_municipalities(columns=["cod_ibge", "nome", "uf"])
    dp = dp.merge(df_mun, on="cod_ibge", how="left")

    for year in years:
        dy = dp[dp["ano"] == year]
        if len(dy) == 0:
            logger.info(f"\n{year}: SEM DADOS SUFICIENTES")
            continue

        df_area = estimate_area(df_target, year)
        dy = dy.merge(df_area, on="cod_ibge", how="left")
        dy["prod_ton"] = dy["pred_kg_ha"] * dy["area_estimada_ha"].fillna(0) / 1000
        prod_mt = dy["prod_ton"].sum() / 1e6

        safra = f"{year - 1}/{str(year)[2:]}"
        dy["uf_cod"] = dy["cod_ibge"].astype(str).str[:2].astype(int)
        dy["regiao"] = np.where(dy["uf_cod"].isin([41, 42, 43]), "Sul", "Cerrado/Outros")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"SAFRA {safra}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Municipios: {len(dy):,}")
        logger.info(f"Area estimada: {dy['area_estimada_ha'].sum() / 1e6:.1f} Mha")
        logger.info(
            f"Produtividade media: {dy['pred_kg_ha'].mean():.0f} kg/ha "
            f"({dy['pred_sc_ha'].mean():.1f} sc/ha)"
        )
        logger.info(f"PRODUCAO TOTAL: {prod_mt:.1f} milhoes de toneladas")

        logger.info("\nPor regiao:")
        for reg in ["Cerrado/Outros", "Sul"]:
            dr = dy[dy["regiao"] == reg]
            if len(dr) == 0:
                continue
            pr = dr["prod_ton"].sum() / 1e6
            logger.info(
                f"  {reg}: {dr['pred_kg_ha'].mean():.0f} kg/ha | "
                f"{dr['area_estimada_ha'].sum() / 1e6:.1f} Mha | {pr:.1f} Mt | {len(dr)} mun"
            )

        logger.info("\nPor UF (top 10):")
        for uf_name in (
            dy.groupby("uf")["prod_ton"].sum().sort_values(ascending=False).head(10).index
        ):
            du = dy[dy["uf"] == uf_name]
            pr = du["prod_ton"].sum() / 1e6
            logger.info(
                f"  {uf_name}: {du['pred_kg_ha'].mean():.0f} kg/ha ({du['pred_sc_ha'].mean():.1f} sc) | "
                f"{du['area_estimada_ha'].sum() / 1e6:.2f} Mha | {pr:.1f} Mt"
            )

    out_path = PROJECT_ROOT / "results" / f"previsao_safra_{years[0]}_{years[-1]}.parquet"
    cols = [
        c
        for c in [
            "cod_ibge",
            "ano",
            "pred_kg_ha",
            "pred_sc_ha",
            "pred_lower_80_kg_ha",
            "pred_upper_80_kg_ha",
            "clima_completo",
            "enso_disponivel",
        ]
        if c in dp.columns
    ]
    dp[cols].to_parquet(out_path, index=False)
    logger.info(f"\nPrevisoes salvas em: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de previsao de soja")
    parser.add_argument("--safra", type=str, help="Safra alvo (ex: 2025/26)", default=None)
    parser.add_argument("--model-version", type=str, default="v2", help="Versao do modelo")
    args = parser.parse_args()

    if args.safra:
        parts = args.safra.split("/")
        year_end = int(parts[0]) + 1 if len(parts[0]) == 4 else int("20" + parts[1])
    else:
        today = date.today()
        year_end = today.year + 1 if today.month >= 10 else today.year

    years_to_predict = [year_end - 1, year_end]
    logger.info(f"Safras alvo: {[f'{y - 1}/{str(y)[2:]}' for y in years_to_predict]}")

    model_info = load_inference_models()
    feat_names = model_info["feature_names"]
    logger.info(f"Modelo: {model_info['model_version']} ({len(feat_names)} features)")

    fc = load_config("features")
    df_climate = load_climate_data()

    df_target = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "target_soja.parquet")
    df_enso = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet")
    df_mun = load_municipalities(columns=["cod_ibge", "lat"])
    lat_lookup = dict(zip(df_mun["cod_ibge"], df_mun["lat"]))

    municipalities = df_target[df_target["ano"] >= 2018]["cod_ibge"].unique().tolist()
    logger.info(f"Municipios produtores (>=1 ano desde 2018): {len(municipalities):,}")

    dp = build_inference_features(
        df_climate,
        df_target,
        df_enso,
        load_ndvi_data(),
        municipalities,
        years_to_predict,
        fc,
        lat_lookup,
    )

    missing_feats = [f for f in feat_names if f not in dp.columns]
    if missing_feats:
        raise ValueError(f"Features exigidas pelo modelo ausentes na inferencia: {missing_feats}")

    dp = generate_predictions_for(model_info, dp)

    dp = dp.rename(
        columns={
            "pred_produtividade_kg_ha": "pred_kg_ha",
            "pred_produtividade_sacas_ha": "pred_sc_ha",
        }
    )

    dp = add_completeness_flags(dp, df_climate, fc, years_to_predict)

    print_report(dp, df_target, years_to_predict)


if __name__ == "__main__":
    main()
