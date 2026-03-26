"""Pipeline completo: atualiza dados, rebuild features, retrain e gera previsoes.

Uso:
    python -m scripts.update_pipeline                    # safra corrente
    python -m scripts.update_pipeline --safra 2025/26    # safra especifica
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import date

import numpy as np
import pandas as pd

from src.common.climate_aggregation import aggregate_climate_duckdb
from src.common.features import (
    add_enso_features,
    add_enso_interactions,
    add_regional_features,
    add_soil_climate_interactions,
    add_soil_features,
    calculate_climate_anomalies,
)
from src.common.io import PROJECT_ROOT, load_config, load_municipalities
from src.common.new_source_features import (
    add_fertilizante_features,
    add_irrigacao_features,
    add_new_source_interactions,
    add_sinistro_features,
    add_uso_solo_features,
)
from src.common.phenology import (
    filter_phenology_window_regional,
    get_default_phenology,
    get_regional_phenology,
)
from src.inference.predict import (
    _fill_missing_anomalies,
    calculate_historical_features,
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


def build_prediction_features(
    df_climate: pd.DataFrame,
    df_target: pd.DataFrame,
    df_enso: pd.DataFrame,
    years_to_predict: list[int],
    features_config: dict,
    lat_lookup: dict[int, float],
) -> pd.DataFrame:
    """Constroi features para predicao. Usa mesma logica de predict.py main()."""
    fc = features_config
    bt, ht = 10.0, 32.0
    for feat in fc["features"]["climate_features"]:
        if feat["name"] == "gdd_accumulated":
            bt = feat.get("base_temp", 10.0)
        if feat["name"] == "hot_days_count":
            ht = feat.get("threshold", 32.0)
    trm = fc.get("features", {}).get("trend_ref_year_min", 2000)
    trx = fc.get("features", {}).get("trend_ref_year_max", 2025)

    # Municipios produtores (min 1 ano desde 2018 — mais inclusivo que predict.py)
    municipalities = df_target[df_target["ano"] >= 2018]["cod_ibge"].unique().tolist()
    logger.info(f"Municipios produtores (>=1 ano desde 2018): {len(municipalities):,}")

    # Agregar clima
    hist_start = min(years_to_predict) - 6
    all_years = list(range(hist_start, min(years_to_predict))) + years_to_predict

    df_f = df_climate[df_climate["cod_ibge"].isin(municipalities)].copy()
    df_f = filter_phenology_window_regional(
        df_f, get_regional_phenology(fc), get_default_phenology(fc)
    )
    df_f = df_f[df_f["crop_year"].isin(all_years)]

    df_all = aggregate_climate_duckdb(df_f, bt, ht, lat_lookup=lat_lookup)
    df_all = add_enso_features(df_all, df_enso)
    df_all = calculate_climate_anomalies(df_all, min_years=5)

    dp = df_all[df_all["ano"].isin(years_to_predict)].copy()

    # Reutiliza calculate_historical_features de predict.py
    dp = calculate_historical_features(dp, df_target, years_to_predict, trm, trx)

    # Reutiliza _fill_missing_anomalies de predict.py
    _fill_missing_anomalies(dp)

    dp = add_regional_features(dp)
    dp = add_enso_interactions(dp)

    soil_path = PROJECT_ROOT / "data" / "processed" / "soil_properties.parquet"
    if soil_path.exists():
        dp = add_soil_features(dp, pd.read_parquet(soil_path))
        dp = add_soil_climate_interactions(dp)

    dp = add_irrigacao_features(dp)
    dp = add_fertilizante_features(dp)
    dp = add_sinistro_features(dp)
    dp = add_uso_solo_features(dp)
    dp = add_new_source_interactions(dp)

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
    cols = [c for c in ["cod_ibge", "ano", "pred_kg_ha", "pred_sc_ha"] if c in dp.columns]
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

    model_path = PROJECT_ROOT / "models" / f"model_{args.model_version}.pkl"
    meta_path = PROJECT_ROOT / "models" / f"model_{args.model_version}_metadata.json"
    if not model_path.exists():
        logger.error(f"Modelo nao encontrado: {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]
    logger.info(f"Modelo: {args.model_version} ({len(feat_names)} features)")

    fc = load_config("features")
    v2p = PROJECT_ROOT / "data" / "processed" / "climate_daily_v2.parquet"
    cp = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
    df_climate = pd.read_parquet(v2p if v2p.exists() else cp)
    df_climate["date"] = pd.to_datetime(df_climate["date"])

    df_target = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "target_soja.parquet")
    df_enso = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet")
    df_mun = load_municipalities(columns=["cod_ibge", "lat"])
    lat_lookup = dict(zip(df_mun["cod_ibge"], df_mun["lat"]))

    dp = build_prediction_features(df_climate, df_target, df_enso, years_to_predict, fc, lat_lookup)

    available = [f for f in feat_names if f in dp.columns]
    dp["pred_kg_ha"] = model.predict(dp[available].values)
    dp["pred_sc_ha"] = dp["pred_kg_ha"] / 60

    print_report(dp, df_target, years_to_predict)


if __name__ == "__main__":
    main()
