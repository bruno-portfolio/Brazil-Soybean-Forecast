"""Pipeline completo: atualiza dados, rebuild features, retrain e gera previsoes.

Uso:
    python scripts/update_pipeline.py                    # safra corrente
    python scripts/update_pipeline.py --safra 2025/26    # safra especifica
    python scripts/update_pipeline.py --skip-ingest      # so rebuild + retrain
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd

# Adicionar raiz ao path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_area(df_target: pd.DataFrame, year: int) -> pd.DataFrame:
    """Estima area colhida para anos futuros usando tendencia linear por municipio."""
    df = df_target.sort_values(["cod_ibge", "ano"])

    # Usar ultimos 5 anos disponiveis para tendencia
    max_year = df["ano"].max()
    recent = df[df["ano"] >= max_year - 4].copy()

    results = []
    for cod_ibge, group in recent.groupby("cod_ibge"):
        if len(group) < 2:
            # Sem tendencia, usar ultimo valor
            last_area = group["area_colhida_ha"].iloc[-1]
        else:
            # Tendencia linear simples
            x = group["ano"].values.astype(float)
            y = group["area_colhida_ha"].values.astype(float)
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                last_area = y[mask][-1] if mask.any() else 0
            else:
                coef = np.polyfit(x[mask], y[mask], 1)
                last_area = max(0, np.polyval(coef, year))

        results.append({"cod_ibge": cod_ibge, "area_colhida_ha": last_area})

    return pd.DataFrame(results)


def generate_predictions(
    model,
    feat_names: list[str],
    df_climate: pd.DataFrame,
    df_target: pd.DataFrame,
    df_enso: pd.DataFrame,
    years_to_predict: list[int],
    features_config: dict,
    lat_lookup: dict,
) -> pd.DataFrame:
    """Gera previsoes para os anos especificados."""
    fc = features_config
    bt = 10.0
    ht = 32.0
    for feat in fc["features"]["climate_features"]:
        if feat["name"] == "gdd_accumulated":
            bt = feat.get("base_temp", 10.0)
        if feat["name"] == "hot_days_count":
            ht = feat.get("threshold", 32.0)
    trm = fc.get("features", {}).get("trend_ref_year_min", 2000)
    trx = fc.get("features", {}).get("trend_ref_year_max", 2025)

    # Municipios produtores (min 1 ano desde 2018 — mais inclusivo)
    recent = df_target[df_target["ano"] >= 2018]
    municipalities = recent["cod_ibge"].unique().tolist()
    logger.info(f"Municipios produtores (>=1 ano desde 2018): {len(municipalities):,}")

    # Filtrar e agregar clima
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

    # Features historicas
    dh = df_target.sort_values(["cod_ibge", "ano"])
    hf = []
    for cod in dp["cod_ibge"].unique():
        mh = dh[dh["cod_ibge"] == cod].sort_values("ano")
        if len(mh) == 0:
            continue
        for y in years_to_predict:
            p = mh[mh["ano"] < y]["produtividade_kg_ha"].values
            if len(p) == 0:
                continue
            hf.append(
                {
                    "cod_ibge": cod,
                    "ano": y,
                    "produtividade_lag1": p[-1],
                    "produtividade_ma3": p[-3:].mean(),
                }
            )
    dp = dp.merge(pd.DataFrame(hf), on=["cod_ibge", "ano"], how="inner")
    dp["trend"] = (dp["ano"] - trm) / (trx - trm)

    # Fill anomalias
    for col in [
        "precip_anomaly",
        "temp_anomaly",
        "hot_days_anomaly",
        "gdd_anomaly",
        "precip_enchimento_anomaly",
        "dry_spell_anomaly",
    ]:
        if col in dp.columns:
            dp[col] = dp[col].fillna(0.0)
        else:
            dp[col] = 0.0

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

    # Predict
    available = [f for f in feat_names if f in dp.columns]
    X = dp[available].values
    dp["pred_kg_ha"] = model.predict(X)
    dp["pred_sc_ha"] = dp["pred_kg_ha"] / 60

    return dp


def print_report(dp: pd.DataFrame, df_target: pd.DataFrame, years: list[int]):
    """Imprime relatorio de previsao."""
    df_mun = load_municipalities(columns=["cod_ibge", "nome", "uf"])
    dp = dp.merge(df_mun, on="cod_ibge", how="left")

    for year in years:
        dy = dp[dp["ano"] == year]
        if len(dy) == 0:
            logger.info(f"\n{year}: SEM DADOS SUFICIENTES")
            continue

        # Area estimada
        df_area = estimate_area(df_target, year)
        dy = dy.merge(df_area, on="cod_ibge", how="left", suffixes=("", "_est"))
        area_col = (
            "area_colhida_ha_est" if "area_colhida_ha_est" in dy.columns else "area_colhida_ha"
        )
        if area_col not in dy.columns:
            dy[area_col] = 0

        dy["prod_ton"] = dy["pred_kg_ha"] * dy[area_col].fillna(0) / 1000
        prod_mt = dy["prod_ton"].sum() / 1e6

        safra = f"{year - 1}/{str(year)[2:]}"
        max_climate = df_target["ano"].max()  # ultimo ano com dados completos
        parcial = " [clima parcial]" if year > max_climate + 2 else ""

        dy["uf_cod"] = dy["cod_ibge"].astype(str).str[:2].astype(int)
        dy["regiao"] = np.where(dy["uf_cod"].isin([41, 42, 43]), "Sul", "Cerrado/Outros")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"SAFRA {safra}{parcial}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Municipios: {len(dy):,}")
        logger.info(f"Area estimada: {dy[area_col].sum() / 1e6:.1f} Mha")
        logger.info(
            f"Produtividade media: {dy['pred_kg_ha'].mean():.0f} kg/ha ({dy['pred_sc_ha'].mean():.1f} sc/ha)"
        )
        logger.info(f"PRODUCAO TOTAL: {prod_mt:.1f} milhoes de toneladas")

        logger.info("\nPor regiao:")
        for reg in ["Cerrado/Outros", "Sul"]:
            dr = dy[dy["regiao"] == reg]
            if len(dr) == 0:
                continue
            pr = dr["prod_ton"].sum() / 1e6
            area_r = dr[area_col].sum() / 1e6
            logger.info(
                f"  {reg}: {dr['pred_kg_ha'].mean():.0f} kg/ha | "
                f"{area_r:.1f} Mha | {pr:.1f} Mt | {len(dr)} mun"
            )

        logger.info("\nPor UF (top 10):")
        for uf_name in (
            dy.groupby("uf")["prod_ton"].sum().sort_values(ascending=False).head(10).index
        ):
            du = dy[dy["uf"] == uf_name]
            pr = du["prod_ton"].sum() / 1e6
            logger.info(
                f"  {uf_name}: {du['pred_kg_ha'].mean():.0f} kg/ha ({du['pred_sc_ha'].mean():.1f} sc) | "
                f"{du[area_col].sum() / 1e6:.2f} Mha | {pr:.1f} Mt"
            )

    # Salvar
    out_path = PROJECT_ROOT / "results" / f"previsao_safra_{years[0]}_{years[-1]}.parquet"
    cols = ["cod_ibge", "ano", "pred_kg_ha", "pred_sc_ha"]
    cols = [c for c in cols if c in dp.columns]
    dp[cols].to_parquet(out_path, index=False)
    logger.info(f"\nPrevisoes salvas em: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de previsao de soja")
    parser.add_argument("--safra", type=str, help="Safra alvo (ex: 2025/26)", default=None)
    parser.add_argument("--skip-ingest", action="store_true", help="Pular ingestao de dados")
    parser.add_argument("--model-version", type=str, default="v2", help="Versao do modelo")
    args = parser.parse_args()

    # Determinar safra
    if args.safra:
        parts = args.safra.split("/")
        year_end = int(parts[0]) + 1 if len(parts[0]) == 4 else int("20" + parts[1])
    else:
        today = date.today()
        year_end = today.year + 1 if today.month >= 10 else today.year

    years_to_predict = [year_end - 1, year_end]
    logger.info(f"Safras alvo: {[f'{y - 1}/{str(y)[2:]}' for y in years_to_predict]}")

    # Carregar modelo
    model_path = PROJECT_ROOT / "models" / f"model_{args.model_version}.pkl"
    meta_path = PROJECT_ROOT / "models" / f"model_{args.model_version}_metadata.json"

    if not model_path.exists():
        logger.error(f"Modelo nao encontrado: {model_path}")
        logger.info("Execute: python -m src.modeling.train v2")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"Modelo: {args.model_version} ({len(meta['feature_names'])} features)")

    # Carregar dados
    fc = load_config("features")

    v2_path = PROJECT_ROOT / "data" / "processed" / "climate_daily_v2.parquet"
    c_path = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
    df_climate = pd.read_parquet(v2_path if v2_path.exists() else c_path)
    df_climate["date"] = pd.to_datetime(df_climate["date"])
    logger.info(f"Clima: {df_climate['date'].min().date()} a {df_climate['date'].max().date()}")

    df_target = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "target_soja.parquet")
    df_enso = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet")

    df_mun = load_municipalities(columns=["cod_ibge", "lat"])
    lat_lookup = dict(zip(df_mun["cod_ibge"], df_mun["lat"]))

    # Gerar previsoes
    dp = generate_predictions(
        model,
        meta["feature_names"],
        df_climate,
        df_target,
        df_enso,
        years_to_predict,
        fc,
        lat_lookup,
    )

    # Relatorio
    print_report(dp, df_target, years_to_predict)


if __name__ == "__main__":
    main()
