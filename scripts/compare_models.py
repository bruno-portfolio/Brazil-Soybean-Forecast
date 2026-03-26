"""Compara model_v1 vs model_v2 para todos os anos pos-treino (2019-2023).

Gera tabela comparativa de metricas + comparacao com valores reais.
"""

import json
import logging
import pickle

import pandas as pd

from src.common.io import PROJECT_ROOT
from src.evaluation.metrics import compute_all_metrics
from src.modeling.split import create_temporal_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"


def load_model(version: str):
    """Carrega modelo por versao."""
    path = MODELS_PATH / f"model_{version}.pkl"
    if not path.exists():
        logger.warning(f"Modelo {path} nao encontrado")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(version: str) -> dict | None:
    """Carrega metadata do modelo."""
    # Prioridade: model metadata (salvo por save_model) > training result
    candidates = [
        MODELS_PATH / f"model_{version}_metadata.json",
        RESULTS_PATH / f"training_result_{version}.json",
        RESULTS_PATH / "training_result.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def evaluate_model_by_year(model, feature_names: list[str], df: pd.DataFrame) -> dict:
    """Avalia modelo por ano, retornando metricas por ano."""
    results = {}
    target = "produtividade_kg_ha"

    for year in sorted(df["ano"].unique()):
        df_year = df[df["ano"] == year].dropna(subset=feature_names + [target])
        if len(df_year) == 0:
            continue

        X = df_year[feature_names].values
        y_true = df_year[target].values
        y_pred = model.predict(X)

        metrics = compute_all_metrics(y_true, y_pred)
        metrics["n_samples"] = len(df_year)
        metrics["n_municipios"] = df_year["cod_ibge"].nunique()
        metrics["y_true_mean"] = float(y_true.mean())
        metrics["y_pred_mean"] = float(y_pred.mean())
        results[int(year)] = metrics

    return results


def main():
    logger.info("=" * 70)
    logger.info("COMPARACAO model_v1 vs model_v2 (anos 2019-2023)")
    logger.info("=" * 70)

    split = create_temporal_split()
    df_val = split.validation
    df_test = split.test
    df_all = pd.concat([df_val, df_test])

    logger.info(f"Validacao: {len(df_val):,} registros ({sorted(df_val['ano'].unique())})")
    logger.info(f"Teste: {len(df_test):,} registros ({sorted(df_test['ano'].unique())})")

    models = {}
    metadata = {}

    for version in ["v1", "v2"]:
        m = load_model(version)
        if m is None:
            continue
        meta = load_metadata(version)
        if meta is None:
            continue
        models[version] = m
        metadata[version] = meta
        logger.info(f"\n{version}: {len(meta['feature_names'])} features")

    if not models:
        logger.error("Nenhum modelo encontrado!")
        return

    all_results = {}
    for version, model in models.items():
        feat_names = metadata[version]["feature_names"]

        available = [f for f in feat_names if f in df_all.columns]
        missing = [f for f in feat_names if f not in df_all.columns]
        if missing:
            logger.warning(f"  {version}: {len(missing)} features ausentes: {missing[:5]}...")

        all_results[version] = evaluate_model_by_year(model, available, df_all)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS POR ANO")
    logger.info("=" * 70)

    header = f"{'Ano':<6} {'Split':<6}"
    for v in models:
        header += f" | {v} MAE(kg/ha) {v} RMSE  {v} MAPE%"
    header += " | Real(media)"
    logger.info(header)
    logger.info("-" * len(header))

    for year in sorted(df_all["ano"].unique()):
        year = int(year)
        split_name = "VAL" if year <= 2021 else "TEST"

        line = f"{year:<6} {split_name:<6}"
        for v in models:
            r = all_results[v].get(year, {})
            mae = r.get("mae_kg_ha", float("nan"))
            rmse = r.get("rmse_kg_ha", float("nan"))
            mape = r.get("mape_percent", float("nan"))
            line += f" | {mae:>10.1f} {rmse:>8.1f} {mape:>7.1f}%"

        first_v = list(models.keys())[0]
        real = all_results[first_v].get(year, {}).get("y_true_mean", float("nan"))
        line += f" | {real:>10.1f}"
        logger.info(line)

    logger.info("\n" + "=" * 70)
    logger.info("RESUMO AGREGADO")
    logger.info("=" * 70)

    for split_name, df_split in [("VALIDACAO (2019-2021)", df_val), ("TESTE (2022-2023)", df_test)]:
        logger.info(f"\n{split_name}:")
        logger.info(
            f"{'Modelo':<10} {'MAE(kg/ha)':>12} {'MAE(sc/ha)':>12} {'RMSE':>10} {'MAPE%':>8} {'N':>8}"
        )
        logger.info("-" * 62)

        for version, model in models.items():
            feat_names = metadata[version]["feature_names"]
            available = [f for f in feat_names if f in df_split.columns]
            target = "produtividade_kg_ha"
            df_clean = df_split.dropna(subset=available + [target])

            if len(df_clean) == 0:
                continue

            X = df_clean[available].values
            y_true = df_clean[target].values
            y_pred = model.predict(X)
            m = compute_all_metrics(y_true, y_pred)

            logger.info(
                f"{version:<10} {m['mae_kg_ha']:>12.1f} {m['mae_sacas_ha']:>12.1f} "
                f"{m['rmse_kg_ha']:>10.1f} {m['mape_percent']:>8.1f} {m['n_samples']:>8}"
            )

    if len(models) == 2:
        logger.info("\n" + "=" * 70)
        logger.info("DELTA v2 vs v1")
        logger.info("=" * 70)

        for split_name, df_split in [("VALIDACAO", df_val), ("TESTE", df_test)]:
            target = "produtividade_kg_ha"

            results_per_v = {}
            for version, model in models.items():
                feat_names = metadata[version]["feature_names"]
                available = [f for f in feat_names if f in df_split.columns]
                df_clean = df_split.dropna(subset=available + [target])
                X = df_clean[available].values
                y_true = df_clean[target].values
                y_pred = model.predict(X)
                results_per_v[version] = compute_all_metrics(y_true, y_pred)

            if "v1" in results_per_v and "v2" in results_per_v:
                mae_v1 = results_per_v["v1"]["mae_kg_ha"]
                mae_v2 = results_per_v["v2"]["mae_kg_ha"]
                delta = mae_v2 - mae_v1
                pct = (delta / mae_v1) * 100
                emoji = "melhor" if delta < 0 else "pior"
                logger.info(
                    f"  {split_name}: MAE v1={mae_v1:.1f} → v2={mae_v2:.1f} ({delta:+.1f} kg/ha, {pct:+.1f}% — {emoji})"
                )

    comparison_path = RESULTS_PATH / "comparison_v1_v2.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResultados salvos em: {comparison_path}")


if __name__ == "__main__":
    main()
