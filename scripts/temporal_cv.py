"""Temporal cross-validation com expanding window para modelos regionais.

Para cada fold, treina Sul + Cerrado separadamente e reporta metricas
por regiao e combinadas. Output: results/temporal_cv_results.json
"""

import json
import logging
import time

import numpy as np
import pandas as pd

from src.common.constants import REGION_SUL
from src.common.io import PROJECT_ROOT
from src.evaluation.metrics import compute_all_metrics
from src.modeling.split import get_feature_columns, load_dataset
from src.modeling.train_regional import (
    add_region_column,
    load_config,
    prepare_data,
    train_lightgbm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "results"

# Folds: test_year de 2016 a 2023, val = test_year - 1, train <= test_year - 2
TEST_YEARS = list(range(2016, 2024))


def run_temporal_cv() -> dict:
    """Executa temporal CV com expanding window."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("TEMPORAL CROSS-VALIDATION (Expanding Window)")
    logger.info("=" * 60)

    config = load_config()
    df = load_dataset()
    df = add_region_column(df)

    feature_cols = get_feature_columns(df)
    feature_cols = [f for f in feature_cols if f not in ["uf_cod", "region"]]
    logger.info(f"Features: {len(feature_cols)}")

    folds = []

    for test_year in TEST_YEARS:
        val_year = test_year - 1
        train_end = test_year - 2

        df_train = df[df["ano"] <= train_end]
        df_val = df[df["ano"] == val_year]
        df_test = df[df["ano"] == test_year]

        if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
            logger.warning(f"  Fold {test_year}: dados insuficientes, pulando")
            continue

        logger.info(f"\n--- Fold test_year={test_year} (train<=  {train_end}, val={val_year}) ---")
        logger.info(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

        fold_result = {
            "test_year": test_year,
            "val_year": val_year,
            "train_years": f"2000-{train_end}",
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
        }

        # --- Treinar Sul ---
        train_sul = df_train[df_train["is_sul"] == 1]
        val_sul = df_val[df_val["is_sul"] == 1]
        test_sul = df_test[df_test["is_sul"] == 1]

        y_test_parts = []
        y_pred_parts = []

        if len(train_sul) > 50 and len(val_sul) > 0 and len(test_sul) > 0:
            X_tr_s, y_tr_s = prepare_data(train_sul, feature_cols)
            X_va_s, y_va_s = prepare_data(val_sul, feature_cols)
            X_te_s, y_te_s = prepare_data(test_sul, feature_cols)

            model_sul, _ = train_lightgbm(
                X_tr_s,
                y_tr_s,
                X_va_s,
                y_va_s,
                feature_cols,
                config.params,
                config.early_stopping_rounds,
                config.early_stopping_enabled,
            )

            y_pred_s = model_sul.predict(X_te_s)
            metrics_sul = compute_all_metrics(y_te_s, y_pred_s)
            fold_result["metrics_sul"] = metrics_sul
            y_test_parts.append(y_te_s)
            y_pred_parts.append(y_pred_s)
            logger.info(f"  Sul:     MAE={metrics_sul['mae_kg_ha']:.1f} kg/ha  n={len(y_te_s)}")
        else:
            fold_result["metrics_sul"] = None

        # --- Treinar Cerrado ---
        train_cer = df_train[df_train["is_sul"] == 0]
        val_cer = df_val[df_val["is_sul"] == 0]
        test_cer = df_test[df_test["is_sul"] == 0]

        if len(train_cer) > 50 and len(val_cer) > 0 and len(test_cer) > 0:
            X_tr_c, y_tr_c = prepare_data(train_cer, feature_cols)
            X_va_c, y_va_c = prepare_data(val_cer, feature_cols)
            X_te_c, y_te_c = prepare_data(test_cer, feature_cols)

            model_cer, _ = train_lightgbm(
                X_tr_c,
                y_tr_c,
                X_va_c,
                y_va_c,
                feature_cols,
                config.params,
                config.early_stopping_rounds,
                config.early_stopping_enabled,
            )

            y_pred_c = model_cer.predict(X_te_c)
            metrics_cer = compute_all_metrics(y_te_c, y_pred_c)
            fold_result["metrics_cerrado"] = metrics_cer
            y_test_parts.append(y_te_c)
            y_pred_parts.append(y_pred_c)
            logger.info(f"  Cerrado: MAE={metrics_cer['mae_kg_ha']:.1f} kg/ha  n={len(y_te_c)}")
        else:
            fold_result["metrics_cerrado"] = None

        # --- Combinado ---
        if y_test_parts:
            y_all = np.concatenate(y_test_parts)
            p_all = np.concatenate(y_pred_parts)
            fold_result["metrics"] = compute_all_metrics(y_all, p_all)
            logger.info(
                f"  Combined MAE={fold_result['metrics']['mae_kg_ha']:.1f} kg/ha  n={len(y_all)}"
            )
        else:
            fold_result["metrics"] = None

        folds.append(fold_result)

    # --- Summary ---
    valid_folds = [f for f in folds if f["metrics"] is not None]
    mae_values = [f["metrics"]["mae_kg_ha"] for f in valid_folds]
    mape_values = [f["metrics"]["mape_percent"] for f in valid_folds]

    summary = {
        "mean_mae_kg_ha": float(np.mean(mae_values)),
        "std_mae_kg_ha": float(np.std(mae_values)),
        "median_mae_kg_ha": float(np.median(mae_values)),
        "mean_mape_percent": float(np.mean(mape_values)),
        "best_year": int(valid_folds[int(np.argmin(mae_values))]["test_year"]),
        "worst_year": int(valid_folds[int(np.argmax(mae_values))]["test_year"]),
        "n_folds": len(valid_folds),
    }

    total_time = time.time() - start_time

    result = {
        "strategy": "expanding_window",
        "n_features": len(feature_cols),
        "folds": folds,
        "summary": summary,
        "total_time_seconds": round(total_time, 1),
    }

    logger.info("\n" + "=" * 60)
    logger.info("RESUMO TEMPORAL CV")
    logger.info("=" * 60)
    logger.info(f"  Folds: {summary['n_folds']}")
    logger.info(
        f"  MAE medio: {summary['mean_mae_kg_ha']:.1f} +/- {summary['std_mae_kg_ha']:.1f} kg/ha"
    )
    logger.info(f"  MAE mediano: {summary['median_mae_kg_ha']:.1f} kg/ha")
    logger.info(f"  MAPE medio: {summary['mean_mape_percent']:.1f}%")
    logger.info(f"  Melhor ano: {summary['best_year']}")
    logger.info(f"  Pior ano: {summary['worst_year']}")
    logger.info(f"  Tempo total: {total_time:.1f}s")

    logger.info("\n  Detalhe por fold:")
    for f in valid_folds:
        m = f["metrics"]
        logger.info(
            f"    {f['test_year']}: MAE={m['mae_kg_ha']:.1f}  MAPE={m['mape_percent']:.1f}%  n={m['n_samples']}"
        )

    return result


def main():
    result = run_temporal_cv()

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_PATH / "temporal_cv_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
