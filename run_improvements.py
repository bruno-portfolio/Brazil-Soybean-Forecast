#!/usr/bin/env python3
"""
Pipeline de Melhorias v2.0 - Brazil Soybean Forecast

Este script executa todas as melhorias implementadas:
1. Baixa dados extras de clima (radiacao, vento) - OPCIONAL
2. Baixa NDVI via Google Earth Engine - OPCIONAL
3. Reconstroi features com novas variaveis
4. Treina ensemble regional
5. Avalia e compara com baseline

Uso:
    python run_improvements.py --all           # Executa tudo
    python run_improvements.py --features      # Apenas reconstroi features
    python run_improvements.py --train         # Apenas treina ensemble
    python run_improvements.py --evaluate      # Apenas avalia
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_download_climate_extras():
    """Baixa dados extras de clima (radiacao e vento)."""
    logger.info("=" * 60)
    logger.info("PASSO 1: Baixando dados extras de clima")
    logger.info("=" * 60)

    try:
        from src.ingest.climate_update import main as update_climate
        update_climate()
        return True
    except Exception as e:
        logger.error(f"Erro ao baixar dados de clima: {e}")
        logger.info("Continuando sem dados de radiacao...")
        return False


def step_download_ndvi():
    """Baixa NDVI via Google Earth Engine."""
    logger.info("=" * 60)
    logger.info("PASSO 2: Baixando NDVI via GEE")
    logger.info("=" * 60)

    try:
        from src.ingest.ndvi_gee import main as download_ndvi
        download_ndvi()
        return True
    except Exception as e:
        logger.error(f"Erro ao baixar NDVI: {e}")
        logger.info("Continuando sem dados NDVI...")
        return False


def step_build_features():
    """Reconstroi features com novas variaveis."""
    logger.info("=" * 60)
    logger.info("PASSO 3: Reconstruindo features")
    logger.info("=" * 60)

    try:
        from src.features.build_features import main as build_features
        df = build_features()
        logger.info(f"Features reconstruidas: {len(df.columns)} colunas")
        return True
    except Exception as e:
        logger.error(f"Erro ao reconstruir features: {e}")
        return False


def step_train_ensemble():
    """Treina ensemble regional."""
    logger.info("=" * 60)
    logger.info("PASSO 4: Treinando ensemble regional")
    logger.info("=" * 60)

    try:
        from src.modeling.ensemble import main as train_ensemble
        ensemble, metrics = train_ensemble()
        logger.info(f"Ensemble treinado - MAE teste: {metrics['mae']:.1f} kg/ha")
        return True, metrics
    except Exception as e:
        logger.error(f"Erro ao treinar ensemble: {e}")
        return False, None


def step_evaluate():
    """Avalia modelo e compara com baseline."""
    logger.info("=" * 60)
    logger.info("PASSO 5: Avaliando modelo")
    logger.info("=" * 60)

    dataset_path = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"
    results_path = PROJECT_ROOT / "results"
    results_path.mkdir(exist_ok=True)

    df = pd.read_parquet(dataset_path)

    target_col = "produtividade_kg_ha"
    exclude_cols = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    df_test = df[df["ano"] >= 2022].dropna(subset=["produtividade_ma3", target_col])

    baseline_ma3 = df_test["produtividade_ma3"].values
    y_true = df_test[target_col].values

    mae_baseline = np.mean(np.abs(y_true - baseline_ma3))

    try:
        from src.modeling.ensemble import RegionalEnsemble
        models_path = PROJECT_ROOT / "models" / "ensemble_regional"

        if models_path.exists():
            ensemble = RegionalEnsemble.load(models_path)
            X_test = df_test[feature_cols]
            y_pred = ensemble.predict(X_test, df_test["cod_ibge"])

            mae_ensemble = np.mean(np.abs(y_true - y_pred))
            improvement = (mae_baseline - mae_ensemble) / mae_baseline * 100

            logger.info("\n" + "=" * 60)
            logger.info("COMPARACAO COM BASELINE")
            logger.info("=" * 60)
            logger.info(f"Baseline (MA3):     {mae_baseline:.1f} kg/ha")
            logger.info(f"Ensemble Regional:  {mae_ensemble:.1f} kg/ha")
            logger.info(f"Melhoria:           {improvement:.1f}%")

            uf_cod = df_test["cod_ibge"].astype(str).str[:2].astype(int)
            sul_mask = uf_cod.isin({41, 42, 43})

            mae_sul = np.mean(np.abs(y_true[sul_mask] - y_pred[sul_mask]))
            mae_outros = np.mean(np.abs(y_true[~sul_mask] - y_pred[~sul_mask]))

            logger.info(f"\nMAE Sul:            {mae_sul:.1f} kg/ha")
            logger.info(f"MAE Cerrado/Outros: {mae_outros:.1f} kg/ha")

            results = {
                "baseline_ma3_mae": float(mae_baseline),
                "ensemble_mae": float(mae_ensemble),
                "improvement_pct": float(improvement),
                "mae_sul": float(mae_sul),
                "mae_outros": float(mae_outros),
                "n_features": len(feature_cols),
                "n_test_samples": len(df_test),
            }

            with open(results_path / "improvement_results.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"\nResultados salvos em: {results_path / 'improvement_results.json'}")

            return True, results

        else:
            logger.warning("Ensemble nao encontrado. Execute --train primeiro.")
            return False, None

    except Exception as e:
        logger.error(f"Erro na avaliacao: {e}")
        return False, None


def main():
    """Pipeline principal."""
    parser = argparse.ArgumentParser(description="Pipeline de Melhorias v2.0")
    parser.add_argument("--all", action="store_true", help="Executa todos os passos")
    parser.add_argument("--climate", action="store_true", help="Baixa dados extras de clima")
    parser.add_argument("--ndvi", action="store_true", help="Baixa NDVI via GEE")
    parser.add_argument("--features", action="store_true", help="Reconstroi features")
    parser.add_argument("--train", action="store_true", help="Treina ensemble")
    parser.add_argument("--evaluate", action="store_true", help="Avalia modelo")
    parser.add_argument("--quick", action="store_true", help="Features + Train + Evaluate (sem downloads)")

    args = parser.parse_args()

    if not any([args.all, args.climate, args.ndvi, args.features, args.train, args.evaluate, args.quick]):
        parser.print_help()
        print("\n" + "=" * 60)
        print("OPCOES RECOMENDADAS:")
        print("=" * 60)
        print("  --quick    : Rapido (usa dados existentes)")
        print("  --all      : Completo (baixa novos dados)")
        print("  --evaluate : Apenas avaliar modelo existente")
        return

    logger.info("=" * 60)
    logger.info("PIPELINE DE MELHORIAS v2.0")
    logger.info("=" * 60)

    if args.all:
        step_download_climate_extras()
        step_download_ndvi()
        step_build_features()
        step_train_ensemble()
        step_evaluate()

    elif args.quick:
        step_build_features()
        step_train_ensemble()
        step_evaluate()

    else:
        if args.climate:
            step_download_climate_extras()

        if args.ndvi:
            step_download_ndvi()

        if args.features:
            step_build_features()

        if args.train:
            step_train_ensemble()

        if args.evaluate:
            step_evaluate()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE CONCLUIDO!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
