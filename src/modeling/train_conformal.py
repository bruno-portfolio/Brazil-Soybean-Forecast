from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.common.constants import REGION_SUL
from src.common.io import PROJECT_ROOT
from src.modeling.split import create_temporal_split, get_feature_columns

MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"


@dataclass
class ConformalResult:
    """Resultado do treinamento conformal."""

    sul_calibrator_path: str
    cerrado_calibrator_path: str
    coverage_80_sul: float
    coverage_80_cerrado: float
    coverage_80_combined: float
    coverage_90_sul: float
    coverage_90_cerrado: float
    coverage_90_combined: float
    interval_width_80: float
    interval_width_90: float


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna de regiao."""
    df = df.copy()
    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)
    df["is_sul"] = df["uf_cod"].isin(REGION_SUL).astype(int)
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "produtividade_kg_ha",
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara dados para treinamento."""
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    return X, y, df_clean


from src.common.conformal import ConformalCalibrator  # noqa: F401


def calculate_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calcula a cobertura real do intervalo."""
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return in_interval.mean()


def calculate_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calcula a largura media do intervalo."""
    return (y_upper - y_lower).mean()


def train_conformal_predictors() -> ConformalResult:
    """Treina calibradores conformal para os modelos regionais."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("CONFORMAL PREDICTION - INTERVALOS CALIBRADOS")
    logger.info("=" * 60)

    sul_path = MODELS_PATH / "model_sul.pkl"
    cerrado_path = MODELS_PATH / "model_cerrado.pkl"

    if not sul_path.exists() or not cerrado_path.exists():
        raise FileNotFoundError(
            "Modelos regionais nao encontrados. Execute train_regional.py primeiro."
        )

    with open(sul_path, "rb") as f:
        model_sul = pickle.load(f)
    with open(cerrado_path, "rb") as f:
        model_cerrado = pickle.load(f)

    logger.info("\nModelos regionais carregados.")

    logger.info("\nCarregando dados...")
    split = create_temporal_split()

    split.train = add_region_column(split.train)
    split.validation = add_region_column(split.validation)
    split.test = add_region_column(split.test)

    feature_cols = get_feature_columns(split.train)
    features_to_exclude = ["uf_cod", "region"]
    feature_cols = [f for f in feature_cols if f not in features_to_exclude]

    logger.info("\n" + "-" * 60)
    logger.info("CALIBRANDO MODELO SUL")
    logger.info("-" * 60)

    val_sul = split.validation[split.validation["is_sul"] == 1]
    test_sul = split.test[split.test["is_sul"] == 1]

    X_val_sul, y_val_sul, _ = prepare_data(val_sul, feature_cols)
    X_test_sul, y_test_sul, _ = prepare_data(test_sul, feature_cols)

    logger.info(f"  Calibracao: {len(X_val_sul):,} amostras")
    logger.info(f"  Teste: {len(X_test_sul):,} amostras")

    y_pred_val_sul = model_sul.predict(X_val_sul)
    y_pred_test_sul = model_sul.predict(X_test_sul)

    calibrator_sul = ConformalCalibrator()
    calibrator_sul.fit(y_val_sul, y_pred_val_sul)

    lower_80_sul, upper_80_sul = calibrator_sul.predict_interval(y_pred_test_sul, alpha=0.20)
    lower_90_sul, upper_90_sul = calibrator_sul.predict_interval(y_pred_test_sul, alpha=0.10)

    coverage_80_sul = calculate_coverage(y_test_sul, lower_80_sul, upper_80_sul)
    coverage_90_sul = calculate_coverage(y_test_sul, lower_90_sul, upper_90_sul)

    logger.info(f"\n  Cobertura 80% Sul: {coverage_80_sul * 100:.1f}% (esperado: 80%)")
    logger.info(f"  Cobertura 90% Sul: {coverage_90_sul * 100:.1f}% (esperado: 90%)")

    width_80_sul = calculate_interval_width(lower_80_sul, upper_80_sul)
    width_90_sul = calculate_interval_width(lower_90_sul, upper_90_sul)

    logger.info(f"  Largura media 80%: {width_80_sul:.0f} kg/ha")
    logger.info(f"  Largura media 90%: {width_90_sul:.0f} kg/ha")

    logger.info("\n" + "-" * 60)
    logger.info("CALIBRANDO MODELO CERRADO")
    logger.info("-" * 60)

    val_cerrado = split.validation[split.validation["is_sul"] == 0]
    test_cerrado = split.test[split.test["is_sul"] == 0]

    X_val_cerrado, y_val_cerrado, _ = prepare_data(val_cerrado, feature_cols)
    X_test_cerrado, y_test_cerrado, _ = prepare_data(test_cerrado, feature_cols)

    logger.info(f"  Calibracao: {len(X_val_cerrado):,} amostras")
    logger.info(f"  Teste: {len(X_test_cerrado):,} amostras")

    y_pred_val_cerrado = model_cerrado.predict(X_val_cerrado)
    y_pred_test_cerrado = model_cerrado.predict(X_test_cerrado)

    calibrator_cerrado = ConformalCalibrator()
    calibrator_cerrado.fit(y_val_cerrado, y_pred_val_cerrado)

    lower_80_cerrado, upper_80_cerrado = calibrator_cerrado.predict_interval(
        y_pred_test_cerrado, alpha=0.20
    )
    lower_90_cerrado, upper_90_cerrado = calibrator_cerrado.predict_interval(
        y_pred_test_cerrado, alpha=0.10
    )

    coverage_80_cerrado = calculate_coverage(y_test_cerrado, lower_80_cerrado, upper_80_cerrado)
    coverage_90_cerrado = calculate_coverage(y_test_cerrado, lower_90_cerrado, upper_90_cerrado)

    logger.info(f"\n  Cobertura 80% Cerrado: {coverage_80_cerrado * 100:.1f}% (esperado: 80%)")
    logger.info(f"  Cobertura 90% Cerrado: {coverage_90_cerrado * 100:.1f}% (esperado: 90%)")

    width_80_cerrado = calculate_interval_width(lower_80_cerrado, upper_80_cerrado)
    width_90_cerrado = calculate_interval_width(lower_90_cerrado, upper_90_cerrado)

    logger.info(f"  Largura media 80%: {width_80_cerrado:.0f} kg/ha")
    logger.info(f"  Largura media 90%: {width_90_cerrado:.0f} kg/ha")

    logger.info("\n" + "-" * 60)
    logger.info("METRICAS COMBINADAS")
    logger.info("-" * 60)

    y_test_all = np.concatenate([y_test_sul, y_test_cerrado])

    lower_80_all = np.concatenate([lower_80_sul, lower_80_cerrado])
    upper_80_all = np.concatenate([upper_80_sul, upper_80_cerrado])

    lower_90_all = np.concatenate([lower_90_sul, lower_90_cerrado])
    upper_90_all = np.concatenate([upper_90_sul, upper_90_cerrado])

    coverage_80_combined = calculate_coverage(y_test_all, lower_80_all, upper_80_all)
    coverage_90_combined = calculate_coverage(y_test_all, lower_90_all, upper_90_all)

    width_80_combined = calculate_interval_width(lower_80_all, upper_80_all)
    width_90_combined = calculate_interval_width(lower_90_all, upper_90_all)

    logger.info(f"\n  Cobertura 80% Combinada: {coverage_80_combined * 100:.1f}% (esperado: 80%)")
    logger.info(f"  Cobertura 90% Combinada: {coverage_90_combined * 100:.1f}% (esperado: 90%)")
    logger.info(
        f"\n  Largura media 80%: {width_80_combined:.0f} kg/ha ({width_80_combined / 60:.1f} sc/ha)"
    )
    logger.info(
        f"  Largura media 90%: {width_90_combined:.0f} kg/ha ({width_90_combined / 60:.1f} sc/ha)"
    )

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    sul_calibrator_path = MODELS_PATH / "conformal_sul.pkl"
    cerrado_calibrator_path = MODELS_PATH / "conformal_cerrado.pkl"

    with open(sul_calibrator_path, "wb") as f:
        pickle.dump(calibrator_sul, f)

    with open(cerrado_calibrator_path, "wb") as f:
        pickle.dump(calibrator_cerrado, f)

    logger.info(f"\n  Calibrador Sul salvo em: {sul_calibrator_path}")
    logger.info(f"  Calibrador Cerrado salvo em: {cerrado_calibrator_path}")

    training_time = time.time() - start_time

    result = ConformalResult(
        sul_calibrator_path=str(sul_calibrator_path),
        cerrado_calibrator_path=str(cerrado_calibrator_path),
        coverage_80_sul=coverage_80_sul,
        coverage_80_cerrado=coverage_80_cerrado,
        coverage_80_combined=coverage_80_combined,
        coverage_90_sul=coverage_90_sul,
        coverage_90_cerrado=coverage_90_cerrado,
        coverage_90_combined=coverage_90_combined,
        interval_width_80=width_80_combined,
        interval_width_90=width_90_combined,
    )

    result_path = RESULTS_PATH / "conformal_result.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("COMPARATIVO: QUANTILE vs CONFORMAL")
    logger.info("=" * 60)

    logger.info("\n{:<25} {:>15} {:>15}".format("Metrica", "Quantile", "Conformal"))
    logger.info("-" * 55)
    logger.info(
        "{:<25} {:>15.1f}% {:>15.1f}%".format(
            "Cobertura 80% Combinada", 50.4, coverage_80_combined * 100
        )
    )
    logger.info(
        "{:<25} {:>15.0f} {:>15.0f}".format("Largura Intervalo 80%", 861, width_80_combined)
    )

    improvement = (coverage_80_combined * 100 - 50.4) / 50.4 * 100
    logger.info(f"\n  Melhoria na cobertura: +{improvement:.0f}%")

    logger.info(f"\n  Tempo de calibracao: {training_time:.1f}s")

    return result


def main() -> None:
    """Pipeline principal de conformal prediction."""
    train_conformal_predictors()


if __name__ == "__main__":
    main()
