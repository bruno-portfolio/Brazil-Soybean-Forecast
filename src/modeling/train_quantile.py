from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.modeling.split import create_temporal_split, get_feature_columns

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"


QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


@dataclass
class QuantileModelResult:
    """Resultado do treinamento de um modelo quantilico."""

    quantile: float
    model_path: str
    best_iteration: int
    pinball_loss_test: float
    mae_test: float


@dataclass
class QuantileTrainingResult:
    """Resultado completo do treinamento quantilico."""

    models: list[QuantileModelResult]
    feature_names: list[str]
    coverage_80: float
    mean_interval_width: float
    training_time_seconds: float


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calcula Pinball Loss (quantile loss)."""
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def calculate_coverage(
    y_true: np.ndarray, y_pred_low: np.ndarray, y_pred_high: np.ndarray
) -> float:
    """Calcula cobertura do intervalo de confianca."""
    in_interval = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    return np.mean(in_interval)


def calculate_interval_width(y_pred_low: np.ndarray, y_pred_high: np.ndarray) -> float:
    """Calcula largura media do intervalo."""
    return np.mean(y_pred_high - y_pred_low)


def load_config() -> dict:
    """Carrega configuracao do model.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["model"]


def prepare_data(
    split_data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "produtividade_kg_ha",
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara dados para treinamento."""
    df_clean = split_data.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    return X, y


def train_quantile_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    quantile: float,
    base_params: dict,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, int]:
    """Treina modelo LightGBM com loss function de quantile."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = base_params.copy()
    params["objective"] = "quantile"
    params["alpha"] = quantile

    is_extreme = quantile in [0.05, 0.10, 0.90, 0.95]

    if is_extreme:
        params["learning_rate"] = 0.02
        params["min_data_in_leaf"] = 3
        params["lambda_l1"] = 0.0
        params["lambda_l2"] = 0.0
        params["num_leaves"] = 255
        params["max_depth"] = 12
        params["feature_fraction"] = 0.8

    num_boost_round = params.pop("n_estimators", 500)

    if is_extreme:
        num_boost_round = 300
        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = [train_data]
        valid_names = ["train"]
    else:
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        valid_sets = [train_data, val_data]
        valid_names = ["train", "val"]

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    best_iter = (
        model.best_iteration
        if hasattr(model, "best_iteration") and model.best_iteration > 0
        else num_boost_round
    )
    return model, best_iter


def save_quantile_model(model: lgb.Booster, quantile: float) -> str:
    """Salva modelo quantilico."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    quantile_str = f"p{int(quantile * 100):02d}"
    model_path = MODELS_PATH / f"quantile_{quantile_str}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return str(model_path)


def train_all_quantiles() -> QuantileTrainingResult:
    """Pipeline completo de treinamento quantilico."""
    import time

    start_time = time.time()

    print("=" * 60)
    print("TREINAMENTO DE MODELOS QUANTILICOS")
    print("=" * 60)
    print(f"\nQuantis a treinar: {QUANTILES}")

    config = load_config()
    base_params = config.get("lightgbm", {}).copy()
    early_stopping_rounds = config["early_stopping"]["rounds"]

    print("\nCarregando dados e criando split temporal...")
    split = create_temporal_split()

    feature_cols = get_feature_columns(split.train)
    print(f"Features: {len(feature_cols)}")

    print("\nPreparando dados...")
    X_train, y_train = prepare_data(split.train, feature_cols)
    X_val, y_val = prepare_data(split.validation, feature_cols)
    X_test, y_test = prepare_data(split.test, feature_cols)

    print(f"  Treino: {len(X_train):,} amostras")
    print(f"  Validacao: {len(X_val):,} amostras")
    print(f"  Teste: {len(X_test):,} amostras")

    model_results = []
    predictions = {}

    print("\n" + "-" * 60)

    for quantile in QUANTILES:
        print(f"\nTreinando modelo para quantil {quantile:.2f}...")

        model, best_iteration = train_quantile_model(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_cols,
            quantile,
            base_params,
            early_stopping_rounds,
        )

        y_pred = model.predict(X_test)
        predictions[quantile] = y_pred

        pb_loss = pinball_loss(y_test, y_pred, quantile)

        mae = np.mean(np.abs(y_test - y_pred))

        model_path = save_quantile_model(model, quantile)

        result = QuantileModelResult(
            quantile=quantile,
            model_path=model_path,
            best_iteration=best_iteration,
            pinball_loss_test=pb_loss,
            mae_test=mae,
        )
        model_results.append(result)

        quantile_str = f"p{int(quantile * 100):02d}"
        print(
            f"  [{quantile_str}] Best iter: {best_iteration}, Pinball: {pb_loss:.2f}, MAE: {mae:.1f} kg/ha"
        )
        print(f"  Salvo em: {model_path}")

    print("\n" + "-" * 60)
    print("\nCalculando metricas de intervalo...")

    y_pred_p05 = predictions.get(0.05)
    y_pred_p10 = predictions[0.10]
    y_pred_p50 = predictions[0.50]
    y_pred_p90 = predictions[0.90]
    y_pred_p95 = predictions.get(0.95)

    coverage_80 = calculate_coverage(y_test, y_pred_p10, y_pred_p90)
    mean_width = calculate_interval_width(y_pred_p10, y_pred_p90)

    coverage_90 = None
    mean_width_90 = None
    if y_pred_p05 is not None and y_pred_p95 is not None:
        coverage_90 = calculate_coverage(y_test, y_pred_p05, y_pred_p95)
        mean_width_90 = calculate_interval_width(y_pred_p05, y_pred_p95)

    training_time = time.time() - start_time

    result = QuantileTrainingResult(
        models=model_results,
        feature_names=feature_cols,
        coverage_80=coverage_80,
        mean_interval_width=mean_width,
        training_time_seconds=training_time,
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DO TREINAMENTO QUANTILICO")
    print("=" * 60)

    print("\nModelos treinados:")
    print("-" * 50)
    print(f"{'Quantil':<10} {'Pinball Loss':<15} {'MAE (kg/ha)':<15}")
    print("-" * 50)
    for m in model_results:
        q_str = f"p{int(m.quantile * 100):02d}"
        print(f"{q_str:<10} {m.pinball_loss_test:<15.2f} {m.mae_test:<15.1f}")

    print("\nMetricas de Intervalo de Confianca:")
    print("-" * 50)
    print(f"Cobertura 80% (p10-p90):  {coverage_80 * 100:.1f}%")
    print("  (Esperado: ~80%, Ideal: 78-82%)")
    print(f"Largura media (p10-p90):  {mean_width:.1f} kg/ha ({mean_width/60:.1f} sc/ha)")

    if coverage_90 is not None:
        print(f"\nCobertura 90% (p05-p95):  {coverage_90 * 100:.1f}%")
        print("  (Esperado: ~90%, Ideal: 88-92%)")
        print(f"Largura media (p05-p95):  {mean_width_90:.1f} kg/ha ({mean_width_90/60:.1f} sc/ha)")

    if coverage_80 < 0.75:
        print("\n[!] Intervalos SUB-COBERTOS (muito estreitos)")
        print("    O modelo subestima a incerteza")
    elif coverage_80 > 0.85:
        print("\n[!] Intervalos SOBRE-COBERTOS (muito largos)")
        print("    O modelo superestima a incerteza")
    else:
        print("\n[OK] Intervalos bem calibrados!")

    print(f"\nTempo de treinamento: {training_time:.1f}s")

    print("\n" + "=" * 60)
    print("EXEMPLO DE PREVISAO COM INTERVALO")
    print("=" * 60)

    idx = np.random.randint(0, len(y_test))
    print(f"\nAmostra de teste #{idx}:")
    print(f"  Real:         {y_test[idx]:.0f} kg/ha ({y_test[idx]/60:.1f} sc/ha)")
    print(f"  Previsao p10: {y_pred_p10[idx]:.0f} kg/ha ({y_pred_p10[idx]/60:.1f} sc/ha)")
    print(f"  Previsao p50: {y_pred_p50[idx]:.0f} kg/ha ({y_pred_p50[idx]/60:.1f} sc/ha)")
    print(f"  Previsao p90: {y_pred_p90[idx]:.0f} kg/ha ({y_pred_p90[idx]/60:.1f} sc/ha)")
    print(f"  Intervalo 80%: [{y_pred_p10[idx]:.0f}, {y_pred_p90[idx]:.0f}] kg/ha")

    in_interval = "SIM" if y_pred_p10[idx] <= y_test[idx] <= y_pred_p90[idx] else "NAO"
    print(f"  Valor real no intervalo: {in_interval}")

    return result


def main() -> None:
    """Pipeline principal de treinamento quantilico."""
    result = train_all_quantiles()

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_PATH / "quantile_training_result.json"

    result_dict = {
        "models": [asdict(m) for m in result.models],
        "feature_names": result.feature_names,
        "coverage_80": result.coverage_80,
        "mean_interval_width": result.mean_interval_width,
        "training_time_seconds": result.training_time_seconds,
        "training_date": datetime.now().isoformat(),
    }

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\nResultado salvo em: {result_path}")


if __name__ == "__main__":
    main()
