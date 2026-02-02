from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics
from src.modeling.split import create_temporal_split, get_feature_columns

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"


@dataclass
class ModelConfig:
    """Configuracao do modelo."""

    algorithm: str
    params: dict[str, Any]
    early_stopping_rounds: int
    early_stopping_enabled: bool


@dataclass
class TrainingResult:
    """Resultado do treinamento."""

    model_path: str
    feature_names: list[str]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_iteration: int
    feature_importance: dict[str, float]
    training_time_seconds: float
    config: dict[str, Any]


def load_config() -> ModelConfig:
    """Carrega configuracao do model.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    algorithm = model_cfg["algorithm"]

    params = model_cfg.get(algorithm, {}).copy()

    return ModelConfig(
        algorithm=algorithm,
        params=params,
        early_stopping_rounds=model_cfg["early_stopping"]["rounds"],
        early_stopping_enabled=model_cfg["early_stopping"]["enabled"],
    )


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


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: ModelConfig,
) -> tuple[lgb.Booster, int]:
    """Treina modelo LightGBM."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = config.params.copy()

    num_boost_round = params.pop("n_estimators", 500)

    callbacks = []
    if config.early_stopping_enabled:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_rounds,
                verbose=False,
            )
        )
    callbacks.append(lgb.log_evaluation(period=0))

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    return model, model.best_iteration


def get_feature_importance(model: lgb.Booster, feature_names: list[str]) -> dict[str, float]:
    """Extrai importancia das features."""
    importance = model.feature_importance(importance_type="gain")
    importance_dict = dict(zip(feature_names, importance))

    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def save_model(
    model: lgb.Booster,
    result: TrainingResult,
    version: str = "v1",
) -> str:
    """Salva modelo e metadados."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_PATH / f"model_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "version": version,
        "algorithm": result.config.get("algorithm", "lightgbm"),
        "training_date": datetime.now().isoformat(),
        "feature_names": result.feature_names,
        "best_iteration": result.best_iteration,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "test_metrics": result.test_metrics,
        "feature_importance": result.feature_importance,
        "training_time_seconds": result.training_time_seconds,
        "config": result.config,
    }

    metadata_path = MODELS_PATH / f"model_{version}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(model_path)


def train_and_evaluate(version: str = "v1") -> TrainingResult:
    """Pipeline completo de treinamento e avaliacao."""
    import time

    start_time = time.time()

    print("=" * 60)
    print("TREINAMENTO DO MODELO PRINCIPAL")
    print("=" * 60)

    config = load_config()
    print(f"\nAlgoritmo: {config.algorithm}")
    print(
        f"Early stopping: {config.early_stopping_enabled} ({config.early_stopping_rounds} rounds)"
    )

    print("\nCarregando dados e criando split temporal...")
    split = create_temporal_split()

    feature_cols = get_feature_columns(split.train)
    print(f"\nFeatures ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    print("\nPreparando dados...")
    X_train, y_train = prepare_data(split.train, feature_cols)
    X_val, y_val = prepare_data(split.validation, feature_cols)
    X_test, y_test = prepare_data(split.test, feature_cols)

    print(f"  Treino: {len(X_train):,} amostras")
    print(f"  Validacao: {len(X_val):,} amostras")
    print(f"  Teste: {len(X_test):,} amostras")

    print("\nTreinando modelo...")
    model, best_iteration = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, config)
    print(f"  Melhor iteracao: {best_iteration}")

    print("\nFazendo predicoes...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("\nCalculando metricas...")
    train_metrics = compute_all_metrics(y_train, y_train_pred)
    val_metrics = compute_all_metrics(y_val, y_val_pred)
    test_metrics = compute_all_metrics(y_test, y_test_pred)

    feature_importance = get_feature_importance(model, feature_cols)

    training_time = time.time() - start_time

    result = TrainingResult(
        model_path="",
        feature_names=feature_cols,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_iteration=best_iteration,
        feature_importance=feature_importance,
        training_time_seconds=training_time,
        config=asdict(config),
    )

    print("\nSalvando modelo...")
    model_path = save_model(model, result, version)
    result.model_path = model_path
    print(f"  Modelo salvo em: {model_path}")

    print("\n" + "=" * 60)
    print("RESULTADOS DO TREINAMENTO")
    print("=" * 60)

    print("\n{:<15} {:>12} {:>12} {:>12}".format("Metrica", "Treino", "Validacao", "Teste"))
    print("-" * 51)
    print(
        "{:<15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "MAE (kg/ha)",
            train_metrics["mae_kg_ha"],
            val_metrics["mae_kg_ha"],
            test_metrics["mae_kg_ha"],
        )
    )
    print(
        "{:<15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "MAE (sc/ha)",
            train_metrics["mae_sacas_ha"],
            val_metrics["mae_sacas_ha"],
            test_metrics["mae_sacas_ha"],
        )
    )
    print(
        "{:<15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "RMSE (kg/ha)",
            train_metrics["rmse_kg_ha"],
            val_metrics["rmse_kg_ha"],
            test_metrics["rmse_kg_ha"],
        )
    )
    print(
        "{:<15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "MAPE (%)",
            train_metrics["mape_percent"],
            val_metrics["mape_percent"],
            test_metrics["mape_percent"],
        )
    )

    print("\n" + "=" * 60)
    print("IMPORTANCIA DAS FEATURES (TOP 10)")
    print("=" * 60)
    print("\n{:<25} {:>15}".format("Feature", "Importancia"))
    print("-" * 40)
    for i, (feat, imp) in enumerate(feature_importance.items()):
        if i >= 10:
            break
        print(f"{feat:<25} {imp:>15.2f}")

    print(f"\nTempo de treinamento: {training_time:.1f}s")

    return result


def main() -> None:
    """Pipeline principal de treinamento."""
    result = train_and_evaluate(version="v1")

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_PATH / "training_result.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"\nResultado salvo em: {result_path}")


if __name__ == "__main__":
    main()
