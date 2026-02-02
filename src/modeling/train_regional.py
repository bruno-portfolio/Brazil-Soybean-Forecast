from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
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

UF_CODES = {
    "RS": 43,
    "SC": 42,
    "PR": 41,
}

REGION_SUL = [41, 42, 43]


@dataclass
class RegionalModelConfig:
    """Configuracao dos modelos regionais."""

    algorithm: str
    params: dict[str, Any]
    early_stopping_rounds: int
    early_stopping_enabled: bool
    sul_params: dict[str, Any] | None = None


@dataclass
class RegionalTrainingResult:
    """Resultado do treinamento regional."""

    sul_model_path: str
    cerrado_model_path: str
    feature_names: list[str]
    sul_metrics: dict[str, Any]
    cerrado_metrics: dict[str, Any]
    combined_metrics: dict[str, Any]
    feature_importance_sul: dict[str, float]
    feature_importance_cerrado: dict[str, float]
    training_time_seconds: float


def load_config() -> RegionalModelConfig:
    """Carrega configuracao do model.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    algorithm = model_cfg["algorithm"]
    params = model_cfg.get(algorithm, {}).copy()

    sul_params = params.copy()
    sul_params["min_data_in_leaf"] = max(10, sul_params.get("min_data_in_leaf", 20))
    sul_params["lambda_l1"] = sul_params.get("lambda_l1", 0.1)
    sul_params["lambda_l2"] = sul_params.get("lambda_l2", 0.1)

    return RegionalModelConfig(
        algorithm=algorithm,
        params=params,
        early_stopping_rounds=model_cfg["early_stopping"]["rounds"],
        early_stopping_enabled=model_cfg["early_stopping"]["enabled"],
        sul_params=sul_params,
    )


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna de regiao (sul vs cerrado)."""
    df = df.copy()
    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)
    df["is_sul"] = df["uf_cod"].isin(REGION_SUL).astype(int)
    df["region"] = df["is_sul"].map({1: "sul", 0: "cerrado"})
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
    return X, y


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    params: dict[str, Any],
    early_stopping_rounds: int,
    early_stopping_enabled: bool,
) -> tuple[lgb.Booster, int]:
    """Treina modelo LightGBM."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = params.copy()
    num_boost_round = params.pop("n_estimators", 500)

    callbacks = []
    if early_stopping_enabled:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
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


def train_regional_models() -> RegionalTrainingResult:
    """Pipeline de treinamento de modelos regionais."""
    start_time = time.time()

    print("=" * 60)
    print("TREINAMENTO DE MODELOS REGIONAIS")
    print("=" * 60)

    config = load_config()
    print(f"\nAlgoritmo: {config.algorithm}")

    print("\nCarregando dados...")
    split = create_temporal_split()

    split.train = add_region_column(split.train)
    split.validation = add_region_column(split.validation)
    split.test = add_region_column(split.test)

    feature_cols = get_feature_columns(split.train)

    features_to_exclude = ["uf_cod", "region"]
    feature_cols = [f for f in feature_cols if f not in features_to_exclude]

    print(f"Features ({len(feature_cols)})")

    print("\n" + "-" * 60)
    print("TREINANDO MODELO SUL (RS, PR, SC)")
    print("-" * 60)

    train_sul = split.train[split.train["is_sul"] == 1]
    val_sul = split.validation[split.validation["is_sul"] == 1]
    test_sul = split.test[split.test["is_sul"] == 1]

    print(f"  Treino Sul: {len(train_sul):,} amostras")
    print(f"  Validacao Sul: {len(val_sul):,} amostras")
    print(f"  Teste Sul: {len(test_sul):,} amostras")

    X_train_sul, y_train_sul = prepare_data(train_sul, feature_cols)
    X_val_sul, y_val_sul = prepare_data(val_sul, feature_cols)
    X_test_sul, y_test_sul = prepare_data(test_sul, feature_cols)

    print("\n  Treinando modelo Sul com params especificos...")
    model_sul, best_iter_sul = train_lightgbm(
        X_train_sul,
        y_train_sul,
        X_val_sul,
        y_val_sul,
        feature_cols,
        config.sul_params,
        config.early_stopping_rounds,
        config.early_stopping_enabled,
    )
    print(f"  Melhor iteracao: {best_iter_sul}")

    y_test_sul_pred = model_sul.predict(X_test_sul)
    test_metrics_sul = compute_all_metrics(y_test_sul, y_test_sul_pred)

    print("\n  Metricas Teste Sul:")
    print(f"    MAE: {test_metrics_sul['mae_kg_ha']:.1f} kg/ha")
    print(f"    MAPE: {test_metrics_sul['mape_percent']:.1f}%")

    fi_sul = get_feature_importance(model_sul, feature_cols)

    print("\n" + "-" * 60)
    print("TREINANDO MODELO CERRADO (demais estados)")
    print("-" * 60)

    train_cerrado = split.train[split.train["is_sul"] == 0]
    val_cerrado = split.validation[split.validation["is_sul"] == 0]
    test_cerrado = split.test[split.test["is_sul"] == 0]

    print(f"  Treino Cerrado: {len(train_cerrado):,} amostras")
    print(f"  Validacao Cerrado: {len(val_cerrado):,} amostras")
    print(f"  Teste Cerrado: {len(test_cerrado):,} amostras")

    X_train_cerrado, y_train_cerrado = prepare_data(train_cerrado, feature_cols)
    X_val_cerrado, y_val_cerrado = prepare_data(val_cerrado, feature_cols)
    X_test_cerrado, y_test_cerrado = prepare_data(test_cerrado, feature_cols)

    print("\n  Treinando modelo Cerrado...")
    model_cerrado, best_iter_cerrado = train_lightgbm(
        X_train_cerrado,
        y_train_cerrado,
        X_val_cerrado,
        y_val_cerrado,
        feature_cols,
        config.params,
        config.early_stopping_rounds,
        config.early_stopping_enabled,
    )
    print(f"  Melhor iteracao: {best_iter_cerrado}")

    y_test_cerrado_pred = model_cerrado.predict(X_test_cerrado)
    test_metrics_cerrado = compute_all_metrics(y_test_cerrado, y_test_cerrado_pred)

    print("\n  Metricas Teste Cerrado:")
    print(f"    MAE: {test_metrics_cerrado['mae_kg_ha']:.1f} kg/ha")
    print(f"    MAPE: {test_metrics_cerrado['mape_percent']:.1f}%")

    fi_cerrado = get_feature_importance(model_cerrado, feature_cols)

    print("\n" + "-" * 60)
    print("METRICAS COMBINADAS (ponderadas por n amostras)")
    print("-" * 60)

    y_test_combined = np.concatenate([y_test_sul, y_test_cerrado])
    y_pred_combined = np.concatenate([y_test_sul_pred, y_test_cerrado_pred])
    combined_metrics = compute_all_metrics(y_test_combined, y_pred_combined)

    print(f"\n  MAE Combinado: {combined_metrics['mae_kg_ha']:.1f} kg/ha")
    print(f"  MAPE Combinado: {combined_metrics['mape_percent']:.1f}%")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    sul_path = MODELS_PATH / "model_sul.pkl"
    cerrado_path = MODELS_PATH / "model_cerrado.pkl"

    with open(sul_path, "wb") as f:
        pickle.dump(model_sul, f)

    with open(cerrado_path, "wb") as f:
        pickle.dump(model_cerrado, f)

    print(f"\n  Modelo Sul salvo em: {sul_path}")
    print(f"  Modelo Cerrado salvo em: {cerrado_path}")

    training_time = time.time() - start_time

    result = RegionalTrainingResult(
        sul_model_path=str(sul_path),
        cerrado_model_path=str(cerrado_path),
        feature_names=feature_cols,
        sul_metrics={
            "test": test_metrics_sul,
            "best_iteration": best_iter_sul,
            "n_train": len(train_sul),
            "n_test": len(test_sul),
        },
        cerrado_metrics={
            "test": test_metrics_cerrado,
            "best_iteration": best_iter_cerrado,
            "n_train": len(train_cerrado),
            "n_test": len(test_cerrado),
        },
        combined_metrics=combined_metrics,
        feature_importance_sul=fi_sul,
        feature_importance_cerrado=fi_cerrado,
        training_time_seconds=training_time,
    )

    result_path = RESULTS_PATH / "regional_training_result.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("COMPARATIVO DE PERFORMANCE")
    print("=" * 60)

    print("\n{:<20} {:>12} {:>12}".format("Regiao", "MAE (kg/ha)", "MAPE (%)"))
    print("-" * 44)
    print(
        "{:<20} {:>12.1f} {:>12.1f}".format(
            "Sul (regional)", test_metrics_sul["mae_kg_ha"], test_metrics_sul["mape_percent"]
        )
    )
    print(
        "{:<20} {:>12.1f} {:>12.1f}".format(
            "Cerrado (regional)",
            test_metrics_cerrado["mae_kg_ha"],
            test_metrics_cerrado["mape_percent"],
        )
    )
    print("-" * 44)
    print(
        "{:<20} {:>12.1f} {:>12.1f}".format(
            "COMBINADO", combined_metrics["mae_kg_ha"], combined_metrics["mape_percent"]
        )
    )

    single_model_path = RESULTS_PATH / "training_result.json"
    if single_model_path.exists():
        with open(single_model_path) as f:
            single_result = json.load(f)
        single_mae = single_result["test_metrics"]["mae_kg_ha"]

        improvement = (single_mae - combined_metrics["mae_kg_ha"]) / single_mae * 100
        print(f"\n  Modelo unico MAE: {single_mae:.1f} kg/ha")
        print(f"  Modelo regional MAE: {combined_metrics['mae_kg_ha']:.1f} kg/ha")
        print(f"  Melhoria: {improvement:+.1f}%")

    print(f"\n  Tempo de treinamento: {training_time:.1f}s")

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES POR REGIAO")
    print("=" * 60)

    print("\n{:<30} {:>15} {:>15}".format("Feature", "Sul", "Cerrado"))
    print("-" * 60)

    all_features = set(list(fi_sul.keys())[:15] + list(fi_cerrado.keys())[:15])
    for feat in list(all_features)[:15]:
        sul_imp = fi_sul.get(feat, 0)
        cer_imp = fi_cerrado.get(feat, 0)
        print(f"{feat[:30]:<30} {sul_imp:>15.0f} {cer_imp:>15.0f}")

    return result


def main() -> None:
    """Pipeline principal de treinamento regional."""
    train_regional_models()


if __name__ == "__main__":
    main()
