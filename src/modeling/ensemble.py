from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuracao de um modelo."""
    name: str
    weight: float
    params: dict[str, Any]


def get_lightgbm_params(region: str = "all") -> dict:
    """Retorna parametros LightGBM otimizados."""
    base_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    if region == "sul":
        base_params.update({
            "num_leaves": 20,
            "learning_rate": 0.03,
            "min_child_samples": 30,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        })

    return base_params


def get_xgboost_params(region: str = "all") -> dict:
    """Retorna parametros XGBoost otimizados."""
    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    if region == "sul":
        base_params.update({
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        })

    return base_params


def get_catboost_params(region: str = "all") -> dict:
    """Retorna parametros CatBoost otimizados."""
    base_params = {
        "loss_function": "MAE",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    if region == "sul":
        base_params.update({
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5,
        })

    return base_params


class EnsembleModel:
    """Ensemble de modelos de regressao."""

    def __init__(self, models: list[ModelConfig] | None = None):
        """Inicializa ensemble."""
        if models is None:
            models = self._get_default_models()
        else:
            total_weight = sum(m.weight for m in models)
            if total_weight > 0:
                for m in models:
                    m.weight /= total_weight

        self.model_configs = models
        self.fitted_models: dict[str, Any] = {}
        self.weights: dict[str, float] = {m.name: m.weight for m in models}

    def _get_default_models(self) -> list[ModelConfig]:
        """Retorna configuracao default de modelos."""
        models = []

        if LIGHTGBM_AVAILABLE:
            models.append(ModelConfig(
                name="lightgbm",
                weight=0.4,
                params=get_lightgbm_params()
            ))

        if XGBOOST_AVAILABLE:
            models.append(ModelConfig(
                name="xgboost",
                weight=0.35,
                params=get_xgboost_params()
            ))

        if CATBOOST_AVAILABLE:
            models.append(ModelConfig(
                name="catboost",
                weight=0.25,
                params=get_catboost_params()
            ))

        if not models:
            raise RuntimeError("Nenhum modelo disponivel. Instale lightgbm, xgboost ou catboost.")

        total_weight = sum(m.weight for m in models)
        for m in models:
            m.weight /= total_weight

        return models

    def _create_model(self, config: ModelConfig) -> Any:
        """Cria instancia do modelo."""
        params = config.params.copy()

        if config.name == "lightgbm":
            params.pop("early_stopping_rounds", None)
            return lgb.LGBMRegressor(**params)

        elif config.name == "xgboost":
            params.pop("early_stopping_rounds", None)
            return xgb.XGBRegressor(**params)

        elif config.name == "catboost":
            params.pop("early_stopping_rounds", None)
            return CatBoostRegressor(**params)

        raise ValueError(f"Modelo desconhecido: {config.name}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None
    ) -> EnsembleModel:
        """Treina todos os modelos do ensemble."""
        logger.info(f"Treinando ensemble com {len(self.model_configs)} modelos...")

        for config in self.model_configs:
            logger.info(f"  Treinando {config.name}...")

            model = self._create_model(config)

            if X_val is not None and y_val is not None:
                if config.name == "lightgbm":
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)]
                    )
                elif config.name == "xgboost":
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif config.name == "catboost":
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        verbose=False
                    )
            else:
                model.fit(X_train, y_train)

            self.fitted_models[config.name] = model
            logger.info(f"    {config.name} treinado")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera predicoes usando media ponderada dos modelos."""
        if not self.fitted_models:
            raise RuntimeError("Modelos nao treinados. Execute fit() primeiro.")

        predictions = np.zeros(len(X))

        for name, model in self.fitted_models.items():
            weight = self.weights[name]
            pred = model.predict(X)
            predictions += weight * pred

        return predictions

    def predict_individual(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Retorna predicoes individuais de cada modelo."""
        return {
            name: model.predict(X)
            for name, model in self.fitted_models.items()
        }

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Retorna importancia agregada das features."""
        importance_dfs = []

        for name, model in self.fitted_models.items():
            weight = self.weights[name]

            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                imp = model.get_feature_importance()
            else:
                continue

            df_imp = pd.DataFrame({
                "feature": feature_names,
                f"importance_{name}": imp * weight
            })
            importance_dfs.append(df_imp)

        if not importance_dfs:
            return pd.DataFrame()

        df_merged = importance_dfs[0]
        for df in importance_dfs[1:]:
            df_merged = df_merged.merge(df, on="feature")

        imp_cols = [c for c in df_merged.columns if c.startswith("importance_")]
        df_merged["importance_total"] = df_merged[imp_cols].sum(axis=1)
        df_merged = df_merged.sort_values("importance_total", ascending=False)

        return df_merged

    def save(self, path: Path) -> None:
        """Salva ensemble em disco."""
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.fitted_models.items():
            model_path = path / f"{name}.pkl"
            joblib.dump(model, model_path)

        meta = {
            "weights": self.weights,
            "model_names": list(self.fitted_models.keys())
        }
        joblib.dump(meta, path / "ensemble_meta.pkl")

        logger.info(f"Ensemble salvo em: {path}")

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        """Carrega ensemble do disco."""
        meta = joblib.load(path / "ensemble_meta.pkl")

        ensemble = cls(models=[])
        ensemble.weights = meta["weights"]

        for name in meta["model_names"]:
            model_path = path / f"{name}.pkl"
            ensemble.fitted_models[name] = joblib.load(model_path)

        logger.info(f"Ensemble carregado de: {path}")
        return ensemble


class RegionalEnsemble:
    """Ensemble com modelos especializados por regiao."""

    def __init__(self):
        """Inicializa ensemble regional."""
        self.ensemble_sul: EnsembleModel | None = None
        self.ensemble_outros: EnsembleModel | None = None
        self.sul_ufs = {41, 42, 43}

    def _get_sul_mask(self, df: pd.DataFrame) -> pd.Series:
        """Retorna mascara para registros do Sul."""
        uf_cod = df["cod_ibge"].astype(str).str[:2].astype(int)
        return uf_cod.isin(self.sul_ufs)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        cod_ibge_train: pd.Series = None,
        cod_ibge_val: pd.Series = None
    ) -> RegionalEnsemble:
        """Treina ensembles regionais."""
        if cod_ibge_train is None:
            raise ValueError("cod_ibge_train necessario para split regional")

        uf_train = cod_ibge_train.astype(str).str[:2].astype(int)
        mask_sul_train = uf_train.isin(self.sul_ufs)

        logger.info("=" * 60)
        logger.info("TREINANDO ENSEMBLE SUL")
        logger.info("=" * 60)

        X_sul = X_train[mask_sul_train]
        y_sul = y_train[mask_sul_train]

        sul_models = []
        if LIGHTGBM_AVAILABLE:
            sul_models.append(ModelConfig("lightgbm", 0.4, get_lightgbm_params("sul")))
        if XGBOOST_AVAILABLE:
            sul_models.append(ModelConfig("xgboost", 0.35, get_xgboost_params("sul")))
        if CATBOOST_AVAILABLE:
            sul_models.append(ModelConfig("catboost", 0.25, get_catboost_params("sul")))

        self.ensemble_sul = EnsembleModel(sul_models)

        if X_val is not None and cod_ibge_val is not None:
            uf_val = cod_ibge_val.astype(str).str[:2].astype(int)
            mask_sul_val = uf_val.isin(self.sul_ufs)
            X_val_sul = X_val[mask_sul_val]
            y_val_sul = y_val[mask_sul_val]
            self.ensemble_sul.fit(X_sul, y_sul, X_val_sul, y_val_sul)
        else:
            self.ensemble_sul.fit(X_sul, y_sul)

        logger.info(f"Sul treinado com {len(X_sul)} amostras")

        logger.info("=" * 60)
        logger.info("TREINANDO ENSEMBLE CERRADO/OUTROS")
        logger.info("=" * 60)

        X_outros = X_train[~mask_sul_train]
        y_outros = y_train[~mask_sul_train]

        outros_models = []
        if LIGHTGBM_AVAILABLE:
            outros_models.append(ModelConfig("lightgbm", 0.4, get_lightgbm_params("all")))
        if XGBOOST_AVAILABLE:
            outros_models.append(ModelConfig("xgboost", 0.35, get_xgboost_params("all")))
        if CATBOOST_AVAILABLE:
            outros_models.append(ModelConfig("catboost", 0.25, get_catboost_params("all")))

        self.ensemble_outros = EnsembleModel(outros_models)

        if X_val is not None and cod_ibge_val is not None:
            X_val_outros = X_val[~mask_sul_val]
            y_val_outros = y_val[~mask_sul_val]
            self.ensemble_outros.fit(X_outros, y_outros, X_val_outros, y_val_outros)
        else:
            self.ensemble_outros.fit(X_outros, y_outros)

        logger.info(f"Cerrado/Outros treinado com {len(X_outros)} amostras")

        return self

    def predict(self, X: pd.DataFrame, cod_ibge: pd.Series) -> np.ndarray:
        """Gera predicoes usando ensemble apropriado por regiao."""
        predictions = np.zeros(len(X))

        uf = cod_ibge.astype(str).str[:2].astype(int)
        mask_sul = uf.isin(self.sul_ufs)

        if mask_sul.any():
            predictions[mask_sul] = self.ensemble_sul.predict(X[mask_sul])

        if (~mask_sul).any():
            predictions[~mask_sul] = self.ensemble_outros.predict(X[~mask_sul])

        return predictions

    def save(self, path: Path) -> None:
        """Salva ensembles regionais."""
        self.ensemble_sul.save(path / "sul")
        self.ensemble_outros.save(path / "outros")
        logger.info(f"Ensembles regionais salvos em: {path}")

    @classmethod
    def load(cls, path: Path) -> RegionalEnsemble:
        """Carrega ensembles regionais."""
        regional = cls()
        regional.ensemble_sul = EnsembleModel.load(path / "sul")
        regional.ensemble_outros = EnsembleModel.load(path / "outros")
        return regional


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula metricas de avaliacao."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


def main():
    """Pipeline principal de treinamento do ensemble."""
    logger.info("=" * 60)
    logger.info("TREINAMENTO ENSEMBLE")
    logger.info("=" * 60)

    logger.info("Carregando dados...")
    df = pd.read_parquet(DATASET_PATH)

    target_col = "produtividade_kg_ha"
    exclude_cols = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    df_train = df[df["ano"] <= 2018]
    df_val = df[(df["ano"] >= 2019) & (df["ano"] <= 2021)]
    df_test = df[df["ano"] >= 2022]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_val = df_val[feature_cols]
    y_val = df_val[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    logger.info(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    regional_ensemble = RegionalEnsemble()
    regional_ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        cod_ibge_train=df_train["cod_ibge"],
        cod_ibge_val=df_val["cod_ibge"]
    )

    logger.info("\n" + "=" * 60)
    logger.info("AVALIACAO")
    logger.info("=" * 60)

    pred_val = regional_ensemble.predict(X_val, df_val["cod_ibge"])
    pred_test = regional_ensemble.predict(X_test, df_test["cod_ibge"])

    metrics_val = calculate_metrics(y_val.values, pred_val)
    metrics_test = calculate_metrics(y_test.values, pred_test)

    logger.info("\nValidacao:")
    logger.info(f"  MAE:  {metrics_val['mae']:.1f} kg/ha")
    logger.info(f"  RMSE: {metrics_val['rmse']:.1f} kg/ha")
    logger.info(f"  MAPE: {metrics_val['mape']:.1f}%")

    logger.info("\nTeste:")
    logger.info(f"  MAE:  {metrics_test['mae']:.1f} kg/ha")
    logger.info(f"  RMSE: {metrics_test['rmse']:.1f} kg/ha")
    logger.info(f"  MAPE: {metrics_test['mape']:.1f}%")

    output_dir = MODELS_DIR / "ensemble_regional"
    regional_ensemble.save(output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("CONCLUIDO!")
    logger.info("=" * 60)

    return regional_ensemble, metrics_test


if __name__ == "__main__":
    main()
