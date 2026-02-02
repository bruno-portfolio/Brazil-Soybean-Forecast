from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calcula Mean Absolute Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calcula Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """Calcula Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def kg_ha_to_sacas_ha(value_kg_ha: float) -> float:
    """Converte kg/ha para sacas/ha."""
    return value_kg_ha / 60.0


def compute_all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calcula todas as metricas de avaliacao."""
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    return {
        "mae_kg_ha": round(mae_val, 2),
        "mae_sacas_ha": round(kg_ha_to_sacas_ha(mae_val), 2),
        "rmse_kg_ha": round(rmse_val, 2),
        "rmse_sacas_ha": round(kg_ha_to_sacas_ha(rmse_val), 2),
        "mape_percent": round(mape_val, 2),
        "n_samples": len(y_true),
    }


if __name__ == "__main__":
    y_true = [3000, 3200, 2800, 3100, 2900]
    y_pred = [2900, 3100, 2700, 3000, 3000]

    metrics = compute_all_metrics(y_true, y_pred)
    print("Teste de metricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
