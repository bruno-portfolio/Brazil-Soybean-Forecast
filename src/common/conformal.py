"""Calibrador conformal para intervalos de predicao."""

from __future__ import annotations

import numpy as np


class ConformalCalibrator:
    """Calibrador conformal simples para intervalos de predicao.

    Gera intervalos calibrados a partir de residuos absolutos |y_true - y_pred|
    no conjunto de calibracao. Garantia de cobertura ~(1-alpha) se o set de
    calibracao for representativo.
    """

    def __init__(self):
        self.conformity_scores = None
        self.n_calib = 0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> ConformalCalibrator:
        """Calibra o predictor usando residuos absolutos."""
        if len(y_true) == 0:
            raise ValueError("Conjunto de calibracao vazio.")
        self.conformity_scores = np.abs(y_true - y_pred)
        self.n_calib = len(y_true)
        return self

    def predict_interval(
        self, y_pred: np.ndarray, alpha: float = 0.20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera intervalos de predicao calibrados."""
        if self.conformity_scores is None:
            raise ValueError("Calibrador nao foi treinado.")

        n = self.n_calib
        adjusted_quantile = min(1.0, (1 - alpha) * (n + 1) / n)
        q = np.quantile(self.conformity_scores, adjusted_quantile)

        lower = np.maximum(0.0, y_pred - q)
        upper = y_pred + q

        return lower, upper
