from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureDriftResult:
    """Resultado da analise de drift para uma feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    mean_train: float
    mean_test: float
    std_train: float
    std_test: float
    drift_detected: bool
    drift_level: str


@dataclass
class DriftReport:
    """Relatorio completo de drift."""

    feature_results: list[FeatureDriftResult]
    error_by_year: dict
    overall_drift_detected: bool
    recommendations: list[str]
    analysis_date: str


class DriftAnalyzer:
    """Analisador de drift para monitoramento do modelo."""

    def __init__(
        self,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_significant: float = 0.2,
        ks_pvalue_threshold: float = 0.01,
    ):
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_significant = psi_threshold_significant
        self.ks_pvalue_threshold = ks_pvalue_threshold

    def calculate_psi(
        self,
        train_values: np.ndarray,
        test_values: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calcula Population Stability Index (PSI)."""
        train_clean = train_values[~np.isnan(train_values)]
        test_clean = test_values[~np.isnan(test_values)]

        if len(train_clean) == 0 or len(test_clean) == 0:
            return 0.0

        min_val = min(train_clean.min(), test_clean.min())
        max_val = max(train_clean.max(), test_clean.max())

        if min_val == max_val:
            return 0.0

        bins = np.linspace(min_val, max_val, n_bins + 1)

        train_counts, _ = np.histogram(train_clean, bins=bins)
        test_counts, _ = np.histogram(test_clean, bins=bins)

        train_pct = train_counts / len(train_clean)
        test_pct = test_counts / len(test_clean)

        train_pct = np.where(train_pct == 0, 0.0001, train_pct)
        test_pct = np.where(test_pct == 0, 0.0001, test_pct)

        psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))

        return abs(psi)

    def calculate_ks_test(
        self,
        train_values: np.ndarray,
        test_values: np.ndarray,
    ) -> tuple[float, float]:
        """Calcula teste Kolmogorov-Smirnov."""
        train_clean = train_values[~np.isnan(train_values)]
        test_clean = test_values[~np.isnan(test_values)]

        if len(train_clean) < 2 or len(test_clean) < 2:
            return 0.0, 1.0

        ks_stat, p_value = stats.ks_2samp(train_clean, test_clean)

        return ks_stat, p_value

    def analyze_feature(
        self,
        feature_name: str,
        train_values: np.ndarray,
        test_values: np.ndarray,
    ) -> FeatureDriftResult:
        """Analisa drift para uma feature especifica."""
        psi = self.calculate_psi(train_values, test_values)

        ks_stat, ks_pvalue = self.calculate_ks_test(train_values, test_values)

        train_clean = train_values[~np.isnan(train_values)]
        test_clean = test_values[~np.isnan(test_values)]

        mean_train = np.mean(train_clean) if len(train_clean) > 0 else 0
        mean_test = np.mean(test_clean) if len(test_clean) > 0 else 0
        std_train = np.std(train_clean) if len(train_clean) > 0 else 0
        std_test = np.std(test_clean) if len(test_clean) > 0 else 0

        if psi >= self.psi_threshold_significant:
            drift_level = "significant"
            drift_detected = True
        elif psi >= self.psi_threshold_moderate or ks_pvalue < self.ks_pvalue_threshold:
            drift_level = "moderate"
            drift_detected = True
        else:
            drift_level = "none"
            drift_detected = False

        return FeatureDriftResult(
            feature=feature_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            mean_train=mean_train,
            mean_test=mean_test,
            std_train=std_train,
            std_test=std_test,
            drift_detected=drift_detected,
            drift_level=drift_level,
        )

    def calculate_error_by_year(
        self,
        df: pd.DataFrame,
        y_true_col: str = "produtividade_kg_ha",
        y_pred_col: str = "pred",
        year_col: str = "ano",
    ) -> dict:
        """Calcula MAE por ano para detectar degradacao temporal."""
        error_by_year = {}

        for year in sorted(df[year_col].unique()):
            df_year = df[df[year_col] == year]
            if len(df_year) > 0 and y_true_col in df_year.columns and y_pred_col in df_year.columns:
                mae = np.mean(np.abs(df_year[y_true_col] - df_year[y_pred_col]))
                error_by_year[int(year)] = float(mae)

        return error_by_year

    def analyze(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_cols: list[str],
        error_by_year: dict = None,
    ) -> DriftReport:
        """Analisa drift completo entre treino e teste."""
        from datetime import datetime

        feature_results = []

        for col in feature_cols:
            if col in df_train.columns and col in df_test.columns:
                result = self.analyze_feature(
                    feature_name=col,
                    train_values=df_train[col].values,
                    test_values=df_test[col].values,
                )
                feature_results.append(result)

        feature_results.sort(key=lambda x: x.psi, reverse=True)

        significant_drift = [r for r in feature_results if r.drift_level == "significant"]

        overall_drift = len(significant_drift) > 0

        recommendations = []

        if len(significant_drift) > 0:
            features_str = ", ".join([r.feature for r in significant_drift[:3]])
            recommendations.append(
                f"DRIFT SIGNIFICATIVO detectado em: {features_str}. "
                "Considerar retreino do modelo com dados mais recentes."
            )

        if error_by_year:
            years = sorted(error_by_year.keys())
            if len(years) >= 2:
                first_error = error_by_year[years[0]]
                last_error = error_by_year[years[-1]]
                if last_error > first_error * 1.5:
                    recommendations.append(
                        f"Erro aumentou {((last_error/first_error)-1)*100:.0f}% entre "
                        f"{years[0]} e {years[-1]}. Modelo pode estar degradando."
                    )

        if len(recommendations) == 0:
            recommendations.append(
                "Nenhum drift significativo detectado. "
                "Modelo parece estavel para os dados analisados."
            )

        return DriftReport(
            feature_results=feature_results,
            error_by_year=error_by_year or {},
            overall_drift_detected=overall_drift,
            recommendations=recommendations,
            analysis_date=datetime.now().isoformat(),
        )

    def format_report(self, report: DriftReport) -> str:
        """Formata relatorio de drift para exibicao."""
        sep = "=" * 70

        lines = [
            sep,
            "ANALISE DE DRIFT: TREINO vs TESTE",
            sep,
            "",
            f"Data da analise: {report.analysis_date[:10]}",
            f"Drift detectado: {'SIM' if report.overall_drift_detected else 'NAO'}",
            "",
            "-" * 70,
            "DRIFT POR FEATURE (ordenado por PSI)",
            "-" * 70,
            "",
            f"{'Feature':<30} {'PSI':>8} {'KS':>8} {'p-value':>10} {'Status':>15}",
            "-" * 70,
        ]

        for r in report.feature_results[:15]:
            if r.drift_level == "significant":
                status = "[!] DRIFT"
            elif r.drift_level == "moderate":
                status = "[~] Moderado"
            else:
                status = "[OK] Estavel"

            lines.append(
                f"{r.feature:<30} {r.psi:>8.3f} {r.ks_statistic:>8.3f} "
                f"{r.ks_pvalue:>10.4f} {status:>15}"
            )

        drift_features = [r for r in report.feature_results if r.drift_detected]
        if drift_features:
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "FEATURES COM DRIFT DETECTADO",
                    "-" * 70,
                    "",
                ]
            )
            for r in drift_features[:5]:
                pct_change = (
                    ((r.mean_test - r.mean_train) / r.mean_train * 100) if r.mean_train != 0 else 0
                )
                lines.append(
                    f"  - {r.feature}: media mudou {pct_change:+.1f}% "
                    f"(treino: {r.mean_train:.2f}, teste: {r.mean_test:.2f})"
                )

        if report.error_by_year:
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "DEGRADACAO DO MODELO (MAE por ano)",
                    "-" * 70,
                    "",
                    f"{'Ano':<10} {'MAE (kg/ha)':>15} {'vs Primeiro':>15}",
                    "-" * 40,
                ]
            )

            years = sorted(report.error_by_year.keys())
            first_error = report.error_by_year[years[0]]

            for year in years:
                error = report.error_by_year[year]
                pct = ((error / first_error) - 1) * 100 if first_error > 0 else 0
                lines.append(f"{year:<10} {error:>15.1f} {pct:>+14.1f}%")

        lines.extend(
            [
                "",
                "-" * 70,
                "RECOMENDACOES",
                "-" * 70,
                "",
            ]
        )
        for rec in report.recommendations:
            lines.append(f"  -> {rec}")

        lines.extend(["", sep])

        return "\n".join(lines)


def main():
    """Executa analise de drift nos dados do modelo."""
    import pickle
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"
    MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
    RESULTS_PATH = PROJECT_ROOT / "results"

    print("=" * 70)
    print("ANALISE DE DRIFT DO MODELO")
    print("=" * 70)

    print("\nCarregando dados...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Total de registros: {len(df):,}")

    df_train = df[df["ano"] <= 2018]
    df_test = df[df["ano"] >= 2019]

    print(f"  Treino (<=2018): {len(df_train):,}")
    print(f"  Teste (>=2019): {len(df_test):,}")

    feature_cols = [
        "precip_total_mm",
        "tmean_avg",
        "tmin_avg",
        "tmax_avg",
        "hot_days_count",
        "gdd_accumulated",
        "precip_enchimento_mm",
        "precip_vegetativo_mm",
        "precip_plantio_mm",
        "dry_spell_max",
        "precip_cv",
        "oni_avg",
        "oni_std",
        "produtividade_lag1",
        "produtividade_ma3",
    ]

    print("\nCarregando modelo e calculando erro por ano...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    model_features = model.feature_name()

    df_pred = df_test.dropna(subset=model_features + ["produtividade_kg_ha"]).copy()
    X_test = df_pred[model_features].values

    y_pred = model.predict(X_test)
    df_pred["pred"] = y_pred

    error_by_year = {}
    for year in sorted(df_pred["ano"].unique()):
        df_year = df_pred[df_pred["ano"] == year]
        mae = np.mean(np.abs(df_year["produtividade_kg_ha"] - df_year["pred"]))
        error_by_year[int(year)] = float(mae)
        print(f"  {year}: MAE = {mae:.1f} kg/ha")

    print("\nAnalisando drift...")
    analyzer = DriftAnalyzer()
    report = analyzer.analyze(df_train, df_test, feature_cols, error_by_year)

    print(analyzer.format_report(report))

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_PATH / "drift_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Relatorio de Analise de Drift\n\n")
        f.write("```\n")
        f.write(analyzer.format_report(report))
        f.write("\n```\n")

    print(f"\nRelatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
