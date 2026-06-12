"""Testes anti-leakage para as features historicas.

O bug historico deste projeto: rolling aplicado na serie plana (fora do
groupby) contamina os primeiros anos de um municipio com dados de outro,
potencialmente de anos-calendario futuros. Estes testes fixam o contrato.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import calculate_historical_features, validate_no_leakage


@pytest.fixture
def two_municipalities():
    return pd.DataFrame(
        {
            "cod_ibge": [1100015] * 4 + [1100023] * 4,
            "ano": [2020, 2021, 2022, 2023] * 2,
            "produtividade_kg_ha": [1000.0, 2000.0, 3000.0, 4000.0, 100.0, 200.0, 300.0, 400.0],
        }
    )


def test_ma3_nao_cruza_municipios(two_municipalities):
    df = calculate_historical_features(two_municipalities)
    mun2 = df[df["cod_ibge"] == 1100023].sort_values("ano")

    assert pd.isna(mun2["produtividade_ma3"].iloc[0])
    assert mun2["produtividade_ma3"].iloc[1] == 100.0
    assert mun2["produtividade_ma3"].iloc[2] == 150.0
    assert mun2["produtividade_ma3"].iloc[3] == 200.0


def test_lag1_nan_no_primeiro_ano(two_municipalities):
    df = calculate_historical_features(two_municipalities)
    first = df.groupby("cod_ibge")["ano"].transform("min") == df["ano"]

    assert df.loc[first, "produtividade_lag1"].isna().all()
    assert df.loc[first, "produtividade_ma3"].isna().all()


def test_hist_mean_exige_tres_anos_anteriores(two_municipalities):
    df = calculate_historical_features(two_municipalities)
    mun1 = df[df["cod_ibge"] == 1100015].sort_values("ano")

    assert mun1["mun_yield_hist_mean"].iloc[:3].isna().all()
    assert mun1["mun_yield_hist_mean"].iloc[3] == 2000.0

    expected_cv = np.std([1000.0, 2000.0, 3000.0], ddof=1) / 2000.0
    assert mun1["mun_yield_volatility"].iloc[3] == pytest.approx(expected_cv, rel=1e-6)


def test_validate_no_leakage_aborta_com_leakage(two_municipalities):
    df = calculate_historical_features(two_municipalities)
    df.loc[df.index[0], "produtividade_ma3"] = 999.0

    with pytest.raises(ValueError, match="Leakage"):
        validate_no_leakage(df)


def test_validate_no_leakage_passa_com_dataset_limpo(two_municipalities):
    df = calculate_historical_features(two_municipalities)
    assert validate_no_leakage(df) is True


def test_inferencia_replica_semantica_do_treino(two_municipalities):
    from src.inference.predict import calculate_historical_features as infer_hist

    df_train = calculate_historical_features(two_municipalities)

    df_infer_base = two_municipalities[["cod_ibge", "ano"]].copy()
    df_infer = infer_hist(
        df_infer_base[df_infer_base["ano"] == 2023],
        two_municipalities,
        years_to_predict=[2023],
    )

    for col in [
        "produtividade_lag1",
        "produtividade_ma3",
        "mun_yield_hist_mean",
        "mun_yield_volatility",
    ]:
        train_vals = df_train[df_train["ano"] == 2023].set_index("cod_ibge")[col]
        infer_vals = df_infer.set_index("cod_ibge")[col]
        pd.testing.assert_series_equal(train_vals, infer_vals, check_names=False, check_dtype=False)
