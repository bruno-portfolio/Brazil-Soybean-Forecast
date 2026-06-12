"""Valida previsoes publicadas contra o PAM real divulgado pelo IBGE.

Busca o PAM do ano alvo via agrobr (sem tocar no target do projeto), faz match
nome+UF com municipalities.parquet e compara com results/predictions_*.parquet:
MAE/MAPE/vies do modelo, baselines lag1/MA3 nas mesmas linhas e cobertura
observada dos intervalos conformal.

Uso:
    python -m scripts.validate_against_pam            # ano 2024
    python -m scripts.validate_against_pam --ano 2025
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import unicodedata

import pandas as pd

from src.common.constants import REGION_SUL
from src.common.io import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions_2024_2025.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()


async def fetch_pam_real(ano: int) -> pd.DataFrame:
    from agrobr.ibge import pam

    df = await pam("soja", ano=[ano], nivel="municipio", variaveis=["rendimento"])
    if len(df) == 0:
        raise ValueError(f"PAM {ano} ainda nao publicado pelo IBGE")
    return df


def resolve_cod_ibge(df_real: pd.DataFrame) -> pd.DataFrame:
    parts = df_real["localidade"].str.rsplit(" - ", n=1)
    df_real = df_real.assign(nome_norm=parts.str[0].map(_norm), uf=parts.str[1].str.strip())
    df_real = df_real.rename(columns={"rendimento": "real_kg_ha"})
    df_real = df_real.dropna(subset=["real_kg_ha"])
    df_real = df_real[(df_real["real_kg_ha"] > 0) & (df_real["real_kg_ha"] <= 10000)]

    mun = pd.read_parquet(MUNICIPALITIES_PATH)[["cod_ibge", "nome", "uf"]]
    mun["nome_norm"] = mun["nome"].map(_norm)

    matched = df_real.merge(
        mun[["cod_ibge", "nome_norm", "uf"]], on=["nome_norm", "uf"], how="inner"
    )
    matched = matched.drop_duplicates(subset="cod_ibge")
    logger.info(f"Match nome+UF: {len(matched)} de {len(df_real)}")
    return matched[["cod_ibge", "real_kg_ha"]]


def main():
    parser = argparse.ArgumentParser(description="Valida previsoes contra PAM real")
    parser.add_argument("--ano", type=int, default=2024)
    args = parser.parse_args()

    real = resolve_cod_ibge(asyncio.run(fetch_pam_real(args.ano)))

    pred = pd.read_parquet(PREDICTIONS_PATH)
    pred = pred[pred["ano"] == args.ano]
    if len(pred) == 0:
        raise ValueError(f"Sem previsoes para {args.ano} em {PREDICTIONS_PATH}")

    m = pred.merge(real, on="cod_ibge", how="inner")
    logger.info(f"Municipios comparados: {len(m)}")

    err = m["pred_produtividade_kg_ha"] - m["real_kg_ha"]
    mae = err.abs().mean()
    mape = (err.abs() / m["real_kg_ha"].clip(lower=100)).mean() * 100

    tgt = pd.read_parquet(TARGET_PATH)
    last3 = tgt[tgt["ano"].between(args.ano - 3, args.ano - 1)]
    ma3 = last3.groupby("cod_ibge")["produtividade_kg_ha"].mean().rename("ma3_kg_ha")
    lag1 = (
        tgt[tgt["ano"] == args.ano - 1]
        .set_index("cod_ibge")["produtividade_kg_ha"]
        .rename("lag1_kg_ha")
    )
    mb = m.merge(ma3, on="cod_ibge").merge(lag1, on="cod_ibge")
    mae_model_same = (mb["pred_produtividade_kg_ha"] - mb["real_kg_ha"]).abs().mean()
    mae_ma3 = (mb["ma3_kg_ha"] - mb["real_kg_ha"]).abs().mean()
    mae_lag1 = (mb["lag1_kg_ha"] - mb["real_kg_ha"]).abs().mean()

    m["is_sul"] = m["cod_ibge"].astype(str).str[:2].astype(int).isin(REGION_SUL)
    by_region = {}
    for nome, sub in [("sul", m[m["is_sul"]]), ("cerrado", m[~m["is_sul"]])]:
        e = sub["pred_produtividade_kg_ha"] - sub["real_kg_ha"]
        by_region[nome] = {
            "mae_kg_ha": round(e.abs().mean(), 1),
            "bias_kg_ha": round(e.mean(), 1),
            "n": len(sub),
        }

    result = {
        "ano_validado": args.ano,
        "n_municipios": len(m),
        "model_mae_kg_ha": round(mae, 1),
        "model_mape_percent": round(mape, 1),
        "model_bias_kg_ha": round(err.mean(), 1),
        "baseline_ma3_mae_kg_ha": round(mae_ma3, 1),
        "baseline_lag1_mae_kg_ha": round(mae_lag1, 1),
        "gain_vs_ma3_percent": round((mae_model_same - mae_ma3) / mae_ma3 * 100, 1),
        "by_region": by_region,
    }

    if "pred_lower_80_kg_ha" in m.columns:
        in80 = (m["real_kg_ha"] >= m["pred_lower_80_kg_ha"]) & (
            m["real_kg_ha"] <= m["pred_upper_80_kg_ha"]
        )
        in90 = (m["real_kg_ha"] >= m["pred_lower_90_kg_ha"]) & (
            m["real_kg_ha"] <= m["pred_upper_90_kg_ha"]
        )
        result["conformal_coverage_80_observed"] = round(in80.mean(), 4)
        result["conformal_coverage_90_observed"] = round(in90.mean(), 4)

    out_path = PROJECT_ROOT / "results" / f"validation_real_{args.ano}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(f"Resultado salvo em: {out_path}")
    return result


if __name__ == "__main__":
    main()
