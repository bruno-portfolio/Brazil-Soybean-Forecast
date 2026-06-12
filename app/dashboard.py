import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd

try:
    import joblib
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import shap
    import streamlit as st
    from plotly.subplots import make_subplots

    from src.common.constants import REGION_SUL
    from src.modeling.train_conformal import ConformalCalibrator

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Erro: Bibliotecas necessárias nao instaladas.")
    print("Execute: pip install streamlit plotly shap matplotlib joblib")
    sys.exit(1)


PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions_2024_2025.parquet"
RISK_PATH = PROJECT_ROOT / "results" / "risk_analysis_2024_2025.parquet"
DRIFT_PATH = PROJECT_ROOT / "results" / "drift_report.md"
REGIONAL_RESULT_PATH = PROJECT_ROOT / "results" / "regional_training_result.json"
BASELINES_PATH = PROJECT_ROOT / "results" / "baselines.json"
TEMPORAL_CV_PATH = PROJECT_ROOT / "results" / "temporal_cv_results.json"


def ano_para_safra(ano: int) -> str:
    """Converte ano PAM para nomenclatura de safra."""
    return f"{ano - 1}/{str(ano)[2:]}"


def safra_para_ano(safra: str) -> int:
    """Converte nomenclatura de safra para ano PAM."""
    return int("20" + safra.split("/")[1])


@st.cache_data
def load_predictions():
    """Carrega previsoes."""
    if PREDICTIONS_PATH.exists():
        df = pd.read_parquet(PREDICTIONS_PATH)
        df["safra"] = df["ano"].apply(ano_para_safra)
        return df
    return None


@st.cache_data
def load_risk_analysis():
    """Carrega analise de risco."""
    if RISK_PATH.exists():
        df = pd.read_parquet(RISK_PATH)
        df["safra"] = df["ano"].apply(ano_para_safra)
        return df
    return None


@st.cache_data
def load_regional_result():
    """Carrega resultado do treinamento regional (modelo de producao)."""
    if REGIONAL_RESULT_PATH.exists():
        with open(REGIONAL_RESULT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_baselines():
    """Carrega metricas dos baselines."""
    if BASELINES_PATH.exists():
        with open(BASELINES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_temporal_cv():
    """Carrega resultados da validacao cruzada temporal."""
    if TEMPORAL_CV_PATH.exists():
        with open(TEMPORAL_CV_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_drift_report():
    """Carrega relatorio de drift."""
    if DRIFT_PATH.exists():
        with open(DRIFT_PATH, encoding="utf-8") as f:
            return f.read()
    return None


def render_model_info_sidebar():
    """Renderiza informacoes do modelo na sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sobre o Modelo")

    regional = load_regional_result()

    if regional:
        combined = regional.get("combined_metrics", {})
        n_train = regional.get("sul_metrics", {}).get("n_train", 0) + regional.get(
            "cerrado_metrics", {}
        ).get("n_train", 0)
        st.sidebar.markdown(
            f"""
        **Versao:** v3 (LightGBM regional Sul/Cerrado)

        **Features:** {len(regional.get("feature_names", []))}

        **Treino (2000-2021):** {n_train:,} amostras

        **Teste (safra 2022/23):**
        - MAE: {combined.get("mae_kg_ha", 0):.0f} kg/ha ({combined.get("mae_sacas_ha", 0):.1f} sc/ha)
        - MAPE: {combined.get("mape_percent", 0):.1f}%
        """
        )
    else:
        st.sidebar.warning("Execute `python -m src.modeling.train_regional` para gerar metricas.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Legenda de Safra")
    st.sidebar.markdown(
        """
    A nomenclatura segue o ciclo agricola:
    - **Safra 2023/24**: plantio Out/2023, colheita 2024
    - **Safra 2024/25**: plantio Out/2024, colheita 2025
    """
    )


def page_visao_geral():
    """Pagina de visao geral com KPIs e graficos."""
    st.header("Visao Geral - Previsoes de Produtividade")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None:
        st.error(
            "Dados de previsao nao encontrados. Execute `python -m src.inference.predict` primeiro."
        )
        return

    regional = load_regional_result()
    mae_txt = ""
    if regional:
        combined = regional.get("combined_metrics", {})
        mae_txt = (
            f"- Testado na safra 2022/23 com MAE de {combined.get('mae_kg_ha', 0):.0f} kg/ha "
            f"({combined.get('mae_sacas_ha', 0):.1f} sacas/ha)\n"
        )

    st.info(
        f"""
    **Sobre estas previsoes:**
    - Modelos LightGBM regionais (Sul / Cerrado) treinados com dados de 2000-2021
    {mae_txt}- Previsoes abaixo usam clima ja observado (modalidade ex-post); o historico de
    produtividade usa o ultimo dado PAM disponivel, que pode estar defasado
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        safras_disponiveis = sorted(df["safra"].unique())
        safra_selecionada = st.selectbox("Safra", safras_disponiveis)
    with col2:
        ufs_disponiveis = ["Todas"] + sorted(df["uf"].unique().tolist())
        uf_selecionada = st.selectbox("UF", ufs_disponiveis)

    ano_selecionado = safra_para_ano(safra_selecionada)
    df_filtrado = df[df["ano"] == ano_selecionado]
    if uf_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado["uf"] == uf_selecionada]

    st.subheader(f"Indicadores - Safra {safra_selecionada}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Municipios Analisados",
            f"{len(df_filtrado):,}",
        )

    with col2:
        media_prod = df_filtrado["pred_produtividade_kg_ha"].mean()
        st.metric(
            "Produtividade Media Prevista",
            f"{media_prod:.0f} kg/ha",
            f"{media_prod / 60:.1f} sc/ha",
        )

    with col3:
        if "pred_lower_80_kg_ha" in df_filtrado.columns:
            media_lower = df_filtrado["pred_lower_80_kg_ha"].mean()
            st.metric(
                "Cenario Pessimista",
                f"{media_lower:.0f} kg/ha",
                help="Limite inferior do intervalo conformal de 80%",
            )

    with col4:
        if df_risk is not None:
            df_risk_filtrado = df_risk[df_risk["ano"] == ano_selecionado]
            if uf_selecionada != "Todas":
                df_risk_filtrado = df_risk_filtrado[df_risk_filtrado["uf"] == uf_selecionada]
            alto_risco = (df_risk_filtrado["rating"].isin(["C", "D"])).sum()
            pct_alto = alto_risco / len(df_risk_filtrado) * 100 if len(df_risk_filtrado) > 0 else 0
            st.metric(
                "Municipios Alto Risco",
                f"{alto_risco:,} ({pct_alto:.1f}%)",
                help="Rating C ou D: probabilidade de quebra > 20%",
            )

    st.subheader(f"Produtividade Prevista por UF - Safra {safra_selecionada}")

    df_ano = df[df["ano"] == ano_selecionado]
    df_uf = (
        df_ano.groupby("uf")
        .agg({"pred_produtividade_kg_ha": "mean", "cod_ibge": "count"})
        .reset_index()
    )
    df_uf.columns = ["UF", "Produtividade Media (kg/ha)", "Municipios"]
    df_uf = df_uf.sort_values("Produtividade Media (kg/ha)", ascending=True)

    fig = px.bar(
        df_uf,
        x="Produtividade Media (kg/ha)",
        y="UF",
        orientation="h",
        color="Produtividade Media (kg/ha)",
        color_continuous_scale="RdYlGn",
        text=df_uf["Produtividade Media (kg/ha)"].apply(lambda x: f"{x:.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Produtividade Prevista (kg/ha)",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    if df_risk is not None:
        st.subheader(f"Distribuicao de Risco - Safra {safra_selecionada}")

        col1, col2 = st.columns(2)

        with col1:
            df_rating = (
                df_risk[df_risk["ano"] == ano_selecionado]["rating"].value_counts().reset_index()
            )
            df_rating.columns = ["Rating", "Quantidade"]
            df_rating = df_rating.sort_values("Rating")

            fig_risk = px.pie(
                df_rating,
                values="Quantidade",
                names="Rating",
                color="Rating",
                color_discrete_map={"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"},
                hole=0.4,
            )
            fig_risk.update_layout(
                title="Distribuicao por Rating",
                annotations=[dict(text="Rating", x=0.5, y=0.5, font_size=14, showarrow=False)],
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            st.markdown(
                """
            **Legenda de Ratings:**

            | Rating | Risco | Prob. Quebra |
            |--------|-------|--------------|
            | **A** | Baixo | < 10% |
            | **B** | Moderado | 10-20% |
            | **C** | Elevado | 20-35% |
            | **D** | Alto | > 35% |

            *Quebra = Receita < Custo de Producao*
            """
            )


def page_validacao():
    """Pagina de validacao do modelo."""
    st.header("Validacao do Modelo")

    st.markdown(
        """
    Esta pagina mostra a qualidade das previsoes do modelo quando comparadas
    com dados reais de produtividade (PAM/IBGE).
    """
    )

    st.subheader("Estrutura de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Divisao Temporal (sem vazamento):**

        | Conjunto | Safra | Ano PAM | Uso |
        |----------|-------|---------|-----|
        | Treino | ate 2020/21 | 2000-2021 | Aprendizado |
        | Validacao | 2021/22 | 2022 | Early stopping |
        | Teste | 2022/23 | 2023 | Avaliacao final |
        | Previsao | 2023/24+ | 2024+ | Producao |
        """
        )

    with col2:
        st.markdown(
            """
        **Por que esta divisao?**

        - O modelo **nunca ve dados futuros** durante o treino
        - Validacao em 2022 (seca historica no RS, La Nina): early stopping
          calibrado no caso mais dificil
        - Teste em 2023 intocado durante o desenvolvimento
        - CV temporal expanding (2016-2023) confirma que o resultado
          nao depende de um unico ano de teste
        """
        )

    st.subheader("Metricas no Conjunto de Teste (safra 2022/23)")

    regional = load_regional_result()
    baselines = load_baselines()

    if regional:
        combined = regional.get("combined_metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mae = combined.get("mae_kg_ha", 0)
            st.metric(
                "MAE", f"{mae:.0f} kg/ha", f"{mae / 60:.1f} sc/ha", help="Erro Medio Absoluto"
            )

        with col2:
            st.metric(
                "MAPE", f"{combined.get('mape_percent', 0):.1f}%", help="Erro Percentual Medio"
            )

        with col3:
            if baselines:
                ma3 = baselines["baseline_ma3"]["test_metrics"]["mae_kg_ha"]
                gain = (ma3 - mae) / ma3 * 100
                st.metric(
                    "vs Baseline (MA3)",
                    f"-{gain:.1f}%",
                    help=f"Reducao de MAE sobre media movel 3 anos ({ma3:.0f} kg/ha)",
                )

        with col4:
            st.metric(
                "Amostras de Teste",
                f"{combined.get('n_samples', 0):,}",
                help="Municipios no teste",
            )

        st.subheader("Erro por Regiao (Teste 2022/23)")

        sul_test = regional.get("sul_metrics", {}).get("test", {})
        cerrado_test = regional.get("cerrado_metrics", {}).get("test", {})
        erro_regiao = pd.DataFrame(
            {
                "Regiao": ["Sul (RS, PR, SC)", "Cerrado/Outros"],
                "MAE (kg/ha)": [sul_test.get("mae_kg_ha", 0), cerrado_test.get("mae_kg_ha", 0)],
                "MAPE (%)": [sul_test.get("mape_percent", 0), cerrado_test.get("mape_percent", 0)],
                "Municipios": [sul_test.get("n_samples", 0), cerrado_test.get("n_samples", 0)],
            }
        )
        st.dataframe(erro_regiao, use_container_width=True, hide_index=True)

        st.markdown(
            """
        O erro maior no Sul reflete a volatilidade climatica da regiao
        (La Nina / El Nino), nao um defeito do modelo: o baseline tambem
        erra mais no Sul.
        """
        )
    else:
        st.error("Execute `python -m src.modeling.train_regional` primeiro.")

    st.subheader("Validacao Cruzada Temporal (Expanding Window)")

    cv = load_temporal_cv()
    if cv:
        folds = pd.DataFrame(
            [
                {
                    "Safra": ano_para_safra(f["test_year"]),
                    "MAE (kg/ha)": f.get("metrics_combined", {}).get("mae_kg_ha"),
                    "MAPE (%)": f.get("metrics_combined", {}).get("mape_percent"),
                    "n": f.get("n_test"),
                }
                for f in cv.get("folds", [])
            ]
        )

        fig2 = px.bar(
            folds,
            x="Safra",
            y="MAE (kg/ha)",
            text=folds["MAE (kg/ha)"].apply(lambda x: f"{x:.0f}"),
        )
        fig2.update_traces(textposition="outside", marker_color="#3498db")
        summary = cv.get("summary", {})
        fig2.add_hline(
            y=summary.get("mean_mae_kg_ha", 0),
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Media: {summary.get('mean_mae_kg_ha', 0):.0f} kg/ha",
        )
        fig2.update_layout(height=400, xaxis_title="Safra de teste", yaxis_title="MAE (kg/ha)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"""
        Cada fold treina do zero com dados ate 2 anos antes da safra de teste
        (validacao no ano intermediario). MAE medio:
        **{summary.get("mean_mae_kg_ha", 0):.0f} +/- {summary.get("std_mae_kg_ha", 0):.0f} kg/ha**
        em {summary.get("n_folds", 0)} safras independentes. O pior ano
        ({ano_para_safra(summary.get("worst_year", 2022))}) corresponde a seca
        historica de La Nina no Sul.
        """
        )
    else:
        st.info("Execute `python -m scripts.temporal_cv` para gerar a validacao cruzada.")


def page_municipio():
    """Pagina de analise por municipio."""
    st.header("Analise por Municipio")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None or df_risk is None:
        st.error("Dados nao encontrados.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        uf = st.selectbox("UF", sorted(df["uf"].unique()))
    with col2:
        municipios = df[df["uf"] == uf]["nome"].unique()
        municipio = st.selectbox("Municipio", sorted(municipios))
    with col3:
        safras = sorted(df["safra"].unique())
        safra = st.selectbox("Safra", safras)

    ano = safra_para_ano(safra)

    df_mun = df[(df["uf"] == uf) & (df["nome"] == municipio) & (df["ano"] == ano)]
    df_risk_mun = df_risk[
        (df_risk["uf"] == uf) & (df_risk["municipio"] == municipio) & (df_risk["ano"] == ano)
    ]

    if len(df_mun) == 0:
        st.warning(f"Dados nao disponiveis para {municipio}/{uf} na safra {safra}.")
        return

    row = df_mun.iloc[0]
    risk_row = df_risk_mun.iloc[0] if len(df_risk_mun) > 0 else None

    st.subheader(f"{municipio}/{uf} - Safra {safra}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Previsao de Produtividade")

        if "pred_lower_80_kg_ha" in row.index:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Intervalo 80%",
                    x=["Previsao"],
                    y=[row["pred_upper_80_kg_ha"] - row["pred_lower_80_kg_ha"]],
                    base=[row["pred_lower_80_kg_ha"]],
                    marker_color="rgba(52, 152, 219, 0.3)",
                    width=0.5,
                )
            )

            fig.add_hline(
                y=row["pred_lower_80_kg_ha"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"inferior 80%: {row['pred_lower_80_kg_ha']:.0f}",
            )

            fig.add_hline(
                y=row["pred_produtividade_kg_ha"],
                line_dash="solid",
                line_color="blue",
                annotation_text=f"previsao: {row['pred_produtividade_kg_ha']:.0f}",
            )

            fig.add_hline(
                y=row["pred_upper_80_kg_ha"],
                line_dash="dash",
                line_color="green",
                annotation_text=f"superior 80%: {row['pred_upper_80_kg_ha']:.0f}",
            )

            fig.update_layout(
                height=350,
                yaxis_title="Produtividade (kg/ha)",
                showlegend=False,
                yaxis=dict(range=[0, row["pred_upper_80_kg_ha"] * 1.2]),
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        | Cenario | Produtividade | Sacas/ha |
        |---------|---------------|----------|
        """
        )
        st.markdown(
            f"| Pessimista (lim. inf. 80%) | {row.get('pred_lower_80_kg_ha', 0):.0f} kg/ha | {row.get('pred_lower_80_kg_ha', 0) / 60:.1f} |"
        )
        st.markdown(
            f"| **Base (previsao)** | **{row['pred_produtividade_kg_ha']:.0f} kg/ha** | **{row['pred_produtividade_kg_ha'] / 60:.1f}** |"
        )
        st.markdown(
            f"| Otimista (lim. sup. 80%) | {row.get('pred_upper_80_kg_ha', 0):.0f} kg/ha | {row.get('pred_upper_80_kg_ha', 0) / 60:.1f} |"
        )

    with col2:
        st.markdown("### Analise de Risco")

        if risk_row is not None:
            rating = risk_row["rating"]
            rating_colors = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
            rating_labels = {
                "A": "BAIXO RISCO",
                "B": "RISCO MODERADO",
                "C": "RISCO ELEVADO",
                "D": "ALTO RISCO",
            }

            st.markdown(
                f"""<div style='background-color: {rating_colors.get(rating, "gray")};
                padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='color: white; margin: 0;'>{rating}</h1>
                <p style='color: white; margin: 0;'>{rating_labels.get(rating, "")}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Prob. Quebra", f"{risk_row['prob_quebra'] * 100:.1f}%")
                st.metric("Custo Producao", f"R$ {risk_row['custo_ha']:,.0f}/ha")
            with col_b:
                st.metric("Spread Sugerido", f"{risk_row['spread_sugerido']}% a.a.")
                st.metric("Preco Soja", f"R$ {risk_row['preco_saca']:.0f}/sc")

            st.markdown("#### Cenarios Financeiros")
            st.markdown(
                f"""
            | Cenario | Receita/ha | Lucro/ha |
            |---------|------------|----------|
            | Pessimista | R$ {risk_row["receita_pessimista"]:,.0f} | R$ {risk_row["lucro_pessimista"]:,.0f} |
            | Base | R$ {risk_row["receita_base"]:,.0f} | R$ {risk_row["lucro_base"]:,.0f} |
            | Otimista | R$ {risk_row["receita_otimista"]:,.0f} | R$ {risk_row["lucro_otimista"]:,.0f} |
            """
            )

    st.markdown("---")
    st.subheader("Perfil do municipio (SHAP da ultima safra observada)")

    with st.spinner("Gerando explicabilidade (isso pode levar alguns segundos)..."):
        try:
            uf_cod = int(str(int(row["cod_ibge"]))[:2])
            is_sul = uf_cod in REGION_SUL
            model_file = "model_sul.pkl" if is_sul else "model_cerrado.pkl"
            gbm = joblib.load(PROJECT_ROOT / "models" / model_file)
            features_treino = gbm.feature_name()

            FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"

            if not FEATURES_PATH.exists():
                st.warning("dataset_final.parquet nao encontrado para gerar o SHAP.")
            else:
                df_features = pd.read_parquet(FEATURES_PATH)

                df_features["cod_ibge"] = df_features["cod_ibge"].astype(int)
                df_features["ano"] = df_features["ano"].astype(int)
                cod_ibge_busca = int(row["cod_ibge"])

                hist_mun = df_features[df_features["cod_ibge"] == cod_ibge_busca]

                if len(hist_mun) == 0:
                    st.info(f"Municipio {cod_ibge_busca} sem historico no dataset de treino.")
                else:
                    ultima = hist_mun[hist_mun["ano"] == hist_mun["ano"].max()]
                    ano_shap = int(ultima["ano"].iloc[0])
                    st.caption(
                        f"As features da previsao {safra} nao sao persistidas; o grafico "
                        f"mostra os drivers da ultima safra observada "
                        f"({ano_para_safra(ano_shap)}) com o modelo "
                        f"{'Sul' if is_sul else 'Cerrado'}."
                    )

                    X_municipio_final = ultima[features_treino]

                    explainer = shap.TreeExplainer(gbm)
                    shap_values = explainer(X_municipio_final)
                    shap_values.feature_names = features_treino

                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig_shap)

        except Exception as e:
            st.warning(f"Erro tecnico ao gerar SHAP: {e}")


def page_regional():
    """Pagina de comparativo regional."""
    st.header("Comparativo Regional")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None:
        st.error("Dados nao encontrados.")
        return

    safra = st.selectbox("Safra", sorted(df["safra"].unique()))
    ano = safra_para_ano(safra)

    st.subheader(f"Ranking de UFs - Safra {safra}")

    df_ano = df[df["ano"] == ano]

    agg_dict = {
        "pred_produtividade_kg_ha": ["mean", "std"],
        "cod_ibge": "count",
    }

    if "pred_lower_80_kg_ha" in df_ano.columns:
        agg_dict["pred_lower_80_kg_ha"] = "mean"
        agg_dict["pred_upper_80_kg_ha"] = "mean"

    df_uf = df_ano.groupby("uf").agg(agg_dict).round(1)
    df_uf.columns = [
        "Prod. Media",
        "Desvio Padrao",
        "Municipios",
        "Inf. 80% Medio",
        "Sup. 80% Medio",
    ]
    df_uf = df_uf.reset_index()
    df_uf = df_uf.sort_values("Prod. Media", ascending=False)

    if df_risk is not None:
        df_risk_ano = df_risk[df_risk["ano"] == ano]
        risco_uf = (
            df_risk_ano.groupby("uf")
            .apply(lambda x: (x["rating"].isin(["C", "D"])).sum() / len(x) * 100)
            .reset_index()
        )
        risco_uf.columns = ["uf", "% Alto Risco"]
        df_uf = df_uf.merge(risco_uf, on="uf", how="left")

    st.dataframe(
        df_uf.style.format(
            {
                "Prod. Media": "{:.0f}",
                "Desvio Padrao": "{:.0f}",
                "Inf. 80% Medio": "{:.0f}",
                "Sup. 80% Medio": "{:.0f}",
                "% Alto Risco": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if df_risk is not None and "% Alto Risco" in df_uf.columns:
        st.subheader("Municipios com Alto Risco por UF")

        df_uf_sorted = df_uf.sort_values("% Alto Risco", ascending=True)

        fig = px.bar(
            df_uf_sorted,
            x="% Alto Risco",
            y="uf",
            orientation="h",
            color="% Alto Risco",
            color_continuous_scale="Reds",
            text=df_uf_sorted["% Alto Risco"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=500,
            xaxis_title="% de Municipios com Rating C ou D",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_monitoramento():
    """Pagina de monitoramento e drift."""
    st.header("Monitoramento do Modelo")

    st.markdown(
        """
    O monitoramento detecta mudancas nas distribuicoes de dados (drift) que
    podem degradar a qualidade das previsoes ao longo do tempo.
    """
    )

    st.subheader("Status Atual")

    regional = load_regional_result()
    n_features = len(regional.get("feature_names", [])) if regional else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Versao", "v3 (regional)")

    with col2:
        st.metric("Features", f"{n_features}")

    with col3:
        st.metric("Treino ate", "Safra 2020/21")

    with col4:
        st.metric("Teste", "Safra 2022/23")

    st.subheader("Analise de Drift")

    drift_report = load_drift_report()

    if drift_report:
        with st.expander("Ver Relatorio Completo de Drift", expanded=False):
            st.code(drift_report, language="text")
    else:
        st.info(
            "Execute `python -m src.monitoring.drift_analysis` para gerar o relatorio de drift."
        )

    st.subheader("Erro por Safra (CV Temporal)")

    cv = load_temporal_cv()
    if cv:
        folds = pd.DataFrame(
            [
                {
                    "Safra": ano_para_safra(f["test_year"]),
                    "MAE (kg/ha)": f.get("metrics_combined", {}).get("mae_kg_ha"),
                }
                for f in cv.get("folds", [])
            ]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=folds["Safra"],
                y=folds["MAE (kg/ha)"],
                marker_color="#3498db",
                text=folds["MAE (kg/ha)"].apply(lambda x: f"{x:.0f}"),
                textposition="outside",
            )
        )
        fig.add_hline(
            y=500,
            line_dash="dash",
            line_color="orange",
            annotation_text="Limite Aceitavel (500 kg/ha)",
        )
        fig.update_layout(
            height=400,
            xaxis_title="Safra",
            yaxis_title="MAE (kg/ha)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Execute `python -m scripts.temporal_cv` para gerar o historico de erro.")


def main():
    """Aplicacao principal do dashboard."""
    st.set_page_config(
        page_title="Soja Produtividade",
        page_icon=":seedling:",
        layout="wide",
    )

    st.title(":seedling: Sistema de Previsao de Produtividade de Soja")

    pagina = st.sidebar.radio(
        "Navegacao",
        [
            "Visao Geral",
            "Validacao do Modelo",
            "Analise por Municipio",
            "Comparativo Regional",
            "Monitoramento",
        ],
    )

    render_model_info_sidebar()

    if pagina == "Visao Geral":
        page_visao_geral()
    elif pagina == "Validacao do Modelo":
        page_validacao()
    elif pagina == "Analise por Municipio":
        page_municipio()
    elif pagina == "Comparativo Regional":
        page_regional()
    elif pagina == "Monitoramento":
        page_monitoramento()


if __name__ == "__main__":
    main()
