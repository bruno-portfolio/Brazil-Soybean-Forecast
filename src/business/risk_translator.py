from dataclasses import dataclass

import pandas as pd


@dataclass
class CenarioFinanceiro:
    """Cenario financeiro para uma previsao."""

    nome: str
    produtividade_kg_ha: float
    produtividade_sacas_ha: float
    receita_ha: float
    custo_ha: float
    lucro_ha: float
    margem_percent: float


@dataclass
class AnaliseRisco:
    """Resultado da analise de risco para um municipio."""

    cod_ibge: int | None
    municipio: str | None
    uf: str
    ano: int

    pred_p10_kg_ha: float
    pred_p50_kg_ha: float
    pred_p90_kg_ha: float

    cenario_pessimista: CenarioFinanceiro
    cenario_base: CenarioFinanceiro
    cenario_otimista: CenarioFinanceiro

    probabilidade_quebra: float
    rating: str
    rating_descricao: str

    preco_saca: float
    custo_ha: float

    recomendacao: str
    spread_sugerido: float | None


CUSTOS_POR_UF = {
    "MT": 4850.0,
    "GO": 4650.0,
    "MS": 4700.0,
    "DF": 4600.0,
    "PR": 4200.0,
    "RS": 4100.0,
    "SC": 4150.0,
    "SP": 4400.0,
    "MG": 4300.0,
    "BA": 4500.0,
    "PI": 4400.0,
    "MA": 4350.0,
    "TO": 4450.0,
    "RO": 4200.0,
    "PA": 4300.0,
    "RR": 4250.0,
    "AC": 4200.0,
    "AM": 4300.0,
    "AP": 4350.0,
    "DEFAULT": 4400.0,
}

RATING_THRESHOLDS = {
    "A": (0.0, 0.10),
    "B": (0.10, 0.20),
    "C": (0.20, 0.35),
    "D": (0.35, 1.0),
}

RATING_DESCRICOES = {
    "A": "Risco Baixo - Margem de seguranca alta",
    "B": "Risco Moderado - Margem adequada",
    "C": "Risco Elevado - Monitorar de perto",
    "D": "Risco Alto - Requer garantias adicionais",
}

SPREAD_POR_RATING = {
    "A": 1.5,
    "B": 2.5,
    "C": 4.0,
    "D": 6.0,
}


class RiskTranslator:
    """Traduz previsoes de produtividade para metricas de risco financeiro."""

    def __init__(
        self,
        custos_por_uf: dict = None,
        preco_saca_default: float = 115.0,
    ):
        """Inicializa o tradutor de risco."""
        self.custos_por_uf = custos_por_uf or CUSTOS_POR_UF
        self.preco_saca_default = preco_saca_default

    def get_custo(self, uf: str) -> float:
        """Retorna custo de producao para uma UF."""
        return self.custos_por_uf.get(uf.upper(), self.custos_por_uf["DEFAULT"])

    def kg_para_sacas(self, kg_ha: float) -> float:
        """Converte kg/ha para sacas/ha (1 saca = 60 kg)."""
        return kg_ha / 60.0

    def calcular_receita(self, produtividade_kg_ha: float, preco_saca: float) -> float:
        """Calcula receita por hectare."""
        sacas_ha = self.kg_para_sacas(produtividade_kg_ha)
        return sacas_ha * preco_saca

    def calcular_cenario(
        self,
        nome: str,
        produtividade_kg_ha: float,
        preco_saca: float,
        custo_ha: float,
    ) -> CenarioFinanceiro:
        """Calcula metricas financeiras para um cenario."""
        sacas_ha = self.kg_para_sacas(produtividade_kg_ha)
        receita_ha = sacas_ha * preco_saca
        lucro_ha = receita_ha - custo_ha
        margem = (lucro_ha / receita_ha * 100) if receita_ha > 0 else 0

        return CenarioFinanceiro(
            nome=nome,
            produtividade_kg_ha=produtividade_kg_ha,
            produtividade_sacas_ha=sacas_ha,
            receita_ha=receita_ha,
            custo_ha=custo_ha,
            lucro_ha=lucro_ha,
            margem_percent=margem,
        )

    def estimar_probabilidade_quebra(
        self,
        pred_p10: float,
        pred_p50: float,
        pred_p90: float,
        custo_ha: float,
        preco_saca: float,
    ) -> float:
        """Estima probabilidade de quebra (receita < custo)."""
        breakeven_kg_ha = (custo_ha / preco_saca) * 60

        if breakeven_kg_ha <= pred_p10:
            return 0.05

        elif breakeven_kg_ha <= pred_p50:
            frac = (breakeven_kg_ha - pred_p10) / (pred_p50 - pred_p10)
            return 0.10 + frac * 0.40

        elif breakeven_kg_ha <= pred_p90:
            frac = (breakeven_kg_ha - pred_p50) / (pred_p90 - pred_p50)
            return 0.50 + frac * 0.40

        else:
            return 0.95

    def atribuir_rating(self, prob_quebra: float) -> tuple[str, str]:
        """Atribui rating de risco baseado na probabilidade de quebra."""
        for rating, (low, high) in RATING_THRESHOLDS.items():
            if low <= prob_quebra < high:
                return rating, RATING_DESCRICOES[rating]

        return "D", RATING_DESCRICOES["D"]

    def gerar_recomendacao(
        self,
        rating: str,
        cenario_pessimista: CenarioFinanceiro,
        uf: str,
    ) -> tuple[str, float]:
        """Gera recomendacao textual e spread sugerido."""
        spread = SPREAD_POR_RATING.get(rating, 6.0)

        if rating == "A":
            if cenario_pessimista.lucro_ha > 500:
                texto = (
                    f"APROVAR CREDITO - Margem de seguranca excelente mesmo no "
                    f"cenario pessimista (lucro R$ {cenario_pessimista.lucro_ha:.0f}/ha). "
                    f"Municipio com baixo risco climatico."
                )
            else:
                texto = (
                    f"APROVAR CREDITO - Baixo risco de quebra. " f"Spread sugerido: {spread}% a.a."
                )

        elif rating == "B":
            texto = (
                f"APROVAR COM CONDICOES - Risco moderado. "
                f"Verificar historico do produtor. "
                f"Spread sugerido: {spread}% a.a."
            )

        elif rating == "C":
            if uf in ["RS", "PR", "SC"]:
                texto = (
                    f"ANALISAR COM CAUTELA - Regiao Sul com alta volatilidade. "
                    f"Considerar garantias adicionais ou seguro agricola. "
                    f"Spread sugerido: {spread}% a.a."
                )
            else:
                texto = (
                    f"ANALISAR COM CAUTELA - Risco elevado. "
                    f"Exigir garantias adicionais. "
                    f"Spread sugerido: {spread}% a.a."
                )

        else:
            texto = (
                f"REJEITAR OU EXIGIR GARANTIAS FORTES - Alto risco de quebra. "
                f"Probabilidade significativa de receita menor que custo. "
                f"Se aprovar, spread minimo: {spread}% a.a. com garantias reais."
            )

        return texto, spread

    def calcular_risco(
        self,
        pred_p10: float,
        pred_p50: float,
        pred_p90: float,
        uf: str,
        ano: int = 2024,
        preco_saca: float = None,
        custo_ha: float = None,
        cod_ibge: int = None,
        municipio: str = None,
    ) -> AnaliseRisco:
        """Calcula analise de risco completa para um municipio."""
        preco = preco_saca or self.preco_saca_default
        custo = custo_ha or self.get_custo(uf)

        cenario_pessimista = self.calcular_cenario("pessimista", pred_p10, preco, custo)
        cenario_base = self.calcular_cenario("base", pred_p50, preco, custo)
        cenario_otimista = self.calcular_cenario("otimista", pred_p90, preco, custo)

        prob_quebra = self.estimar_probabilidade_quebra(pred_p10, pred_p50, pred_p90, custo, preco)

        rating, rating_descricao = self.atribuir_rating(prob_quebra)

        recomendacao, spread = self.gerar_recomendacao(rating, cenario_pessimista, uf)

        return AnaliseRisco(
            cod_ibge=cod_ibge,
            municipio=municipio,
            uf=uf,
            ano=ano,
            pred_p10_kg_ha=pred_p10,
            pred_p50_kg_ha=pred_p50,
            pred_p90_kg_ha=pred_p90,
            cenario_pessimista=cenario_pessimista,
            cenario_base=cenario_base,
            cenario_otimista=cenario_otimista,
            probabilidade_quebra=prob_quebra,
            rating=rating,
            rating_descricao=rating_descricao,
            preco_saca=preco,
            custo_ha=custo,
            recomendacao=recomendacao,
            spread_sugerido=spread,
        )

    def analisar_dataframe(
        self,
        df: pd.DataFrame,
        col_p10: str = "pred_p10_kg_ha",
        col_p50: str = "pred_p50_kg_ha",
        col_p90: str = "pred_p90_kg_ha",
        col_uf: str = "uf",
        col_ano: str = "ano",
        col_cod_ibge: str = "cod_ibge",
        col_municipio: str = "nome",
        preco_saca: float = None,
    ) -> pd.DataFrame:
        """Analisa risco para um DataFrame de previsoes."""
        resultados = []

        for _, row in df.iterrows():
            analise = self.calcular_risco(
                pred_p10=row[col_p10],
                pred_p50=row[col_p50],
                pred_p90=row[col_p90],
                uf=row[col_uf],
                ano=row.get(col_ano, 2024),
                preco_saca=preco_saca,
                cod_ibge=row.get(col_cod_ibge),
                municipio=row.get(col_municipio),
            )

            resultados.append(
                {
                    "cod_ibge": analise.cod_ibge,
                    "municipio": analise.municipio,
                    "uf": analise.uf,
                    "ano": analise.ano,
                    "pred_p10_kg_ha": analise.pred_p10_kg_ha,
                    "pred_p50_kg_ha": analise.pred_p50_kg_ha,
                    "pred_p90_kg_ha": analise.pred_p90_kg_ha,
                    "receita_pessimista": analise.cenario_pessimista.receita_ha,
                    "receita_base": analise.cenario_base.receita_ha,
                    "receita_otimista": analise.cenario_otimista.receita_ha,
                    "lucro_pessimista": analise.cenario_pessimista.lucro_ha,
                    "lucro_base": analise.cenario_base.lucro_ha,
                    "lucro_otimista": analise.cenario_otimista.lucro_ha,
                    "margem_base_pct": analise.cenario_base.margem_percent,
                    "prob_quebra": analise.probabilidade_quebra,
                    "rating": analise.rating,
                    "rating_descricao": analise.rating_descricao,
                    "spread_sugerido": analise.spread_sugerido,
                    "custo_ha": analise.custo_ha,
                    "preco_saca": analise.preco_saca,
                }
            )

        return pd.DataFrame(resultados)

    def gerar_relatorio_texto(self, analise: AnaliseRisco) -> str:
        """Gera relatorio de risco em formato texto."""
        sep = "=" * 70

        relatorio = f"""
{sep}
ANALISE DE RISCO - {analise.municipio or 'N/A'}/{analise.uf} - SAFRA {analise.ano}
{sep}

PREVISAO DE PRODUTIVIDADE:
  Cenario Pessimista (p10):   {analise.pred_p10_kg_ha:.0f} kg/ha ({analise.pred_p10_kg_ha/60:.1f} sc/ha)
  Cenario Base (p50):         {analise.pred_p50_kg_ha:.0f} kg/ha ({analise.pred_p50_kg_ha/60:.1f} sc/ha)
  Cenario Otimista (p90):     {analise.pred_p90_kg_ha:.0f} kg/ha ({analise.pred_p90_kg_ha/60:.1f} sc/ha)

ANALISE FINANCEIRA:
  Preco considerado:          R$ {analise.preco_saca:.2f}/saca
  Custo de producao:          R$ {analise.custo_ha:.2f}/ha

  | Cenario    | Receita/ha   | Lucro/ha     | Margem  |
  |------------|--------------|--------------|---------|
  | Pessimista | R$ {analise.cenario_pessimista.receita_ha:,.2f} | R$ {analise.cenario_pessimista.lucro_ha:,.2f} | {analise.cenario_pessimista.margem_percent:.1f}% |
  | Base       | R$ {analise.cenario_base.receita_ha:,.2f} | R$ {analise.cenario_base.lucro_ha:,.2f} | {analise.cenario_base.margem_percent:.1f}% |
  | Otimista   | R$ {analise.cenario_otimista.receita_ha:,.2f} | R$ {analise.cenario_otimista.lucro_ha:,.2f} | {analise.cenario_otimista.margem_percent:.1f}% |

RISCO DE QUEBRA:
  Probabilidade de Receita < Custo: {analise.probabilidade_quebra*100:.1f}%

  RATING: [{analise.rating}] {analise.rating_descricao}

RECOMENDACAO:
  {analise.recomendacao}

  Spread sugerido: {analise.spread_sugerido}% a.a.
{sep}
"""
        return relatorio


def main():
    """Exemplo de uso do RiskTranslator."""
    print("=" * 70)
    print("EXEMPLO DE USO DO RISK TRANSLATOR")
    print("=" * 70)

    translator = RiskTranslator()

    print("\n--- Exemplo 1: Sorriso/MT (Centro-Oeste, estavel) ---")
    analise_mt = translator.calcular_risco(
        pred_p10=3200,
        pred_p50=3450,
        pred_p90=3700,
        uf="MT",
        ano=2024,
        municipio="Sorriso",
        preco_saca=115.0,
    )
    print(translator.gerar_relatorio_texto(analise_mt))

    print("\n--- Exemplo 2: Nao-Me-Toque/RS (Sul, volatil) ---")
    analise_rs = translator.calcular_risco(
        pred_p10=1800,
        pred_p50=2400,
        pred_p90=3000,
        uf="RS",
        ano=2024,
        municipio="Nao-Me-Toque",
        preco_saca=115.0,
    )
    print(translator.gerar_relatorio_texto(analise_rs))


if __name__ == "__main__":
    main()
