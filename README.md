# 🌐 GeoQuant – Institutional Macro Research Terminal
### *Advanced Geopolitical Risk Modeling & Volatility Forecasting Architecture*
**GeoQuant** é um terminal de pesquisa macro-quantitativa de nível institucional projetado para extrair, modelar e simular o impacto de choques geopolíticos e macroeconômicos estruturais nos preços e na volatilidade de commodities globais (foco em WTI, Brent, Metais e Grãos).
A plataforma integra modelos avançados de econometria financeira para capturar caudas pesadas (*tail risk*) e spillovers de volatilidade exógena, fornecendo uma infraestrutura completa de gerenciamento de risco e tomada de decisão para *Hedge Funds*, *Asset Managers* e mesas de *Prop Trading*.
## 🔬 Arquitetura Quantitativa & Modelagem
O core matemático do terminal opera através de uma abordagem híbrida em quatro estágios principais:
### 1. Filtragem de Volatilidade: GARCH-X com Encolhimento Bayesiano
 * **GARCH-X(1,1):** A volatilidade condicional é estimada utilizando o **GeoFactor** (um índice proprietário de estresse geopolítico e macro composto por variáveis de spreads, moedas e commodities agrícolas) como variável exógena (X). Isso permite isolar componentes endógenos de mercado de choques exógenos reais.
 * **Bayes Shrinkage:** Para evitar *overfitting* e instabilidade em janelas curtas, as volatilidades estimadas sofrem um encolhimento bayesiano (*shrinkage*) contra um *prior* histórico de longo prazo ponderado pela intensidade do regime atual.
### 2. Dinâmica de Correlação Multivariada: DCC-GARCH Corrigido
 * A evolução das correlações entre os ativos (ex: WTI vs Brent) é modelada via **DCC (Dynamic Conditional Correlation)**, capturando a quebra de correlações históricas e o aumento de cointegração em momentos de estresse sistêmico.
### 3. Teoria do Valor Extremo (EVT) para Caudas Pesadas
 * Os resíduos padronizados do modelo passam por um mapeamento de cauda utilizando a distribuição **GPD (Generalized Pareto Distribution)** acima de limiares ótimos. Isso garante que o cálculo do *Value at Risk* (VaR) e do *Expected Shortfall* (ES) respeite a assimetria e a curtose excessiva do mercado de energia.
### 4. Motor de Simulação Monte Carlo Projetivo
 * **Jump-Diffusion Process:** A simulação de múltiplos caminhos (*paths*) para horizontes de até 30 dias integra processos de difusão com saltos assimétricos (*Poisson Jumps*) calibrados dinamicamente com base no regime de estresse atual e no comportamento de commodities de inteligência geopolítica (como fertilizantes, uréia e DAP).
## 📊 Estrutura do Terminal (Módulos)
O dashboard é organizado em 8 abas funcionais sob a filosofia visual *Clean Minimalist*:
 * **Market & Risk:** Visão em tempo real de preços, superfícies de volatilidade estendida e o monitor de *Black Swans* no mercado de fertilizantes via dados CRU/Green Markets.
 * **Geopolitical:** Evolução do *GeoFactor* normatizado e comportamento de grãos (*Wheat, Corn, Soy*) indexados a partir de datas-marco de conflitos.
 * **Monte Carlo:** Gráficos de leque (*Fan Charts*) probabilísticos de projeção de preço, cálculo de VaR 95%, CVaR (Expected Shortfall) e densidade de probabilidade de cenários de estresse ($40 ou $150).
 * **Quant Stats:** Análise estatística de momentos superiores (Skewness, Kurtosis) e persistência de parâmetros DCC (\alpha, \beta).
 * **Macro & Corr:** Matrizes de correlação dinâmica e o *Composite Stress Index* global.
 * **Institutional Backtest:** Framework rigoroso de validação regulatória do modelo de risco contendo os testes de **Kupiec** (cobertura incondicional), **Christoffersen** (cobertura condicional), **Dynamic Quantile (DQ)** e a estatística Z de Acerbi para ES.
 * **Advanced Analytics:** Parâmetros calibrados de EVT e diagnósticos de resíduos (Ljung-Box e ARCH-LM).
 * **Executive Dashboard:** Resumo executivo contendo o *Information Coefficient* (IC) do GeoFactor e contribuições marginais de atributos via penalização LASSO.
## 🛠️ Tecnologias Utilizadas
 * **Core:** Python 3.9+
 * **Interface:** Streamlit
 * **Econometria & Math:** arch_model (ARCH/GARCH), scipy.stats (GPD, fitting), statsmodels (VAR, Ljung-Box, Logit)
 * **Machine Learning:** scikit-learn (LassoCV para pesos dinâmicos)
 * **Data Ingestion:** yfinance para dados de mercado, PchipInterpolator para dados macro espaçados.
 * **Visualização:** Plotly Graph Objects (Engine de alta performance com subplots de eixos secundários interativos).
## 💻 Instalação e Execução
### Pré-requisitos
Certifique-se de ter o Python instalado (v3.9 ou superior recomendado).
 1. **Clone o repositório:**
   ```bash
   
   ```
git clone https://github.com/seu-usuario/geoquant.git
cd geoquant
```

2. **Instale as dependências:**
   ```bash
pip install -r requirements.txt

```
 3. **Execute o Terminal Streamlit:**
   ```bash
   
   ```
streamlit run app.py
```

---

## 📋 Arquivo `requirements.txt` necessário

Para garantir o funcionamento perfeito do motor econométrico, o seu ambiente deve conter:
```text
streamlit>=1.30.0
numpy>=1.22.0
pandas>=1.4.0
plotly>=5.10.0
scipy>=1.8.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
yfinance>=0.2.0
arch>=5.3.0
pytz

```
## 📄 Licença
Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
## 👨‍💻 Autor
 * **Eduardo Moraes** - *Quant Data Scientist & Economics* - GitHub / LinkedIn
> **Disclaimer Institucional:** Este terminal foi desenvolvido estritamente para fins de pesquisa quantitativa e educacional. Backtests e simulações de cenários passados ou futuros não garantem retornos reais. Nenhuma parte deste código ou das saídas do dashboard constitui recomendação de investimento.
> 
