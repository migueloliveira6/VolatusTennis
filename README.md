# VolatusTennis

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![GitHub Pages](https://img.shields.io/badge/Deploy-GitHub%20Pages-222222.svg)](https://pages.github.com/)
[![Daily Tennis Predictions](https://github.com/migueloliveira6/VolatusTennis/actions/workflows/predictions.yml/badge.svg)](https://github.com/migueloliveira6/VolatusTennis/actions/workflows/predictions.yml)

Um sistema automatizado de previsão de resultados de ténis utilizando Machine Learning, com pipeline CI/CD completa e interface web interativa.

## Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Pipeline CI/CD](#-pipeline-cicd)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [API e Dados](#-api-e-dados)
- [Contribuir](#-contribuir)
- [Contacto](#-contacto)

  
## Agradecimentos

- [TennisExplorer](https://www.tennisexplorer.com/) - Fonte de dados
- [XGBoost](https://xgboost.readthedocs.io/) - Framework de ML
- [Chart.js](https://www.chartjs.org/) - Visualizações
- [GitHub Actions](https://github.com/features/actions) - CI/CD
- Comunidade Python e Machine Learning
- [Repositório JeffSackman](https://github.com/JeffSackmann/tennis_atp) - Fonte de dados

## Sobre o Projeto

O **VolatusTennis** é um sistema completo de previsão de resultados de ténis que combina:

- **Machine Learning**: Modelo XGBoost com sistema de ELO dinâmico por superfície
- **Web Scraping**: Extração automática de dados de partidas e odds
- **Automação**: Pipeline CI/CD com GitHub Actions
- **Interface Web**: Dashboard interativo hospedado no GitHub Pages
- **Notificações**: Sistema de alertas via Telegram

### Características Principais

- ✅ Previsões diárias automáticas
- ✅ Sistema de ELO com decaimento temporal
- ✅ Análise por superfície (Hard, Clay, Grass)
- ✅ Cálculo de ROI e valor esperado
- ✅ Histórico Head-to-Head (H2H)
- ✅ Dashboard web responsivo
- ✅ Deploy automático via GitHub Pages

## Funcionalidades

### 1. Previsões Automáticas

- Scraping diário do site TennisExplorer
- Processamento de dados com validação de superfície
- Geração de previsões usando modelo treinado
- Exportação para CSV e JSON

### 2. Sistema de Rating ELO

- **ELO dinâmico por superfície**: Hard, Clay, Grass
- **K-Factor variável**: 
  - ATP: 32
  - Challenger: 20
- **Decaimento temporal**: Partidas mais antigas têm menor impacto
- **Fator mínimo**: 30% para jogos muito antigos

### 3. Interface Web

- Filtros por torneio e superfície
- Design responsivo (mobile-friendly)

### 4. Notificações Telegram

- Envio automático de previsões
- Formatação clara e concisa
- Suporte para mensagens longas (divisão automática)

## Tecnologias

### Backend

- **Python 3.8+**
  - pandas: Manipulação de dados
  - numpy: Operações numéricas
  - scikit-learn: Preprocessing e métricas
  - XGBoost: Modelo de ML
  - BeautifulSoup4: Web scraping
  - requests: Requisições HTTP
  - joblib: Serialização de modelos

### Frontend

- **HTML5/CSS3/JavaScript**
- **Chart.js**: Visualização de dados
- **Design Responsivo**: Mobile-first

### DevOps

- **GitHub Actions**: CI/CD
- **GitHub Pages**: Hosting
- **Python Virtual Environment**: Isolamento de dependências

## Arquitetura

```
┌─────────────────┐
│  GitHub Actions │
│   (Scheduler)   │
└────────┬────────┘
         │ Trigger diário (22:50 UTC)
         ▼
┌─────────────────┐
│  Scraping Bot   │
│  (Python)       │
└────────┬────────┘
         │ Extrai dados
         ▼
┌─────────────────┐
│ Data Processing │
│ (Pandas/NumPy)  │
└────────┬────────┘
         │ Normaliza e valida
         ▼
┌─────────────────┐
│  ML Model       │
│  (XGBoost+ELO)  │
└────────┬────────┘
         │ Gera previsões
         ▼
┌─────────────────┐
│  Export Data    │
│  (CSV/JSON)     │
└────────┬────────┘
         │ Salva em docs/predicts/
         ▼
┌─────────────────┐
│  GitHub Pages   │
│  (Web Deploy)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Telegram Bot    │
│ (Notifications) │
└─────────────────┘
```

## Instalação

### Pré-requisitos

- Python 3.12 ou superior
- pip (gestor de pacotes Python)
- Git

### Clonar o Repositório

```bash
git clone https://github.com/username/VolatusTennis.git
cd volatustennis
```

### Configurar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Paths
MODEL_PATH=models/
DATA_PATH=data/atp_chall_matches_2025_elo_temporal.csv
RAW_DATA_ATP_PATH=data/atp_matches_
RAW_DATA_CHALL_PATH=data/atp_matches_qual_chall_
DATA_2025_PATH=data/dataset_2025_normalizado.csv
OUTPUT_PATH=data/

# Model Parameters
K_FACTOR=32
INITIAL_ELO=1500

# Telegram (opcional)
TELEGRAM_BOT_TOKEN=seu_token_aqui
TELEGRAM_CHAT_ID=seu_chat_id_aqui
```

## Uso

### Executar Scraping e Previsões

```bash
# Executar notebook principal
jupyter notebook notebooks/scraping_bottelegram_optimized.ipynb
```

### Treinar Modelo

```bash
python src/model_elo_xgboost.py
```

### Visualizar Resultados

Abra o arquivo `docs/index.html` num browser ou aceda à versão online:

```
https://volatustennis.me
```

### 🔎 Comparar Previsões com Resultados ✅

Use o script `src/compare_predictions_results.py` para comparar as previsões geradas com os resultados reais e obter métricas simples (matched, accuracy, coverage).

Comportamento padrão:
- Se `--pred-file` não for fornecido, o script escolhe o ficheiro mais recente em `PREVISOES_PATH` (default: `previsoes/`).
- Se a data não for fornecida em `--results-date`, o script tenta extrair uma data do nome do ficheiro (formato YYYY-MM-DD). Caso não consiga, usa a data de ontem.
- Gera um CSV em `previsoes/comparacao_previsoes_YYYY-MM-DD.csv` com a correspondência entre previsão e resultado e flags `Matched` e `Correct`.

Exemplos:

```bash
# Comparar usando o ficheiro de previsões mais recente
python src/compare_predictions_results.py

# Comparar com um ficheiro específico e data explícita
python src/compare_predictions_results.py \
  --pred-file previsoes/previsoes_tenis_2025-07-01.csv \
  --results-date 2025-07-01
```

Saída:
- `comparacao_previsoes_YYYY-MM-DD.csv` — lista de previsões com colunas `Matched` (se foi encontrado jogo correspondente) e `Correct` (se o vencedor previsto corresponde ao vencedor real).


## 🔄 Pipeline CI/CD

O projeto utiliza GitHub Actions para automação completa:

### Workflow Principal

```yaml
name: NeuroTennis Daily Predictions

on:
  schedule:
    - cron: '0 23 * * *'  # Executa às 23:00 UTC diariamente
  workflow_dispatch:      # Permite execução manual

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - Checkout do código
      - Configurar Python 3.8
      - Instalar dependências
      - Executar scraping
      - Gerar previsões
      - Commit e push dos resultados
      - Deploy no GitHub Pages
      - Enviar notificações Telegram
```

### GitHub Pages

O deploy é automático após cada commit em `main`:

- **Branch**: `gh-pages` (ou `main/docs`)
- **Diretório**: `/docs`
- **URL**: `https://migueloliveira6.github.io/neurotennis/`

## 📁 Estrutura do Projeto

```
neurotennis/
├── .github/
│   └── workflows/
│       └── predict.yml          # GitHub Actions workflow
├── data/
│   ├── atp_matches_*.csv        # Dados históricos ATP
│   ├── atp_matches_qual_chall_*.csv
│   └── dataset_2025_normalizado.csv
├── docs/                        # GitHub Pages (Frontend)
│   ├── index.html               # Dashboard principal
│   ├── predicts/
│   │   ├── predictions.json     # Previsões atuais
│   │   ├── analytics.json       # Dados de análise
│   │   └── previsoes_*.csv      # Histórico
│   └── NeuroTennis.ico
├── models/
│   ├── tennis_surface_elo_model_xgboost.pkl
│   ├── tennis_surface_elo_scaler_xgboost.pkl
│   └── tennis_surface_elo_data_xgboost.pkl
├── notebooks/
│   ├── scraping_bottelegram_optimized.ipynb
│   └── atp_chall_matches_2025_elo_kfactor.ipynb
├── src/
│   ├── model_elo_xgboost.py     # Modelo principal
│   └── utils/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Modelo de Machine Learning

### Sistema de ELO

O sistema de rating utiliza uma implementação modificada do ELO:

```python
ELO_novo = ELO_antigo + K × (Resultado - Probabilidade_Esperada) × Fator_Temporal
```

Onde:
- **K**: Fator de ajuste (32 para ATP, 20 para Challenger)
- **Fator Temporal**: Decaimento exponencial baseado na data
- **Superfície**: ELO separado para Hard, Clay e Grass

### XGBoost Model

```python
Parâmetros otimizados:
- n_estimators: 1100
- learning_rate: 0.025
- max_depth: 5
- min_child_weight: 6
- subsample: 0.8
- colsample_bytree: 0.8
```

### Features Utilizadas

- ELO do jogador (por superfície)
- ELO do adversário (por superfície)
- Diferença de ELO
- Taxa de vitória H2H
- Número de confrontos H2H
- Taxa de vitória na superfície
- Número de partidas na superfície

### Métricas de Performance

```
Accuracy: ~65%
F1-Score: ~0.65
Log Loss: ~0.64
Brier Score: ~0.22
```

## API e Dados

### Formato JSON (predictions.json)

```json
[
  {
    "Torneio": "Shanghai",
    "Jogador 1": "Carlos Alcaraz",
    "Jogador 2": "Jannik Sinner",
    "Vencedor Previsto": "Carlos Alcaraz",
    "Confiança (%)": 67.3,
    "ELO Diff": 45.2,
    "H2H": "5-3 (62% para Carlos Alcaraz)",
    "Odd 1": 1.85,
    "Odd 2": 2.10,
    "Superfície": "Hard",
    "Valor Aposta": 0.125,
    "ROI Esperado (%)": 8.5
  }
]
```

### Formato CSV

O mesmo formato está disponível em CSV em `docs/predicts/previsoes_YYYY-MM-DD.csv`

## 🤝 Contribuir

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para a sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit as mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Guidelines

- Siga PEP 8 para código Python
- Adicione testes para novas features
- Atualize a documentação quando necessário
- Mantenha mensagens de commit claras e descritivas

## Contacto

LinkedIn: [LinkedIn](https://www.linkedin.com/in/luis-oliveira6)

Link do Projeto: [https://github.com/migueloliveira6/VolatusTennis](https://github.com/migueloliveira6/VolatusTennis)

Website: [https://migueloliveira6.github.io/VolatusTennis/](https://migueloliveira6.github.io/VolatusTennis/)

---

⭐ **Se este projeto te foi útil, considera dar uma estrela!** ⭐

## Clustering de Estilo de Jogo

Foi adicionado um pipeline separado para clustering de jogadores ATP por temporada, usando a base SQLite existente em `datasets/tennis_data.db`.

### O que faz

- Carrega as partidas diretamente da tabela `matches`
- Constrói observações no nível player-season
- Calcula features de serviço, retorno, superfície e agressividade
- Compara KMeans, GMM, clustering hierárquico e HDBSCAN
- Exporta tabelas, métricas e gráficos para `outputs/`

### Como executar

```bash
python main.py --db-path datasets/tennis_data.db --start-year 2024 --end-year 2024
```

### Saídas geradas

- `outputs/player_season_clusters.csv`
- `outputs/cluster_profiles.csv`
- `outputs/clustering_metrics.csv`
- `outputs/cluster_report.json`
- `outputs/figures/`

### Dependências extras

Para gerar visualizações completas, instale também:

- seaborn
- plotly
- umap-learn
- hdbscan








