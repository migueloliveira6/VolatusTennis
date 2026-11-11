# VolatusTennis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Pages](https://img.shields.io/badge/Deploy-GitHub%20Pages-222222.svg)](https://volatustennis.me)
[![Daily Tennis Predictions](https://github.com/migueloliveira6/VolatusTennis/actions/workflows/predictions.yml/badge.svg)](https://github.com/migueloliveira6/VolatusTennis/actions/workflows/predictions.yml)

Previsões diárias de jogos de ténis ATP com machine learning: probabilidades calibradas, ELO por superfície, valor das odds e histórico público de acertos.

🌐 **Site:** [volatustennis.me](https://volatustennis.me)

## Índice

- [Sobre o projeto](#sobre-o-projeto)
- [O site](#o-site)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [Modelo de machine learning](#modelo-de-machine-learning)
- [Pipeline diária (CI/CD)](#pipeline-diária-cicd)
- [Formatos de dados](#formatos-de-dados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Clustering de estilos de jogo](#clustering-de-estilos-de-jogo)
- [Contribuir](#contribuir)
- [Agradecimentos e contacto](#agradecimentos-e-contacto)

## Sobre o projeto

O **VolatusTennis** prevê o vencedor dos jogos ATP do dia seguinte e publica as previsões num site estático. Combina:

- **Modelo XGBoost** treinado com jogos ATP, Challenger e qualifying desde 2000, com ELO por superfície calculado dentro do modelo, ranking, forma recente, confrontos diretos e estatísticas de serviço/resposta.
- **Probabilidades calibradas** e **apostas com Kelly fracionado**: o site mostra a probabilidade de cada jogador, se as odds têm valor e o stake sugerido em % da banca.
- **Pipeline automática** em GitHub Actions: compara as previsões de ontem com os resultados reais, faz scraping dos jogos de amanhã, gera previsões e publica no GitHub Pages.
- **Histórico público**: cada previsão avaliada fica registada com o resultado, as odds e o lucro da aposta sugerida.
- **Notificações no Telegram** com as previsões e o resumo do dia anterior.

> As previsões são informação estatística e não aconselhamento de apostas.

## O site

| Separador | Conteúdo |
|---|---|
| **Previsões** | Jogos de amanhã por torneio: vencedor previsto, probabilidade, diferença de ELO, H2H, odds, stake sugerido e ROI esperado. Filtros por torneio e "Só com valor". |
| **Estatísticas** | Taxa de acerto global, evolução mensal, acerto por faixa de confiança, Top 10 por superfície e evolução do ELO. |
| **Histórico** | Cada previsão avaliada, jogo a jogo, com filtros (período, torneio, superfície, confiança, jogador, só apostas), ROI e lucro das apostas sugeridas, curva da banca, maior queda e gráfico de calibração (confiança vs. acerto real). |
| **Clustering** | Estilos de jogo dos jogadores por temporada: mapa 2D, distribuição por cluster, evolução e radar do perfil, com pesquisa por jogador. |

O site é uma página única (`docs/index.html`) em HTML, CSS e JavaScript com [Chart.js](https://www.chartjs.org/). Funciona em telemóvel e com teclado e leitores de ecrã (separadores e pesquisa com WAI-ARIA).

### Instalar no telemóvel

O site instala-se como uma app (PWA), com ícone no ecrã principal, ecrã inteiro e funcionamento offline com os últimos dados guardados:

- **Android:** abrir o site no Chrome e tocar em **Instalar app** (no site ou no menu do Chrome).
- **iPhone:** abrir o site no Safari, tocar em **Partilhar** e escolher **Adicionar ao ecrã principal**. O botão **Instalar app** do site mostra estes passos.

Os avisos diários das previsões chegam pelo canal do Telegram (botão **Telegram** no site).

## Arquitetura

```
 ┌────────────── Atualização manual dos dados ──────────────┐
 │                                                           │
 │  TennisAbstract (Sackmann) ── tennis_scraper_2026.py      │
 │        │                                                  │
 │        ▼                                                  │
 │  notebook clean_player_name ── CSV normalizado            │
 │        │                                                  │
 │        ▼                                                  │
 │  update_from_csv.py ──► BD SQLite (datasets/*.db)         │
 │                              │                            │
 │                              ▼                            │
 │                     train_model.py ──► models/*.pkl ──► commit
 └───────────────────────────────────────────────────────────┘

 ┌────────────── Pipeline diária (GitHub Actions) ──────────┐
 │  cron-job.org ── repository_dispatch                      │
 │        ▼                                                  │
 │  restaura estado do gh-pages (analytics + histórico)      │
 │        ▼                                                  │
 │  compare_predictions_results.py  (previsões de ontem      │
 │        │                          vs. resultados reais)   │
 │        ▼                                                  │
 │  engine.py  (TennisExplorer: jogos + odds de amanhã       │
 │        │     → previsões → Telegram)                      │
 │        ▼                                                  │
 │  generate_predictions.py · generate_analytics.py          │
 │        ▼                                                  │
 │  deploy de docs/ para o gh-pages ──► volatustennis.me     │
 └───────────────────────────────────────────────────────────┘
```

A base de dados SQLite tem mais de 100 MB e **não está no git**. O modelo treinado (`models/*.pkl`, cerca de 19 MB) está, e é o que a pipeline diária usa.

## Instalação

### Pré-requisitos

- Python 3.10 ou superior (o CI usa 3.10)
- Git

### Passos

```bash
git clone https://github.com/migueloliveira6/VolatusTennis.git
cd VolatusTennis
python -m venv .venv
```

Ativar o ambiente virtual:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um ficheiro `.env` na raiz (não vai para o git):

| Variável | Para quê | Default |
|---|---|---|
| `DB_PATH` | BD SQLite com os jogos | `datasets/tennis_data.db` (o `update_from_csv.py` usa `tennis_data_2026_v3.db`) |
| `MODEL_PATH` | Pasta dos modelos | `models` |
| `PREVISOES_PATH` | Pasta dos CSV de previsões e comparações | `previsoes` |
| `CSV_PATH` | CSV de jogos novos para o `update_from_csv.py` | `datasets/dataset_2026_normalizado_v5.csv` |
| `NAME_LOOKUP` | Mapeamento de nomes TennisExplorer → Sackmann | `notebooks/name_lookup.csv` |
| `TOKEN_BOT`, `CHAT_ID` | Bot do Telegram (opcional) | — |
| `PREDICTIONS_JSON_URL` | `predictions.json` publicado (fallback do compare) | — |
| `KELLY_FRACTION`, `MIN_EDGE`, `MAX_STAKE` | Gestão de banca | `0.25`, `0.02`, `0.05` |
| `OPTUNA_TRIALS` | Nº de tentativas de afinação no treino | `10` |

Exemplo:

```env
DB_PATH=datasets/tennis_data_2026_v4.db
MODEL_PATH=models
PREVISOES_PATH=previsoes
TOKEN_BOT=o_token_do_bot
CHAT_ID=o_chat_id
```

## Uso

> ⚠️ O `src/engine.py` faz scraping e **envia mensagens para o Telegram**. Não o use para testar alterações; para isso use os testes ou `predict_match` diretamente.

### Atualizar a base de dados com jogos novos

O Sackmann publica os resultados dos torneios já terminados no TennisAbstract. Depois disso:

1. Acrescentar os links dos torneios a `src/tennisabstract links tourn 26.txt` e o calendário (data e superfície) em `src/tennis_scraper_2026.py`, e correr o scraper.
2. Normalizar os nomes dos jogadores com o notebook `notebooks/clean_player_name.ipynb`, que gera o CSV normalizado.
3. Inserir os jogos novos na BD (primeiro em modo de teste):

```bash
python src/update_from_csv.py --dry-run
```

```bash
python src/update_from_csv.py
```

O script só insere jogos que ainda não existem (mesmo vencedor, perdedor e ronda no mesmo torneio ou data), por isso pode ser corrido várias vezes.

Para recalcular as colunas de ELO da BD (por exemplo, depois de entrarem torneios atrasados), use uma cópia (a BD original não é alterada):

```bash
python src/recompute_elo.py --source-db datasets/tennis_data_2026_v3.db --output-db datasets/tennis_data_2026_v4.db
```

### Treinar o modelo

```bash
python src/train_model.py
```

Treina com os jogos de 2000 até ao presente (cerca de 15 minutos). No fim:
- substitui o modelo em `models/`;
- regista as métricas em `models/manifest.json` e `metrics/metrics_history.csv`.

Depois é só fazer commit dos dois `.pkl` canónicos para a pipeline diária os usar.

Opções úteis:

```bash
# treino rápido de teste, sem tocar em models/
python src/train_model.py --start-year 2022 --trials 5 --model-dir /tmp/modelo_teste

# ver o resumo do modelo de produção
python src/model_elo_xgboost.py
```

### Comparar previsões com resultados

```bash
python src/compare_predictions_results.py --pred-json-file predictions_2026-11-05.json --results-date 2026-11-05
```

O script gera `previsoes/comparacao_previsoes_<data>.csv` com o acerto, as odds, o stake e o lucro de cada previsão. Depois, `python src/generate_analytics.py` atualiza os JSON do site (estatísticas e histórico).

### Testes

```bash
pytest -q
```

### Ver o site localmente

```bash
cd docs
python -m http.server 8000
```

Depois abra `http://localhost:8000`.

## Modelo de machine learning

### ELO por superfície

O ELO é calculado dentro do modelo, numa passagem cronológica pelos jogos (data → torneio → ronda), com uma classificação global e uma por superfície:

```
ELO_novo = ELO_antigo + K × (resultado − probabilidade_esperada)
K = 250 / (n + 5)^0.4 × multiplicador
```

- `n` é o nº de jogos que o jogador já disputou. Jogadores novos sobem e descem depressa; veteranos estabilizam.
- O multiplicador vale 1.1 em Grand Slams e 0.85 em Challengers e qualifying.
- Não há decaimento por ano civil.

### Jogos especiais

| Tipo | ELO | Exemplo de treino |
|---|---|---|
| Walkover (W/O) | não conta | não |
| Abandono (RET), desclassificação, jogo interrompido | K × 0.5 | não |
| Laver Cup (exibição) | não conta | não |
| Davis Cup, Jogos Olímpicos, United/ATP Cup, qualifying | normal | sim |

### Features

Todas as features usam apenas jogos anteriores ao jogo a prever:

| Grupo | Features |
|---|---|
| ELO | ELO global e por superfície, combinação dos dois, probabilidade ELO, nº de jogos |
| Ranking | ranking (log), pontos, idade |
| Forma | vitórias nos últimos 5/10/25/50 jogos, taxa de vitória na superfície |
| H2H | taxa de vitória e nº de confrontos diretos |
| Serviço/resposta | médias móveis exponenciais de pontos ganhos no serviço e na resposta (ajustadas à qualidade do adversário), aces e duplas faltas |
| Contexto | superfície, nível do torneio, melhor de 3/5 sets, qualifying |

Existem também features de **momentum** (variação do ELO em 30/90 dias) e **fadiga** (dias de descanso, jogos e minutos recentes). Estão desligadas por defeito porque, em produção, o estado dos jogadores fica congelado na data do último treino. Ativam-se com `--exclude-groups` sem valores.

### Treino, avaliação e calibração

- XGBoost com afinação de hiperparâmetros (Optuna + validação cruzada temporal).
- Avaliação nos 15% de jogos mais recentes, nunca vistos no treino. Depois, novo treino com 100% dos dados.
- **Previsão simétrica:** média de P(A vence B) e 1 − P(B vence A).
- **Calibração** das probabilidades (Platt ou isotónica), escolhida num bloco temporal próprio.

Resultados no holdout (cerca de 39 mil jogos completos, ago/2023 – out/2026):

| | Log loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| **VolatusTennis** | **0.604** | **0.209** | **66.8%** | **0.733** |
| Só ELO | 0.625 | 0.218 | 65.0% | 0.707 |
| Pontos de ranking | 0.658 | 0.229 | 62.7% | 0.681 |
| Mais bem classificado ganha | — | — | 63.5% | — |

Só jogos ATP: log loss 0.600, accuracy 66.7%.

### Valor das apostas

O "Stake sugerido" é calculado com **¼ do critério de Kelly**:
- só há aposta quando a probabilidade do modelo supera a implícita na odd em pelo menos 2 pontos percentuais;
- o stake nunca passa de 5% da banca.

Estes valores são configuráveis no `.env`.

## Pipeline diária (CI/CD)

O workflow `.github/workflows/predictions.yml` corre todos os dias. É disparado pelo [cron-job.org](https://cron-job.org) através de um `repository_dispatch` (`run-daily-predictions`).

1. Restaura do branch `gh-pages` o estado acumulado: `docs/analytics/*.json` (estatísticas e histórico) e os snapshots `docs/predicts/history/`.
2. **Compara** as previsões de ontem (snapshot do dia) com os resultados reais.
3. **Faz scraping** dos jogos e odds de amanhã no TennisExplorer, **gera as previsões** com o modelo em `models/` e **envia-as para o Telegram**.
4. Gera `docs/predicts/predictions.json` e atualiza as estatísticas e o `track_record.json`.
5. Guarda o snapshot do dia e publica `docs/` no `gh-pages`, que serve [volatustennis.me](https://volatustennis.me).

O estado do site vive no `gh-pages`. Os JSON de `docs/analytics` e `docs/predicts` no `main` não são a fonte de verdade.

## Formatos de dados

### `docs/predicts/predictions.json`

```json
[
  {
    "Torneio": "Paris",
    "Jogador 1": "Jannik Sinner",
    "Jogador 2": "Ben Shelton",
    "Vencedor Previsto": "Jannik Sinner",
    "Confiança (%)": 74.2,
    "ELO Diff": 210.5,
    "H2H": "5-1 (83% para Jannik Sinner)",
    "Odd 1": 1.35,
    "Odd 2": 3.2,
    "Superfície": "Hard",
    "Valor Aposta": 0.012,
    "ROI Esperado (%)": 0.2
  }
]
```

`Valor Aposta` é o stake sugerido como fração da banca (0.012 = 1,2%). O CSV equivalente fica em `previsoes/previsoes_tenis_<data>.csv`.

### `docs/analytics/track_record.json`

Uma linha por previsão avaliada (alimenta o separador Histórico):

```json
{"date": "2026-11-05", "tourney": "Paris", "surface": "Hard", "p1": "Jannik Sinner", "p2": "Ben Shelton",
 "pick": "Jannik Sinner", "winner": "Jannik Sinner", "score": "6-4 7-6(3)", "conf": 74.2,
 "correct": true, "odd": 1.35, "stake": 0.012, "profit": 0.0042}
```

## Estrutura do projeto

```
VolatusTennis/
├── .github/workflows/
│   └── predictions.yml         # pipeline diária
├── docs/                       # site (GitHub Pages)
│   ├── index.html
│   ├── css/ · js/              # estilos e lógica do site
│   ├── sw.js                   # service worker (app instalável e offline)
│   ├── img/                    # logo, ícones, imagem de partilha
│   ├── predicts/               # previsões do dia + snapshots
│   ├── analytics/              # estatísticas, histórico, clustering
│   └── data/                   # top 10 e evolução do ELO
├── models/                     # modelo de produção (+ manifest.json)
├── src/
│   ├── engine.py               # produção: scraping → previsões → Telegram
│   ├── model_elo_xgboost.py    # TennisPredictor (treino e previsão)
│   ├── features.py             # estado dos jogadores e features
│   ├── evaluation.py           # métricas, baselines, calibração
│   ├── train_model.py          # ponto único de treino
│   ├── elo_calculator.py       # ELO com K dinâmico
│   ├── match_policy.py         # walkovers, abandonos, exibições, ordem das rondas
│   ├── db.py                   # acesso à BD SQLite
│   ├── update_from_csv.py      # inserir jogos novos na BD
│   ├── recompute_elo.py        # recalcular o ELO numa cópia da BD
│   ├── tennis_scraper_2026.py  # scraping TennisAbstract
│   ├── compare_predictions_results.py
│   ├── generate_predictions.py
│   ├── generate_analytics.py
│   └── track_record.py         # histórico jogo a jogo
├── clustering/style_clustering/ # clustering de estilos de jogo
├── notebooks/                  # exploração e limpeza de nomes
├── tests/                      # pytest
├── main.py                     # CLI do clustering
├── streamlit_app.py            # explorador do clustering
└── requirements.txt
```

## Clustering de estilos de jogo

Pipeline separada que agrupa jogadores ATP por temporada segundo o estilo de jogo (serviço, resposta, superfície, agressividade). Compara KMeans, GMM, clustering hierárquico e HDBSCAN.

```bash
python main.py --db-path datasets/tennis_data.db --start-year 2024 --end-year 2024
```

```bash
python scripts/export_clustering_json.py   # gera os JSON do separador Clustering
```

```bash
streamlit run streamlit_app.py             # explorador interativo
```

As saídas ficam em `outputs/`: `player_season_clusters.csv`, `cluster_profiles.csv`, `clustering_metrics.csv`, `cluster_report.json` e `figures/`.

## Contribuir

1. Faça fork do projeto e crie um branch (`git checkout -b feature/nova-funcionalidade`).
2. Corra os testes (`pytest -q`) antes de submeter.
3. Abra um Pull Request a descrever a alteração.

O código, os comentários e as colunas de dados estão em português; mantenha essa convenção.

## Agradecimentos e contacto

- [Jeff Sackmann / tennis_atp](https://github.com/JeffSackmann/tennis_atp) e [Tennis Abstract](https://www.tennisabstract.com/): dados históricos ATP
- [TennisExplorer](https://www.tennisexplorer.com/): jogos e odds do dia
- [XGBoost](https://xgboost.readthedocs.io/), [Optuna](https://optuna.org/), [scikit-learn](https://scikit-learn.org/), [Chart.js](https://www.chartjs.org/)

**LinkedIn:** [luis-oliveira6](https://www.linkedin.com/in/luis-oliveira6) · **Site:** [volatustennis.me](https://volatustennis.me) · **Repositório:** [github.com/migueloliveira6/VolatusTennis](https://github.com/migueloliveira6/VolatusTennis)

⭐ Se este projeto te foi útil, considera dar uma estrela!
