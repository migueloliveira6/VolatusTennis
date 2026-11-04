# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VolatusTennis (formerly NeuroTennis): daily ATP/Challenger match predictions from an XGBoost model on surface-specific ELO features. It scrapes upcoming matches, predicts winners, compares yesterday's predictions to real results, and publishes static JSON to a GitHub Pages site (`docs/`, served at volatustennis.me). It can also send Telegram notifications. Code, comments, logs and data column names are mostly in **Portuguese** (e.g. `previsoes`, `Jogador 1`, `Vencedor Previsto`, `Confiança (%)`, `Superfície`). Keep that convention.

## Commands

```bash
pip install -r requirements.txt

# Tests (conftest.py puts src/ on sys.path, so modules are imported as `from db import ...`)
pytest -q
pytest tests/test_compare.py -q
pytest tests/test_db.py::test_name -q

# Daily pipeline, in the order the workflow runs it
python src/compare_predictions_results.py --pred-json-file <predictions_YYYY-MM-DD.json> --results-date YYYY-MM-DD
python src/engine.py                 # scrape tomorrow's matches + predict -> previsoes/previsoes_tenis_*.csv, Telegram
python src/generate_predictions.py   # latest previsoes CSV -> docs/predicts/predictions.json
python src/generate_analytics.py     # latest comparacao_previsoes_*.csv -> docs/analytics/*.json (incremental)

# Training
python src/train_model.py                       # the only training entrypoint: canonical + versioned pkl, models/manifest.json, logs/training_*.json
python src/train_model.py --start-year 2022 --trials 5 --model-dir <tmp>   # quick smoke run that doesn't touch models/
python src/model_elo_xgboost.py                 # load-only: prints a summary of the production model

# DB maintenance
python src/daily_update.py           # scrape yesterday's results, insert matches, update ELO in SQLite
python src/recompute_elo.py --source-db datasets/tennis_data_2026.db --output-db datasets/tennis_data_2026_v2.db   # recompute ELO on a copy with match_policy

# Player-style clustering (separate subsystem)
python main.py --db-path datasets/tennis_data.db --output-dir outputs
streamlit run streamlit_app.py
python scripts/export_clustering_json.py   # -> docs/analytics/clustering_*.json
```

Run scripts from the repo root. Many paths (`docs/`, `datasets/`, `previsoes/`, `notebooks/name_lookup.csv`) are relative to the current working directory.

## Architecture

**Data store**: `datasets/tennis_data.db` (SQLite; override with `DB_PATH`). `src/db.py` is the access layer: `load_matches_for_training`, `load_todays_matches`, `save_predictions`. It renames DB columns to what the model expects (`winner_surface_elo`, `loser_surface_elo`, ...). Historical source data is Jeff Sackmann's `tennis_atp` (`data/tennis_atp`). `src/rebuild_atp_db.py`, `atp_data_migration.py` and `migrate_csv_to_sqlite_FULL.py` build the DB.

**ELO**: `src/elo_calculator.py` holds surface-separated ELO (Hard/Clay/Grass) with a K-factor per tournament level (ATP 32, Challenger 20) and a temporal-decay factor (30% floor for old matches). `daily_update.py` applies it incrementally to new results. Special matches follow `src/match_policy.py` everywhere (ELO rebuild, `daily_update`, training):
- W/O and the Laver Cup (exhibition) are out of ELO and training.
- RET/DEF/abandoned matches update ELO with K×0.5 and count in history, but are not training examples.
- Davis Cup, Olympics, United/ATP Cup and qualifying count normally.

`recompute_elo.py` reapplies this to a copy of an existing DB. It keeps the scraped Jun–Oct 2026 rows that aren't in the Sackmann CSVs, which is why `atp_data_migration.py` (CSV rebuild) is not used for that.

**Model**: `TennisPredictor` in `src/model_elo_xgboost.py` is the central class. `load_data()` (from the DB) → `preprocess_data()` walks matches chronologically and builds point-in-time player history, H2H, surface stats and profiles, so features never leak future data → `train_model()` (XGBoost + Optuna, `TimeSeriesSplit`, fixed 0.5 threshold). It evaluates on the most recent 15%, then refits on 100% of the data and returns a metrics dict → `save_model()` / `load_saved_model()`. `predict_match(p1, p2, surface, date, tourney_level)` serves predictions. Probabilities are calibrated (Platt or isotonic, chosen on a temporal calibration block in `src/evaluation.py`, stored as `calibrator` in the data pkl and applied in `predict_match`). Each training run reports baselines (ELO-only, rank points, higher-ranked) and ATP-only metrics on the same holdout. Bet sizing in `engine.calcular_stake_kelly` is fractional Kelly (`KELLY_FRACTION`, `MIN_EDGE`, `MAX_STAKE` env, defaults 0.25 / 0.02 / 0.05). Training and prediction must share `_build_feature_row()` and `_to_model_matrix()` (one-hot of `surface` + `tourney_level`); `tests/test_predict_alignment.py` guards this. Artifacts live in `MODEL_PATH` (default `models/`) as `tennis_surface_elo_{model,data}_xgboost.pkl`. The `_data` pickle holds the player-state dicts needed at inference. `engine.py` only loads the model and exits with an error if it is missing; training is `src/train_model.py`. **Never run `engine.py` to test changes**: it scrapes and sends Telegram messages. Other `model_elo_*.py` / `train_model3.py` / `first_model.py` files are experiments or variants (WTA, random forest, ensemble), not the production path.

**Scraping & orchestration**: `src/engine.py` is the production entrypoint. It has `NeuroTennis` (TennisExplorer scraper, grouped by tournament), `SuperficieDetector` (infers surface), `TennisPredicaoAnalyzer` (runs predictions, EV/ROI from odds), Telegram helpers, and `salvar_status_site`. When there are no matches it writes `docs/predicts/status.json` with `status: "no_matches"`, and `generate_predictions.py` then publishes an empty list. `scraper_results.py` scrapes finished results. `tennis_scraper_2026.py` scrapes TennisAbstract tournament pages listed in `src/tennisabstract links tourn 26.txt`.

**Evaluation loop**: `compare_predictions_results.py` matches predictions to actual results by normalized player names and writes `previsoes/comparacao_previsoes_YYYY-MM-DD.csv` (`Matched`, `Correct`). `generate_analytics.py` merges it into the cumulative `docs/analytics/*.json`. `analytics_state.json` records which comparisons were already counted, so it doesn't double-count.

**Frontend**: `docs/index.html` is a single static page (Chart.js) that fetches `predicts/*.json`, `analytics/*.json` and `data/*.json` (the last from `src/extract_features_v3.py`, `src/DR_top10.py`). Changing a JSON shape means updating the matching `fetch` consumer in `index.html`.

**Clustering**: `clustering/style_clustering/` is a self-contained package (config → data_loader → feature_engineering → preprocessing → clustering (KMeans/HDBSCAN + stability) → visualization), driven by `main.py`, `streamlit_app.py` and `scripts/export_clustering_json.py`.

## CI (`.github/workflows/`)

- `predictions.yml` is the live pipeline. It runs on `repository_dispatch` type `run-daily-predictions` (triggered externally, not by cron), on Python 3.10. Cumulative state is **not** stored in `main`: the job checks out the `gh-pages` branch, restores `analytics/*.json` and `predicts/history/` from it, runs compare → engine → generate_predictions → generate_analytics, snapshots `docs/predicts/history/predictions_<date>.json` (input for the next day's compare), and deploys `./docs` to `gh-pages`. Locally committed `docs/analytics` / `docs/predicts` files are not the source of truth.
- `predictions_latest.yml` is disabled (`on: {}`). It is a planned DB-based pipeline: pytest → `daily_update.py` → `train_model.py` → `predict_today.py` → `generate_predictions.py`, with `PYTHONPATH=src`. It references `src/bot_telegram_git.py`, which does not exist.

Env vars/secrets used: `TOKEN_BOT`, `CHAT_ID`, `MODEL_PATH`, `PREVISOES_PATH`, `NAME_LOOKUP`, `PREDICTIONS_JSON_URL`, `DB_PATH`. Locally they come from `.env` via python-dotenv.

## Gotchas

- `.gitignore` ignores `notebooks/*`, `/data/`, `/datasets/`, `*.db` and all `models/*.pkl` except the two canonical ones, yet some notebooks are already tracked. The SQLite DB is >100 MB and stays out of git.
- Imports are inconsistent: some modules use `from db import ...` (needs `src/` on the path), and others fall back to `from src.db import ...`. Running `python src/<script>.py` from the root works; importing across modules from elsewhere may need `PYTHONPATH=src`.
- Several scripts (`generate_predictions.py`, `generate_analytics.py`, `extract_features_v3.py`) run their logic at module top level instead of inside `main()`, so importing them has side effects.
