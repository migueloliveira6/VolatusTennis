#!/usr/bin/env python3
"""Treina o modelo XGBoost a partir da BD SQLite e grava o modelo canónico + versão + manifest.

É o único ponto de treino: `engine.py` e `model_elo_xgboost.py` apenas carregam o modelo.

Usage:
    python src/train_model.py
    python src/train_model.py --db ./datasets/tennis_data.db --start-year 2000 --trials 20
    python src/train_model.py --model-dir /tmp/models_teste   # não toca em models/
"""
import os
import sys
import json
import time
import argparse
import joblib
import logging
import subprocess
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_elo_xgboost import TennisPredictor, MODEL_PATH, MODEL_FILENAME, DATA_FILENAME

DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2026
LOG_DIR = 'logs'
METRICS_DIR = 'metrics'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_model')


def _get_short_git_hash():
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def time_split(df_in, train_frac=0.7, val_frac=0.15):
    """Divide por ordem cronológica em treino/validação/teste (sem baralhar)."""
    df_sorted = df_in.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    train = df_sorted.iloc[:train_end].reset_index(drop=True)
    val = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test = df_sorted.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def build_parser():
    parser = argparse.ArgumentParser(description='Treina o modelo de previsão de ténis (XGBoost + ELO)')
    parser.add_argument('--db', default=None, help='Caminho da BD SQLite (default: DB_PATH ou datasets/tennis_data.db)')
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR)
    parser.add_argument('--end-year', type=int, default=DEFAULT_END_YEAR)
    parser.add_argument('--trials', type=int, default=None, help='Nº de trials Optuna (default: OPTUNA_TRIALS ou 10)')
    parser.add_argument('--model-dir', default=MODEL_PATH, help='Pasta de saída dos modelos (default: MODEL_PATH)')
    parser.add_argument('--no-refit', action='store_true', help='Não refazer o fit final com 100%% dos dados')
    return parser


def _update_manifest(model_dir, entry):
    manifest_path = os.path.join(model_dir, 'manifest.json')
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Manifest ilegível, a recriar: {e}")
    manifest.insert(0, entry)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return manifest_path


def _append_metrics_history(entry):
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_csv = os.path.join(METRICS_DIR, 'metrics_history.csv')
    metrics = entry['metrics']
    row = pd.DataFrame([{
        'timestamp': entry['created_at'],
        'model_file': entry['model_file'],
        'git': entry['git'],
        'auc': metrics.get('auc'),
        'accuracy': metrics.get('accuracy'),
        'f1': metrics.get('f1'),
        'logloss': metrics.get('logloss'),
        'brier': metrics.get('brier'),
        'train_time_seconds': metrics.get('train_time_seconds'),
        'n_rows': entry['n_rows'],
        'min_date': entry['train_period']['min_date'],
        'max_date': entry['train_period']['max_date'],
    }])
    row.to_csv(metrics_csv, mode='a', header=not os.path.exists(metrics_csv), index=False)
    return metrics_csv


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.trials is not None:
        os.environ['OPTUNA_TRIALS'] = str(args.trials)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'training_{timestamp}.log'), encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info('🔧 Iniciando treino do modelo XGBoost')
    logger.info(f"Anos: {args.start_year}-{args.end_year} | Model dir: {args.model_dir}")

    predictor = TennisPredictor()

    start_load = time.time()
    predictor.load_data(
        db_path=args.db,
        start_date=f'{args.start_year}-01-01',
        end_date=f'{args.end_year + 1}-01-01',
    )
    df = predictor.preprocess_data()
    load_time = time.time() - start_load
    logger.info(f"Dados preparados em {load_time:.1f}s — {len(df)} linhas")

    if df.empty:
        logger.error('Nenhum dado disponível para treino')
        return 1

    start_train = time.time()
    metrics = predictor.train_model(df, refit_full=not args.no_refit)
    metrics['train_time_seconds'] = round(time.time() - start_train, 1)
    metrics['preprocess_time_seconds'] = round(load_time, 1)
    logger.info(f"📊 Métricas no holdout temporal: {metrics}")

    # Modelo canónico (o que o engine carrega)
    predictor.save_model(args.model_dir)

    # Cópia versionada
    short_hash = _get_short_git_hash()
    git_part = f'_git{short_hash}' if short_hash else ''
    versioned_model_fname = f"model_{timestamp}{git_part}.pkl"
    versioned_data_fname = f"model_{timestamp}{git_part}_data.pkl"
    joblib.dump(predictor.model, os.path.join(args.model_dir, versioned_model_fname))
    joblib.dump(predictor.get_state(), os.path.join(args.model_dir, versioned_data_fname))

    entry = {
        'model_file': versioned_model_fname,
        'data_file': versioned_data_fname,
        'canonical_files': [MODEL_FILENAME, DATA_FILENAME],
        'created_at': timestamp,
        'git': short_hash,
        'metrics': metrics,
        'n_rows': int(len(df)),
        'n_features': len(predictor.feature_columns or []),
        'years': [args.start_year, args.end_year],
        'train_period': {
            'min_date': str(df['date'].min()),
            'max_date': str(df['date'].max()),
        },
    }
    manifest_path = _update_manifest(args.model_dir, entry)

    metrics_path = os.path.join(LOG_DIR, f'training_{timestamp}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'entry': entry}, f, indent=2, ensure_ascii=False, default=str)
    metrics_csv = _append_metrics_history(entry)

    logger.info(f"✅ Modelo canónico: {os.path.join(args.model_dir, MODEL_FILENAME)}")
    logger.info(f"✅ Versão: {versioned_model_fname}")
    logger.info(f"✅ Manifest: {manifest_path} | Log: {metrics_path} | Histórico: {metrics_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
