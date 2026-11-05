#!/usr/bin/env python3
"""Recalcula o ELO por superfície de uma BD existente aplicando a política de `match_policy.py`.

Trabalha sobre uma CÓPIA da BD (a original nunca é alterada), para manter os jogos que vieram
do scraping e não existem nos CSV do Sackmann. Usa o ELO com K dinâmico (nº de jogos do jogador,
sem ano de referência) e a ordem data → torneio → ronda → match_num.

Usage:
    python src/recompute_elo.py --source-db datasets/tennis_data_2026.db --output-db datasets/tennis_data_2026_v2.db
"""
import argparse
import os
import shutil
import sqlite3
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calculator import INITIAL_ELO, level_k_multiplier, update_elo_dynamic
from match_policy import classify_result, elo_k_multiplier, is_exhibition, is_training_target, sort_matches

SURFACES = ('hard', 'clay', 'grass', 'carpet')

# Correções de dados vindos do scraping
TOURNAMENT_TYPE_FIXES = {'regular': 'ATP'}
LEVEL_FIXES = {'W': 'G'}  # "Us-Open" raspado com nível 'W'

POLICY_COLUMNS = {
    'result_type': 'TEXT',
    'counts_for_elo': 'INTEGER',
    'training_target': 'INTEGER',
    'elo_k_multiplier': 'REAL',
}


def build_parser():
    parser = argparse.ArgumentParser(description='Recalcula o ELO de uma BD com a política de jogos especiais')
    parser.add_argument('--source-db', default=os.getenv('DB_PATH', 'datasets/tennis_data_2026.db'))
    parser.add_argument('--output-db', required=True, help='Nova BD (cópia da original com o ELO recalculado)')
    parser.add_argument('--force', action='store_true', help='Substituir a BD de saída se já existir')
    return parser


def _ensure_policy_columns(cursor):
    existing = {row[1] for row in cursor.execute('PRAGMA table_info(matches)')}
    for column, sql_type in POLICY_COLUMNS.items():
        if column not in existing:
            cursor.execute(f'ALTER TABLE matches ADD COLUMN {column} {sql_type}')


def recompute(source_db: str, output_db: str, force: bool = False) -> dict:
    if os.path.abspath(source_db) == os.path.abspath(output_db):
        raise ValueError('A BD de saída tem de ser diferente da original')
    if not os.path.exists(source_db):
        raise FileNotFoundError(source_db)
    if os.path.exists(output_db):
        if not force:
            raise FileExistsError(f'{output_db} já existe. Use --force para substituir.')
        os.remove(output_db)
    shutil.copy2(source_db, output_db)

    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    _ensure_policy_columns(cursor)

    matches = pd.read_sql_query(
        'SELECT match_id, tourney_date, tourney_id, round, match_num, winner_name, loser_name, surface, '
        'score, tourney_name, tourney_level, tournament_type FROM matches', conn
    )
    matches = sort_matches(matches)

    elo = {}       # (jogador, superfície) -> ELO atual
    n_matches = {}  # (jogador, superfície) -> nº de jogos que contaram para o ELO
    last_date = {}
    updates = []
    stats = Counter()

    for row in matches.itertuples(index=False):
        level = LEVEL_FIXES.get(row.tourney_level, row.tourney_level)
        tournament_type = TOURNAMENT_TYPE_FIXES.get(row.tournament_type, row.tournament_type) or 'ATP'
        surface_key = (row.surface or 'Hard').lower()
        winner_key, loser_key = (row.winner_name, surface_key), (row.loser_name, surface_key)
        result_type = classify_result(row.score)
        policy_multiplier = elo_k_multiplier(row.score, row.tourney_name)
        multiplier = policy_multiplier * level_k_multiplier(level, row.round)
        stats[result_type] += 1
        if is_exhibition(row.tourney_name):
            stats['exhibition'] += 1

        winner_before = elo.get(winner_key, INITIAL_ELO)
        loser_before = elo.get(loser_key, INITIAL_ELO)

        if multiplier > 0:
            winner_after, loser_after, k_winner, _ = update_elo_dynamic(
                winner_before, loser_before, n_matches.get(winner_key, 0), n_matches.get(loser_key, 0), multiplier
            )
            elo[winner_key], elo[loser_key] = winner_after, loser_after
            n_matches[winner_key] = n_matches.get(winner_key, 0) + 1
            n_matches[loser_key] = n_matches.get(loser_key, 0) + 1
        else:
            winner_after, loser_after, k_winner = winner_before, loser_before, 0.0
            stats['excluded_from_elo'] += 1

        updates.append((
            level, tournament_type, winner_before, loser_before, winner_after, loser_after,
            # k_factor_used = K do vencedor (dinâmico); temporal_factor deixa de ser usado (1.0)
            k_winner, k_winner, 1.0,
            result_type, int(policy_multiplier > 0), int(is_training_target(row.score, row.tourney_name)),
            multiplier, int(row.match_id),
        ))

    cursor.executemany(
        'UPDATE matches SET tourney_level = ?, tournament_type = ?, '
        'winner_elo_before = ?, loser_elo_before = ?, winner_elo_after = ?, loser_elo_after = ?, '
        'k_factor_used = ?, effective_k_factor = ?, temporal_factor = ?, '
        'result_type = ?, counts_for_elo = ?, training_target = ?, elo_k_multiplier = ? '
        'WHERE match_id = ?',
        updates,
    )

    # ELO atual por superfície na tabela players
    for (date, name) in cursor.execute(
        'SELECT MAX(tourney_date), name FROM (SELECT tourney_date, winner_name AS name FROM matches WHERE counts_for_elo = 1 '
        'UNION ALL SELECT tourney_date, loser_name FROM matches WHERE counts_for_elo = 1) GROUP BY name'
    ).fetchall():
        last_date[name] = date
    player_updates = [
        tuple(elo.get((name, s), INITIAL_ELO) for s in SURFACES) + (last_date.get(name), name)
        for (name,) in cursor.execute('SELECT name FROM players').fetchall()
    ]
    cursor.executemany(
        'UPDATE players SET current_elo_hard = ?, current_elo_clay = ?, current_elo_grass = ?, '
        'current_elo_carpet = ?, last_match_date = COALESCE(?, last_match_date) WHERE name = ?',
        player_updates,
    )
    conn.commit()
    conn.close()

    summary = {'matches': len(matches), **stats}
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = recompute(args.source_db, args.output_db, force=args.force)
    print(f'✅ ELO recalculado em {args.output_db}')
    for key, value in summary.items():
        print(f'   {key}: {value}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
