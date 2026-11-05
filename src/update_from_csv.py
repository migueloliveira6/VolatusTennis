#!/usr/bin/env python3
"""
Atualiza a BD (matches + players) a partir de um CSV de jogos novos (scraping TennisAbstract →
notebook clean_player_name).

CSV esperado (ex.: dataset_2026_normalizado_v4.csv), colunas Sackmann:
    tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,
    winner_name,loser_name,score,DRW,best_of,round,minutes,w_ace,w_df,w_svpt,...,
    winner_rank,loser_rank,tournament_type

- Só insere jogos que ainda não existem (mesmo vencedor + perdedor + ronda no mesmo torneio ou
  na mesma data), em vez de filtrar pela última data da BD: assim não se perdem torneios da
  mesma semana nem torneios publicados mais tarde. O match_num não é usado (vem sempre 1 no
  CSV raspado).
- ELO com K dinâmico e a política de jogos especiais (iguais ao recompute_elo.py).

Usage:
    python src/update_from_csv.py                     # usa DB_PATH e CSV_PATH do .env
    python src/update_from_csv.py --csv datasets/novo.csv --dry-run
"""

import argparse
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calculator import INITIAL_ELO, level_k_multiplier, update_elo_dynamic
from match_policy import classify_result, elo_k_multiplier, is_training_target, sort_matches
from recompute_elo import LEVEL_FIXES, TOURNAMENT_TYPE_FIXES

DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '..', 'datasets', 'tennis_data_2026_v3.db'))
CSV_PATH = os.getenv('CSV_PATH', os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset_2026_normalizado_v5.csv'))

# Colunas copiadas do CSV tal como vêm (se existirem no CSV e na tabela)
CSV_COLUMNS = [
    'tourney_id', 'tourney_name', 'surface', 'draw_size', 'match_num', 'round', 'best_of', 'minutes',
    'winner_id', 'winner_seed', 'winner_entry', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
    'winner_rank', 'winner_rank_points', 'loser_id', 'loser_seed', 'loser_entry', 'loser_hand', 'loser_ht',
    'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points', 'score', 'DRW',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
]
SURFACES = ('hard', 'clay', 'grass', 'carpet')


def _clean(value):
    """NaN/NaT → None para o SQLite; numpy → tipos Python."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, 'item') else value


def parse_dates(series: pd.Series) -> pd.Series:
    """Aceita YYYYMMDD (Sackmann) ou YYYY-MM-DD."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format='%Y%m%d', errors='coerce')
    return parsed.fillna(pd.to_datetime(text, errors='coerce'))


def load_existing_keys(conn):
    """Chaves dos jogos já na BD: (torneio, ronda, vencedor, perdedor) e (data, ronda, vencedor, perdedor).

    O match_num não entra na chave: no CSV raspado vem sempre 1 dentro de cada torneio.
    """
    by_tourney, by_date = set(), set()
    for tid, date, winner, loser, rnd in conn.execute(
        'SELECT tourney_id, tourney_date, winner_name, loser_name, round FROM matches'
    ):
        if tid is not None:
            by_tourney.add((str(tid), rnd, winner, loser))
        by_date.add((str(date)[:10], rnd, winner, loser))
    return by_tourney, by_date


def select_new_matches(df: pd.DataFrame, by_tourney: set, by_date: set) -> pd.DataFrame:
    """Filtra os jogos do CSV que ainda não estão na BD (também remove duplicados dentro do CSV).

    Um jogo é duplicado se já existir o mesmo vencedor/perdedor na mesma ronda do mesmo torneio
    (tourney_id) ou na mesma data de torneio.
    """
    keep, seen_tourney, seen_date = [], set(by_tourney), set(by_date)
    for row in df.itertuples(index=False):
        rnd = _clean(getattr(row, 'round', None))
        tid = _clean(row.tourney_id)
        tourney_key = (str(tid), rnd, row.winner_name, row.loser_name) if tid is not None else None
        date_key = (row.tourney_date.strftime('%Y-%m-%d'), rnd, row.winner_name, row.loser_name)
        duplicate = (tourney_key is not None and tourney_key in seen_tourney) or date_key in seen_date
        keep.append(not duplicate)
        if tourney_key is not None:
            seen_tourney.add(tourney_key)
        seen_date.add(date_key)
    return df[keep]


def load_elo_state(conn):
    """ELO atual por (jogador, superfície) da tabela players e nº de jogos que contaram para o ELO."""
    elo = {}
    for name, *values in conn.execute(
        'SELECT name, current_elo_hard, current_elo_clay, current_elo_grass, current_elo_carpet FROM players'
    ):
        for surface, value in zip(SURFACES, values):
            if value is not None:
                elo[(name, surface)] = float(value)
    n_matches = {}
    for surface, name, count in conn.execute(
        "SELECT lower(surface), name, COUNT(*) FROM ("
        "  SELECT surface, winner_name AS name, score FROM matches UNION ALL"
        "  SELECT surface, loser_name, score FROM matches"
        ") WHERE score IS NULL OR score NOT LIKE '%W/O%' GROUP BY 1, 2"
    ):
        n_matches[(name, surface)] = count
    return elo, n_matches


def update_db(db_path: str, csv_path: str, dry_run: bool = False) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    table_columns = {row[1] for row in cur.execute('PRAGMA table_info(matches)')}
    players_columns = {row[1] for row in cur.execute('PRAGMA table_info(players)')}
    last_date = pd.to_datetime(cur.execute('SELECT MAX(tourney_date) FROM matches').fetchone()[0])

    df = pd.read_csv(csv_path)
    df['tourney_date'] = parse_dates(df['tourney_date'])
    df = df.dropna(subset=['tourney_date', 'winner_name', 'loser_name', 'surface'])
    for column in ('tourney_id', 'match_num', 'round'):
        if column not in df.columns:
            df[column] = None
    print(f"📄 CSV: {csv_path} ({len(df)} jogos) | 📁 BD: {db_path} (último jogo: {last_date.date() if pd.notna(last_date) else '—'})")

    by_tourney, by_date = load_existing_keys(conn)
    df_new = sort_matches(select_new_matches(df, by_tourney, by_date))
    summary = {'csv_rows': len(df), 'new': len(df_new), 'already_in_db': len(df) - len(df_new)}
    if df_new.empty:
        print('⚠️ Não há jogos novos para inserir.')
        conn.close()
        return summary

    backfilled = int((df_new['tourney_date'] <= last_date).sum()) if pd.notna(last_date) else 0
    summary['backfilled'] = backfilled
    print(f"✅ {len(df_new)} jogos novos ({df_new['tourney_date'].min().date()} a {df_new['tourney_date'].max().date()}); "
          f"{summary['already_in_db']} já estavam na BD")
    if backfilled:
        print(f"ℹ️ {backfilled} jogos são anteriores ou iguais à última data da BD (torneios publicados mais tarde). "
              "O treino reordena tudo; para as colunas de ELO da BD ficarem exatas corra o recompute_elo.py.")
    if dry_run:
        print('🔎 --dry-run: nada foi gravado.')
        conn.close()
        return summary

    elo, n_matches = load_elo_state(conn)
    for row in df_new.to_dict(orient='records'):
        surface = row['surface']
        surface_key = surface.lower()
        winner, loser = row['winner_name'], row['loser_name']
        level = LEVEL_FIXES.get(row.get('tourney_level'), row.get('tourney_level'))
        tournament_type = _clean(row.get('tournament_type'))
        tournament_type = TOURNAMENT_TYPE_FIXES.get(tournament_type, tournament_type) or ('CHALLENGER' if level == 'C' else 'ATP')

        policy_multiplier = elo_k_multiplier(row.get('score'), row.get('tourney_name'))
        multiplier = policy_multiplier * level_k_multiplier(level, row.get('round'))
        w_key, l_key = (winner, surface_key), (loser, surface_key)
        w_before, l_before = elo.get(w_key, INITIAL_ELO), elo.get(l_key, INITIAL_ELO)
        if multiplier > 0:
            w_after, l_after, k_winner, _ = update_elo_dynamic(
                w_before, l_before, n_matches.get(w_key, 0), n_matches.get(l_key, 0), multiplier)
            elo[w_key], elo[l_key] = w_after, l_after
            n_matches[w_key] = n_matches.get(w_key, 0) + 1
            n_matches[l_key] = n_matches.get(l_key, 0) + 1
        else:
            w_after, l_after, k_winner = w_before, l_before, 0.0

        values = {col: _clean(row.get(col)) for col in CSV_COLUMNS if col in row}
        values.update({
            'winner_name': winner, 'loser_name': loser, 'surface': surface,
            'tourney_level': _clean(level), 'tourney_date': row['tourney_date'].strftime('%Y-%m-%d'),
            'tournament_type': tournament_type,
            'winner_elo_before': w_before, 'loser_elo_before': l_before,
            'winner_elo_after': w_after, 'loser_elo_after': l_after,
            'k_factor_used': k_winner, 'temporal_factor': 1.0, 'effective_k_factor': k_winner,
            'result_type': classify_result(row.get('score')), 'counts_for_elo': int(policy_multiplier > 0),
            'training_target': int(is_training_target(row.get('score'), row.get('tourney_name'))),
            'elo_k_multiplier': multiplier,
        })
        values = {k: v for k, v in values.items() if k in table_columns}
        cur.execute(
            f"INSERT INTO matches ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
            list(values.values()),
        )

        date = values['tourney_date']
        for name, won, new_elo in ((winner, True, w_after), (loser, False, l_after)):
            if cur.execute('SELECT 1 FROM players WHERE name = ?', (name,)).fetchone() is None:
                cur.execute('INSERT INTO players (name, current_elo_hard, current_elo_clay, current_elo_grass, '
                            'current_elo_carpet) VALUES (?, ?, ?, ?, ?)', (name, *(INITIAL_ELO,) * 4))
            counters = ''
            if {'total_wins', 'total_losses', 'total_matches'} <= players_columns:
                counters = (', total_wins = COALESCE(total_wins, 0) + 1' if won else
                            ', total_losses = COALESCE(total_losses, 0) + 1') + ', total_matches = COALESCE(total_matches, 0) + 1'
            cur.execute(f'UPDATE players SET current_elo_{surface_key} = ?, '
                        f'last_match_date = MAX(COALESCE(last_match_date, ?), ?){counters} WHERE name = ?',
                        (new_elo, date, date, name))

    conn.commit()
    conn.close()
    print(f"🎾 {len(df_new)} jogos novos inseridos e ELOs atualizados.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Insere jogos novos de um CSV (formato Sackmann) na BD')
    parser.add_argument('--db', default=DB_PATH)
    parser.add_argument('--csv', default=CSV_PATH)
    parser.add_argument('--dry-run', action='store_true', help='Só mostra o que seria inserido')
    args = parser.parse_args(argv)
    update_db(args.db, args.csv, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
