import sqlite3

import pandas as pd
import pytest

from elo_calculator import dynamic_k
from train_model import DEFAULT_EXCLUDED_GROUPS, build_parser
from update_from_csv import update_db


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE matches (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT, tourney_id TEXT, tourney_name TEXT, surface TEXT,
        tourney_level TEXT, tourney_date TEXT, match_num INTEGER, round TEXT, best_of INTEGER, minutes REAL,
        winner_name TEXT, loser_name TEXT, winner_rank REAL, loser_rank REAL, score TEXT,
        w_ace REAL, w_df REAL, w_svpt REAL, w_1stWon REAL, w_2ndWon REAL,
        l_ace REAL, l_df REAL, l_svpt REAL, l_1stWon REAL, l_2ndWon REAL,
        tournament_type TEXT, winner_elo_before REAL, loser_elo_before REAL,
        winner_elo_after REAL, loser_elo_after REAL, k_factor_used REAL, temporal_factor REAL,
        effective_k_factor REAL, result_type TEXT, counts_for_elo INTEGER, training_target INTEGER,
        elo_k_multiplier REAL)''')
    conn.execute('''CREATE TABLE players (name TEXT PRIMARY KEY, current_elo_hard REAL, current_elo_clay REAL,
        current_elo_grass REAL, current_elo_carpet REAL, last_match_date TEXT, total_matches INTEGER,
        total_wins INTEGER, total_losses INTEGER)''')
    conn.execute("INSERT INTO matches (tourney_id, tourney_name, surface, tourney_level, tourney_date, match_num, "
                 "round, winner_name, loser_name, score) VALUES "
                 "('2026-0001', 'Shanghai', 'Hard', 'M', '2026-10-07', 300, 'F', 'A', 'B', '6-3 6-4')")
    conn.commit()
    conn.close()


def _write_csv(path, rows):
    base = {'tourney_name': 'T', 'surface': 'Hard', 'tourney_level': 'A', 'best_of': 3, 'minutes': 90,
            'score': '6-3 6-4', 'tournament_type': 'ATP', 'w_svpt': 70, 'w_df': 2, 'w_1stWon': 30, 'w_2ndWon': 12,
            'w_ace': 5, 'l_svpt': 70, 'l_df': 3, 'l_1stWon': 25, 'l_2ndWon': 10, 'l_ace': 2}
    pd.DataFrame([{**base, **r} for r in rows]).to_csv(path, index=False)


def test_update_inserts_same_week_and_backfilled_tournaments_but_not_duplicates(tmp_path):
    db, csv = tmp_path / 'db.sqlite', tmp_path / 'new.csv'
    _make_db(db)
    _write_csv(csv, [
        # duplicado do jogo já na BD (mesmo tourney_id + match_num)
        {'tourney_id': '2026-0001', 'match_num': 300, 'round': 'F', 'tourney_date': 20261007,
         'winner_name': 'A', 'loser_name': 'B', 'tourney_name': 'Shanghai', 'tourney_level': 'M'},
        # torneio da mesma semana do último jogo (antes era ignorado por tourney_date > última data)
        {'tourney_id': '2026-0002', 'match_num': 1, 'round': 'R32', 'tourney_date': 20261007,
         'winner_name': 'C', 'loser_name': 'D'},
        # torneio publicado mais tarde com data anterior
        {'tourney_id': '2026-0003', 'match_num': 1, 'round': 'R32', 'tourney_date': 20260601,
         'winner_name': 'E', 'loser_name': 'F', 'surface': 'Grass', 'tourney_level': 'G'},
        # walkover: entra na BD mas não mexe no ELO
        {'tourney_id': '2026-0004', 'match_num': 1, 'round': 'R32', 'tourney_date': 20261014,
         'winner_name': 'C', 'loser_name': 'G', 'score': 'W/O'},
    ])

    summary = update_db(str(db), str(csv))

    assert summary['new'] == 3 and summary['already_in_db'] == 1 and summary['backfilled'] == 2
    conn = sqlite3.connect(db)
    assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 4
    c_d = conn.execute("SELECT winner_elo_after, w_svpt, w_df, counts_for_elo FROM matches "
                       "WHERE tourney_id = '2026-0002'").fetchone()
    assert c_d[0] == pytest.approx(1500 + dynamic_k(0) * 0.5)  # K dinâmico, jogador novo
    assert c_d[1] == 70 and c_d[2] == 2  # w_svpt e w_df já não ficam a NULL
    assert c_d[3] == 1
    wo = conn.execute("SELECT winner_elo_before, winner_elo_after, counts_for_elo, training_target FROM matches "
                      "WHERE tourney_id = '2026-0004'").fetchone()
    assert wo[0] == wo[1] and wo[2] == 0 and wo[3] == 0
    assert conn.execute("SELECT total_matches FROM players WHERE name = 'C'").fetchone()[0] == 2

    # Correr outra vez não duplica nada
    assert update_db(str(db), str(csv))['new'] == 0


def test_train_model_excludes_fatigue_and_momentum_by_default():
    assert build_parser().parse_args([]).exclude_groups == list(DEFAULT_EXCLUDED_GROUPS)
    assert build_parser().parse_args(['--exclude-groups']).exclude_groups == []
