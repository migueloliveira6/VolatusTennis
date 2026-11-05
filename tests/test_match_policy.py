import sqlite3

import pandas as pd
import pytest

from match_policy import (add_policy_columns, classify_result, counts_for_history, elo_k_multiplier,
                          is_training_target)
from elo_calculator import dynamic_k, expected_score
from recompute_elo import recompute


@pytest.mark.parametrize('score, expected', [
    ('7-6(5) 6-3', 'completed'),
    (None, 'completed'),
    ('W/O', 'walkover'),
    ('Walkover', 'walkover'),
    ('6-4 2-1 RET', 'retired'),
    ("Ret'd", 'retired'),
    ('Played and abandoned', 'retired'),
    ('DEF', 'default'),
    ('Def.', 'default'),
])
def test_classify_result(score, expected):
    assert classify_result(score) == expected


def test_policy_rules():
    assert elo_k_multiplier('6-3 6-4') == 1.0
    assert elo_k_multiplier('6-4 1-0 RET') == 0.5
    assert elo_k_multiplier('W/O') == 0.0
    assert elo_k_multiplier('6-3 6-4', 'Laver Cup') == 0.0
    assert elo_k_multiplier('6-3 6-4', 'Davis Cup WG R1') == 1.0
    assert counts_for_history('6-4 1-0 RET') and not is_training_target('6-4 1-0 RET')
    assert not counts_for_history('W/O') and not is_training_target('W/O')


def test_add_policy_columns_without_score_treats_all_as_completed():
    df = add_policy_columns(pd.DataFrame({'winner_name': ['A']}))
    assert df.loc[0, 'result_type'] == 'completed'
    assert bool(df.loc[0, 'training_target'])


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE matches (
        match_id INTEGER PRIMARY KEY, tourney_date TEXT, tourney_id TEXT, round TEXT, match_num INTEGER, winner_name TEXT, loser_name TEXT, surface TEXT,
        score TEXT, tourney_name TEXT, tourney_level TEXT, tournament_type TEXT, temporal_factor REAL,
        winner_elo_before REAL, loser_elo_before REAL, winner_elo_after REAL, loser_elo_after REAL,
        k_factor_used REAL, effective_k_factor REAL)''')
    conn.execute('''CREATE TABLE players (name TEXT, current_elo_hard REAL, current_elo_clay REAL,
        current_elo_grass REAL, current_elo_carpet REAL, last_match_date TEXT)''')
    rows = [
        (1, '2025-01-06', 'A', 'B', 'Hard', '6-3 6-4', 'Brisbane', 'A', 'ATP', 1.0),
        (2, '2025-01-13', 'A', 'B', 'Hard', 'W/O', 'Adelaide', 'A', 'ATP', 1.0),
        (3, '2025-01-20', 'B', 'A', 'Hard', '6-4 1-0 RET', 'Australian Open', 'G', 'ATP', 1.0),
        (4, '2025-09-20', 'A', 'B', 'Hard', '6-3 6-4', 'Laver Cup', 'A', 'regular', 1.0),
        (5, '2025-08-25', 'A', 'B', 'Hard', '6-3 6-4', 'Us-Open', 'W', 'regular', 1.0),
    ]
    conn.executemany('INSERT INTO matches (match_id, tourney_date, winner_name, loser_name, surface, score, '
                     'tourney_name, tourney_level, tournament_type, temporal_factor, tourney_id, round, match_num) '
                     'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [r + (f'T{r[0]}', 'R32', 1) for r in rows])
    conn.executemany('INSERT INTO players VALUES (?, 1500, 1500, 1500, 1500, NULL)', [('A',), ('B',)])
    conn.commit()
    conn.close()


def test_recompute_elo_applies_policy_on_a_copy(tmp_path):
    source = tmp_path / 'source.db'
    output = tmp_path / 'output.db'
    _make_db(source)

    summary = recompute(str(source), str(output))

    assert summary['walkover'] == 1 and summary['retired'] == 1 and summary['exhibition'] == 1
    conn = sqlite3.connect(output)
    m = {r[0]: r for r in conn.execute(
        'SELECT match_id, winner_elo_before, winner_elo_after, effective_k_factor, counts_for_elo, '
        'training_target, tourney_level, tournament_type FROM matches ORDER BY match_id')}
    # 1: jogo normal ATP, K dinâmico de um jogador novo (n=0): 1500 -> 1500 + K/2
    assert m[1][2] == pytest.approx(1500 + dynamic_k(0) * 0.5)
    # 2: walkover não mexe no ELO
    assert m[2][1] == m[2][2] and m[2][4] == 0 and m[2][5] == 0
    # 3: abandono num Grand Slam: K dinâmico (n=1, o W/O não conta) x 0.5 x 1.1, fora do treino
    assert m[3][3] == pytest.approx(dynamic_k(1) * 0.5 * 1.1) and m[3][4] == 1 and m[3][5] == 0
    # 4: Laver Cup excluída
    assert m[4][4] == 0 and m[4][5] == 0
    # 5: correções de dados do scraping
    assert m[5][6] == 'G' and m[5][7] == 'ATP'
    # A BD original não foi alterada
    original = sqlite3.connect(source).execute('SELECT winner_elo_after FROM matches WHERE match_id = 1').fetchone()[0]
    assert original is None
    # Tabela players com o ELO final
    elo_a = conn.execute("SELECT current_elo_hard FROM players WHERE name = 'A'").fetchone()[0]
    assert elo_a == pytest.approx(m[5][2])


def test_recompute_refuses_to_overwrite_source(tmp_path):
    source = tmp_path / 'source.db'
    _make_db(source)
    with pytest.raises(ValueError):
        recompute(str(source), str(source))
