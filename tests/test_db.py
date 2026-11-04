import sqlite3
import pandas as pd
from db import get_connection, load_matches_for_training


def _create_inmemory_matches_db():
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tourney_date DATE,
            winner_name TEXT,
            loser_name TEXT,
            surface TEXT,
            winner_rank REAL,
            loser_rank REAL,
            winner_rank_points REAL,
            loser_rank_points REAL,
            winner_age REAL,
            loser_age REAL,
            winner_elo_after REAL,
            loser_elo_after REAL
        )
    ''')
    # Insert sample rows
    rows = [
        ('2025-01-01', 'A', 'B', 'Hard', 10, 20, 3000, 1500, 25.0, 27.0, 1520, 1480),
        ('2025-02-01', 'C', 'D', 'Clay', 30, 40, 1200, 900, 22.0, 30.0, 1510, 1490),
        ('2025-03-01', 'E', 'F', 'Grass', 50, 60, 800, 700, 24.0, 26.0, 1530, 1470),
    ]
    cur.executemany(
        'INSERT INTO matches (tourney_date, winner_name, loser_name, surface, winner_rank, loser_rank, '
        'winner_rank_points, loser_rank_points, winner_age, loser_age, winner_elo_after, loser_elo_after) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    return conn


def test_load_matches_for_training_all():
    conn = _create_inmemory_matches_db()
    df = load_matches_for_training(conn)
    assert len(df) == 3
    assert 'tourney_date' in df.columns
    conn.close()


def test_load_matches_for_training_with_date_range():
    conn = _create_inmemory_matches_db()
    df = load_matches_for_training(conn, start_date='2025-02-01', end_date='2025-04-01')
    assert len(df) == 2
    assert set(df['surface']) == {'Clay', 'Grass'}
    conn.close()
