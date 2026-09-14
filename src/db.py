"""SQLite helper utilities for the project.
Provides a simple connection factory and common queries used by training/prediction scripts.
"""
import os
import sqlite3
import pandas as pd
from typing import Optional

DEFAULT_DB = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '..', 'datasets', 'tennis_data.db'))


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a sqlite3.Connection using DB_PATH or provided path."""
    path = db_path or DEFAULT_DB
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    # Return rows as tuples / dicts is handled in pandas
    return conn


def load_matches_for_training(conn: sqlite3.Connection, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Load matches suitable for training and normalize column names expected by existing code.

    Returns columns: tourney_date (datetime), winner_name, loser_name, surface,
    winner_surface_elo, loser_surface_elo, and optional tourney_level
    """
    table_info = pd.read_sql_query("PRAGMA table_info(matches)", conn)
    available_columns = set(table_info['name'].astype(str).tolist())

    optional_selects = []
    if 'tourney_level' in available_columns:
        optional_selects.append("COALESCE(tourney_level, 'UNK') as tourney_level")

    optional_select_sql = ""
    if optional_selects:
        optional_select_sql = ",\n        " + ",\n        ".join(optional_selects)

    query = f"""
    SELECT
        tourney_date as tourney_date,
        winner_name as winner_name,
        loser_name as loser_name,
        surface as surface,
        winner_rank as winner_rank,
        loser_rank as loser_rank,
        winner_rank_points as winner_rank_points,
        loser_rank_points as loser_rank_points,
        winner_age as winner_age,
        loser_age as loser_age,
        winner_elo_after as winner_surface_elo,
        loser_elo_after as loser_surface_elo{optional_select_sql}
    FROM matches
    WHERE winner_name IS NOT NULL
      AND loser_name IS NOT NULL
      AND surface IS NOT NULL
      AND winner_elo_after IS NOT NULL
      AND loser_elo_after IS NOT NULL
    """

    conditions = []
    params = {}
    if start_date:
        conditions.append("tourney_date >= :start_date")
        params['start_date'] = start_date
    if end_date:
        conditions.append("tourney_date < :end_date")
        params['end_date'] = end_date

    if conditions:
        query += "\n  AND " + " AND ".join(conditions)

    df = pd.read_sql_query(query, conn, parse_dates=['tourney_date'], params=params)
    return df


def load_todays_matches(conn: sqlite3.Connection, target_date: str) -> pd.DataFrame:
    """Load matches scheduled for a specific date (used for prediction)."""
    query = """
    SELECT
        match_id,
        tourney_date as tourney_date,
        tourney_name as tournament,
        winner_name as player1,
        loser_name as player2,
        surface
    FROM matches
    WHERE tourney_date = :target_date
    """
    df = pd.read_sql_query(query, conn, parse_dates=['tourney_date'], params={'target_date': target_date})
    return df


def ensure_predictions_table(conn: sqlite3.Connection):
    """Create predictions table if it does not exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        match_date DATE,
        tournament TEXT,
        player1 TEXT,
        player2 TEXT,
        odd1 REAL,
        odd2 REAL,
        model_version TEXT,
        prob REAL,
        predicted_winner TEXT,
        value_bet REAL,
        roi_expected REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);")
    conn.commit()


def save_predictions(conn: sqlite3.Connection, df_preds: pd.DataFrame, model_version: str):
    """Save a DataFrame of predictions to the `predictions` table.

    Expects df_preds with columns similar to the bot output:
      'Torneio','Jogador 1','Jogador 2','Odd 1','Odd 2','Vencedor Previsto','Confiança (%)','Valor Aposta','ROI Esperado (%)'

    The function maps/normalizes columns and inserts rows in batch.
    """
    ensure_predictions_table(conn)
    cursor = conn.cursor()

    records = []
    for _, r in df_preds.iterrows():
        prob_pct = r.get('Confiança (%)', 0)
        try:
            prob = float(prob_pct) / 100.0
        except Exception:
            prob = None

        try:
            value_bet = float(r.get('Valor Aposta', 0))
        except Exception:
            value_bet = None

        try:
            roi_pct = r.get('ROI Esperado (%)', 0)
            roi_expected = float(roi_pct) / 100.0
        except Exception:
            roi_expected = None

        records.append((
            None,  # match_id (unknown when predictions come from scraper)
            None,  # match_date
            r.get('Torneio'),
            r.get('Jogador 1'),
            r.get('Jogador 2'),
            r.get('Odd 1'),
            r.get('Odd 2'),
            model_version,
            prob,
            r.get('Vencedor Previsto'),
            value_bet,
            roi_expected
        ))

    cursor.executemany("""
        INSERT INTO predictions (
            match_id, match_date, tournament, player1, player2,
            odd1, odd2, model_version, prob, predicted_winner,
            value_bet, roi_expected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
