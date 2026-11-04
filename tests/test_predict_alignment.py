import numpy as np
import pandas as pd

from model_elo_xgboost import TennisPredictor


PLAYERS = ['P1', 'P2', 'P3', 'P4']
LEVELS = ['G', 'M', 'A']
SURFACES = ['Hard', 'Clay']


def _synthetic_matches(n=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    elo = {p: 1500.0 for p in PLAYERS}
    for i in range(n):
        winner, loser = rng.choice(PLAYERS, size=2, replace=False)
        elo[winner] += 8
        elo[loser] -= 8
        rows.append({
            'tourney_date': pd.Timestamp('2020-01-06') + pd.Timedelta(days=7 * (i // 4)),
            'winner_name': winner,
            'loser_name': loser,
            'surface': SURFACES[i % 2],
            'tourney_level': LEVELS[i % 3],
            'winner_rank': float(PLAYERS.index(winner) + 1),
            'loser_rank': float(PLAYERS.index(loser) + 1),
            'winner_rank_points': 1000.0,
            'loser_rank_points': 800.0,
            'winner_age': 25.0,
            'loser_age': 27.0,
            'winner_surface_elo': elo[winner],
            'loser_surface_elo': elo[loser],
        })
    return pd.DataFrame(rows)


class _CapturingModel:
    """Substitui o XGBoost e guarda a matriz recebida em predict_proba."""

    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X
        return np.array([[0.4, 0.6]])


def _prepared_predictor():
    predictor = TennisPredictor()
    predictor.matches = _synthetic_matches()
    df = predictor.preprocess_data()
    return predictor, df


def test_predict_match_uses_training_columns_and_encodes_tourney_level():
    predictor, df = _prepared_predictor()
    predictor.feature_columns = predictor._to_model_matrix(df).columns.tolist()
    predictor.model = _CapturingModel()

    winner, confidence, details = predictor.predict_match('P1', 'P2', 'Hard', date='2030-01-01', tourney_level='M')

    X = predictor.model.last_X
    assert list(X.columns) == predictor.feature_columns
    assert X['tourney_level_M'].iloc[0] == 1
    assert X['tourney_level_G'].iloc[0] == 0
    assert X['surface_Hard'].iloc[0] == 1
    assert winner == 'P1'
    assert details['tourney_level'] == 'M'


def test_training_and_prediction_build_the_same_features():
    predictor, df = _prepared_predictor()
    predictor.feature_columns = predictor._to_model_matrix(df).columns.tolist()
    predictor.model = _CapturingModel()

    predictor.predict_match('P3', 'P4', 'Clay', date='2030-01-01', tourney_level='G')
    X_pred = predictor.model.last_X
    X_train = predictor._to_model_matrix(df, expected_columns=predictor.feature_columns)

    assert list(X_pred.columns) == list(X_train.columns)
    assert X_pred['tourney_level_G'].iloc[0] == 1
    assert X_pred['surface_Clay'].iloc[0] == 1


def test_preprocess_skips_walkovers_and_keeps_retirements_out_of_training():
    matches = _synthetic_matches(n=40)
    matches['score'] = '6-3 6-4'
    matches.loc[10, 'score'] = 'W/O'
    matches.loc[11, 'score'] = '6-4 1-0 RET'
    matches['tourney_name'] = 'Torneio'
    predictor = TennisPredictor()
    predictor.matches = matches

    df = predictor.preprocess_data()

    assert len(df) == 38
    total_history = sum(len(h) for h in predictor.player_history.values())
    assert total_history == 2 * 39  # o abandono conta no histórico, o walkover não


def test_train_model_returns_metrics_and_refits_on_all_data(monkeypatch):
    monkeypatch.setenv('OPTUNA_TRIALS', '1')
    predictor, df = _prepared_predictor()

    metrics = predictor.train_model(df, refit_full=True)

    for key in ('accuracy', 'auc', 'logloss', 'brier', 'n_test', 'n_estimators'):
        assert key in metrics
    assert metrics['refit_full'] is True
    assert predictor.last_match_date == df['date'].max()
    state = predictor.get_state()
    assert state['feature_columns'] == predictor.feature_columns
    assert state['training_metrics'] == metrics
    assert state['calibrator']['method'] == metrics['calibration']['method']
    assert {'elo', 'rank_points'} <= set(metrics['baselines'])
