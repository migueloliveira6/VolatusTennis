import numpy as np
import pandas as pd
import pytest

from features import FEATURE_GROUPS, FeatureBuilder
from model_elo_xgboost import TennisPredictor, features_for_groups


PLAYERS = ['P1', 'P2', 'P3', 'P4']
LEVELS = ['G', 'M', 'A']
SURFACES = ['Hard', 'Clay']
ROUNDS = ['R32', 'R16', 'QF', 'SF']


def _synthetic_matches(n=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        winner, loser = rng.choice(PLAYERS, size=2, replace=False)
        rows.append({
            'tourney_date': pd.Timestamp('2020-01-06') + pd.Timedelta(days=7 * (i // 4)),
            'tourney_id': f'T{i // 4}',
            'match_num': i % 4,
            'round': ROUNDS[i % 4],
            'winner_name': winner,
            'loser_name': loser,
            'surface': SURFACES[(i // 4) % 2],
            'tourney_level': LEVELS[(i // 4) % 3],
            'best_of': 5 if LEVELS[(i // 4) % 3] == 'G' else 3,
            'minutes': 100.0,
            'score': '6-3 6-4',
            'tourney_name': 'Torneio',
            'winner_rank': float(PLAYERS.index(winner) + 1),
            'loser_rank': float(PLAYERS.index(loser) + 1),
            'winner_rank_points': 1000.0,
            'loser_rank_points': 800.0,
            'winner_age': 25.0,
            'loser_age': 27.0,
            'w_svpt': 80.0, 'w_1stWon': 35.0, 'w_2ndWon': 15.0, 'w_ace': 6.0, 'w_df': 2.0,
            'l_svpt': 80.0, 'l_1stWon': 30.0, 'l_2ndWon': 12.0, 'l_ace': 3.0, 'l_df': 4.0,
        })
    return pd.DataFrame(rows)


class _CapturingModel:
    """Substitui o XGBoost: guarda a matriz recebida e devolve p=0.6 para a 1.ª linha e 0.3 para a 2.ª."""

    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X
        p = np.where(np.arange(len(X)) == 0, 0.6, 0.3)
        return np.column_stack([1 - p, p])


def _prepared_predictor(matches=None):
    predictor = TennisPredictor()
    predictor.matches = _synthetic_matches() if matches is None else matches
    df = predictor.preprocess_data()
    return predictor, df


def test_predict_match_uses_training_columns_and_is_symmetric():
    predictor, df = _prepared_predictor()
    predictor.feature_columns = predictor._to_model_matrix(df).columns.tolist()
    predictor.model = _CapturingModel()

    winner, confidence, details = predictor.predict_match('P1', 'P2', 'Hard', date='2030-01-01', tourney_level='M')

    X = predictor.model.last_X
    assert list(X.columns) == predictor.feature_columns
    assert len(X) == 2  # (P1, P2) e (P2, P1)
    assert X['tourney_level_M'].iloc[0] == 1 and X['tourney_level_G'].iloc[0] == 0
    assert X['surface_Hard'].iloc[0] == 1
    assert X['elo_diff'].iloc[0] == pytest.approx(-X['elo_diff'].iloc[1])
    # p simétrica = (0.6 + 1 - 0.3) / 2 = 0.65 (sem calibrador)
    assert winner == 'P1' and confidence == pytest.approx(0.65)
    assert details['tourney_level'] == 'M' and details['best_of'] == 3


def test_grand_slam_defaults_to_best_of_five():
    predictor, df = _prepared_predictor()
    predictor.feature_columns = predictor._to_model_matrix(df).columns.tolist()
    predictor.model = _CapturingModel()
    _, _, details = predictor.predict_match('P3', 'P4', 'Clay', date='2030-01-01', tourney_level='G')
    assert details['best_of'] == 5
    assert predictor.model.last_X['best_of'].iloc[0] == 5


def test_swap_rows_mirror_main_rows():
    predictor, df = _prepared_predictor()
    swap = predictor.swap_features.loc[df['match_idx']]
    assert np.allclose(df['elo_diff'].to_numpy(), -swap['elo_diff'].to_numpy())
    assert (df['target'].to_numpy() == 1 - swap['target'].to_numpy()).all()


def test_features_use_only_previous_matches():
    """O 1.º jogo de cada jogador é visto com o estado inicial (sem informação do próprio jogo)."""
    predictor, df = _prepared_predictor()
    first = df.iloc[0]
    assert first['player_elo'] == 1500 and first['opponent_elo'] == 1500
    assert first['h2h_matches'] == 0
    assert first['player_stats_matches'] == 0


def test_elo_updates_within_tournament_follow_round_order():
    builder = FeatureBuilder()
    base = {'tourney_date': pd.Timestamp('2024-01-01'), 'surface': 'Hard', 'tourney_level': 'A',
            'score': '6-3 6-4', 'tourney_name': 'T'}
    builder.update({**base, 'winner_name': 'A', 'loser_name': 'B', 'round': 'R16'})
    elo_after_first = builder.players['A']['elo']
    features = builder.features('A', 'C', 'Hard', 'A', pd.Timestamp('2024-01-01'))
    # O jogo anterior do mesmo torneio (mesma data) já conta no estado
    assert features['player_elo'] == pytest.approx(elo_after_first) and elo_after_first > 1500
    assert features['player_days_rest'] == 0 and features['player_matches_14d'] == 1


def test_preprocess_skips_walkovers_and_keeps_retirements_out_of_training():
    matches = _synthetic_matches(n=40)
    matches.loc[10, 'score'] = 'W/O'
    matches.loc[11, 'score'] = '6-4 1-0 RET'
    predictor, df = _prepared_predictor(matches)

    assert len(df) == 38
    total_matches = sum(state['n'] for state in predictor.feature_builder.players.values())
    assert total_matches == 2 * 39  # o abandono conta no histórico, o walkover não


def test_exclude_feature_groups_removes_columns():
    predictor, df = _prepared_predictor()
    predictor.excluded_features = features_for_groups(['fatigue', 'context'])
    X = predictor._to_model_matrix(df)
    assert not any(col in X.columns for col in FEATURE_GROUPS['fatigue'])
    assert not any(col.startswith('surface_') and col in ('surface_Hard', 'surface_Clay') for col in X.columns)
    with pytest.raises(ValueError):
        features_for_groups(['nao_existe'])


def test_train_model_returns_metrics_and_refits_on_all_data(monkeypatch, tmp_path):
    monkeypatch.setenv('OPTUNA_TRIALS', '1')
    predictor, df = _prepared_predictor()

    metrics = predictor.train_model(df, refit_full=True)

    for key in ('accuracy', 'auc', 'logloss', 'brier', 'n_test', 'n_estimators', 'single_perspective'):
        assert key in metrics
    assert metrics['refit_full'] is True
    assert predictor.last_match_date == df['date'].max()
    state = predictor.get_state()
    assert state['feature_columns'] == predictor.feature_columns
    assert state['calibrator']['method'] == metrics['calibration']['method']
    assert {'elo', 'rank_points'} <= set(metrics['baselines'])

    # Gravar e carregar mantém a previsão
    predictor.save_model(str(tmp_path))
    loaded = TennisPredictor()
    assert loaded.load_saved_model(str(tmp_path))
    before = predictor.predict_match('P1', 'P2', 'Hard', date='2030-01-01', tourney_level='A')[1]
    after = loaded.predict_match('P1', 'P2', 'Hard', date='2030-01-01', tourney_level='A')[1]
    assert after == pytest.approx(before)
