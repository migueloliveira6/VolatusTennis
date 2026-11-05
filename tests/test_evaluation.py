import numpy as np
import pandas as pd
import pytest

from evaluation import (apply_calibration, baseline_metrics, elo_probability, expected_calibration_error,
                        fit_calibrator, rank_points_probability, select_calibration_method)
from engine import calcular_stake_kelly, inferir_nivel_torneio


def test_ece_is_zero_for_perfectly_calibrated_bins():
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_isotonic_and_platt_reduce_ece_on_overconfident_probabilities():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.3, 0.7, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(int)
    overconfident = np.clip(0.5 + 2 * (true_p - 0.5), 0.01, 0.99)

    raw_ece = expected_calibration_error(y, overconfident)
    for method in ('platt', 'isotonic'):
        calibrator = fit_calibrator(method, overconfident[:2000], y[:2000])
        calibrated = apply_calibration(calibrator, overconfident[2000:])
        assert expected_calibration_error(y[2000:], calibrated) < raw_ece


def test_select_calibration_method_returns_known_method_with_metrics():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    best, results = select_calibration_method(p[:500], y[:500], p[500:], y[500:])
    assert best in ('none', 'platt', 'isotonic')
    assert set(results) == {'none', 'platt', 'isotonic'}


def test_apply_calibration_without_calibrator_is_identity():
    p = np.array([0.2, 0.5, 0.9])
    assert np.allclose(apply_calibration(None, p), p)


def test_baseline_probabilities():
    assert elo_probability(0) == pytest.approx(0.5)
    assert elo_probability(400) == pytest.approx(10 / 11)
    assert rank_points_probability(3000, 1000) == pytest.approx(0.75)
    assert rank_points_probability(np.nan, 1000) == pytest.approx(0.5)


def test_baseline_metrics_higher_ranked_accuracy():
    df = pd.DataFrame({
        'elo_diff': [100, -50, 20, -10],
        'player_rank_points': [2000, 500, 800, 900],
        'opponent_rank_points': [1000, 1500, 700, 1000],
        'player_rank': [5, 40, 20, np.nan],
        'opponent_rank': [10, 3, 25, 15],
    })
    y = np.array([1, 0, 0, 1])
    result = baseline_metrics(df, y)
    assert result['higher_ranked']['accuracy'] == pytest.approx(2 / 3)
    assert result['higher_ranked']['coverage'] == pytest.approx(0.75)
    assert 'logloss' in result['elo']


def test_kelly_fractional_with_min_edge_and_cap():
    # Sem edge suficiente: odd 2.0 implica 50%, modelo 51% < 2 p.p.
    assert calcular_stake_kelly(0.51, 2.0, fracao=0.25, edge_minimo=0.02, stake_maximo=0.05) == (0, 0)
    # Kelly completo = (0.6*2-1)/(2-1) = 0.2 → 1/4 = 0.05
    stake, roi = calcular_stake_kelly(0.6, 2.0, fracao=0.25, edge_minimo=0.02, stake_maximo=0.10)
    assert stake == pytest.approx(0.05)
    assert roi == pytest.approx(0.2)
    # Limite máximo de stake
    stake, _ = calcular_stake_kelly(0.9, 3.0, fracao=0.25, edge_minimo=0.02, stake_maximo=0.05)
    assert stake == pytest.approx(0.05)
    # Odds inválidas
    assert calcular_stake_kelly(0.7, 1.0) == (0, 0)


def test_inferir_nivel_torneio():
    assert inferir_nivel_torneio("Australian Open") == "G"
    assert inferir_nivel_torneio("French Open") == "G"
    assert inferir_nivel_torneio("Paris") == "M"
    assert inferir_nivel_torneio("Masters Cup") == "F"
    assert inferir_nivel_torneio("Metz") == "A"
