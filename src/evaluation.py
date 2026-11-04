"""Métricas de avaliação, baselines e calibração de probabilidades do modelo de previsão.

O calibrador é guardado como dicionário simples (método + objeto sklearn/parâmetros), para que
o pkl não dependa do caminho de import deste módulo.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Níveis do circuito ATP (códigos Sackmann) — o que o site efetivamente prevê
ATP_LEVELS = ('G', 'M', 'A', 'F', 'D')
CALIBRATION_METHODS = ('none', 'platt', 'isotonic')
EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def reliability_table(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Tabela da curva de calibração: por bin de probabilidade prevista, média prevista vs frequência real."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    table = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        table.append({
            'bin_low': round(float(edges[b]), 2),
            'bin_high': round(float(edges[b + 1]), 2),
            'mean_pred': float(y_prob[mask].mean()),
            'frac_pos': float(y_true[mask].mean()),
            'count': int(mask.sum()),
        })
    return table


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: média ponderada (pelo nº de jogos) de |prob. prevista − frequência real| por bin."""
    table = reliability_table(y_true, y_prob, n_bins)
    total = sum(row['count'] for row in table)
    if total == 0:
        return float('nan')
    return float(sum(row['count'] * abs(row['mean_pred'] - row['frac_pos']) for row in table) / total)


def probability_metrics(y_true, y_prob) -> dict:
    """Métricas de probabilidade para uma previsão binária (threshold 0.5 na accuracy)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _clip(y_prob)
    metrics = {
        'n': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        'logloss': float(log_loss(y_true, y_prob, labels=[0, 1])),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'ece': expected_calibration_error(y_true, y_prob),
    }
    metrics['auc'] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float('nan')
    return metrics


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def elo_probability(elo_diff):
    """Probabilidade de vitória pela fórmula ELO clássica a partir da diferença de ELO."""
    return 1.0 / (1.0 + 10 ** (-np.asarray(elo_diff, dtype=float) / 400.0))


def rank_points_probability(player_points, opponent_points):
    """Probabilidade proporcional aos pontos ATP (pts / (pts + pts_adv)); 0.5 quando falta informação."""
    p = np.asarray(player_points, dtype=float)
    o = np.asarray(opponent_points, dtype=float)
    total = p + o
    with np.errstate(divide='ignore', invalid='ignore'):
        prob = np.where((total > 0) & np.isfinite(total), p / total, 0.5)
    return prob


def baseline_metrics(df: pd.DataFrame, y_true) -> dict:
    """Baselines no mesmo conjunto: só ELO, pontos de ranking e 'ganha o mais bem classificado'."""
    y_true = np.asarray(y_true, dtype=int)
    result = {
        'elo': probability_metrics(y_true, elo_probability(df['elo_diff'])),
        'rank_points': probability_metrics(
            y_true, rank_points_probability(df['player_rank_points'], df['opponent_rank_points'])
        ),
    }
    ranks_known = df['player_rank'].notna() & df['opponent_rank'].notna() & (df['player_rank'] != df['opponent_rank'])
    if ranks_known.any():
        picks = (df.loc[ranks_known, 'player_rank'] < df.loc[ranks_known, 'opponent_rank']).astype(int)
        result['higher_ranked'] = {
            'accuracy': float((picks.to_numpy() == y_true[ranks_known.to_numpy()]).mean()),
            'coverage': float(ranks_known.mean()),
        }
    return result


# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------

def fit_calibrator(method: str, y_prob, y_true) -> dict:
    """Ajusta um calibrador ('none', 'platt' ou 'isotonic') sobre probabilidades fora da amostra."""
    if method == 'none':
        return {'method': 'none'}
    if method == 'platt':
        lr = LogisticRegression(C=1e6)
        lr.fit(_logit(y_prob).reshape(-1, 1), np.asarray(y_true, dtype=int))
        return {'method': 'platt', 'coef': float(lr.coef_[0][0]), 'intercept': float(lr.intercept_[0])}
    if method == 'isotonic':
        iso = IsotonicRegression(out_of_bounds='clip', y_min=EPS, y_max=1 - EPS)
        iso.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
        return {'method': 'isotonic', 'model': iso}
    raise ValueError(f"Método de calibração desconhecido: {method}")


def apply_calibration(calibrator: dict | None, y_prob):
    """Aplica o calibrador; sem calibrador (pkl antigo) devolve as probabilidades originais."""
    y_prob = np.asarray(y_prob, dtype=float)
    if not calibrator or calibrator.get('method', 'none') == 'none':
        return y_prob
    if calibrator['method'] == 'platt':
        z = calibrator['coef'] * _logit(y_prob) + calibrator['intercept']
        return _clip(1.0 / (1.0 + np.exp(-z)))
    if calibrator['method'] == 'isotonic':
        return _clip(calibrator['model'].predict(y_prob))
    raise ValueError(f"Método de calibração desconhecido: {calibrator['method']}")


def select_calibration_method(p_calib, y_calib, p_eval, y_eval) -> tuple[str, dict]:
    """Ajusta cada método no bloco de calibração e escolhe o de menor log loss no bloco de avaliação."""
    results = {}
    for method in CALIBRATION_METHODS:
        calibrator = fit_calibrator(method, p_calib, y_calib)
        results[method] = probability_metrics(y_eval, apply_calibration(calibrator, p_eval))
    best = min(results, key=lambda m: results[m]['logloss'])
    return best, results
