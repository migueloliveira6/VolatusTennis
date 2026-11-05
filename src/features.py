"""Estado incremental dos jogadores e construção das features do modelo.

Uma única passagem cronológica pelos jogos: para cada jogo calculam-se as features com o estado
*antes* do jogo e só depois se atualiza o estado. O mesmo estado (gravado no pkl) serve a previsão.
O estado é feito só de dicionários/deques, para o pkl não depender de classes deste módulo.
"""
import math
from collections import deque

import numpy as np
import pandas as pd

try:
    from elo_calculator import INITIAL_ELO, expected_score, level_k_multiplier, update_elo_dynamic
    from match_policy import elo_k_multiplier
except ImportError:
    from src.elo_calculator import INITIAL_ELO, expected_score, level_k_multiplier, update_elo_dynamic
    from src.match_policy import elo_k_multiplier

FORM_WINDOWS = (5, 10, 25, 50)
SURFACE_ELO_WEIGHT = 0.5          # ELO combinado = 50% superfície + 50% global
EWMA_HALF_LIFE_MATCHES = 20       # stats de serviço/resposta: meia-vida de 20 jogos com stats
EWMA_ALPHA = 1 - 0.5 ** (1 / EWMA_HALF_LIFE_MATCHES)
# Valores iniciais (≈ médias do circuito): funcionam como shrinkage para jogadores com poucos jogos
SERVE_PRIORS = {'spw': 0.62, 'rpw': 0.38, 'ace_rate': 0.07, 'df_rate': 0.035, 'spw_adj': 0.0, 'rpw_adj': 0.0}
ELO_HISTORY_LEN = 80
RECENT_MATCHES_LEN = 30
MAX_REST_DAYS = 365

FEATURE_GROUPS = {
    'elo': ['player_elo', 'opponent_elo', 'elo_diff', 'player_surface_elo', 'opponent_surface_elo',
            'surface_elo_diff', 'blend_elo_diff', 'elo_win_prob', 'player_elo_matches', 'opponent_elo_matches'],
    'ranking': ['player_rank', 'opponent_rank', 'log_rank_diff', 'player_rank_points', 'opponent_rank_points',
                'rank_points_diff', 'player_age', 'opponent_age', 'age_diff'],
    'form': [f'win_last_{w}_diff' for w in FORM_WINDOWS] + [
        'player_surface_win_rate', 'opponent_surface_win_rate', 'surface_win_rate_diff',
        'player_surface_matches', 'opponent_surface_matches', 'surface_matches_diff'],
    'h2h': ['h2h_win_rate', 'h2h_matches'],
    'serve_return': ['player_spw', 'opponent_spw', 'spw_diff', 'player_rpw', 'opponent_rpw', 'rpw_diff',
                     'spw_adj_diff', 'rpw_adj_diff', 'ace_rate_diff', 'df_rate_diff',
                     'player_stats_matches', 'opponent_stats_matches'],
    'momentum': ['player_elo_delta_30', 'opponent_elo_delta_30', 'player_elo_delta_90', 'opponent_elo_delta_90',
                 'elo_delta_90_diff'],
    'fatigue': ['player_days_rest', 'opponent_days_rest', 'days_rest_diff', 'player_matches_14d',
                'opponent_matches_14d', 'matches_14d_diff', 'player_minutes_7d', 'opponent_minutes_7d',
                'minutes_7d_diff'],
    'context': ['best_of', 'is_qualifying', 'surface', 'tourney_level'],
}


def _num(value):
    """Converte para float, devolvendo NaN para valores em falta."""
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _diff(a, b):
    return np.nan if (pd.isna(a) or pd.isna(b)) else float(a) - float(b)


def _new_player():
    return {
        'elo': INITIAL_ELO, 'n': 0,
        'surface_elo': {}, 'surface_n': {},
        'surface_w': {}, 'surface_l': {},
        'results': deque(maxlen=max(FORM_WINDOWS)),
        'elo_hist': deque(maxlen=ELO_HISTORY_LEN),   # (data, ELO global depois do jogo)
        'recent': deque(maxlen=RECENT_MATCHES_LEN),   # (data, minutos)
        'last_date': None,
        'ewma': dict(SERVE_PRIORS), 'stats_n': 0,
        'rank': np.nan, 'rank_points': np.nan, 'age': np.nan,
    }


class FeatureBuilder:
    """Mantém o estado de todos os jogadores e calcula features ponto-no-tempo."""

    def __init__(self, state=None):
        state = state or {}
        self.players = state.get('players', {})
        self.h2h = state.get('h2h', {})  # (a, b) -> vitórias de a sobre b
        self.last_date = state.get('last_date')

    def get_state(self):
        return {'players': self.players, 'h2h': self.h2h, 'last_date': self.last_date}

    def knows(self, name):
        return name in self.players

    def player(self, name):
        if name not in self.players:
            self.players[name] = _new_player()
        return self.players[name]

    # ------------------------------------------------------------------ features

    @staticmethod
    def _elo_delta(state, date, days):
        """Variação do ELO global nos últimos `days` dias (0 sem histórico)."""
        hist = state['elo_hist']
        if not hist:
            return 0.0
        cutoff = date - pd.Timedelta(days=days)
        for hist_date, elo in reversed(hist):
            if hist_date <= cutoff:
                return state['elo'] - elo
        return state['elo'] - INITIAL_ELO if state['n'] <= len(hist) else state['elo'] - hist[0][1]

    @staticmethod
    def _fatigue(state, date):
        if state['last_date'] is None:
            rest = MAX_REST_DAYS
        else:
            rest = min(max((date - state['last_date']).days, 0), MAX_REST_DAYS)
        matches_14d = sum(1 for d, _ in state['recent'] if (date - d).days <= 14)
        minutes_7d = sum(m for d, m in state['recent'] if (date - d).days <= 7 and not pd.isna(m))
        return rest, matches_14d, minutes_7d

    def features(self, player, opponent, surface, tourney_level, date, best_of=3, is_qualifying=0,
                 player_rank=None, opponent_rank=None, player_rank_points=None, opponent_rank_points=None,
                 player_age=None, opponent_age=None):
        """Features do ponto de vista de `player` contra `opponent`, com o estado antes do jogo.

        Ranking/idade: valores passados (treino, do próprio jogo) ou, se None, os últimos conhecidos.
        """
        p = self.players.get(player) or _new_player()
        o = self.players.get(opponent) or _new_player()
        date = pd.Timestamp(date)

        p_surface_elo = p['surface_elo'].get(surface, INITIAL_ELO)
        o_surface_elo = o['surface_elo'].get(surface, INITIAL_ELO)
        p_blend = SURFACE_ELO_WEIGHT * p_surface_elo + (1 - SURFACE_ELO_WEIGHT) * p['elo']
        o_blend = SURFACE_ELO_WEIGHT * o_surface_elo + (1 - SURFACE_ELO_WEIGHT) * o['elo']

        pr = p['rank'] if player_rank is None else _num(player_rank)
        orank = o['rank'] if opponent_rank is None else _num(opponent_rank)
        ppts = p['rank_points'] if player_rank_points is None else _num(player_rank_points)
        opts = o['rank_points'] if opponent_rank_points is None else _num(opponent_rank_points)
        page = p['age'] if player_age is None else _num(player_age)
        oage = o['age'] if opponent_age is None else _num(opponent_age)
        log_rank_diff = (math.log(pr) - math.log(orank)) if (pr > 0 and orank > 0) else np.nan

        wins = self.h2h.get((player, opponent), 0)
        losses = self.h2h.get((opponent, player), 0)
        h2h_total = wins + losses

        p_sw, p_sl = p['surface_w'].get(surface, 0), p['surface_l'].get(surface, 0)
        o_sw, o_sl = o['surface_w'].get(surface, 0), o['surface_l'].get(surface, 0)
        p_swr = p_sw / (p_sw + p_sl) if p_sw + p_sl else 0.5
        o_swr = o_sw / (o_sw + o_sl) if o_sw + o_sl else 0.5

        p_results, o_results = list(p['results']), list(o['results'])
        form = {f'win_last_{w}_diff': sum(p_results[-w:]) - sum(o_results[-w:]) for w in FORM_WINDOWS}

        pe, oe = p['ewma'], o['ewma']
        p_rest, p_m14, p_min7 = self._fatigue(p, date)
        o_rest, o_m14, o_min7 = self._fatigue(o, date)
        p_d30, p_d90 = self._elo_delta(p, date, 30), self._elo_delta(p, date, 90)
        o_d30, o_d90 = self._elo_delta(o, date, 30), self._elo_delta(o, date, 90)

        return {
            # elo
            'player_elo': p['elo'], 'opponent_elo': o['elo'], 'elo_diff': p['elo'] - o['elo'],
            'player_surface_elo': p_surface_elo, 'opponent_surface_elo': o_surface_elo,
            'surface_elo_diff': p_surface_elo - o_surface_elo,
            'blend_elo_diff': p_blend - o_blend,
            'elo_win_prob': expected_score(p_blend, o_blend),
            'player_elo_matches': math.log1p(p['n']), 'opponent_elo_matches': math.log1p(o['n']),
            # ranking
            'player_rank': pr, 'opponent_rank': orank, 'log_rank_diff': log_rank_diff,
            'player_rank_points': ppts, 'opponent_rank_points': opts, 'rank_points_diff': _diff(ppts, opts),
            'player_age': page, 'opponent_age': oage, 'age_diff': _diff(page, oage),
            # form
            **form,
            'player_surface_win_rate': p_swr, 'opponent_surface_win_rate': o_swr,
            'surface_win_rate_diff': p_swr - o_swr,
            'player_surface_matches': p_sw + p_sl, 'opponent_surface_matches': o_sw + o_sl,
            'surface_matches_diff': (p_sw + p_sl) - (o_sw + o_sl),
            # h2h
            'h2h_win_rate': wins / h2h_total if h2h_total else 0.5, 'h2h_matches': h2h_total,
            # serve_return
            'player_spw': pe['spw'], 'opponent_spw': oe['spw'], 'spw_diff': pe['spw'] - oe['spw'],
            'player_rpw': pe['rpw'], 'opponent_rpw': oe['rpw'], 'rpw_diff': pe['rpw'] - oe['rpw'],
            'spw_adj_diff': pe['spw_adj'] - oe['spw_adj'], 'rpw_adj_diff': pe['rpw_adj'] - oe['rpw_adj'],
            'ace_rate_diff': pe['ace_rate'] - oe['ace_rate'], 'df_rate_diff': pe['df_rate'] - oe['df_rate'],
            'player_stats_matches': math.log1p(p['stats_n']), 'opponent_stats_matches': math.log1p(o['stats_n']),
            # momentum
            'player_elo_delta_30': p_d30, 'opponent_elo_delta_30': o_d30,
            'player_elo_delta_90': p_d90, 'opponent_elo_delta_90': o_d90, 'elo_delta_90_diff': p_d90 - o_d90,
            # fatigue
            'player_days_rest': p_rest, 'opponent_days_rest': o_rest, 'days_rest_diff': p_rest - o_rest,
            'player_matches_14d': p_m14, 'opponent_matches_14d': o_m14, 'matches_14d_diff': p_m14 - o_m14,
            'player_minutes_7d': p_min7, 'opponent_minutes_7d': o_min7, 'minutes_7d_diff': p_min7 - o_min7,
            # context
            'best_of': int(best_of) if not pd.isna(best_of) else 3,
            'is_qualifying': int(is_qualifying),
            'surface': surface,
            'tourney_level': str(tourney_level) if tourney_level is not None and not pd.isna(tourney_level) else 'UNK',
        }

    # ------------------------------------------------------------------ atualização

    def _update_serve_stats(self, w, l, row):
        w_svpt, l_svpt = _num(row.get('w_svpt')), _num(row.get('l_svpt'))
        if not (w_svpt > 0 and l_svpt > 0):
            return
        w_won = _num(row.get('w_1stWon')) + _num(row.get('w_2ndWon'))
        l_won = _num(row.get('l_1stWon')) + _num(row.get('l_2ndWon'))
        if pd.isna(w_won) or pd.isna(l_won):
            return
        w_spw, l_spw = w_won / w_svpt, l_won / l_svpt
        observations = {
            id(w): {'spw': w_spw, 'rpw': 1 - l_spw,
                    'ace_rate': _num(row.get('w_ace')) / w_svpt, 'df_rate': _num(row.get('w_df')) / w_svpt,
                    # ajustadas ao adversário: acima do que o adversário costuma conceder
                    'spw_adj': w_spw - (1 - l['ewma']['rpw']), 'rpw_adj': (1 - l_spw) - (1 - l['ewma']['spw'])},
            id(l): {'spw': l_spw, 'rpw': 1 - w_spw,
                    'ace_rate': _num(row.get('l_ace')) / l_svpt, 'df_rate': _num(row.get('l_df')) / l_svpt,
                    'spw_adj': l_spw - (1 - w['ewma']['rpw']), 'rpw_adj': (1 - w_spw) - (1 - w['ewma']['spw'])},
        }
        for state in (w, l):
            for key, value in observations[id(state)].items():
                if not pd.isna(value):
                    state['ewma'][key] += EWMA_ALPHA * (value - state['ewma'][key])
            state['stats_n'] += 1

    def update(self, row):
        """Atualiza o estado com um jogo já disputado (linha com nomes/colunas do Sackmann)."""
        multiplier = elo_k_multiplier(row.get('score'), row.get('tourney_name'))
        if multiplier <= 0:
            return
        winner_name, loser_name = row['winner_name'], row['loser_name']
        w, l = self.player(winner_name), self.player(loser_name)
        surface = row['surface']
        date = pd.Timestamp(row['tourney_date'])
        multiplier *= level_k_multiplier(row.get('tourney_level'), row.get('round'))

        # ELO global e por superfície (K dinâmico pelo nº de jogos de cada jogador)
        w['elo'], l['elo'], _, _ = update_elo_dynamic(w['elo'], l['elo'], w['n'], l['n'], multiplier)
        w_se, l_se, _, _ = update_elo_dynamic(
            w['surface_elo'].get(surface, INITIAL_ELO), l['surface_elo'].get(surface, INITIAL_ELO),
            w['surface_n'].get(surface, 0), l['surface_n'].get(surface, 0), multiplier)
        w['surface_elo'][surface], l['surface_elo'][surface] = w_se, l_se

        self._update_serve_stats(w, l, row)

        minutes = _num(row.get('minutes'))
        for state, won, prefix in ((w, 1, 'winner'), (l, 0, 'loser')):
            state['n'] += 1
            state['surface_n'][surface] = state['surface_n'].get(surface, 0) + 1
            bucket = 'surface_w' if won else 'surface_l'
            state[bucket][surface] = state[bucket].get(surface, 0) + 1
            state['results'].append(won)
            state['elo_hist'].append((date, state['elo']))
            state['recent'].append((date, minutes))
            state['last_date'] = date
            for field in ('rank', 'rank_points', 'age'):
                value = _num(row.get(f'{prefix}_{field}'))
                if not pd.isna(value):
                    state[field] = value

        self.h2h[(winner_name, loser_name)] = self.h2h.get((winner_name, loser_name), 0) + 1
        if self.last_date is None or date > self.last_date:
            self.last_date = date
