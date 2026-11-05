"""Política de tratamento de jogos especiais (abandonos, walkovers, exibições) para ELO e treino.

Regras (decididas para o VolatusTennis):
- W/O (walkover): o jogo não foi disputado → fora do ELO e do treino.
- RET (abandono) e DEF (desclassificação): contam para o ELO com K × 0.5 e para o histórico
  (forma, H2H, superfície), mas não são exemplos de treino (o resultado é ruído).
- Laver Cup: exibição → fora do ELO e do treino.
- Davis Cup, Jogos Olímpicos, United Cup, ATP Cup e qualifying contam normalmente.
"""
import re

import pandas as pd

RESULT_COMPLETED = 'completed'
RESULT_RETIRED = 'retired'
RESULT_DEFAULT = 'default'
RESULT_WALKOVER = 'walkover'

ELO_K_MULTIPLIER = {
    RESULT_COMPLETED: 1.0,
    RESULT_RETIRED: 0.5,
    RESULT_DEFAULT: 0.5,
    RESULT_WALKOVER: 0.0,
}

EXHIBITION_TOURNAMENTS = ('laver cup',)

_WALKOVER_RE = re.compile(r'\bW/?O\b|walkover', re.IGNORECASE)
# Abandonos e jogos não terminados ('Played and abandoned', 'ABD', 'unfinished') tratam-se como RET
_RETIRED_RE = re.compile(r'\bRET\b|\bRE\b|retired|abandoned|unfinished|\bABD\b', re.IGNORECASE)
_DEFAULT_RE = re.compile(r'\bDEF\b|\bDefault\b|\bDisq', re.IGNORECASE)


def classify_result(score) -> str:
    """Classifica o resultado pelo texto do score (Sackmann: 'W/O', '6-4 2-1 RET', 'DEF')."""
    if score is None or (not isinstance(score, str) and pd.isna(score)):
        return RESULT_COMPLETED
    text = str(score)
    if _WALKOVER_RE.search(text):
        return RESULT_WALKOVER
    if _RETIRED_RE.search(text):
        return RESULT_RETIRED
    if _DEFAULT_RE.search(text):
        return RESULT_DEFAULT
    return RESULT_COMPLETED


def is_exhibition(tourney_name) -> bool:
    """Exibições que não contam para o ELO nem para o treino."""
    if tourney_name is None or (not isinstance(tourney_name, str) and pd.isna(tourney_name)):
        return False
    name = str(tourney_name).lower()
    return any(exhibition in name for exhibition in EXHIBITION_TOURNAMENTS)


def elo_k_multiplier(score, tourney_name=None) -> float:
    """Multiplicador do K do ELO para o jogo (0 = não atualiza o ELO)."""
    if is_exhibition(tourney_name):
        return 0.0
    return ELO_K_MULTIPLIER[classify_result(score)]


def counts_for_history(score, tourney_name=None) -> bool:
    """O jogo entra no histórico do jogador (ELO, forma, H2H, superfície)?"""
    return elo_k_multiplier(score, tourney_name) > 0


def is_training_target(score, tourney_name=None) -> bool:
    """O jogo pode ser um exemplo de treino (jogo completo e oficial)?"""
    return not is_exhibition(tourney_name) and classify_result(score) == RESULT_COMPLETED


# Ordem das rondas dentro de um torneio (o match_num do Sackmann não segue as rondas)
ROUND_ORDER = {
    'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'ER': 4,
    'R128': 5, 'R64': 6, 'R32': 7, 'R16': 8, 'RR': 9,
    'QF': 10, 'SF': 11, 'BR': 12, 'F': 13,
}


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena cronologicamente: data do torneio, torneio, ronda e match_num (ordenação estável)."""
    out = df.copy()
    out['_round_order'] = out['round'].map(ROUND_ORDER).fillna(4.5) if 'round' in out.columns else 0
    keys = [c for c in ('tourney_date', 'tourney_id', '_round_order', 'match_num') if c in out.columns]
    out = out.sort_values(keys, kind='mergesort', na_position='last')
    return out.drop(columns='_round_order')


def add_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `result_type`, `counts_for_history` e `training_target` a um DataFrame de jogos.

    Sem colunas `score`/`tourney_name` (BD antiga) todos os jogos contam como completos.
    """
    out = df.copy()
    scores = out['score'] if 'score' in out.columns else pd.Series([None] * len(out), index=out.index)
    names = out['tourney_name'] if 'tourney_name' in out.columns else pd.Series([None] * len(out), index=out.index)
    out['result_type'] = scores.map(classify_result)
    exhibition = names.map(is_exhibition)
    out['counts_for_history'] = (out['result_type'] != RESULT_WALKOVER) & ~exhibition
    out['training_target'] = (out['result_type'] == RESULT_COMPLETED) & ~exhibition
    return out
