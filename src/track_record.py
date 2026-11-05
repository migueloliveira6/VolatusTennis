"""Histórico jogo a jogo das previsões (track record) publicado em docs/analytics/track_record.json.

Cada comparação diária (comparacao_previsoes_<data>.csv) acrescenta uma linha por previsão com
resultado: previsão, confiança, vencedor real, odds, stake sugerido e lucro da aposta.
A junção é idempotente: reprocessar o mesmo dia substitui as linhas desse dia, não as duplica.
"""
import pandas as pd

TRACK_RECORD_FILENAME = 'track_record.json'


def _value(row, column):
    value = row.get(column) if column in row else None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _float(row, column, digits=3):
    value = _value(row, column)
    try:
        return round(float(value), digits) if value is not None else None
    except (TypeError, ValueError):
        return None


def _same_name(a, b):
    return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()


def build_track_rows(df: pd.DataFrame) -> list[dict]:
    """Converte um CSV de comparação nas linhas do histórico (só previsões com resultado)."""
    if df is None or df.empty:
        return []
    matched = df[df['Matched'].astype(str).str.lower().isin(['true', '1'])] if 'Matched' in df.columns else df
    rows = []
    for _, row in matched.iterrows():
        pick = _value(row, 'Vencedor Previsto')
        p1, p2 = _value(row, 'Jogador 1'), _value(row, 'Jogador 2')
        # odd do jogador escolhido pelo modelo
        odd = _float(row, 'Odd 1', 2) if _same_name(pick, p1) else _float(row, 'Odd 2', 2) if _same_name(pick, p2) else None
        stake = _float(row, 'Valor Aposta', 4) or 0.0
        correct = str(_value(row, 'Correct')).lower() in ('true', '1')
        profit = _float(row, 'Lucro Aposta', 4)
        if stake > 0 and profit is None:
            profit = round(stake * (odd - 1), 4) if correct and odd else (-stake if not correct else None)
        rows.append({
            'date': str(_value(row, 'Pred_Date'))[:10],
            'tourney': _value(row, 'Pred_Torneio') or _value(row, 'Match_Tourney'),
            'surface': _value(row, 'Superfície'),
            'p1': p1,
            'p2': p2,
            'pick': pick,
            'winner': _value(row, 'Actual_Winner'),
            'score': _value(row, 'Score'),
            'conf': _float(row, 'Confiança (%)', 1),
            'correct': correct,
            'odd': odd,
            'stake': stake,
            'profit': profit if stake > 0 else 0.0,
        })
    return rows


def _key(row):
    players = tuple(sorted([str(row.get('p1')), str(row.get('p2'))]))
    return (row.get('date'), *players)


def merge_track_record(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Junta linhas novas ao histórico (as novas substituem as do mesmo dia e jogo) e ordena por data."""
    merged = {_key(row): row for row in existing or []}
    for row in new_rows:
        merged[_key(row)] = row
    return sorted(merged.values(), key=lambda r: (r.get('date') or '', r.get('tourney') or '', r.get('p1') or ''))
