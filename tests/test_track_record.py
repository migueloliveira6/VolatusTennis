import pandas as pd
import pytest

from track_record import build_track_rows, merge_track_record


def _comparison(rows):
    base = {'Pred_File': 'predictions_2026-11-05.json', 'Pred_Date': '2026-11-05', 'Pred_Torneio': 'Paris',
            'Superfície': 'Hard', 'Matched': True, 'Score': '6-4 6-4'}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_build_rows_uses_pick_odds_and_bet_profit():
    df = _comparison([
        # aposta ganha: odd do escolhido (Jogador 2)
        {'Jogador 1': 'A', 'Jogador 2': 'B', 'Vencedor Previsto': 'B', 'Confiança (%)': 62.0,
         'Odd 1': 2.4, 'Odd 2': 1.8, 'Valor Aposta': 0.03, 'Lucro Aposta': 0.024,
         'Actual_Winner': 'B', 'Correct': True},
        # aposta perdida, sem coluna de lucro preenchida
        {'Jogador 1': 'C', 'Jogador 2': 'D', 'Vencedor Previsto': 'C', 'Confiança (%)': 71.0,
         'Odd 1': 1.6, 'Odd 2': 2.5, 'Valor Aposta': 0.02, 'Lucro Aposta': None,
         'Actual_Winner': 'D', 'Correct': False},
        # sem aposta
        {'Jogador 1': 'E', 'Jogador 2': 'F', 'Vencedor Previsto': 'E', 'Confiança (%)': 55.0,
         'Odd 1': 1.9, 'Odd 2': 1.9, 'Valor Aposta': 0.0, 'Actual_Winner': 'E', 'Correct': True},
        # sem resultado: fica de fora
        {'Jogador 1': 'G', 'Jogador 2': 'H', 'Vencedor Previsto': 'G', 'Matched': False, 'Correct': False},
    ])

    rows = build_track_rows(df)

    assert len(rows) == 3
    won, lost, no_bet = rows
    assert won['odd'] == 1.8 and won['stake'] == 0.03 and won['profit'] == pytest.approx(0.024)
    assert won['surface'] == 'Hard' and won['correct'] is True
    assert lost['odd'] == 1.6 and lost['profit'] == pytest.approx(-0.02) and lost['correct'] is False
    assert no_bet['stake'] == 0.0 and no_bet['profit'] == 0.0


def test_build_rows_handles_old_comparison_csv_without_odds_or_stake():
    df = pd.DataFrame([{'Pred_Date': '2026-01-05', 'Pred_Torneio': 'Brisbane', 'Jogador 1': 'A', 'Jogador 2': 'B',
                        'Vencedor Previsto': 'A', 'Confiança (%)': 66.6, 'Actual_Winner': 'A',
                        'Matched': 'True', 'Correct': 'True'}])
    row = build_track_rows(df)[0]
    assert row['odd'] is None and row['stake'] == 0.0 and row['surface'] is None and row['correct'] is True


def test_merge_is_idempotent_and_sorted():
    day1 = [{'date': '2026-11-05', 'p1': 'A', 'p2': 'B', 'correct': True}]
    day0 = [{'date': '2026-11-04', 'p1': 'C', 'p2': 'D', 'correct': False}]
    merged = merge_track_record(day1, day0)
    assert [r['date'] for r in merged] == ['2026-11-04', '2026-11-05']
    # reprocessar o mesmo jogo (mesmo com jogadores trocados) substitui, não duplica
    again = merge_track_record(merged, [{'date': '2026-11-05', 'p1': 'B', 'p2': 'A', 'correct': False}])
    assert len(again) == 2
    assert again[-1]['correct'] is False
