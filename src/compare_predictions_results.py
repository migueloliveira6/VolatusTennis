#!/usr/bin/env python3
"""
Comparador de Previsões vs Resultados
- Localiza o arquivo de previsões mais recente em `PREVISOES_PATH` (ou usa --pred-file)
- Determina a data alvo (pode ser extraída do nome do ficheiro ou passada via --results-date)
- Faz scraping dos resultados com `scraper_results.ResultsScraper.scrape_results`
- Normaliza nomes e faz o matching das partidas
- Calcula métricas (matched, accuracy) e salva CSV de comparação
"""

import os
import re
import argparse
from datetime import datetime, timedelta
import unicodedata
import pandas as pd
import requests
from typing import Optional

# Importar função do scraper local
try:
    from scraper_results import ResultsScraper
except Exception:
    # Tentar ajustar import se for executado de outro working dir
    import sys
    sys.path.append(os.path.dirname(__file__))
    from scraper_results import ResultsScraper


PREDICTION_COLUMN_ALIASES = {
    'Torneio': ['Torneio', 'tourney', 'tournament', 'Tournament', 'torneio'],
    'Jogador 1': ['Jogador 1', 'player1', 'player_1', 'Player 1', 'jogador1', 'jogador_1'],
    'Jogador 2': ['Jogador 2', 'player2', 'player_2', 'Player 2', 'jogador2', 'jogador_2'],
    'Vencedor Previsto': ['Vencedor Previsto', 'winner_pred', 'predicted_winner', 'winner_prediction', 'predictedWinner'],
    'Confiança (%)': ['Confiança (%)', 'confidence', 'Confidence', 'confidence_pct', 'probability'],
    'Valor Aposta': ['Valor Aposta', 'valor_aposta', 'stake', 'bet_value'],
    'ROI Esperado (%)': ['ROI Esperado (%)', 'roi_esperado', 'expected_roi', 'roi_expected']
}


def find_latest_prediction_file(previsoes_path: str) -> Optional[str]:
    files = [f for f in os.listdir(previsoes_path) if f.startswith('previsoes_tenis') and f.endswith('.csv')]
    if not files:
        return None
    files_full = [os.path.join(previsoes_path, f) for f in files]
    files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files_full[0]


def find_prediction_file_for_date(previsoes_path: str, target_date: datetime) -> Optional[str]:
    files = [f for f in os.listdir(previsoes_path) if f.startswith('previsoes_tenis') and f.endswith('.csv')]
    if not files:
        return None

    target = target_date.date()
    candidates = []
    for f in files:
        dt = extract_date_from_filename(f)
        if dt and dt.date() == target:
            candidates.append(os.path.join(previsoes_path, f))

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if m:
        try:
            return datetime.strptime(m.group(1), '%Y-%m-%d')
        except ValueError:
            return None
    return None


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Normalize all whitespace (including newlines/tabs) before filtering punctuation.
    s = re.sub(r'\s+', ' ', s)
    # Keep letters, numbers and spaces
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def normalize_prediction_dataframe(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Map known column aliases to canonical names used by comparison logic."""
    if df_pred is None or df_pred.empty:
        return df_pred

    rename_map = {}
    columns_set = set(df_pred.columns)

    for canonical, aliases in PREDICTION_COLUMN_ALIASES.items():
        if canonical in columns_set:
            continue
        for alias in aliases:
            if alias in columns_set:
                rename_map[alias] = canonical
                break

    if rename_map:
        df_pred = df_pred.rename(columns=rename_map)

    return df_pred


def match_prediction(pred_row: pd.Series, results_df: pd.DataFrame) -> Optional[pd.Series]:
    p1 = normalize_name(pred_row.get('Jogador 1') or pred_row.get('player1') or '')
    p2 = normalize_name(pred_row.get('Jogador 2') or pred_row.get('player2') or '')
    torneio_pred = normalize_name(pred_row.get('Torneio') or pred_row.get('tourney') or '')

    if not p1 or not p2:
        return None

    for _, res in results_df.iterrows():
        winner = normalize_name(res.get('winner', ''))
        loser = normalize_name(res.get('loser', ''))
        tourney = normalize_name(res.get('tourney_name', ''))

        # verificar se os jogadores coincidem (independente da ordem)
        if {p1, p2} == {winner, loser}:
            # preferir que o torneio coincida ou que seja substring
            if torneio_pred and torneio_pred in tourney:
                return res
            else:
                # se não houver match de torneio, ainda podemos aceitar
                return res

    # Nenhum match exato encontrado
    return None


def compare(pred_file: Optional[str], results_date: datetime, out_dir: str = 'previsoes', df_pred: Optional[pd.DataFrame] = None, pred_file_label: Optional[str] = None):
    if df_pred is None:
        if not pred_file:
            raise ValueError("pred_file é obrigatório quando df_pred não é fornecido")
        df_pred = pd.read_csv(pred_file)
        pred_file_label = pred_file_label or os.path.basename(pred_file)
    else:
        pred_file_label = pred_file_label or (os.path.basename(pred_file) if pred_file else "predictions.json")

    df_pred = normalize_prediction_dataframe(df_pred)

    missing_core_cols = [c for c in ['Jogador 1', 'Jogador 2'] if c not in df_pred.columns]
    if missing_core_cols:
        raise ValueError(
            "Colunas essenciais ausentes nas previsões: "
            f"{missing_core_cols}. Colunas disponíveis: {list(df_pred.columns)}"
        )

    # Scrape results for the results_date
    scraper = ResultsScraper()
    results_df = scraper.scrape_results(results_date)

    if results_df.empty:
        print('⚠️  Nenhum resultado encontrado para a data:', results_date.strftime('%Y-%m-%d'))
        return

    # Prepare results DataFrame
    out_rows = []
    total = 0
    correct = 0
    matched = 0
    bets_placed = 0
    bets_won = 0
    bets_lost = 0
    stake_total = 0.0
    return_total = 0.0
    profit_total = 0.0

    def _safe_float(value) -> float:
        try:
            if value is None or pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    for _, pred in df_pred.iterrows():
        total += 1
        matched_row = match_prediction(pred, results_df)
        stake = _safe_float(pred.get('Valor Aposta'))
        bet_placed = stake > 0
        bet_result = 'NO_BET'
        bet_return = 0.0
        bet_profit = 0.0

        if matched_row is not None:
            matched += 1
            actual_winner = matched_row.get('winner')
            actual_loser = matched_row.get('loser')
            score = matched_row.get('score')
            tourney = matched_row.get('tourney_name')
            is_correct = normalize_name(pred.get('Vencedor Previsto', '')) == normalize_name(actual_winner)
            if is_correct:
                correct += 1

            if bet_placed:
                bets_placed += 1
                if is_correct:
                    bets_won += 1
                    odd_vencedor = _safe_float(
                        pred.get('Odd 1')
                        if normalize_name(pred.get('Jogador 1', '')) == normalize_name(actual_winner)
                        else pred.get('Odd 2')
                    )
                    if odd_vencedor > 0:
                        bet_return = round(stake * odd_vencedor, 3)
                        bet_profit = round(bet_return - stake, 3)
                    else:
                        bet_return = round(stake, 3)
                        bet_profit = 0.0
                    bet_result = 'WIN'
                else:
                    bets_lost += 1
                    bet_return = 0.0
                    bet_profit = round(-stake, 3)
                    bet_result = 'LOSS'

                stake_total += stake
                return_total += bet_return
                profit_total += bet_profit
        else:
            actual_winner = None
            actual_loser = None
            score = None
            tourney = None
            is_correct = False
            if bet_placed:
                bet_result = 'UNMATCHED'

        out_rows.append({
            'Pred_File': pred_file_label,
            'Pred_Date': results_date.strftime('%Y-%m-%d'),
            'Pred_Torneio': pred.get('Torneio'),
            'Jogador 1': pred.get('Jogador 1'),
            'Jogador 2': pred.get('Jogador 2'),
            'Vencedor Previsto': pred.get('Vencedor Previsto'),
            'Confiança (%)': pred.get('Confiança (%)'),
            'Superfície': pred.get('Superfície'),
            'Odd 1': pred.get('Odd 1'),
            'Odd 2': pred.get('Odd 2'),
            'Valor Aposta': stake,
            'Aposta Realizada': bool(bet_placed),
            'Resultado Aposta': bet_result,
            'Retorno Aposta': round(bet_return, 3),
            'Lucro Aposta': round(bet_profit, 3),
            'Actual_Winner': actual_winner,
            'Actual_Loser': actual_loser,
            'Score': score,
            'Match_Tourney': tourney,
            'Matched': bool(matched_row is not None),
            'Correct': bool(is_correct)
        })

    df_out = pd.DataFrame(out_rows)

    summary = {
        'predictions_total': total,
        'matched': matched,
        'correct': correct,
        'accuracy': (correct / matched) if matched else None,
        'coverage': (matched / total) if total else None,
        'bets_placed': bets_placed,
        'bets_won': bets_won,
        'bets_lost': bets_lost,
        'stake_total': round(stake_total, 3),
        'return_total': round(return_total, 3),
        'profit_total': round(profit_total, 3),
        'bet_accuracy': (bets_won / bets_placed) if bets_placed else None,
        'bet_roi': (profit_total / stake_total) if stake_total else None
    }

    # Salvar CSV
    date_str = results_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_dir, f'comparacao_previsoes_{date_str}.csv')
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print('\n✅ Comparação concluída')
    print(f"Total matches: {total}")
    print(f"Previsões encontradas: {matched}")
    print(f"Previsões corretas: {correct}")
    if summary['accuracy'] is not None:
        print(f"Accuracy (sobre matched): {summary['accuracy']:.2%}")
    if summary['coverage'] is not None:
        print(f"Cobertura (matched/total): {summary['coverage']:.2%}")
    if summary['bets_placed']:
        print(f"Apostas feitas: {summary['bets_placed']}")
        print(f"Apostas ganhas: {summary['bets_won']}")
        print(f"Apostas perdidas: {summary['bets_lost']}")
        print(f"Valor apostado: {summary['stake_total']:.3f}")
        print(f"Retorno total: {summary['return_total']:.3f}")
        print(f"Lucro/prejuízo: {summary['profit_total']:.3f}")
        if summary['bet_accuracy'] is not None:
            print(f"Acerto das apostas: {summary['bet_accuracy']:.2%}")
        if summary['bet_roi'] is not None:
            print(f"ROI das apostas: {summary['bet_roi']:.2%}")

    print(f"CSV salvo em: {out_path}\n")

    return df_out, summary


def _load_predictions_from_json_url(pred_json_url: str) -> pd.DataFrame:
    res = requests.get(pred_json_url, timeout=30)
    res.raise_for_status()
    data = res.json()
    if not isinstance(data, list):
        raise ValueError("JSON de previsões inválido: esperado uma lista")
    return normalize_prediction_dataframe(pd.DataFrame(data))


def _load_predictions_from_json_file(pred_json_file: str) -> pd.DataFrame:
    df = pd.read_json(pred_json_file)
    if df.empty:
        raise ValueError("JSON de previsões vazio")
    return normalize_prediction_dataframe(df)


def main():
    parser = argparse.ArgumentParser(description='Comparar previsões com resultados reais')
    parser.add_argument('--pred-file', type=str, help='Arquivo de previsões CSV (se não fornecido, procura o ficheiro do dia alvo em PREVISOES_PATH)')
    parser.add_argument('--preds-path', type=str, default=os.getenv('PREVISOES_PATH', 'previsoes'), help='Pasta onde estão as previsões')
    parser.add_argument('--results-date', type=str, help='Data dos resultados (YYYY-MM-DD). Se omitido, usa ontem')
    parser.add_argument('--out-dir', type=str, default='previsoes', help='Diretório para salvar o CSV de comparação')
    parser.add_argument('--pred-json-url', type=str, help='URL do predictions.json (ex: gh-pages)')
    parser.add_argument('--pred-json-file', type=str, help='Caminho local para predictions.json')

    args = parser.parse_args()

    pred_json_url = args.pred_json_url or os.getenv('PREDICTIONS_JSON_URL')
    pred_json_file = args.pred_json_file

    pred_file = args.pred_file
    df_pred = None
    pred_file_label = None

    # Default operacional: comparar sempre o dia anterior
    yesterday = datetime.now() - timedelta(days=1)
    default_results_date = datetime.combine(yesterday.date(), datetime.min.time())

    # Data dos resultados
    if args.results_date:
        results_date = datetime.strptime(args.results_date, '%Y-%m-%d')
    else:
        results_date = default_results_date

    if pred_json_file:
        df_pred = _load_predictions_from_json_file(pred_json_file)
        pred_file_label = os.path.basename(pred_json_file)
    elif pred_json_url:
        df_pred = _load_predictions_from_json_url(pred_json_url)
        pred_file_label = pred_json_url
    else:
        if not pred_file:
            pred_file = find_prediction_file_for_date(args.preds_path, results_date)
            if not pred_file:
                raise FileNotFoundError(
                    f"Nenhum arquivo de previsões encontrado para {results_date.strftime('%Y-%m-%d')} em {args.preds_path}"
                )
        elif not args.results_date:
            # Se o ficheiro for fornecido manualmente e a data não for explícita,
            # usar a data do nome do ficheiro quando disponível.
            dt = extract_date_from_filename(os.path.basename(pred_file))
            if dt:
                results_date = dt

    result = compare(pred_file, results_date, args.out_dir, df_pred=df_pred, pred_file_label=pred_file_label)
    if not result:
        return

    df_out, summary = result

if __name__ == '__main__':
    main()
