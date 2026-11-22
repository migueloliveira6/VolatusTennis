#!/usr/bin/env python3
"""Verificador de jogos novos (corre ~6h depois do workflow principal).

Alguns jogos do dia só aparecem no TennisExplorer depois de o workflow principal correr
(a ronda anterior acaba tarde). Este script procura, para o dia já publicado no site, jogos
com odds que ainda não começaram e que não estão no predictions.json, prevê só esses e
acrescenta-os ao predictions.json e ao snapshot do dia (que o comparador usa no dia seguinte).

Usage:
    python src/check_new_matches.py --site-dir site                 # site-dir = conteúdo do gh-pages
    python src/check_new_matches.py --site-dir site --dry-run       # não escreve nem envia Telegram
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_predictions_results import normalize_name
from engine import (CHAT_ID, MODEL_PATH, TOKEN_BOT, NeuroTennis, TennisPredicaoAnalyzer,
                    enviar_resultados_telegram, enviar_telegram)

logger = logging.getLogger('check_new_matches')


def par_jogadores(jogador1: str, jogador2: str) -> frozenset:
    """Identifica um jogo pelo par de jogadores, em qualquer ordem."""
    return frozenset({normalize_name(jogador1), normalize_name(jogador2)})


def ler_json(path: str, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def escrever_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def selecionar_jogos_novos(jogos_por_torneio: dict, publicadas: list[dict]) -> tuple[dict, dict]:
    """Jogos com odds, por começar e ainda não publicados. Devolve (jogos por torneio, contagens)."""
    vistos = {par_jogadores(p['Jogador 1'], p['Jogador 2']) for p in publicadas}
    novos, contagens = {}, {'sem_odds': 0, 'iniciados': 0, 'ja_publicados': 0, 'novos': 0}
    for torneio, jogos in jogos_por_torneio.items():
        for jogo in jogos:
            if jogo.odd1 is None or jogo.odd2 is None:
                contagens['sem_odds'] += 1
                continue
            if getattr(jogo, 'iniciado', False):
                contagens['iniciados'] += 1
                continue
            chave = par_jogadores(jogo.player1, jogo.player2)
            if chave in vistos:
                contagens['ja_publicados'] += 1
                continue
            vistos.add(chave)
            novos.setdefault(torneio, []).append(jogo)
            contagens['novos'] += 1
    return novos, contagens


def mensagem_novos_jogos(df_novas: pd.DataFrame) -> str:
    com_valor = int((pd.to_numeric(df_novas['Valor Aposta'], errors='coerce') > 0).sum())
    linhas = [f"🎾 Novos jogos para hoje: {len(df_novas)} (com valor: {com_valor})"]
    for _, row in df_novas.iterrows():
        linhas.append(f"• {row['Jogador 1']} vs {row['Jogador 2']} ({row['Torneio']}): "
                      f"{row['Vencedor Previsto']} {row['Confiança (%)']:.1f}%")
    return "\n".join(linhas)


def _carregar_modelo():
    from model_elo_xgboost import TennisPredictor
    predictor = TennisPredictor()
    if not predictor.load_saved_model(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado ou inválido em '{MODEL_PATH}'")
    return predictor


def verificar(site_dir: str, data_alvo: date | None = None, dry_run: bool = False,
              scraper=None, predictor=None, hoje: date | None = None) -> dict:
    """Procura e acrescenta jogos novos ao dia publicado. Devolve um resumo."""
    predicts_dir = os.path.join(site_dir, 'predicts')
    status_path = os.path.join(predicts_dir, 'status.json')
    predictions_path = os.path.join(predicts_dir, 'predictions.json')

    status = ler_json(status_path, {})
    if data_alvo is None:
        try:
            data_alvo = datetime.strptime(str(status.get('date', ''))[:10], '%Y-%m-%d').date()
        except ValueError:
            return {'has_new': False, 'motivo': 'status.json sem data'}
    hoje = hoje or datetime.now(timezone.utc).date()
    if data_alvo < hoje:
        return {'has_new': False, 'data': str(data_alvo), 'motivo': 'a data publicada já passou'}

    publicadas = ler_json(predictions_path, [])
    scraper = scraper or NeuroTennis(timeout=15)
    jogos = scraper.extrair_jogos_agrupados_por_torneio(data_alvo=data_alvo)
    novos, contagens = selecionar_jogos_novos(jogos, publicadas)
    resumo = {'has_new': False, 'data': str(data_alvo), **contagens}
    if not novos:
        return resumo

    analyzer = TennisPredicaoAnalyzer(predictor=predictor or _carregar_modelo(), max_workers=4, calcular_roi=True)
    resultados, _ = analyzer.analisar_jogos(
        novos, data=datetime.combine(data_alvo, datetime.min.time()), processar_paralelo=False)
    novas = [r.to_dict() for r in resultados]
    resumo['previstos'] = len(novas)  # jogadores desconhecidos do modelo ficam de fora
    if not novas:
        return resumo

    todas = sorted(publicadas + novas, key=lambda p: p.get('Confiança (%)', 0), reverse=True)
    resumo.update(has_new=True, jogos_novos=[f"{p['Jogador 1']} vs {p['Jogador 2']}" for p in novas])
    if dry_run:
        return resumo

    escrever_json(predictions_path, todas)
    escrever_json(os.path.join(predicts_dir, 'history', f'predictions_{data_alvo}.json'), todas)
    if status.get('status') != 'ready':
        escrever_json(status_path, {'status': 'ready', 'message': 'Previsões disponíveis.', 'date': str(data_alvo)})

    if TOKEN_BOT and CHAT_ID:
        df_novas = pd.DataFrame(novas)
        enviar_telegram(mensagem_novos_jogos(df_novas), TOKEN_BOT, CHAT_ID)
        enviar_resultados_telegram(df_novas, TOKEN_BOT, CHAT_ID)
    return resumo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Acrescenta ao dia publicado os jogos que apareceram depois')
    parser.add_argument('--site-dir', required=True, help='Pasta com o conteúdo do gh-pages (tem predicts/)')
    parser.add_argument('--date', default=None, help='Data alvo YYYY-MM-DD (default: a do status.json)')
    parser.add_argument('--dry-run', action='store_true', help='Não escreve ficheiros nem envia Telegram')
    args = parser.parse_args(argv)

    data_alvo = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else None
    resumo = verificar(args.site_dir, data_alvo=data_alvo, dry_run=args.dry_run)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))

    output = os.getenv('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as f:
            f.write(f"has_new={'true' if resumo['has_new'] and not args.dry_run else 'false'}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
