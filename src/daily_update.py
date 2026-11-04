#!/usr/bin/env python3
"""
Script de Atualização Diária - Fase 3B
Faz scraping dos resultados de ontem e atualiza a base de dados SQLite

Uso:
    python daily_update.py
    python daily_update.py --date 2024-11-15  # Para uma data específica
    python daily_update.py --csv results.csv  # De ficheiro CSV
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging
from elo_calculator import update_elo, get_k_factor, calculate_temporal_factor
from match_policy import elo_k_multiplier
from scraper_results import scrape_yesterday_results

# CONFIGURAÇÃO
import os

DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '..', 'datasets', 'tennis_data.db'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FUNÇÕES DE BASE DE DADOS

def get_player_elo(cursor, player_name, surface):
    """
    Consulta o ELO atual de um jogador numa superfície.

    Args:
        cursor: SQLite cursor
        player_name: str
        surface: 'Hard', 'Clay', 'Grass', ou 'Carpet'

    Returns:
        float: ELO atual ou 1500 se jogador novo
    """
    col_name = f"current_elo_{surface.lower()}"

    result = cursor.execute(f"""
        SELECT {col_name} FROM players WHERE name = ?
    """, (player_name,)).fetchone()

    if result:
        return result[0]
    else:
        # Jogador novo - inserir na BD
        logger.info(f"  🆕 Novo jogador: {player_name}")
        cursor.execute("""
            INSERT INTO players (
                name, current_elo_hard, current_elo_clay, 
                current_elo_grass, current_elo_carpet
            ) VALUES (?, 1500, 1500, 1500, 1500)
        """, (player_name,))
        return 1500.0


def update_player_elo(cursor, player_name, surface, new_elo, match_date):
    """Atualiza o ELO de um jogador na BD."""
    col_name = f"current_elo_{surface.lower()}"

    cursor.execute(f"""
        UPDATE players 
        SET {col_name} = ?, last_match_date = ?
        WHERE name = ?
    """, (new_elo, match_date, player_name))


def insert_match(cursor, match_data):
    """
    Insere uma nova partida na tabela matches.

    Args:
        match_data: dict com todas as colunas necessárias
    """
    cursor.execute("""
        INSERT INTO matches (
            tourney_date, tourney_name, surface, round, score,
            winner_name, loser_name, tournament_type,
            winner_elo_before, loser_elo_before,
            winner_elo_after, loser_elo_after,
            k_factor_used, effective_k_factor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match_data['date'],
        match_data['tourney_name'],
        match_data['surface'],
        match_data.get('round', 'Unknown'),
        match_data.get('score', ''),
        match_data['winner'],
        match_data['loser'],
        match_data['tournament_type'],
        match_data['winner_elo_before'],
        match_data['loser_elo_before'],
        match_data['winner_elo_after'],
        match_data['loser_elo_after'],
        match_data['k_factor_base'],
        match_data['k_factor_effective']
    ))


def increment_player_stats(cursor, winner_name, loser_name):
    """Atualiza contadores de vitórias/derrotas."""
    # Vencedor
    cursor.execute("""
        UPDATE players 
        SET total_wins = total_wins + 1, 
            total_matches = total_matches + 1
        WHERE name = ?
    """, (winner_name,))

    # Perdedor
    cursor.execute("""
        UPDATE players 
        SET total_losses = total_losses + 1, 
            total_matches = total_matches + 1
        WHERE name = ?
    """, (loser_name,))

# PROCESSAMENTO PRINCIPAL

def process_daily_results(results_df, db_path=DB_PATH):
    """
    Processa os resultados e atualiza a base de dados.

    Args:
        results_df: DataFrame com os resultados
        db_path: Caminho para a BD SQLite
    """
    if results_df.empty:
        logger.warning("⚠️  Nenhum resultado para processar")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.info(f"\n{'='*70}")
    logger.info(f"💾 Processando {len(results_df)} partidas...")
    logger.info(f"{'='*70}\n")

    processed = 0
    errors = 0

    for idx, row in results_df.iterrows():
        try:
            # 1. Obter ELOs atuais
            surface = row['surface']
            winner_elo_before = get_player_elo(cursor, row['winner'], surface)
            loser_elo_before = get_player_elo(cursor, row['loser'], surface)

            # 2. Calcular novos ELOs usando o módulo
            tournament_type = row.get('tournament_type', 'ATP')
            k_base = get_k_factor(tournament_type)

            # Converter data para datetime se for string
            match_date = row['date']
            if isinstance(match_date, str):
                match_date = datetime.strptime(match_date, '%Y-%m-%d')

            # Política de jogos especiais: W/O e exibições não mexem no ELO; abandonos com K × 0.5
            k_multiplier = elo_k_multiplier(row.get('score'), row.get('tourney_name'))
            if k_multiplier > 0:
                # Para jogos recentes, temporal_factor ≈ 1.0
                winner_elo_after, loser_elo_after, k_eff = update_elo(
                    winner_elo_before,
                    loser_elo_before,
                    tournament_type,
                    temporal_factor=calculate_temporal_factor(match_date) * k_multiplier
                )
            else:
                winner_elo_after, loser_elo_after, k_eff = winner_elo_before, loser_elo_before, 0.0

            logger.info(f"  🎾 {row['winner']} def. {row['loser']}")
            logger.info(f"     {surface} | {row['tourney_name']} | {row.get('score', 'N/A')}")
            logger.info(f"     ELO: {winner_elo_before:.0f} → {winner_elo_after:.0f} | "
                       f"{loser_elo_before:.0f} → {loser_elo_after:.0f} "
                       f"(K={k_eff:.1f})")

            # 3. Atualizar ELOs na tabela players
            update_player_elo(cursor, row['winner'], surface, winner_elo_after, match_date)
            update_player_elo(cursor, row['loser'], surface, loser_elo_after, match_date)

            # 4. Atualizar contadores
            increment_player_stats(cursor, row['winner'], row['loser'])

            # 5. Inserir partida na tabela matches
            match_data = {
                'date': match_date.strftime('%Y-%m-%d'),
                'tourney_name': row['tourney_name'],
                'surface': surface,
                'round': row.get('round', 'Unknown'),
                'score': row.get('score', ''),
                'winner': row['winner'],
                'loser': row['loser'],
                'tournament_type': tournament_type,
                'winner_elo_before': winner_elo_before,
                'loser_elo_before': loser_elo_before,
                'winner_elo_after': winner_elo_after,
                'loser_elo_after': loser_elo_after,
                'k_factor_base': k_base,
                'k_factor_effective': k_eff
            }

            insert_match(cursor, match_data)
            processed += 1

        except Exception as e:
            logger.error(f"  ❌ Erro ao processar {row.get('winner', '?')} vs {row.get('loser', '?')}: {e}")
            errors += 1

    # Commit todas as alterações
    conn.commit()
    conn.close()

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Atualização concluída!")
    logger.info(f"   Processadas: {processed} partidas")
    if errors > 0:
        logger.warning(f"   Erros: {errors}")
    logger.info(f"{'='*70}")

# MAIN

def main():
    parser = argparse.ArgumentParser(
        description='Atualização diária da BD de ténis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Exemplos:
    python daily_update.py                    # Resultados de ontem
    python daily_update.py --date 2024-11-15  # Data específica
    python daily_update.py --csv results.csv  # De ficheiro CSV
        """
    )
    parser.add_argument('--date', type=str, 
                       help='Data no formato YYYY-MM-DD (default: ontem)')
    parser.add_argument('--csv', type=str, 
                       help='Carregar de CSV em vez de scraping')

    args = parser.parse_args()

    # Determinar data alvo
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now() - timedelta(days=1)

    logger.info("=" * 70)
    logger.info("🎾 ATUALIZAÇÃO DIÁRIA - TENIS ML-AI")
    logger.info("=" * 70)
    logger.info(f"📅 Data alvo: {target_date.strftime('%Y-%m-%d')}")
    logger.info(f"💾 Base de dados: {DB_PATH}")

    # Obter resultados
    if args.csv:
        logger.info(f"📄 Carregando de CSV: {args.csv}")
        results_df = pd.read_csv(args.csv)
        results_df['date'] = pd.to_datetime(results_df['date'])
    else:
        logger.info("🌐 Iniciando scraping...")
        results_df = scrape_yesterday_results(target_date)

    # Processar
    if not results_df.empty:
        process_daily_results(results_df)
    else:
        logger.warning("\n⚠️  Nenhum resultado para processar")
        logger.info("💡 Dica: Verifica se houve jogos nesta data")


if __name__ == '__main__':
    main()