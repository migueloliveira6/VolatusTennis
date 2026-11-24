#!/usr/bin/env python3
"""
Scraper de resultados de partidas (BetExplorer)

Lê os resultados ATP de singulares de um dia em /tennis/results/ (ver src/betexplorer.py).
O vencedor é o jogador em <strong> na lista; os nomes completos vêm da página de cada jogo,
com a mesma conversão e o mesmo name_lookup das previsões (engine.py).
Jogos sem vencedor (W.O., cancelados) ficam de fora; desistências (RET) contam.
"""

from datetime import date, datetime, timedelta
from typing import Optional
import re
import logging

import pandas as pd
import requests

try:
    from betexplorer import BetExplorerClient, JogoLista
except ImportError:
    from src.betexplorer import BetExplorerClient, JogoLista

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_surface_from_tournament(tourney_name: str) -> str:
    """Detecta a superfície baseada no nome do torneio"""
    tourney_lower = tourney_name.lower()

    # Grass
    grass_keywords = ['wimbledon', 'queen', 'halle', 'stuttgart grass', 
                      'eastbourne', 'mallorca', 'newport', 'hertogenbosch']
    if any(kw in tourney_lower for kw in grass_keywords):
        return 'Grass'

    # Clay
    clay_keywords = ['french', 'roland garros', 'monte carlo', 'rome', 
                     'madrid', 'barcelona', 'hamburg', 'munich', 'estoril',
                     'geneva', 'lyon', 'bucharest', 'budapest', 'umag',
                     'gstaad', 'bastad', 'kitzbühel', 'casablanca', 'marrakech']
    if any(kw in tourney_lower for kw in clay_keywords):
        return 'Clay'

    # Default: Hard
    return 'Hard'

# CLASSE DE SCRAPING DE RESULTADOS

class ResultsScraper:
    """Scraper de RESULTADOS de partidas"""

    BLACKLIST_TORNEIO = frozenset({
        "itf", "utr", "junior", "exhibition", "boodles tennis challenge",
        "boodles tennis cup", "boodles tennis series", "series",
        "world university games", "olympic", "youth olympic",
        "fed cup", "hopman cup", "laver cup", "atp cup", "league", "challenger", "chall."
    })

    def __init__(self, timeout: int = 10, cliente: Optional[BetExplorerClient] = None):
        self.timeout = timeout
        self.cliente = cliente or BetExplorerClient(timeout=timeout)
        logger.info("🚀 ResultsScraper inicializado!")

    def _is_valid_tournament(self, nome: str) -> bool:
        """Verifica se é um torneio válido"""
        if not nome:
            return False
        nome_lower = nome.lower()
        return not any(blacklisted in nome_lower for blacklisted in self.BLACKLIST_TORNEIO)

    def _detect_tournament_type(self, tourney_name: str) -> str:
        """Só se leem torneios ATP de singulares; os Challenger já ficam de fora"""
        return 'CHALLENGER' if 'challenger' in tourney_name.lower() else 'ATP'

    @staticmethod
    def _score_do_vencedor(jogo: JogoLista) -> str:
        """Parciais do ponto de vista do vencedor: '6-4 6-7(5) 7-5' (+ ' RET')."""
        sets = jogo.parciais if jogo.vencedor == 'casa' else [(b, a) for a, b in jogo.parciais]

        def set_str(a: str, b: str) -> str:
            # o BetExplorer põe o tie-break no lado de quem perdeu o set; aqui vai no fim: 6-7(5)
            tb = re.findall(r'\((\d+)\)', a + b)
            limpo = lambda s: re.sub(r'\(\d+\)', '', s)
            return f"{limpo(a)}-{limpo(b)}" + (f"({tb[0]})" if tb else "")

        score = ' '.join(set_str(a, b) for a, b in sets)
        if jogo.estado == 'RET':
            score = f"{score} RET".strip()
        return score

    def _resultado(self, jogo: JogoLista, dia: date) -> Optional[dict]:
        if not jogo.terminado or jogo.vencedor is None or jogo.estado in ('WO', 'CAN'):
            return None
        if not self._is_valid_tournament(jogo.torneio):
            return None
        casa, fora = self.cliente.nomes_completos(jogo)
        winner, loser = (casa, fora) if jogo.vencedor == 'casa' else (fora, casa)
        if 'Unknown' in (winner, loser):
            logger.warning(f"  ⚠️  Jogador desconhecido em {jogo.href}")
        return {
            "date": dia.strftime("%Y-%m-%d"),
            "tourney_name": jogo.torneio,
            "surface": jogo.superficie or detect_surface_from_tournament(jogo.torneio),
            "round": "Unknown",  # a lista do BetExplorer não indica a ronda
            "winner": winner,
            "loser": loser,
            "score": self._score_do_vencedor(jogo),
            "tournament_type": self._detect_tournament_type(jogo.torneio),
        }

    def scrape_results(self, target_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Faz scraping dos resultados de uma data específica

        Args:
            target_date: Data alvo (default: ontem)

        Returns:
            DataFrame com colunas: date, tourney_name, surface, round,
                                   winner, loser, score, tournament_type
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)
        dia = target_date.date() if isinstance(target_date, datetime) else target_date
        logger.info(f"📅 Scraping resultados de {dia:%Y-%m-%d}")

        try:
            jogos = self.cliente.jogos_do_dia(dia, "results")
        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição: {e}")
            return pd.DataFrame()

        results = []
        for jogo in jogos:
            try:
                resultado = self._resultado(jogo, dia)
            except Exception as e:
                logger.warning(f"Erro ao processar {jogo.href}: {e}")
                continue
            if resultado:
                results.append(resultado)
                logger.info(f"  ✅ {resultado['winner']} def. {resultado['loser']} ({resultado['score']})")

        if not results:
            logger.warning("⚠️  Nenhum resultado encontrado para esta data")
            return pd.DataFrame()
        logger.info(f"✅ Total: {len(results)} resultados encontrados")
        return pd.DataFrame(results)

    def __del__(self):
        if hasattr(self, 'cliente'):
            self.cliente.close()

# FUNÇÃO DE CONVENIÊNCIA

def scrape_yesterday_results(target_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Função wrapper para facilitar uso

    Args:
        target_date: Data alvo (default: ontem)

    Returns:
        DataFrame com resultados
    """
    scraper = ResultsScraper()
    try:
        return scraper.scrape_results(target_date)
    finally:
        del scraper

# TESTE

if __name__ == '__main__':
    print("=" * 70)
    print("TESTE DO SCRAPER DE RESULTADOS")
    print("=" * 70)

    # Testar com uma data conhecida
    # Ajusta para uma data que SABES que teve jogos
    test_date = datetime(2025, 10, 27)  # ATP Finals 2025

    print(f"\nTestando com: {test_date.strftime('%Y-%m-%d')}")
    print("Se não encontrar resultados, muda a data no código")

    df = scrape_yesterday_results(test_date)

    if not df.empty:
        print(f"\n📊 Resultados encontrados: {len(df)}")
        print("\n" + "="*70)
        print(df.head(10).to_string(index=False))
        print("="*70)

        # Salvar para testar depois
        df.to_csv('test_results.csv', index=False)
        print(f"\n💾 Salvo em: test_results.csv")
    else:
        print("\n⚠️  Nenhum resultado encontrado")
        print("\nDicas:")
        print("   - Verifica se houve jogos ATP nessa data")
        print("   - Verifica manualmente: https://www.betexplorer.com/tennis/results/")