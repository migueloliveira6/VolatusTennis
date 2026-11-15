#!/usr/bin/env python3
"""
Scraper de Resultados de Partidas - Tennis Explorer
Faz scraping de resultados de partidas já realizadas do Tennis Explorer

Estrutura do HTML:
- Pares de linhas: r10/r10b, r11/r11b, etc.
- Vencedor determinado por td.result (maior número ganha)
- Scores em múltiplas células td.score
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import re

try:
    from name_lookup import apply_name_lookup
except ImportError:
    from src.name_lookup import apply_name_lookup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache global para nomes de jogadores
nome_cache = {}

# FUNÇÕES AUXILIARES

def getPlayersFullName(playerUrl: str) -> str:
    """
    Obtém o nome completo do jogador com cache
    """
    if playerUrl in nome_cache:
        return nome_cache[playerUrl]

    try:
        player_url = "https://www.tennisexplorer.com" + playerUrl
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        player_response = requests.get(player_url, headers=headers, timeout=10)
        player_response.raise_for_status()
        player_soup = BeautifulSoup(player_response.content, "html.parser")

        player_table = player_soup.find("table", {"class": "plDetail"})
        if not player_table:
            nome_cache[playerUrl] = "Unknown"
            return "Unknown"

        player_table_body = player_table.find("tbody")
        player_name = player_table_body.find_all("h3")

        if not player_name:
            nome_cache[playerUrl] = "Unknown"
            return "Unknown"

        name = " ".join(player_name[0].text.split())
        splitname = name.split(" ")

        if len(splitname) >= 2:
            first_name = splitname[-1]
            last_name = name.replace(" " + first_name, "")
            name = first_name + " " + last_name

        name = name.strip().replace("-", " ")
        # Mesmas correções de nomes que o engine.py aplica às previsões
        name = apply_name_lookup(name)
        nome_cache[playerUrl] = name
        return name

    except Exception as e:
        logger.warning(f"Erro ao extrair nome do jogador {playerUrl}: {e}")
        nome_cache[playerUrl] = "Unknown"
        return "Unknown"


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
    """Scraper otimizado para RESULTADOS de partidas"""

    BLACKLIST_TORNEIO = frozenset({
        "itf", "utr", "junior", "exhibition", "boodles tennis challenge", 
        "boodles tennis cup", "boodles tennis series", "series",
        "world university games", "olympic", "youth olympic",
        "fed cup", "hopman cup", "laver cup", "atp cup", "league", "challenger", "chall."
    })

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.8,en;q=0.5",
        "Connection": "keep-alive"
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        logger.info("🚀 ResultsScraper inicializado!")

    def _is_valid_tournament(self, nome: str) -> bool:
        """Verifica se é um torneio válido"""
        if not nome:
            return False
        nome_lower = nome.lower()
        return not any(blacklisted in nome_lower for blacklisted in self.BLACKLIST_TORNEIO)

    def _build_url(self, target_date: datetime) -> str:
        """Constrói URL para uma data específica"""
        return (f"https://www.tennisexplorer.com/matches/"
                f"?type=atp-single&year={target_date.year}"
                f"&month={target_date.month:02d}"
                f"&day={target_date.day:02d}")

    def _extract_set_scores(self, row) -> List[str]:
        """
        Extrai os scores de sets de uma linha.
        Se houver tiebreak (ex: 7<sup>6</sup>), devolve "7(6)".
        """
        scores = []
        for td in row.find_all('td', class_='score'):
            # Mantemos o HTML bruto para apanhar <sup>
            raw_html = str(td)
            text = td.get_text(strip=True).replace('\xa0', '').replace('&nbsp;', '')
            if not text:
                continue

            # Detetar padrão tipo "7<sup>6</sup>"
            # Vamos procurar o sup dentro do HTML
            if '<sup>' in raw_html:
                # Número principal = último dígito do texto (ex: "76" -> "7")
                main = text[:-1] if len(text) > 1 else text
                # Tiebreak = conteúdo do sup (ex: "6")
                from bs4 import BeautifulSoup
                sup = BeautifulSoup(raw_html, 'html.parser').find('sup')
                tb = sup.get_text(strip=True) if sup else ''
                if tb:
                    scores.append(f"{main}({tb})")
                else:
                    scores.append(main)
            else:
                scores.append(text)
        return scores

    def _build_score_string(self, scores1: List[str], scores2: List[str]) -> str:
        """
        Constrói string de placar a partir de duas listas

        Args:
            scores1: Scores do jogador 1
            scores2: Scores do jogador 2

        Returns:
            String tipo "6-4 7-5 6-3"
        """
        if not scores1 or not scores2:
            return ""

        # Limpar scores com tiebreaks (ex: "32" de "3<sup>2</sup>")
        def clean_score(s):
            # Se tiver mais de 2 dígitos, é provável que seja tiebreak mal parseado
            # Ex: "32" -> pode ser "7" com tiebreak 7-6(2)
            # Por agora, vamos manter como está e melhorar depois
            return s

        pairs = []
        for s1, s2 in zip(scores1, scores2):
            s1_clean = clean_score(s1)
            s2_clean = clean_score(s2)
            pairs.append(f"{s1_clean}-{s2_clean}")

        return ' '.join(pairs)

    def _extract_round(self, row) -> str:
        """Extrai a ronda do jogo (F, SF, QF, etc.)"""
        round_cell = row.find("td", class_="t-type")
        if round_cell:
            round_text = round_cell.get_text(strip=True)
            if round_text:
                return round_text
        return "Unknown"

    def _detect_tournament_type(self, tourney_name: str) -> str:
        """Detecta se é ATP ou CHALLENGER"""
        tourney_lower = tourney_name.lower()

        if 'challenger' in tourney_lower:
            return 'CHALLENGER'
        elif any(x in tourney_lower for x in ['atp', 'masters', 'grand slam', 'open']):
            return 'ATP'
        else:
            return 'ATP'

    def _process_match_pair(self, row1, row2, current_tourney, current_surface, current_type):
        """
        Processa um par de linhas (r10/r10b) e extrai o resultado

        Args:
            row1: Primeira linha (r10)
            row2: Segunda linha (r10b)
            current_tourney: Nome do torneio
            current_surface: Superfície
            current_type: Tipo do torneio (ATP/CHALLENGER)

        Returns:
            dict com resultado ou None se inválido
        """
        try:
            # 1. Verificar se ambas as linhas têm jogadores
            name1_td = row1.find("td", class_="t-name")
            name2_td = row2.find("td", class_="t-name")

            if not name1_td or not name2_td:
                return None

            # 2. Verificar se têm resultados (sets ganhos)
            result1_td = row1.find("td", class_="result")
            result2_td = row2.find("td", class_="result")

            if not result1_td or not result2_td:
                return None

            # Verificar se tem números (jogo terminado)
            try:
                res1 = int(result1_td.get_text(strip=True))
                res2 = int(result2_td.get_text(strip=True))
            except (ValueError, AttributeError):
                # Não é um resultado final (pode ser jogo agendado)
                return None

            # 3. Determinar vencedor e perdedor
            if res1 > res2:
                winner_row = row1
                loser_row = row2
            elif res2 > res1:
                winner_row = row2
                loser_row = row1
            else:
                # casos de erro
                return None

            # 4. Extrair links dos jogadores
            winner_link = winner_row.find("td", class_="t-name").a
            loser_link = loser_row.find("td", class_="t-name").a

            if not winner_link or not loser_link:
                return None

            winner_href = winner_link.get('href', '')
            loser_href = loser_link.get('href', '')

            if not winner_href or not loser_href:
                return None

            # 5. Obter nomes completos
            winner_name = getPlayersFullName(winner_href)
            loser_name = getPlayersFullName(loser_href)

            if winner_name == "Unknown" or loser_name == "Unknown":
                logger.warning(f"  ⚠️  Jogador desconhecido: {winner_href} ou {loser_href}")
                # Continuar mesmo assim

            # 6. Extrair scores
            scores1 = self._extract_set_scores(row1)
            scores2 = self._extract_set_scores(row2)

            score = self._build_score_string(scores1, scores2)

            # 7. Extrair ronda
            round_info = self._extract_round(row1)

            return {
                'winner': winner_name,
                'loser': loser_name,
                'score': score if score else f"{res1}-{res2}",  # Fallback
                'round': round_info
            }

        except Exception as e:
            logger.debug(f"Erro ao processar par: {e}")
            return None

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

        url = self._build_url(target_date)
        logger.info(f"📅 Scraping resultados de {target_date.strftime('%Y-%m-%d')}")
        logger.info(f"🌐 URL: {url}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            tables = soup.select("table.result")
            if not tables:
                logger.warning("⚠️  Nenhuma tabela encontrada")
                return pd.DataFrame()

            results = []
            current_tourney = None
            current_surface = None
            current_type = None

            for table in tables:
                rows = table.select("tr")
                i = 0

                while i < len(rows):
                    row = rows[i]
                    classes = row.get("class", [])

                    # Cabeçalho de torneio = início de um novo bloco
                    if "head" in classes and "flags" in classes:
                        tourney_td = row.find("td", class_="t-name")
                        if tourney_td and tourney_td.a:
                            tourney_name = tourney_td.a.get_text(strip=True)

                            if self._is_valid_tournament(tourney_name):
                                current_tourney = tourney_name
                                current_surface = detect_surface_from_tournament(tourney_name)
                                current_type = self._detect_tournament_type(tourney_name)
                                logger.info(f"🏟️  {current_tourney} ({current_surface}, {current_type})")
                            else:
                                # Se o torneio estiver na blacklist, limpamos o contexto
                                current_tourney = None
                                current_surface = None
                                current_type = None

                        i += 1
                        continue

                    #  Se não estamos dentro de um torneio válido, ignorar linhas até ao próximo head flags
                    if not current_tourney:
                        i += 1
                        continue

                    #  Processar pares r10/r10b, r11/r11b, ... APENAS dentro do bloco atual
                    row_id = row.get("id", "")

                    # Linha candidata a ser a de cima do par (ex: r10, r11, r12...)
                    if row_id and row_id.startswith("r") and not row_id.endswith("b"):
                        if i + 1 < len(rows):
                            next_row = rows[i + 1]
                            next_id = next_row.get("id", "")

                            # Confirmar que é o par correspondente (r10b, r11b, etc.)
                            if next_id == row_id + "b":
                                match_data = self._process_match_pair(
                                    row, next_row,
                                    current_tourney, current_surface, current_type
                                )

                                if match_data:
                                    result = {
                                        "date": target_date.strftime("%Y-%m-%d"),
                                        "tourney_name": current_tourney,
                                        "surface": current_surface,
                                        "round": match_data["round"],
                                        "winner": match_data["winner"],
                                        "loser": match_data["loser"],
                                        "score": match_data["score"],
                                        "tournament_type": current_type,
                                    }
                                    results.append(result)
                                    logger.info(
                                        f"  ✅ {match_data['winner']} def. {match_data['loser']} ({match_data['score']})"
                                    )

                                # Avançar 2 linhas (par processado)
                                i += 2
                                continue

                    # 4️⃣ Se não caiu em nenhum caso especial, avançar 1 linha
                    i += 1

            if results:
                df = pd.DataFrame(results)
                logger.info(f"\n✅ Total: {len(df)} resultados encontrados")
                return df
            else:
                logger.warning("⚠️  Nenhum resultado encontrado para esta data")
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()

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
        print("   - Verifica manualmente: https://www.tennisexplorer.com/matches/")