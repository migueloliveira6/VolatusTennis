#!/usr/bin/env python3
"""
NeuroTennis - Sistema de Scraping e Previsão de Jogos de Ténis
Versão Otimizada com Processamento Paralelo e ROI
"""

import sys
import os
import logging
import requests
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
TOKEN_BOT = os.getenv('TOKEN_BOT')
CHAT_ID = os.getenv('CHAT_ID')
MODEL_PATH = os.getenv('MODEL_PATH', 'models')
PREVISOES_PATH = os.getenv('PREVISOES_PATH', 'previsoes')
NAME_LOOKUP = os.getenv('NAME_LOOKUP', 'notebooks/name_lookup.csv')
SITE_STATUS_PATH = os.getenv('SITE_STATUS_PATH', os.path.join('docs', 'predicts', 'status.json'))
# Cache global para nomes de jogadores
nome_cache = {}

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def getPlayersFullName(playerUrl: str) -> str:
    """
    Obtém o nome completo do jogador com cache
    
    Args:
        playerUrl: URL relativa do jogador
        
    Returns:
        Nome completo normalizado do jogador
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
        
        # Aplicar substituições personalizadas se existir
        try:
            if os.path.exists(NAME_LOOKUP):
                name_dict = pd.read_csv(NAME_LOOKUP)
            for _, item in name_dict.iterrows():
                name = name.replace(item.old, item.new)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.info(f"Erro ao aplicar substituições de nome: {e}")
        
        nome_cache[playerUrl] = name
        return name
        
    except Exception as e:
        logger.warning(f"Erro ao extrair nome do jogador {playerUrl}: {e}")
        nome_cache[playerUrl] = "Unknown"
        return "Unknown"


# ============================================================================
# CLASSES DE DADOS
# ============================================================================

@dataclass
class Jogo:
    """Classe para representar um jogo de ténis"""
    player1: str
    player2: str
    odd1: Optional[float]
    odd2: Optional[float]
    
    def __str__(self):
        odds_info = f" (Odds: {self.odd1} vs {self.odd2})" if self.odd1 and self.odd2 else ""
        return f"🎾 {self.player1} vs {self.player2}{odds_info}"


@dataclass
class ResultadoPrevisao:
    """Classe para representar resultado de uma previsão"""
    torneio: str
    jogador1: str
    jogador2: str
    vencedor_previsto: str
    confianca: float
    elo_diff: float
    h2h: str
    odd1: Optional[float]
    odd2: Optional[float]
    superficie: str
    valor_aposta: Optional[float] = None
    roi_esperado: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "Torneio": self.torneio,
            "Jogador 1": self.jogador1,
            "Jogador 2": self.jogador2,
            "Vencedor Previsto": self.vencedor_previsto,
            "Confiança (%)": round(self.confianca * 100, 1),
            "ELO Diff": round(self.elo_diff, 1),
            "H2H": self.h2h,
            "Odd 1": self.odd1,
            "Odd 2": self.odd2,
            "Superfície": self.superficie,
            "Valor Aposta": round(self.valor_aposta, 3) if self.valor_aposta is not None and not pd.isna(self.valor_aposta) else 0,
            "ROI Esperado (%)": round(self.roi_esperado * 100, 1) if self.roi_esperado is not None and not pd.isna(self.roi_esperado) else 0
        }


# ============================================================================
# CLASSE DE SCRAPING
# ============================================================================

class NeuroTennis:
    """Classe otimizada para scraping do Tennis Explorer"""
    
    BLACKLIST_TORNEIO = frozenset({
        "itf", "utr", "junior", "exhibition", "boodles tennis challenge", 
        "boodles tennis cup", "boodles tennis series", "series",
        "pro", "match", "world university games", "challenger", "ultimate tennis"
    })
    
    BLACKLIST_NOMES = frozenset({
        "challenger", "itf", "wta", "utr", "series", "pro", "match", "cup", 
        "junior", "wimbledon", "french", "us open", "australian", "masters", 
        "finals", "exhibition", "eastbourne", "miami", "indian wells", "paris", 
        "rome", "shanghai", "tokyo", "beijing", "doha", "dubai", "sydney", 
        "montreal", "cincinnati", "stuttgart", "birmingham", "brussels", 
        "istanbul", "moscow", "st petersburg", "basel", "rotterdam", 
        "hamburg", "vienna", "london"
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
        logger.info("🚀 NeuroTennis inicializado!")
    
    def _is_atp_torneio(self, nome: str) -> bool:
        """Verifica se é um torneio ATP válido"""
        if not nome:
            return False
        nome_lower = nome.lower()
        return not any(blacklisted in nome_lower for blacklisted in self.BLACKLIST_TORNEIO)
    
    def _is_nome_valido(self, nome: str) -> bool:
        """Verifica se o nome do jogador é válido"""
        if not nome or len(nome.split()) > 5:
            return False
        nome_lower = nome.lower()
        return not any(blocked in nome_lower for blocked in self.BLACKLIST_NOMES)
    
    def _limpar_nome_jogador(self, nome_bruto: str) -> str:
        """Limpa o nome do jogador removendo seeds e iniciais"""
        if not nome_bruto:
            return ""
        nome = re.sub(r'\([^)]*\)', '', nome_bruto)
        nome = re.sub(r'\b[A-Z]\.\s*', '', nome)
        return ' '.join(nome.split()).strip()
    
    def _extrair_odds(self, linha) -> Tuple[Optional[float], Optional[float]]:
        """Extrai as odds de uma linha"""
        try:
            odds_elements = linha.find_all("td", class_=["course", "coursew"])
            if len(odds_elements) < 2:
                return None, None
            
            if "coursew" in odds_elements[0].get("class", []):
                odd1 = float(odds_elements[0].get_text(strip=True))
                odd2 = float(odds_elements[1].get_text(strip=True))
            elif "coursew" in odds_elements[1].get("class", []):
                odd1 = float(odds_elements[1].get_text(strip=True))
                odd2 = float(odds_elements[0].get_text(strip=True))
            else:
                odd1 = float(odds_elements[0].get_text(strip=True))
                odd2 = float(odds_elements[1].get_text(strip=True))
            
            return odd1, odd2
        except (ValueError, AttributeError, IndexError) as e:
            logger.debug(f"Erro ao extrair odds: {e}")
            return None, None
    
    def _processar_par_jogadores(self, linha1, linha2, torneio_atual: str) -> Optional[Jogo]:
        """Processa um par de linhas para extrair informações dos jogadores"""
        try:
            jogador1_td = linha1.find("td", class_="t-name")
            jogador2_td = linha2.find("td", class_="t-name")
            
            if not all([jogador1_td, jogador2_td, jogador1_td.a, jogador2_td.a]):
                return None
            
            href1 = jogador1_td.a.get("href", "")
            href2 = jogador2_td.a.get("href", "")
            nome_bruto1 = jogador1_td.get_text(strip=True)
            nome_limpo1 = self._limpar_nome_jogador(nome_bruto1)
            nome_bruto2 = jogador2_td.get_text(strip=True)
            nome_limpo2 = self._limpar_nome_jogador(nome_bruto2)
            
            if not (self._is_nome_valido(nome_limpo1) and self._is_nome_valido(nome_limpo2)):
                return None
            
            odd1, odd2 = self._extrair_odds(linha1)
            jogador1 = getPlayersFullName(href1)
            jogador2 = getPlayersFullName(href2)
            
            return Jogo(
                player1=jogador1,
                player2=jogador2,
                odd1=odd1,
                odd2=odd2
            )
        except Exception as e:
            logger.warning(f"Erro ao processar par de jogadores: {e}")
            return None
    
    def _construir_url(self, dias_offset: int = 1) -> str:
        """Constrói a URL para scraping"""
        hoje = datetime.today()
        data_target = hoje + timedelta(days=dias_offset)
        return (f"https://www.tennisexplorer.com/matches/"
                f"?type=atp-single&year={data_target.year}"
                f"&month={data_target.month:02d}"
                f"&day={data_target.day:02d}")
    
    def _extrair_nome_torneio(self, linha) -> Optional[str]:
        """Extrai o nome do torneio de uma linha de cabeçalho"""
        try:
            td_nome = linha.find("td", class_="t-name")
            if td_nome and td_nome.a:
                nome_torneio = td_nome.a.get_text(strip=True)
                if nome_torneio == "Tampere challenger":
                    return None
                if self._is_atp_torneio(nome_torneio):
                    return nome_torneio
            return None
        except Exception as e:
            logger.debug(f"Erro ao extrair nome do torneio: {e}")
            return None
    
    def extrair_jogos_agrupados_por_torneio(self, dias_offset: int = 1) -> Dict[str, List[Jogo]]:
        """Extrai jogos agrupados por torneio"""
        url = self._construir_url(dias_offset)
        logger.info(f"📅 URL sendo acessada: {url}")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            tabelas = soup.select("table.result")
            
            if not tabelas:
                logger.warning("⚠️ Nenhuma tabela encontrada.")
                return {}
            
            torneios = {}
            torneio_atual = None
            
            for tabela in tabelas:
                linhas = tabela.select("tr")
                i = 0
                
                while i < len(linhas):
                    linha = linhas[i]
                    classes_linha = linha.get("class", [])
                    
                    if "head" in classes_linha and "flags" in classes_linha:
                        torneio_atual = self._extrair_nome_torneio(linha)
                        if torneio_atual:
                            torneios.setdefault(torneio_atual, [])
                            logger.info(f"🏟️ Torneio encontrado: {torneio_atual}")
                        i += 1
                        continue
                    
                    if torneio_atual and i + 1 < len(linhas):
                        linha_atual = linhas[i]
                        linha_seguinte = linhas[i + 1]
                        
                        tem_jogador_atual = linha_atual.find("td", class_="t-name") is not None
                        tem_jogador_seguinte = linha_seguinte.find("td", class_="t-name") is not None
                        
                        classes_atual = linha_atual.get("class", [])
                        classes_seguinte = linha_seguinte.get("class", [])
                        eh_header_atual = "head" in classes_atual or "flags" in classes_atual
                        eh_header_seguinte = "head" in classes_seguinte or "flags" in classes_seguinte
                        
                        if (tem_jogador_atual and tem_jogador_seguinte and 
                            not eh_header_atual and not eh_header_seguinte):
                            
                            jogo = self._processar_par_jogadores(linha_atual, linha_seguinte, torneio_atual)
                            
                            if jogo:
                                torneios[torneio_atual].append(jogo)
                                i += 2
                            else:
                                i += 2
                        else:
                            i += 1
                    else:
                        i += 1
            
            torneios = {k: v for k, v in torneios.items() if v}
            logger.info(f"✅ Total de torneios encontrados: {len(torneios)}")
            return torneios
            
        except requests.RequestException as e:
            logger.error(f"❌ Erro na requisição HTTP: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            return {}
    
    def __del__(self):
        """Limpa recursos ao destruir o objeto"""
        if hasattr(self, 'session'):
            self.session.close()


# ============================================================================
# DETECTOR DE SUPERFÍCIE
# ============================================================================

class SuperficieDetector:
    """Classe para detectar a superfície do torneio"""
    
    SUPERFICIE_PATTERNS = {
        "Grass": {
            "torneios": {
                "wimbledon", "queen's club", "queens club", "halle", "stuttgart open",
                "eastbourne", "mallorca", "newport", "birmingham", "nottingham",
                "hertogenbosch", "libema open", "nature valley"
            },
            "patterns": [
                r"grass\s*(court|championship)?",
                r"lawn\s*tennis",
                r"queen'?s?\s*club"
            ]
        },
        "Hard": {
            "torneios": {
                "us open", "brisbane", "hong kong", "adelaide", "auckland", "indian wells", "miami",
                "shanghai", "beijing", "tokyo", "montpellier", "marseille", "dallas",
                "toronto", "montreal", "cincinnati", "washington", "atlanta",
                "los cabos", "san diego", "winston salem", "new york", "acapulco",
                "australian open", "atp finals", "masters cup", "davis cup finals",
                "laver cup", "hopman cup", "atp cup", "united cup", "doha", "dubai",
                "chengdu", "hangzhou", "almaty", "stockholm", "brussels", "metz",
                "rotterdam", "basel", "vienna", "paris", "athens", "sydney", "next gen finals"
            },
            "patterns": [
                r"hard\s*(court)?",
                r"us\s*open",
                r"australian\s*open"
            ]
        },
        "Clay": {
            "torneios": {
                "french open", "roland garros", "monte carlo", "rome", "madrid",
                "barcelona", "hamburg", "munich", "estoril", "geneva", "lyon",
                "bucharest", "budapest", "umag", "gstaad", "bastad", "kitzbühel",
                "casablanca", "marrakech", "houston", "charleston", "bogota", "buenos aires",
                "santiago", "rio de janeiro"
            },
            "patterns": [
                r"clay\s*(court)?",
                r"french\s*open",
                r"roland\s*garros"
            ]
        }
    }
    
    _cache = {}
    
    @classmethod
    def detectar_superficie(cls, nome_torneio: str, data: datetime = None) -> str:
        """Detecta a superfície do torneio"""
        if not nome_torneio:
            return "Hard"
        
        cache_key = nome_torneio.lower().strip()
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        nome_lower = nome_torneio.lower()
        superficie_detectada = cls._detectar_por_nome_e_patterns(nome_lower)
        
        if not superficie_detectada and data:
            superficie_detectada = cls._detectar_por_sazonalidade(data)
        
        superficie_final = superficie_detectada or "Hard"
        cls._cache[cache_key] = superficie_final
        return superficie_final
    
    @classmethod
    def _detectar_por_nome_e_patterns(cls, nome_lower: str) -> Optional[str]:
        """Detecta superfície por nome e padrões"""
        for superficie, config in cls.SUPERFICIE_PATTERNS.items():
            for torneio in config["torneios"]:
                if torneio in nome_lower:
                    return superficie
            for pattern in config["patterns"]:
                if re.search(pattern, nome_lower, re.IGNORECASE):
                    return superficie
        return None
    
    @classmethod
    def _detectar_por_sazonalidade(cls, data: datetime) -> Optional[str]:
        """Detecta superfície baseada na sazonalidade"""
        mes = data.month
        if mes in [6, 7]:
            return "Grass"
        if mes in [4, 5]:
            return "Clay"
        return "Hard"


# ============================================================================
# ANALISADOR DE PREVISÕES
# ============================================================================

class TennisPredicaoAnalyzer:
    """Classe principal para análise de previsões"""
    
    def __init__(self, predictor, max_workers: int = 4, calcular_roi: bool = True):
        self.predictor = predictor
        self.max_workers = max_workers
        self.calcular_roi = calcular_roi
        self.superficie_detector = SuperficieDetector()
        logger.info(f"🔮 NeuroTennis iniciado!")
    
    def _filtrar_jogos_validos(self, jogos_dict: Dict) -> List[Tuple[str, Dict]]:
        """Filtra jogos válidos (com odds)"""
        jogos_validos = []
        for torneio, jogos in jogos_dict.items():
            for jogo in jogos:
                if isinstance(jogo, dict):
                    odd1 = jogo.get("odd1")
                    odd2 = jogo.get("odd2")
                else:
                    odd1 = getattr(jogo, 'odd1', None)
                    odd2 = getattr(jogo, 'odd2', None)
                
                if odd1 is not None and odd2 is not None:
                    jogos_validos.append((torneio, jogo))
        return jogos_validos
    
    def _processar_jogo_individual(self, torneio: str, jogo: Any, data: datetime) -> Optional[ResultadoPrevisao]:
        """Processa um jogo individual"""
        try:
            if isinstance(jogo, dict):
                p1 = jogo.get("player1")
                p2 = jogo.get("player2")
                odd1 = jogo.get("odd1")
                odd2 = jogo.get("odd2")
            else:
                p1 = getattr(jogo, 'player1', None)
                p2 = getattr(jogo, 'player2', None)
                odd1 = getattr(jogo, 'odd1', None)
                odd2 = getattr(jogo, 'odd2', None)
            
            if not all([p1, p2, odd1, odd2]):
                return None
            
            superficie = self.superficie_detector.detectar_superficie(torneio, data)
            winner, prob, detalhes = self.predictor.predict_match(p1, p2, superficie, date=data)
            
            valor_aposta = None
            roi_esperado = None
            
            if self.calcular_roi:
                valor_aposta, roi_esperado = self._calcular_metricas_aposta(
                    winner, prob, p1, p2, odd1, odd2
                )
            
            return ResultadoPrevisao(
                torneio=torneio,
                jogador1=p1,
                jogador2=p2,
                vencedor_previsto=winner,
                confianca=prob,
                elo_diff=detalhes.get("elo_diff", 0),
                h2h=detalhes.get("h2h", "N/A"),
                odd1=odd1,
                odd2=odd2,
                superficie=superficie,
                valor_aposta=valor_aposta,
                roi_esperado=roi_esperado
            )
        except Exception as e:
            logger.warning(f"⚠️ Erro ao processar {p1} vs {p2}: {e}")
            return None
    
    def _calcular_metricas_aposta(self, winner: str, prob: float, p1: str, p2: str,
                                 odd1: float, odd2: float) -> Tuple[Optional[float], Optional[float]]:
        """Calcula métricas de aposta (Kelly Criterion)"""
        try:
            odd_vencedor = odd1 if winner == p1 else odd2
            
            if not odd_vencedor or odd_vencedor <= 0:
                return 0, None
            
            prob_implicita = 1 / odd_vencedor
            
            if prob > prob_implicita and odd_vencedor > 1:
                valor_aposta = (prob * odd_vencedor - 1) / (odd_vencedor - 1)
                roi_esperado = prob * (odd_vencedor - 1) - (1 - prob)
                
                if pd.isna(valor_aposta) or pd.isna(roi_esperado):
                    return 0, 0
                
                return max(0, min(round(valor_aposta, 3), 1)), round(roi_esperado, 3)
            
            return 0, 0
        except (ZeroDivisionError, TypeError, ValueError):
            return 0, 0
    
    def analisar_jogos(self, jogos_dict: Dict, data: datetime = None,
                      processar_paralelo: bool = True) -> Tuple[List[ResultadoPrevisao], Dict[str, Any]]:
        """Analisa todos os jogos"""
        if data is None:
            data = datetime.today()
        
        logger.info("📄 Iniciando análise de jogos...")
        jogos_validos = self._filtrar_jogos_validos(jogos_dict)
        
        logger.info(f"📊 Jogos encontrados: {len(jogos_dict)} torneios, {len(jogos_validos)} válidos")
        
        if not jogos_validos:
            logger.warning("⚠️ Nenhum jogo válido encontrado!")
            return [], {}
        
        if processar_paralelo and len(jogos_validos) > 1:
            resultados = self._processar_paralelo(jogos_validos, data)
        else:
            resultados = self._processar_sequencial(jogos_validos, data)
        
        resultados_validos = [r for r in resultados if r is not None]
        estatisticas = self._gerar_estatisticas(resultados_validos, jogos_dict)
        
        logger.info(f"✅ Análise concluída: {len(resultados_validos)} previsões geradas")
        return resultados_validos, estatisticas
    
    def _processar_paralelo(self, jogos_validos: List, data: datetime) -> List[Optional[ResultadoPrevisao]]:
        """Processa jogos em paralelo"""
        logger.info(f"⚡ Processando {len(jogos_validos)} jogos em paralelo...")
        resultados = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_jogo = {
                executor.submit(self._processar_jogo_individual, torneio, jogo, data): (torneio, jogo)
                for torneio, jogo in jogos_validos
            }
            
            for future in as_completed(future_to_jogo):
                try:
                    resultado = future.result(timeout=30)
                    resultados.append(resultado)
                except Exception as e:
                    logger.warning(f"⚠️ Erro no processamento: {e}")
                    resultados.append(None)
        
        return resultados
    
    def _processar_sequencial(self, jogos_validos: List, data: datetime) -> List[Optional[ResultadoPrevisao]]:
        """Processa jogos sequencialmente"""
        logger.info(f"📄 Processando {len(jogos_validos)} jogos sequencialmente...")
        resultados = []
        
        for i, (torneio, jogo) in enumerate(jogos_validos, 1):
            if i % 10 == 0:
                logger.info(f"   Processando jogo {i}/{len(jogos_validos)}...")
            resultado = self._processar_jogo_individual(torneio, jogo, data)
            resultados.append(resultado)
        
        return resultados
    
    def _gerar_estatisticas(self, resultados: List[ResultadoPrevisao], jogos_dict: Dict) -> Dict[str, Any]:
        """Gera estatísticas da análise"""
        if not resultados:
            return {"previsoes_geradas": 0, "erros": 0}
        
        total_jogos = sum(len(jogos) for jogos in jogos_dict.values())
        
        superficie_stats = {}
        for resultado in resultados:
            if resultado:
                sup = resultado.superficie
                if sup not in superficie_stats:
                    superficie_stats[sup] = {"count": 0, "confianca_media": 0}
                superficie_stats[sup]["count"] += 1
                superficie_stats[sup]["confianca_media"] += resultado.confianca
        
        for sup_data in superficie_stats.values():
            if sup_data["count"] > 0:
                sup_data["confianca_media"] /= sup_data["count"]
        
        roi_stats = {}
        if self.calcular_roi:
            apostas_recomendadas = [r for r in resultados if r and r.valor_aposta and r.valor_aposta > 0]
            if apostas_recomendadas:
                roi_stats = {
                    "apostas_recomendadas": len(apostas_recomendadas),
                    "roi_medio": sum(r.roi_esperado for r in apostas_recomendadas) / len(apostas_recomendadas),
                    "valor_medio_aposta": sum(r.valor_aposta for r in apostas_recomendadas) / len(apostas_recomendadas)
                }
        
        return {
            "total_torneios": len(jogos_dict),
            "total_jogos": total_jogos,
            "jogos_validos": len(resultados),
            "previsoes_geradas": len(resultados),
            "erros": total_jogos - len(resultados),
            "superficie_stats": superficie_stats,
            "roi_stats": roi_stats
        }
    
    def gerar_dataframe(self, resultados: List[ResultadoPrevisao]) -> pd.DataFrame:
        """Converte resultados para DataFrame"""
        if not resultados:
            return pd.DataFrame()
        
        data = [resultado.to_dict() for resultado in resultados]
        df = pd.DataFrame(data)
        
        if "Confiança (%)" in df.columns:
            df = df.sort_values("Confiança (%)", ascending=False)
        
        return df.reset_index(drop=True)


# ============================================================================
# FUNÇÕES DE TELEGRAM
# ============================================================================

def enviar_telegram(mensagem: str, token: str, chat_id: str) -> bool:
    """Envia mensagem via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, data={"chat_id": chat_id, "text": mensagem})
        return res.status_code == 200
    except Exception as e:
        logger.error(f"Erro ao enviar Telegram: {e}")
        return False


def salvar_status_site(status: str, mensagem: str, data: datetime) -> bool:
    """Atualiza o estado curto exibido pelo site."""
    try:
        import json

        pasta_status = os.path.dirname(SITE_STATUS_PATH)
        if pasta_status:
            os.makedirs(pasta_status, exist_ok=True)

        with open(SITE_STATUS_PATH, 'w', encoding='utf-8') as arquivo:
            json.dump({
                "status": status,
                "message": mensagem,
                "date": data.strftime("%Y-%m-%d")
            }, arquivo, ensure_ascii=False, indent=2)
        logger.info(f"✅ Status do site atualizado em '{SITE_STATUS_PATH}'")
        return True
    except OSError as e:
        logger.error(f"❌ Erro ao atualizar status do site: {e}")
        return False


def notificar_sem_jogos(data: datetime) -> Tuple[bool, bool]:
    """Informa a ausência de jogos no Telegram e no site."""
    data_formatada = data.strftime("%d/%m/%Y")
    mensagem = f"Não há jogos de ténis disponíveis para {data_formatada}."
    telegram_enviado = enviar_telegram(mensagem, TOKEN_BOT, CHAT_ID)
    status_salvo = salvar_status_site("no_matches", mensagem, data)
    return telegram_enviado, status_salvo


def enviar_resultados_telegram(df: pd.DataFrame, token: str, chat_id: str) -> bool:
    """Envia resultados formatados via Telegram"""
    try:
        df_telegram = df[pd.to_numeric(df['Valor Aposta'], errors='coerce') > 0]
        if df_telegram.empty:
            logger.info("ℹ️ Nenhuma previsão com valor de aposta positivo para enviar ao Telegram")
            return True

        linhas = []
        for _, row in df_telegram.iterrows():
            valor_aposta = round(row['Valor Aposta'], 3) if pd.notna(row['Valor Aposta']) else 'N/A'
            roi_esperado = round(row['ROI Esperado (%)'], 3) if pd.notna(row['ROI Esperado (%)']) else 'N/A'
            linha = (
                f"Jogo: {row['Jogador 1']} vs {row['Jogador 2']}\n"
                f"🎯 Previsto: {row['Vencedor Previsto']} ({row['Confiança (%)']:.1f}%)\n"
                f"ELO Diff: {row['ELO Diff']} | H2H: {row['H2H']}\n"
                f"Valor Aposta: {valor_aposta}\n"
                f"ROI Esperado: {roi_esperado}\n"
                f"Odds: {row['Odd 1']} / {row['Odd 2']}\n"
                f"Torneio: {row['Torneio']}\n"
                f"──────────────"
            )
            linhas.append(linha)
        
        mensagem = "\n".join(linhas)
        
        # Tentar enviar mensagem completa
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, data={"chat_id": chat_id, "text": mensagem})
        
        if res.status_code == 200:
            logger.info("✅ Mensagem enviada com sucesso!")
            return True
        
        # Se falhar, dividir em partes
        logger.warning("⚠️ Mensagem muito longa, dividindo...")
        
        # Dividir em 2 partes
        meio = len(linhas) // 2
        mensagem1 = "\n".join(linhas[:meio])
        mensagem2 = "\n".join(linhas[meio:])
        
        res1 = requests.post(url, data={"chat_id": chat_id, "text": mensagem1})
        res2 = requests.post(url, data={"chat_id": chat_id, "text": mensagem2})
        
        if res1.status_code == 200 and res2.status_code == 200:
            logger.info("✅ Mensagens enviadas em 2 partes!")
            return True
        
        # Se ainda falhar, dividir em 3 partes
        logger.warning("⚠️ Dividindo em 3 partes...")
        terco = len(linhas) // 3
        mensagem1 = "\n".join(linhas[:terco])
        mensagem2 = "\n".join(linhas[terco:terco*2])
        mensagem3 = "\n".join(linhas[terco*2:])
        
        res1 = requests.post(url, data={"chat_id": chat_id, "text": mensagem1})
        res2 = requests.post(url, data={"chat_id": chat_id, "text": mensagem2})
        res3 = requests.post(url, data={"chat_id": chat_id, "text": mensagem3})
        
        if all(r.status_code == 200 for r in [res1, res2, res3]):
            logger.info("✅ Mensagens enviadas em 3 partes!")
            return True
        
        logger.error("❌ Falha ao enviar mensagens")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erro ao enviar Telegram: {e}")
        return False


def enviar_estatisticas_telegram(estatisticas: Dict[str, Any], token: str, chat_id: str) -> bool:
    """Envia estatísticas via Telegram"""
    try:
        total = estatisticas.get('predictions_total', 'N/A')
        matched = estatisticas.get('matched', 'N/A')
        correct = estatisticas.get('correct', 'N/A')
        accuracy = estatisticas.get('accuracy', None)
        coverage = estatisticas.get('coverage', None)
        bets_placed = estatisticas.get('bets_placed', None)
        bets_won = estatisticas.get('bets_won', None)
        bets_lost = estatisticas.get('bets_lost', None)
        stake_total = estatisticas.get('stake_total', None)
        return_total = estatisticas.get('return_total', None)
        profit_total = estatisticas.get('profit_total', None)
        bet_accuracy = estatisticas.get('bet_accuracy', None)
        bet_roi = estatisticas.get('bet_roi', None)

        linhas = [
            "*Resumo das Comparações de Previsões*",
            f"Total previsões: {total}",
            f"Matches encontrados: {matched}",
            f"Previsões corretas: {correct}",
        ]

        if accuracy is not None:
            try:
                linhas.append(f"Accuracy (sobre matched): {accuracy:.2%}")
            except Exception:
                linhas.append(f"Accuracy (sobre matched): {accuracy}")
        else:
            linhas.append("Accuracy (sobre matched): N/A")

        if coverage is not None:
            try:
                linhas.append(f"Cobertura (matched/total): {coverage:.2%}")
            except Exception:
                linhas.append(f"Cobertura (matched/total): {coverage}")
        else:
            linhas.append("Cobertura (matched/total): N/A")

        if bets_placed is not None:
            linhas.append("")
            linhas.append("*Resumo das Apostas*")
            linhas.append(f"Apostas feitas: {bets_placed}")
            linhas.append(f"Apostas ganhas: {bets_won if bets_won is not None else 'N/A'}")
            linhas.append(f"Apostas perdidas: {bets_lost if bets_lost is not None else 'N/A'}")
            linhas.append(f"Valor apostado: {stake_total:.3f}" if isinstance(stake_total, (int, float)) else "Valor apostado: N/A")
            linhas.append(f"Retorno total: {return_total:.3f}" if isinstance(return_total, (int, float)) else "Retorno total: N/A")
            linhas.append(f"Lucro/prejuízo: {profit_total:.3f}" if isinstance(profit_total, (int, float)) else "Lucro/prejuízo: N/A")
            if bet_accuracy is not None:
                linhas.append(f"Accuracy das apostas: {bet_accuracy:.2%}")
            else:
                linhas.append("Accuracy das apostas: N/A")
            if bet_roi is not None:
                linhas.append(f"ROI das apostas: {bet_roi:.2%}")
            else:
                linhas.append("ROI das apostas: N/A")

        mensagem = "\n".join(linhas)

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, data={"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"})

        if res.status_code == 200:
            logger.info("✅ Estatísticas enviadas com sucesso via Telegram")
            return True

        logger.error(f"❌ Falha ao enviar estatísticas (status {res.status_code}): {res.text}")
        return False

    except Exception as e:
        logger.error(f"❌ Erro ao enviar estatísticas via Telegram: {e}")
        return False

def find_latest_comparacao_file(previsoes_path: str = PREVISOES_PATH) -> Optional[str]:
    """Retorna o caminho do arquivo `comparacao_previsoes_YYYY-MM-DD.csv` mais recente em `previsoes_path` ou None se não existir."""
    try:
        files = [f for f in os.listdir(previsoes_path) if f.startswith('comparacao_previsoes_') and f.endswith('.csv')]
        if not files:
            return None
        files_full = [os.path.join(previsoes_path, f) for f in files]
        files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files_full[0]
    except Exception as e:
        logger.error(f"Erro ao listar arquivos de comparacao: {e}")
        return None


def enviar_comparacao_telegram(file_path: Optional[str], token: str, chat_id: str, previsoes_path: str = PREVISOES_PATH) -> bool:
    """Lê o CSV de comparação, calcula resumo e envia por Telegram usando `enviar_estatisticas_telegram`."""
    try:
        path = file_path or find_latest_comparacao_file(previsoes_path)
        if not path:
            logger.warning("⚠️ Nenhum arquivo de comparação encontrado para enviar.")
            return False

        df = pd.read_csv(path)
        total = len(df)
        matched = int(df['Matched'].astype(bool).sum()) if 'Matched' in df.columns else 0
        correct = int(df['Correct'].astype(bool).sum()) if 'Correct' in df.columns else 0
        accuracy = (correct / matched) if matched else None
        coverage = (matched / total) if total else None
        stake_series = pd.to_numeric(df['Valor Aposta'], errors='coerce').fillna(0) if 'Valor Aposta' in df.columns else pd.Series([0] * len(df))
        bet_mask = stake_series > 0

        if 'Resultado Aposta' in df.columns:
            bet_result_series = df['Resultado Aposta'].fillna('NO_BET').astype(str).str.upper()
        else:
            bet_result_series = pd.Series(['NO_BET'] * len(df))

        bets_placed = int(bet_mask.sum())
        bets_won = int((bet_mask & (bet_result_series == 'WIN')).sum())
        bets_lost = int((bet_mask & (bet_result_series == 'LOSS')).sum())
        stake_total = float(stake_series[bet_mask].sum())
        return_total = float(pd.to_numeric(df['Retorno Aposta'], errors='coerce').fillna(0)[bet_mask].sum()) if 'Retorno Aposta' in df.columns else 0.0
        profit_total = float(pd.to_numeric(df['Lucro Aposta'], errors='coerce').fillna(0)[bet_mask].sum()) if 'Lucro Aposta' in df.columns else 0.0
        bet_accuracy = (bets_won / bets_placed) if bets_placed else None
        bet_roi = (profit_total / stake_total) if stake_total else None

        summary = {
            'predictions_total': total,
            'matched': matched,
            'correct': correct,
            'accuracy': accuracy,
            'coverage': coverage,
            'bets_placed': bets_placed,
            'bets_won': bets_won,
            'bets_lost': bets_lost,
            'stake_total': stake_total,
            'return_total': return_total,
            'profit_total': profit_total,
            'bet_accuracy': bet_accuracy,
            'bet_roi': bet_roi
        }

        logger.info(f"📤 Enviando resumo da comparação ({os.path.basename(path)}) via Telegram")
        return enviar_estatisticas_telegram(summary, token, chat_id)
    except Exception as e:
        logger.error(f"❌ Erro ao processar arquivo de comparação: {e}")
        return False

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal do script"""
    try:
        logger.info("=" * 60)
        logger.info("🎾 NeuroTennis - Sistema de Previsão de Ténis")
        logger.info("=" * 60)
        
        # 1. Importar modelo de previsão
        logger.info("\n📦 Carregando modelo de previsão...")
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
        from model_elo_xgboost import TennisPredictor
        
        predictor = TennisPredictor()
        
        # Verificar se modelo existe
        model_files = [
            'tennis_surface_elo_model_xgboost.pkl',
            'tennis_surface_elo_scaler_xgboost.pkl',
            'tennis_surface_elo_data_xgboost.pkl'
        ]
        
        if all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files):
            logger.info("✅ Modelo encontrado. Carregando...")
            if not predictor.load_saved_model():
                logger.warning("⚠️ Falha ao carregar. Treinando novo...")
                predictor.load_data()
                df = predictor.preprocess_data()
                predictor.train_model(df)
                predictor.save_model()
        else:
            logger.info("⚠️ Nenhum modelo encontrado. Treinando novo...")
            predictor.load_data()
            df = predictor.preprocess_data()
            predictor.train_model(df)
            predictor.save_model()
        
        # 2. Scraping dos jogos
        logger.info("\n🕷️ Iniciando scraping de jogos...")
        scraper = NeuroTennis(timeout=15)
        jogos_hoje = scraper.extrair_jogos_agrupados_por_torneio(dias_offset=1)
        
        if not jogos_hoje:
            logger.warning("⚠️ Nenhum jogo encontrado para hoje!")
            data_jogos = datetime.today() + timedelta(days=1)
            telegram_enviado, status_salvo = notificar_sem_jogos(data_jogos)
            logger.info(f"📱 Telegram sem jogos: {'enviado' if telegram_enviado else 'falhou'}")
            logger.info(f"🌐 Status do site sem jogos: {'salvo' if status_salvo else 'falhou'}")
            return
        
        logger.info(f"✅ {len(jogos_hoje)} torneios encontrados")
        
        # Mostrar resumo dos jogos
        total_jogos = 0
        for torneio, jogos in jogos_hoje.items():
            # Filtrar jogos válidos
            jogos_validos = [j for j in jogos if j.odd1 is not None and j.odd2 is not None]
            
            # Limpar espaços dos nomes
            for j in jogos_validos:
                if j.player1.startswith(" "):
                    j.player1 = j.player1.lstrip()
                if j.player2.startswith(" "):
                    j.player2 = j.player2.lstrip()
            
            if jogos_validos:
                total_jogos += len(jogos_validos)
                logger.info(f"\n🏟️ {torneio}: {len(jogos_validos)} jogos")
                for j in jogos_validos:
                    logger.info(f"  {j.player1} vs {j.player2} — Odds: {j.odd1} / {j.odd2}")
        
        logger.info(f"\n🎾 Total de jogos válidos: {total_jogos}")

        data_jogos = datetime.today() + timedelta(days=1)
        if total_jogos == 0:
            logger.warning("⚠️ Nenhum jogo válido encontrado para hoje!")
            telegram_enviado, status_salvo = notificar_sem_jogos(data_jogos)
            logger.info(f"📱 Telegram sem jogos: {'enviado' if telegram_enviado else 'falhou'}")
            logger.info(f"🌐 Status do site sem jogos: {'salvo' if status_salvo else 'falhou'}")
            return

        salvar_status_site("ready", "Previsões disponíveis.", data_jogos)
        
        # 3. Análise e previsões
        logger.info("\n🔮 Iniciando análise e previsões...")
        analyzer = TennisPredicaoAnalyzer(
            predictor=predictor,
            max_workers=4,
            calcular_roi=True
        )
        
        resultados, estatisticas = analyzer.analisar_jogos(
            jogos_hoje,
            processar_paralelo=True
        )
        
        # Mostrar estatísticas
        logger.info("\n📊 Estatísticas da Análise:")
        logger.info("-" * 50)
        logger.info(f"🏟️ Total de torneios: {estatisticas['total_torneios']}")
        logger.info(f"🎾 Total de jogos: {estatisticas['total_jogos']}")
        logger.info(f"✅ Jogos válidos: {estatisticas['jogos_validos']}")
        logger.info(f"🔮 Previsões geradas: {estatisticas['previsoes_geradas']}")
        logger.info(f"❌ Erros: {estatisticas['erros']}")
        
        if estatisticas.get('superficie_stats'):
            logger.info(f"\n🏟️ Por Superfície:")
            for superficie, dados in estatisticas['superficie_stats'].items():
                logger.info(f"   - {superficie}: {dados['count']} jogos (confiança média: {dados['confianca_media']:.1%})")
        
        if estatisticas.get('roi_stats'):
            roi_stats = estatisticas['roi_stats']
            logger.info(f"\n💰 Estatísticas de ROI:")
            logger.info(f"   - Apostas recomendadas: {roi_stats.get('apostas_recomendadas', 0)}")
            logger.info(f"   - ROI médio esperado: {roi_stats.get('roi_medio', 0):.1%}")
            logger.info(f"   - Valor médio de aposta: {roi_stats.get('valor_medio_aposta', 0):.1%}")
        
        # 4. Gerar DataFrame e salvar
        logger.info("\n💾 Salvando resultados...")
        resultados_dict = [r.to_dict() for r in resultados]
        df_resultados = pd.DataFrame(resultados_dict)
        df_resultados = df_resultados.sort_values(by="Confiança (%)", ascending=False).reset_index(drop=True)

        # Também salvar sem data no nome (para compatibilidade)
        data_str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        csv_path = os.path.join(PREVISOES_PATH, f"previsoes_tenis_{data_str}.csv")
        df_resultados.to_csv(csv_path, index=False)
        logger.info(f"📁 Previsões salvas em '{csv_path}'")
        
        # 5. Enviar notificação Telegram
        logger.info("\n📱 Enviando notificação Telegram...")
        
        # Mensagem inicial
        mensagem_inicial = "🎾 Previsões de Ténis prontas para amanhã!"
        if enviar_telegram(mensagem_inicial, TOKEN_BOT, CHAT_ID):
            logger.info("✅ Notificação enviada!")
        
        # Enviar resultados detalhados
        if enviar_resultados_telegram(df_resultados, TOKEN_BOT, CHAT_ID):
            logger.info("✅ Resultados detalhados enviados!")
        
        # Enviar resumo da comparação de previsões anteriores
        if enviar_comparacao_telegram(None, TOKEN_BOT, CHAT_ID, PREVISOES_PATH):
            logger.info("✅ Resumo da comparação enviado!")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Processo concluído com sucesso!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {str(e)}", exc_info=True)
        raise

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()