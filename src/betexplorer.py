"""Leitura das páginas de ténis do BetExplorer (substitui o TennisExplorer, que fecha a 16/12/2026).

Páginas usadas:
- Jogos de um dia: /tennis/next/?year=&month=&day=  (por começar, com odds)
- Resultados de um dia: /tennis/results/?year=&month=&day=
- Página de cada jogo, só para o nome completo dos jogadores (a lista mostra "Sinner J.")

Estrutura da lista (table.table-main):
- um <tbody> por torneio e dia, com o cabeçalho tr.js-tournament
  ("ATP - Singles: Rome, clay", "Challenger Men - Singles: ...", "ATP - Doubles: ...");
- uma linha tr[data-dt] por jogo, com data-dt="d,m,aaaa,h,min" (hora do servidor, CET) e
  data-dt-now com a hora do servidor no momento do pedido;
- jogador da casa e visitante em .table-main__teamLine--home/--away; o vencedor vem em <strong>;
- odds em td.table-main__odds button[data-odd]; resultado em td.table-main__result
  ("2:1", "0:2 RET.", "WO.", "CAN."), parciais em td.table-main__partial ("(6:4, 7:6<sup>5</sup>)").

A página de um dia mostra também jogos dos dias vizinhos, por isso filtra-se pela data de data-dt.
O resultado ao vivo é preenchido por JavaScript: no HTML do servidor um jogo ao vivo parece por
começar, por isso "iniciado" também conta jogos cuja hora marcada já passou.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    from name_lookup import apply_name_lookup
except ImportError:
    from src.name_lookup import apply_name_lookup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.betexplorer.com"
PREFIXO_ATP_SINGLES = "ATP - Singles:"
SUPERFICIES = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en;q=0.9,pt-PT;q=0.8",
}


@dataclass
class JogoLista:
    """Um jogo tal como aparece na lista do dia (nomes ainda abreviados)."""
    torneio: str                     # "Rome", "Finals - Turin", "Davis Cup - World Group"
    superficie: Optional[str]        # "Hard"/"Clay"/"Grass" do cabeçalho, ou None
    href: str                        # página do jogo (dá os nomes completos)
    casa: str                        # "Sinner J."
    fora: str
    inicio: datetime                 # hora marcada (servidor)
    odd_casa: Optional[float] = None
    odd_fora: Optional[float] = None
    terminado: bool = False          # tem resultado (FIN)
    iniciado: bool = False           # terminado ou hora marcada já passou
    estado: str = ""                 # "", "RET", "WO", "CAN"
    vencedor: Optional[str] = None   # "casa" / "fora" / None
    parciais: List[Tuple[str, str]] = field(default_factory=list)  # [("6", "4"), ("7", "6(5)")]


def url_dia(pagina: str, dia: date) -> str:
    """pagina = 'next' (jogos do dia) ou 'results'."""
    return f"{BASE_URL}/tennis/{pagina}/?year={dia.year}&month={dia.month:02d}&day={dia.day:02d}"


def _data_hora(valor: Optional[str]) -> Optional[datetime]:
    try:
        d, m, a, h, mi = (int(x) for x in (valor or "").split(","))
        return datetime(a, m, d, h, mi)
    except (ValueError, TypeError):
        return None


def separar_cabecalho(cabecalho: str) -> Tuple[str, Optional[str]]:
    """'ATP - Singles: Rome, clay' -> ('Rome', 'Clay'); sem piso -> (nome, None)."""
    nome = cabecalho.split(":", 1)[1].strip() if ":" in cabecalho else cabecalho.strip()
    superficie = None
    if "," in nome:
        base, sufixo = nome.rsplit(",", 1)
        chave = sufixo.strip().lower().split()[0] if sufixo.strip() else ""
        if chave in SUPERFICIES:
            nome, superficie = base.strip(), SUPERFICIES[chave]
    return nome, superficie


def _odd(td) -> Optional[float]:
    botao = td.find("button") if td else None
    try:
        return float(botao["data-odd"]) if botao and botao.get("data-odd") else None
    except ValueError:
        return None


def _parciais(td) -> List[Tuple[str, str]]:
    """'(6:4, 6<sup>5</sup>:7)' -> [('6','4'), ('6(5)','7')]."""
    if not td:
        return []
    for sup in td.find_all("sup"):
        sup.replace_with(f"({sup.get_text(strip=True)})")
    texto = td.get_text("", strip=True).strip("()")
    sets = []
    for parte in texto.split(","):
        if ":" in parte:
            a, b = parte.strip().split(":", 1)
            sets.append((a.strip(), b.strip()))
    return sets


def _linha_para_jogo(tr, torneio: str, superficie: Optional[str]) -> Optional[JogoLista]:
    link = tr.select_one("a.table-main__teamsLink")
    casa = tr.select_one(".table-main__teamLine--home")
    fora = tr.select_one(".table-main__teamLine--away")
    inicio = _data_hora(tr.get("data-dt"))
    if not (link and casa and fora and inicio):
        return None

    agora = _data_hora(tr.get("data-dt-now"))
    hora = tr.select_one(".table-main__time")
    terminado = bool(hora and "table-main__time--fin" in (hora.get("class") or []))
    resultado = tr.select_one("td.table-main__result")
    texto_resultado = resultado.get_text(" ", strip=True).upper() if resultado else ""
    estado = next((e for e in ("RET", "WO", "CAN") if e in texto_resultado.replace(".", "")), "")

    vencedor = None
    if casa.find("strong"):
        vencedor = "casa"
    elif fora.find("strong"):
        vencedor = "fora"

    odds = tr.select("td.table-main__odds")
    return JogoLista(
        torneio=torneio,
        superficie=superficie,
        href=link.get("href", ""),
        casa=casa.get_text(" ", strip=True),
        fora=fora.get_text(" ", strip=True),
        inicio=inicio,
        odd_casa=_odd(odds[0]) if len(odds) >= 2 else None,
        odd_fora=_odd(odds[1]) if len(odds) >= 2 else None,
        terminado=terminado,
        iniciado=terminado or bool(texto_resultado) or (agora is not None and inicio <= agora),
        estado=estado,
        vencedor=vencedor,
        parciais=_parciais(tr.select_one("td.table-main__partial")),
    )


def ler_lista(html: str, dia: date, prefixo: str = PREFIXO_ATP_SINGLES) -> List[JogoLista]:
    """Jogos da data `dia` em torneios cujo cabeçalho começa por `prefixo` (ATP singulares)."""
    soup = BeautifulSoup(html, "lxml")
    jogos, vistos = [], set()
    for tbody in soup.select("table.table-main tbody"):
        cab = tbody.select_one("a.table-main__tournament")
        cabecalho = cab.get_text(" ", strip=True) if cab else ""
        if not cabecalho.startswith(prefixo):
            continue
        torneio, superficie = separar_cabecalho(cabecalho)
        for tr in tbody.select("tr[data-dt]"):
            jogo = _linha_para_jogo(tr, torneio, superficie)
            if jogo and jogo.inicio.date() == dia and jogo.href not in vistos:
                vistos.add(jogo.href)
                jogos.append(jogo)
    return jogos


def ler_nomes_jogo(html: str) -> Tuple[Optional[str], Optional[str]]:
    """Nomes completos ("Auger-Aliassime Felix") da página de um jogo: casa e fora, por ordem."""
    soup = BeautifulSoup(html, "lxml")
    nomes = []
    for a in soup.select('a[href*="/tennis/player/"]'):
        nome = " ".join(a.get_text(" ", strip=True).split())
        if nome and nome not in nomes:
            nomes.append(nome)
    if len(nomes) < 2:
        return None, None
    return nomes[0], nomes[1]


def converter_nome(nome: str, lookup_path: Optional[str] = None) -> str:
    """'Auger-Aliassime Felix' -> 'Felix Auger Aliassime' (último token passa para a frente).

    É a mesma conversão que se fazia aos nomes do TennisExplorer, para o name_lookup.csv e os
    nomes já publicados continuarem a bater certo.
    """
    nome = " ".join((nome or "").split())
    partes = nome.split(" ")
    if len(partes) >= 2:
        primeiro = partes[-1]
        nome = primeiro + " " + " ".join(partes[:-1])
    nome = nome.strip().replace("-", " ")
    return apply_name_lookup(nome, lookup_path)


class BetExplorerClient:
    """Pedidos ao BetExplorer com sessão e cache dos nomes por jogo."""

    def __init__(self, timeout: int = 15, lookup_path: Optional[str] = None):
        self.timeout = timeout
        self.lookup_path = lookup_path
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._nomes: Dict[str, Tuple[str, str]] = {}

    def html(self, url: str) -> str:
        resposta = self.session.get(url, timeout=self.timeout)
        resposta.raise_for_status()
        return resposta.text

    def jogos_do_dia(self, dia: date, pagina: str = "next") -> List[JogoLista]:
        url = url_dia(pagina, dia)
        logger.info(f"📅 URL sendo acessada: {url}")
        return ler_lista(self.html(url), dia)

    def nomes_completos(self, jogo: JogoLista) -> Tuple[str, str]:
        """Nomes já convertidos e corrigidos pelo name_lookup; 'Unknown' se a página falhar."""
        if jogo.href not in self._nomes:
            try:
                casa, fora = ler_nomes_jogo(self.html(BASE_URL + jogo.href))
            except requests.RequestException as e:
                logger.warning(f"Erro ao ler os nomes de {jogo.href}: {e}")
                casa = fora = None
            self._nomes[jogo.href] = (
                converter_nome(casa, self.lookup_path) if casa else "Unknown",
                converter_nome(fora, self.lookup_path) if fora else "Unknown",
            )
        return self._nomes[jogo.href]

    def close(self):
        self.session.close()
