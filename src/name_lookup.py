"""Correções de nomes de jogadores (notebooks/name_lookup.csv, colunas old,new).

Aplicadas da mesma forma aos jogos do dia (engine.py) e aos resultados (scraper_results.py):
se só um dos lados fosse corrigido, o comparador de resultados deixava de encontrar o jogo.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NAME_LOOKUP = 'notebooks/name_lookup.csv'
_lookup_cache = {}


def load_name_lookup(path: str | None = None) -> list[tuple[str, str]]:
    """Pares (old, new) do ficheiro, pela ordem do ficheiro. Lido uma vez por caminho."""
    path = path or os.getenv('NAME_LOOKUP', DEFAULT_NAME_LOOKUP)
    if path not in _lookup_cache:
        pairs = []
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                pairs = [(str(old), str(new)) for old, new in zip(df['old'], df['new'])
                         if pd.notna(old) and pd.notna(new)]
            else:
                logger.warning(f"Ficheiro de correção de nomes não encontrado: {path}")
        except Exception as e:
            logger.warning(f"Erro ao ler correções de nomes ({path}): {e}")
        _lookup_cache[path] = pairs
    return _lookup_cache[path]


def apply_name_lookup(name: str, path: str | None = None) -> str:
    """Aplica as substituições old → new ao nome, pela ordem do ficheiro."""
    for old, new in load_name_lookup(path):
        name = name.replace(old, new)
    return name.strip()
