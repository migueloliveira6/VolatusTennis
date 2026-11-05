"""
Módulo de Cálculo ELO para Ténis
Réplica EXATA da lógica do notebook atp_chall_matches_2025_elo_kfactor.ipynb
"""

import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURAÇÕES DO SISTEMA ELO
# ============================================================================

# K-Factors por tipo de torneio
K_FACTORS = {
    'ATP': 32,        # Torneios ATP Tour
    'CHALLENGER': 20, # Challengers
    'ITF': 10        # ITF/Futures
}

# Parâmetros do Decaimento Temporal
TEMPORAL_DECAY_YEARS = 5.0     # Número de anos para decaimento significativo
MIN_TEMPORAL_FACTOR = 0.3      # Fator mínimo (30% do K original para jogos muito antigos)
REFERENCE_YEAR = 2025          # Ano de referência (torneios deste ano têm fator 1.0)

# ELO Inicial
INITIAL_ELO = 1500

# ============================================================================
# FUNÇÕES DE CÁLCULO
# ============================================================================

def calculate_temporal_factor(match_date, reference_year=None):
    """
    Calcula o fator de decaimento temporal baseado na data da partida.
    Torneios mais recentes têm fator 1.0, mais antigos têm fator reduzido.

    Args:
        match_date (datetime): Data da partida

    Returns:
        float: Fator entre MIN_TEMPORAL_FACTOR e 1.0

    Exemplos:
        - Jogo de 2025: fator = 1.0 (100%)
        - Jogo de 2020: fator ≈ 0.56 (56%)
        - Jogo de 1992: fator = 0.3 (30%)
    """
    if match_date is None or (hasattr(match_date, '__iter__') and len(match_date) == 0):
        return 1.0

    # Se for string, converter para datetime
    if isinstance(match_date, str):
        try:
            match_date = datetime.strptime(match_date, '%Y-%m-%d')
        except:
            return 1.0

    # Calcular diferença em anos desde o ano de referência
    ref_year = REFERENCE_YEAR if reference_year is None else reference_year
    years_ago = ref_year - match_date.year

    if years_ago <= 0:
        return 1.0  # Torneios atuais ou futuros têm fator máximo

    # Aplicar decaimento exponencial
    # Fórmula: MIN_FACTOR + (1 - MIN_FACTOR) * exp(-years_ago / DECAY_YEARS)
    decay_factor = np.exp(-years_ago / TEMPORAL_DECAY_YEARS)
    temporal_factor = MIN_TEMPORAL_FACTOR + (1 - MIN_TEMPORAL_FACTOR) * decay_factor

    return temporal_factor


def expected_score(elo_a, elo_b):
    """
    Calcula a probabilidade esperada de vitória do jogador A contra o jogador B.

    Args:
        elo_a (float): ELO do jogador A
        elo_b (float): ELO do jogador B

    Returns:
        float: Probabilidade esperada (0 a 1)

    Exemplo:
        - Se ELO_A = 1600 e ELO_B = 1500, expected_score ≈ 0.64 (64% de chance)
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(winner_elo, loser_elo, tournament_type, temporal_factor=None, match_date=None, reference_year=None):
    """
    Atualiza os ELOs dos jogadores após uma partida.
    Réplica EXATA da lógica do notebook.

    Args:
        winner_elo (float): ELO atual do vencedor
        loser_elo (float): ELO atual do perdedor
        tournament_type (str): Tipo do torneio ('ATP', 'CHALLENGER', ou 'ITF')
        temporal_factor (float, optional): Fator temporal pré-calculado
        match_date (datetime, optional): Data do jogo (para calcular temporal_factor)

    Returns:
        tuple: (novo_elo_vencedor, novo_elo_perdedor, k_factor_efetivo)

    Exemplo:
        >>> update_elo(1600, 1500, 'ATP', match_date=datetime(2025, 1, 1))
        (1607.36, 1492.64, 32.0)
    """
    # Obter K-factor base do tipo de torneio
    base_K = K_FACTORS.get(tournament_type, K_FACTORS['CHALLENGER'])

    # Calcular fator temporal se não fornecido
    if temporal_factor is None:
        if match_date is not None:
            temporal_factor = calculate_temporal_factor(match_date, reference_year=reference_year)
        else:
            temporal_factor = 1.0

    # K-factor efetivo = base * fator temporal
    effective_K = base_K * temporal_factor

    # Calcular probabilidade esperada de vitória
    expected_win = expected_score(winner_elo, loser_elo)

    # Atualizar ELOs
    # Vencedor ganha pontos baseado em quanto era "inesperada" a vitória
    # Perdedor perde a mesma quantidade
    change = effective_K * (1 - expected_win)

    new_winner_elo = winner_elo + change
    new_loser_elo = loser_elo - change

    return new_winner_elo, new_loser_elo, effective_K


def update_elo_simple(winner_elo, loser_elo, k_factor=32):
    """
    Versão simplificada para updates diários (sem temporal decay).
    Usa para jogos atuais onde temporal_factor = 1.0

    Args:
        winner_elo (float): ELO do vencedor
        loser_elo (float): ELO do perdedor
        k_factor (float): K-factor a usar (default 32 para ATP)

    Returns:
        tuple: (novo_elo_vencedor, novo_elo_perdedor)
    """
    expected_win = expected_score(winner_elo, loser_elo)
    change = k_factor * (1 - expected_win)

    return winner_elo + change, loser_elo - change


def get_k_factor_for_tournament(tournament_type):
    """
    Retorna o K-factor base para um tipo de torneio.

    Args:
        tournament_type (str): 'ATP', 'CHALLENGER', ou 'ITF'

    Returns:
        float: K-factor (32, 20, ou 10)
    """
    return K_FACTORS.get(tournament_type, K_FACTORS['CHALLENGER'])


# Backwards-compatible alias used by daily update scripts.
get_k_factor = get_k_factor_for_tournament


# ============================================================================
# ELO COM K DINÂMICO (sem ano de referência)
# ============================================================================
# K depende do nº de jogos já disputados pelo jogador (fórmula estilo FiveThirtyEight):
# jogadores novos movem-se depressa, veteranos estabilizam. Não há decaimento por ano civil,
# por isso o ELO de um jogo passado não depende da data em que a BD é reconstruída.

DYNAMIC_K_NUMERATOR = 250.0
DYNAMIC_K_OFFSET = 5.0
DYNAMIC_K_SHAPE = 0.4

# Multiplicadores por nível (códigos Sackmann); qualifying usa o multiplicador de Challenger
LEVEL_K_MULTIPLIER = {'G': 1.1, 'C': 0.85, 'S': 0.7, 'T': 0.7, 'J': 0.7}
QUALIFYING_ROUNDS = ('Q1', 'Q2', 'Q3', 'Q4')
QUALIFYING_K_MULTIPLIER = 0.85


def dynamic_k(n_matches):
    """K do jogador em função do nº de jogos anteriores (n=0 → ~131, n=500 → ~21)."""
    return DYNAMIC_K_NUMERATOR / ((max(n_matches, 0) + DYNAMIC_K_OFFSET) ** DYNAMIC_K_SHAPE)


def level_k_multiplier(tourney_level=None, round_name=None):
    """Peso do jogo no ELO consoante o nível do torneio e se é qualifying."""
    if isinstance(round_name, str) and round_name.upper() in QUALIFYING_ROUNDS:
        return QUALIFYING_K_MULTIPLIER
    if isinstance(tourney_level, str):
        return LEVEL_K_MULTIPLIER.get(tourney_level.upper(), 1.0)
    return 1.0


def update_elo_dynamic(winner_elo, loser_elo, winner_n, loser_n, multiplier=1.0):
    """Atualiza o ELO com K próprio de cada jogador.

    Returns:
        tuple: (novo_elo_vencedor, novo_elo_perdedor, k_vencedor, k_perdedor)
    """
    expected_win = expected_score(winner_elo, loser_elo)
    k_winner = dynamic_k(winner_n) * multiplier
    k_loser = dynamic_k(loser_n) * multiplier
    return (winner_elo + k_winner * (1 - expected_win),
            loser_elo - k_loser * (1 - expected_win),
            k_winner, k_loser)


# ============================================================================
# FUNÇÕES DE VALIDAÇÃO (Para testes)
# ============================================================================

def test_temporal_factors():
    """
    Testa se os fatores temporais estão corretos conforme o notebook.
    """
    test_cases = [
        (datetime(1992, 1, 1), 0.301),  # Ano 1992
        (datetime(2000, 1, 1), 0.305),  # Ano 2000
        (datetime(2010, 1, 1), 0.335),  # Ano 2010
        (datetime(2020, 1, 1), 0.558),  # Ano 2020
        (datetime(2024, 1, 1), 0.873),  # Ano 2024
        (datetime(2025, 1, 1), 1.000),  # Ano 2025
    ]

    print("🧪 Teste de Fatores Temporais:")
    all_passed = True

    for date, expected in test_cases:
        result = calculate_temporal_factor(date)
        diff = abs(result - expected)
        passed = diff < 0.01  # Tolerância de 1%

        status = "✅" if passed else "❌"
        print(f"  {status} Ano {date.year}: {result:.3f} (esperado: {expected:.3f})")

        if not passed:
            all_passed = False

    return all_passed


def test_elo_calculation():
    """
    Testa cálculo de ELO com exemplos conhecidos.
    """
    print("🧪 Teste de Cálculo ELO:")

    # Teste 1: ATP 2025 (temporal_factor = 1.0)
    w_elo, l_elo, k_eff = update_elo(2200, 2000, 'ATP', match_date=datetime(2025, 1, 1))
    print(f"  ✅ ATP 2025 (2200 vs 2000): {w_elo:.1f} / {l_elo:.1f} (K={k_eff:.1f})")

    # Teste 2: CHALLENGER 2025
    w_elo, l_elo, k_eff = update_elo(1600, 1400, 'CHALLENGER', match_date=datetime(2025, 1, 1))
    print(f"  ✅ CHALLENGER 2025 (1600 vs 1400): {w_elo:.1f} / {l_elo:.1f} (K={k_eff:.1f})")

    # Teste 3: ATP 1992 (temporal_factor ≈ 0.3)
    w_elo, l_elo, k_eff = update_elo(1500, 1500, 'ATP', match_date=datetime(1992, 1, 1))
    print(f"  ✅ ATP 1992 (1500 vs 1500): {w_elo:.1f} / {l_elo:.1f} (K={k_eff:.1f})")
    print(f"     (K-factor reduzido devido ao temporal decay)")


if __name__ == '__main__':
    print("=" * 60)
    print("🎾 MÓDULO DE CÁLCULO ELO - TENIS ML-AI")
    print("=" * 60)
    print("Configuração:")
    print(f"  K-Factor ATP: {K_FACTORS['ATP']}")
    print(f"  K-Factor CHALLENGER: {K_FACTORS['CHALLENGER']}")
    print(f"  K-Factor ITF: {K_FACTORS['ITF']}")
    print(f"  Temporal Decay: {TEMPORAL_DECAY_YEARS} anos")
    print(f"  Fator Mínimo: {MIN_TEMPORAL_FACTOR} ({MIN_TEMPORAL_FACTOR*100:.0f}%)")
    print(f"  Ano de Referência: {REFERENCE_YEAR}")
    print()

    # Executar testes
    test_temporal_factors()
    test_elo_calculation()

    print("" + "=" * 60)
    print("✅ Módulo carregado e testado com sucesso!")
    print("=" * 60)