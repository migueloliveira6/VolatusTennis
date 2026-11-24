"""Leitura das páginas do BetExplorer (estrutura real de 24/11/2026, reduzida)."""
from datetime import date

import pandas as pd

import betexplorer as bx
from engine import NeuroTennis, inferir_nivel_torneio
from scraper_results import ResultsScraper


def _cabecalho(texto, href='/tennis/atp-singles/x/'):
    return (f'<tr class="js-tournament"><th class="h-text-left" colspan="2">'
            f'<a href="{href}" class="table-main__tournament table-main__tournament--tennis">'
            f'<i><img alt="Italy"></i><i class="tennis_tournament_type tennis_tournament_type--ms"></i>{texto}</a>'
            f'</th><th class="table-main__odds"></th><th class="table-main__odds"></th></tr>')


def _por_comecar(dt, casa, fora, href, odds=('1.67', '2.16'), agora='24,11,2026,12,00'):
    celulas = ''.join(f'<td class="table-main__odds"><button data-odd="{o}"></button></td>' for o in odds)
    return (f'<tr data-dt="{dt}" data-dt-now="{agora}"><td class="table-main__tt">'
            f'<span class="table-main__time">18:40</span>'
            f'<a class="table-main__teamsLink" href="{href}">'
            f'<span class="table-main__teamLine table-main__teamLine--home">{casa}</span>'
            f'<span class="table-main__teamLine table-main__teamLine--away">{fora}</span></a></td>'
            f'<td class="table-main__streams" data-live-cell="score"><a href="{href}"></a></td>{celulas}</tr>')


def _terminado(dt, casa, fora, href, resultado, parciais, vencedor='casa'):
    casa = f'<strong>{casa}</strong>' if vencedor == 'casa' else casa
    fora = f'<strong>{fora}</strong>' if vencedor == 'fora' else fora
    return (f'<tr data-dt="{dt}" data-dt-now="24,11,2026,20,00"><td class="table-main__tt">'
            f'<span class="table-main__time table-main__time--fin">FIN</span>'
            f'<a class="table-main__teamsLink" href="{href}">'
            f'<span class="table-main__teamLine table-main__teamLine--home">{casa}</span>'
            f'<span class="table-main__teamLine table-main__teamLine--away">{fora}</span></a></td>'
            f'<td class="table-main__result"><a href="{href}"><strong>{resultado}</strong></a></td>'
            f'<td class="table-main__partial" colspan="2">{parciais}</td></tr>')


def _pagina(*tbodies):
    corpo = ''.join(f'<tbody data-key="{i}">{t}</tbody>' for i, t in enumerate(tbodies))
    return f'<html><body><table class="table-main only12">{corpo}</table></body></html>'


PAGINA_DIA = _pagina(
    _cabecalho('ATP - Singles: Rome, clay')
    + _por_comecar('25,11,2026,14,00', 'Sinner J.', 'Alcaraz C.', '/tennis/atp-singles/rome/sinner-alcaraz/A1/')
    + _por_comecar('24,11,2026,11,00', 'Ruud C.', 'Fritz T.', '/tennis/atp-singles/rome/ruud-fritz/A2/')  # já passou
    + _por_comecar('26,11,2026,10,00', 'Rune H.', 'Paul T.', '/tennis/atp-singles/rome/rune-paul/A3/'),  # outro dia
    _cabecalho('ATP - Singles: Davis Cup - World Group')
    + _por_comecar('25,11,2026,16,20', 'Machac T.', 'Draxl L.', '/tennis/atp-singles/davis/machac-draxl/D1/', odds=()),
    _cabecalho('Challenger Men - Singles: Yokohama, hard', '/tennis/challenger-men-singles/yokohama/')
    + _por_comecar('25,11,2026,03,00', 'Zhu E.', 'Cook E.', '/tennis/challenger-men-singles/yokohama/zhu-cook/C1/'),
    _cabecalho('ATP - Doubles: Davis Cup - World Group', '/tennis/atp-doubles/davis/')
    + _por_comecar('25,11,2026,19,00', 'Machac/Pavlasek', 'Diallo/Galarneau', '/tennis/atp-doubles/davis/d/D2/'),
)

PAGINA_RESULTADOS = _pagina(
    _cabecalho('ATP - Singles: Finals - Turin, hard')
    + _terminado('15,11,2026,14,00', 'Zverev A.', 'Jodar R.', '/tennis/atp-singles/finals/z-j/F1/', '2:0', '(6:4, 6:4)')
    + _terminado('15,11,2026,20,30', 'Fils A.', 'Shelton B.', '/tennis/atp-singles/finals/f-s/F2/', '1:2',
                 '(2:6, 7:6<sup>5</sup>, 3:6)', vencedor='fora')
    + _terminado('15,11,2026,22,00', 'Fritz T.', 'Rune H.', '/tennis/atp-singles/finals/f-r/F3/',
                 'WO.', '', vencedor='casa')
    + _terminado('15,11,2026,23,00', 'Ruud C.', 'De Minaur A.', '/tennis/atp-singles/finals/r-d/F4/',
                 '0:1 <span title="Retired">RET.</span>', '(6<sup>3</sup>:7, 1:0)', vencedor='fora')
    + _terminado('14,11,2026,20,00', 'Sinner J.', 'Musetti L.', '/tennis/atp-singles/finals/s-m/F5/', '2:0', '(6:1, 6:1)'),
)

NOMES = {
    '/tennis/atp-singles/rome/sinner-alcaraz/A1/': ('Sinner Jannik', 'Alcaraz Carlos'),
    '/tennis/atp-singles/rome/ruud-fritz/A2/': ('Ruud Casper', 'Fritz Taylor'),
    '/tennis/atp-singles/davis/machac-draxl/D1/': ('Machac Tomas', 'Draxl Liam'),
    '/tennis/atp-singles/finals/z-j/F1/': ('Zverev Alexander', 'Jodar Rafael'),
    '/tennis/atp-singles/finals/f-s/F2/': ('Fils Arthur', 'Shelton Ben'),
    '/tennis/atp-singles/finals/f-r/F3/': ('Fritz Taylor', 'Rune Holger'),
    '/tennis/atp-singles/finals/r-d/F4/': ('Ruud Casper', 'De Minaur Alex'),
}


def _pagina_jogo(casa, fora):
    # cada jogador aparece duas vezes (foto sem texto e nome), como na página real
    links = ''.join(f'<a href="/tennis/player/{n.lower().replace(" ", "-")}/x/"></a>'
                    f'<h2><a href="/tennis/player/{n.lower().replace(" ", "-")}/x/">{n}</a></h2>'
                    for n in (casa, fora))
    return f'<html><body><h1>{casa} vs {fora} H2H Stats &amp; Odds</h1>{links}</body></html>'


class FakeCliente(bx.BetExplorerClient):
    """Cliente sem rede: devolve as páginas acima."""
    def __init__(self, lista):
        super().__init__(lookup_path='nao_existe.csv')
        self.lista = lista
        self.pedidos = []

    def html(self, url):
        self.pedidos.append(url)
        caminho = url.replace(bx.BASE_URL, '')
        if caminho in NOMES:
            return _pagina_jogo(*NOMES[caminho])
        return self.lista


def test_header_gives_tournament_and_surface():
    assert bx.separar_cabecalho('ATP - Singles: Rome, clay') == ('Rome', 'Clay')
    assert bx.separar_cabecalho('ATP - Singles: Finals - Turin, hard') == ('Finals - Turin', 'Hard')
    assert bx.separar_cabecalho('ATP - Singles: Davis Cup - World Group') == ('Davis Cup - World Group', None)


def test_list_keeps_only_atp_singles_of_the_requested_day():
    jogos = bx.ler_lista(PAGINA_DIA, date(2026, 11, 25))
    assert [(j.torneio, j.casa) for j in jogos] == [('Rome', 'Sinner J.'), ('Davis Cup - World Group', 'Machac T.')]
    sinner, machac = jogos
    assert (sinner.odd_casa, sinner.odd_fora, sinner.superficie) == (1.67, 2.16, 'Clay')
    assert not sinner.iniciado and not sinner.terminado
    assert (machac.odd_casa, machac.odd_fora) == (None, None)


def test_match_is_started_when_its_time_has_passed():
    # ao vivo não aparece no HTML do servidor: conta a hora marcada face a data-dt-now
    jogos = bx.ler_lista(PAGINA_DIA, date(2026, 11, 24))
    assert [(j.casa, j.iniciado) for j in jogos] == [('Ruud C.', True)]


def test_names_from_match_page_are_converted_like_before():
    assert bx.ler_nomes_jogo(_pagina_jogo('Auger-Aliassime Felix', 'Lehecka Jiri')) == \
        ('Auger-Aliassime Felix', 'Lehecka Jiri')
    assert bx.converter_nome('Auger-Aliassime Felix', 'nao_existe.csv') == 'Felix Auger Aliassime'
    assert bx.converter_nome('De Minaur Alex', 'nao_existe.csv') == 'Alex De Minaur'


def test_engine_groups_matches_with_full_names_and_surface():
    cliente = FakeCliente(PAGINA_DIA)
    jogos = NeuroTennis(cliente=cliente).extrair_jogos_agrupados_por_torneio(data_alvo=date(2026, 11, 25))
    assert list(jogos) == ['Rome', 'Davis Cup - World Group']
    sinner = jogos['Rome'][0]
    assert (sinner.player1, sinner.player2, sinner.odd1, sinner.odd2, sinner.superficie) == \
        ('Jannik Sinner', 'Carlos Alcaraz', 1.67, 2.16, 'Clay')
    assert cliente.pedidos[0] == 'https://www.betexplorer.com/tennis/next/?year=2026&month=11&day=25'


def test_tournament_levels_for_betexplorer_names():
    assert inferir_nivel_torneio('Finals - Turin') == 'F'
    assert inferir_nivel_torneio('Davis Cup - World Group') == 'D'
    assert inferir_nivel_torneio('Australian Open') == 'G'
    assert inferir_nivel_torneio('Rome') == 'M'
    assert inferir_nivel_torneio('Stockholm') == 'A'


def test_results_winner_score_and_walkovers():
    cliente = FakeCliente(PAGINA_RESULTADOS)
    df = ResultsScraper(cliente=cliente).scrape_results(pd.Timestamp('2026-11-15').to_pydatetime())
    assert df[['winner', 'loser', 'score']].values.tolist() == [
        ['Alexander Zverev', 'Rafael Jodar', '6-4 6-4'],
        ['Ben Shelton', 'Arthur Fils', '6-2 6-7(5) 6-3'],   # do ponto de vista do vencedor
        ['Alex De Minaur', 'Casper Ruud', '7-6(3) 0-1 RET'],  # desistência conta; W.O. não
    ]
    assert set(df['tourney_name']) == {'Finals - Turin'} and set(df['surface']) == {'Hard'}
    assert set(df['date']) == {'2026-11-15'}
