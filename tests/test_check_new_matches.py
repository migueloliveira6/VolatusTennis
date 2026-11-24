import json
from datetime import date

import pytest

import check_new_matches as cnm
from engine import Jogo, NeuroTennis

DIA = date(2026, 11, 23)


class FakeScraper:
    def __init__(self, jogos):
        self.jogos = jogos
        self.datas = []

    def extrair_jogos_agrupados_por_torneio(self, dias_offset=1, data_alvo=None):
        self.datas.append(data_alvo)
        return self.jogos


class FakePredictor:
    """Prevê sempre o jogador 1 com 60%, sem modelo."""
    def predict_match(self, p1, p2, surface, date=None, tourney_level='UNK'):
        return p1, 0.6, {'elo_diff': 10.0, 'h2h': '0-0'}


def _publicado(j1, j2):
    return {'Torneio': 'Stockholm', 'Jogador 1': j1, 'Jogador 2': j2, 'Vencedor Previsto': j1,
            'Confiança (%)': 70.0, 'ELO Diff': 50.0, 'H2H': '-', 'Odd 1': 1.5, 'Odd 2': 2.5,
            'Superfície': 'Hard', 'Valor Aposta': 0.0, 'ROI Esperado (%)': 0.0}


def _site(tmp_path, publicadas, status='ready'):
    predicts = tmp_path / 'predicts'
    (predicts / 'history').mkdir(parents=True)
    (predicts / 'predictions.json').write_text(json.dumps(publicadas), encoding='utf-8')
    (predicts / f'history/predictions_{DIA}.json').write_text(json.dumps(publicadas), encoding='utf-8')
    (predicts / 'status.json').write_text(json.dumps({'status': status, 'date': str(DIA)}), encoding='utf-8')
    return tmp_path


def _ler(site, rel):
    return json.loads((site / rel).read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def sem_telegram(monkeypatch):
    enviados = []
    monkeypatch.setattr(cnm, 'TOKEN_BOT', 'token')
    monkeypatch.setattr(cnm, 'CHAT_ID', 'chat')
    monkeypatch.setattr(cnm, 'enviar_telegram', lambda msg, *a: enviados.append(msg) or True)
    monkeypatch.setattr(cnm, 'enviar_resultados_telegram', lambda df, *a: True)
    return enviados


def test_only_new_not_started_matches_with_odds_are_added(tmp_path, sem_telegram):
    site = _site(tmp_path, [_publicado('Taylor Fritz', 'Holger Rune')])
    scraper = FakeScraper({'Stockholm': [
        Jogo('Holger Rune', 'Taylor Fritz', 1.4, 2.8),                 # já publicado (ordem trocada)
        Jogo('Lorenzo Musetti', 'Ugo Humbert', 1.5, 2.6),              # novo
        Jogo('Alex Michelsen', 'Alejandro Tabilo', 1.3, 3.5, iniciado=True),  # já começou
        Jogo('Casper Ruud', 'Jaume Munar', None, None),                # sem odds
    ]})

    resumo = cnm.verificar(str(site), scraper=scraper, predictor=FakePredictor(), hoje=DIA)

    assert scraper.datas == [DIA]
    assert resumo['has_new'] and resumo['novos'] == 1
    assert (resumo['ja_publicados'], resumo['iniciados'], resumo['sem_odds']) == (1, 1, 1)
    predictions = _ler(site, 'predicts/predictions.json')
    assert [p['Jogador 1'] for p in predictions] == ['Taylor Fritz', 'Lorenzo Musetti']  # ordenado por confiança
    assert _ler(site, f'predicts/history/predictions_{DIA}.json') == predictions
    assert len(sem_telegram) == 1 and 'Lorenzo Musetti vs Ugo Humbert' in sem_telegram[0]


def test_second_run_adds_nothing(tmp_path, sem_telegram):
    site = _site(tmp_path, [])
    scraper = FakeScraper({'Stockholm': [Jogo('Lorenzo Musetti', 'Ugo Humbert', 1.5, 2.6)]})
    assert cnm.verificar(str(site), scraper=scraper, predictor=FakePredictor(), hoje=DIA)['has_new']
    segundo = cnm.verificar(str(site), scraper=scraper, predictor=FakePredictor(), hoje=DIA)
    assert not segundo['has_new'] and segundo['ja_publicados'] == 1
    assert len(_ler(site, 'predicts/predictions.json')) == 1
    assert len(sem_telegram) == 1


def test_no_matches_status_becomes_ready(tmp_path):
    site = _site(tmp_path, [], status='no_matches')
    scraper = FakeScraper({'Stockholm': [Jogo('Lorenzo Musetti', 'Ugo Humbert', 1.5, 2.6)]})
    cnm.verificar(str(site), scraper=scraper, predictor=FakePredictor(), hoje=DIA)
    status = _ler(site, 'predicts/status.json')
    assert status['status'] == 'ready' and status['date'] == str(DIA)


def test_dry_run_writes_and_sends_nothing(tmp_path, sem_telegram):
    site = _site(tmp_path, [])
    antes = (site / 'predicts/predictions.json').read_text(encoding='utf-8')
    scraper = FakeScraper({'Stockholm': [Jogo('Lorenzo Musetti', 'Ugo Humbert', 1.5, 2.6)]})
    resumo = cnm.verificar(str(site), dry_run=True, scraper=scraper, predictor=FakePredictor(), hoje=DIA)
    assert resumo['has_new'] and resumo['jogos_novos'] == ['Lorenzo Musetti vs Ugo Humbert']
    assert (site / 'predicts/predictions.json').read_text(encoding='utf-8') == antes
    assert sem_telegram == []


def test_past_date_is_skipped_without_scraping(tmp_path):
    site = _site(tmp_path, [])
    scraper = FakeScraper({})
    resumo = cnm.verificar(str(site), scraper=scraper, predictor=FakePredictor(), hoje=date(2026, 11, 24))
    assert not resumo['has_new'] and scraper.datas == []


def test_url_uses_explicit_date():
    url = NeuroTennis(timeout=5)._construir_url(data_alvo=date(2026, 11, 23))
    assert url == 'https://www.betexplorer.com/tennis/next/?year=2026&month=11&day=23'
