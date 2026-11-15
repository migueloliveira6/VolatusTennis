import pandas as pd

import name_lookup
import scraper_results


def _lookup_file(tmp_path):
    path = tmp_path / 'name_lookup.csv'
    # formato real do ficheiro (alguns "new" têm um espaço à frente)
    path.write_text('old,new\nA. Stephens Sloane, Sloane Stephens\n'
                    'Agustin Tirante Thiago,Thiago Agustin Tirante\n', encoding='utf-8')
    return str(path)


def test_apply_name_lookup_replaces_and_strips(tmp_path):
    path = _lookup_file(tmp_path)
    assert name_lookup.apply_name_lookup('Agustin Tirante Thiago', path) == 'Thiago Agustin Tirante'
    assert name_lookup.apply_name_lookup('A. Stephens Sloane', path) == 'Sloane Stephens'
    assert name_lookup.apply_name_lookup('Taylor Fritz', path) == 'Taylor Fritz'


def test_missing_lookup_file_keeps_names(tmp_path):
    assert name_lookup.apply_name_lookup('Taylor Fritz', str(tmp_path / 'nao_existe.csv')) == 'Taylor Fritz'


class _FakeResponse:
    content = b'<table class="plDetail"><tbody><tr><td><h3>Tirante Thiago Agustin</h3></td></tr></tbody></table>'

    def raise_for_status(self):
        pass


def test_results_scraper_applies_same_lookup_as_predictions(tmp_path, monkeypatch):
    """Caso real de 14/11/2026: o resultado tem de sair com o nome corrigido, igual à previsão."""
    monkeypatch.setenv('NAME_LOOKUP', _lookup_file(tmp_path))
    monkeypatch.setattr(scraper_results.requests, 'get', lambda *a, **k: _FakeResponse())
    scraper_results.nome_cache.clear()

    assert scraper_results.getPlayersFullName('/player/tirante/') == 'Thiago Agustin Tirante'
