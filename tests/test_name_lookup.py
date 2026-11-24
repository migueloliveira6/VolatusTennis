import pandas as pd

import betexplorer
import name_lookup


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


def test_results_scraper_applies_same_lookup_as_predictions(tmp_path, monkeypatch):
    """Caso real de 14/11/2026: o resultado tem de sair com o nome corrigido, igual à previsão."""
    lookup = _lookup_file(tmp_path)
    monkeypatch.setenv('NAME_LOOKUP', lookup)
    # resultados (caminho do ambiente) e previsões (caminho explícito) dão o mesmo nome
    assert betexplorer.converter_nome('Tirante Thiago Agustin') == 'Thiago Agustin Tirante'
    assert betexplorer.converter_nome('Tirante Thiago Agustin', lookup) == 'Thiago Agustin Tirante'
