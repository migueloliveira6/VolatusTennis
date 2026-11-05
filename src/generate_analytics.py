# scripts/generate_analysis.py

import os
import json
import re
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_record import TRACK_RECORD_FILENAME, build_track_rows, merge_track_record

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARISONS_DIR = os.path.join(BASE_DIR, "previsoes")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "analytics")

# Cria pastas se não existirem
os.makedirs(DOCS_DATA_DIR, exist_ok=True)


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if data is not None else default
    except Exception:
        return default


def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def numeric_series(df, column_name, default=0.0):
    if column_name not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column_name], errors='coerce').fillna(default)


def upsert_count_entry(existing_items, key_name, key_value, delta_total, delta_correct):
    items_by_key = {str(item.get(key_name)): dict(item) for item in existing_items if key_name in item}

    current = items_by_key.get(str(key_value), {
        key_name: key_value,
        "total_predictions": 0,
        "correct": 0,
        "accuracy": 0.0
    })

    current_total = int(current.get("total_predictions", 0)) + int(delta_total)
    current_correct = int(current.get("correct", 0)) + int(delta_correct)

    if current_total <= 0:
        items_by_key.pop(str(key_value), None)
    else:
        current["total_predictions"] = current_total
        current["correct"] = current_correct
        current["accuracy"] = round((current_correct / current_total) * 100, 2)
        items_by_key[str(key_value)] = current

    return list(items_by_key.values())


def find_latest_comparison_file(comparisons_dir):
    csv_files = [
        f for f in os.listdir(comparisons_dir)
        if f.startswith("comparacao_previsoes_") and f.endswith(".csv")
    ]

    if not csv_files:
        return None, None

    dated_files = []
    undated_files = []
    for filename in csv_files:
        match = re.match(r"comparacao_previsoes_(\d{4}-\d{2}-\d{2})\.csv$", filename)
        full_path = os.path.join(comparisons_dir, filename)
        if match:
            dated_files.append((match.group(1), full_path))
        else:
            undated_files.append(full_path)

    if dated_files:
        dated_files.sort(key=lambda item: item[0], reverse=True)
        latest_date, latest_file = dated_files[0]
        return latest_file, latest_date

    undated_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    latest_file = undated_files[0]
    latest_date = datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime("%Y-%m-%d")
    return latest_file, latest_date

# === 1️⃣ Localizar CSVs de comparação ===
latest_csv_path, comparison_date = find_latest_comparison_file(COMPARISONS_DIR)

if not latest_csv_path:
    print("⚠️  Nenhum arquivo de comparação encontrado. Execute compare_predictions_results.py primeiro.")
    exit(0)

latest_csv_name = os.path.basename(latest_csv_path)
print(f"Usando arquivo de comparação mais recente: {latest_csv_name}")

# === 2️⃣ Carregar CSV mais recente ===
df = pd.read_csv(latest_csv_path)
print(f"   ✓ {latest_csv_name}")

# === 3️⃣ Validar colunas esperadas ===
expected_cols = [
    "Pred_File", "Pred_Date", "Pred_Torneio", "Jogador 1", "Jogador 2",
    "Vencedor Previsto", "Confiança (%)", "Actual_Winner",
    "Matched", "Correct"
]

missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

# === 4️⃣ Filtrar apenas previsões matched (que tiveram resultado) ===
df_matched = df[df['Matched'] == True].copy()

if df_matched.empty:
    print("⚠️  Nenhuma previsão matched encontrada. Não há dados para análise.")
    exit(0)

print(f"\n📈 Total de previsões matched: {len(df_matched)}")

stake_series = numeric_series(df_matched, 'Valor Aposta', 0.0)
bet_mask = stake_series > 0

if 'Resultado Aposta' in df_matched.columns:
    bet_result_series = df_matched['Resultado Aposta'].fillna('NO_BET').astype(str).str.upper()
else:
    bet_result_series = pd.Series(
        [
            'WIN' if bool(correct) and bool(bet) else 'LOSS' if bool(bet) else 'NO_BET'
            for correct, bet in zip(df_matched['Correct'].fillna(False), bet_mask)
        ],
        index=df_matched.index,
        dtype=str,
    )

bet_wins_series = bet_mask & (bet_result_series == 'WIN')
bet_losses_series = bet_mask & (bet_result_series == 'LOSS')
bet_stake_total = float(stake_series[bet_mask].sum())
bet_return_total = float(numeric_series(df_matched, 'Retorno Aposta', 0.0)[bet_mask].sum())
bet_profit_total = float(numeric_series(df_matched, 'Lucro Aposta', 0.0)[bet_mask].sum())
bet_bets_total = int(bet_mask.sum())
bet_wins_total = int(bet_wins_series.sum())
bet_losses_total = int(bet_losses_series.sum())
bet_accuracy = (bet_wins_total / bet_bets_total * 100) if bet_bets_total > 0 else 0
bet_roi = (bet_profit_total / bet_stake_total * 100) if bet_stake_total > 0 else 0

# === 5️⃣ Preparar paths e estado persistente ===
global_stats_path = os.path.join(DOCS_DATA_DIR, "global_stats.json")
accuracy_by_date_path = os.path.join(DOCS_DATA_DIR, "accuracy_by_date.json")
accuracy_by_surface_path = os.path.join(DOCS_DATA_DIR, "accuracy_by_surface.json")
accuracy_by_confidence_path = os.path.join(DOCS_DATA_DIR, "accuracy_by_confidence.json")
analytics_state_path = os.path.join(DOCS_DATA_DIR, "analytics_state.json")

analytics_state = load_json_file(analytics_state_path, {"processed_comparison_files": []})
processed_files = set(analytics_state.get("processed_comparison_files", []))
already_processed = latest_csv_name in processed_files

# === 6️⃣ Agregação por data (patch em cima do histórico existente) ===
reference_date = comparison_date or datetime.now().strftime("%Y-%m-%d")
total = len(df_matched)
correct = df_matched['Correct'].sum()
wrong = total - correct
acc = (correct / total * 100) if total > 0 else 0

new_entry = {
    "date": reference_date,
    "total_predictions": int(total),
    "correct": int(correct),
    "wrong": int(wrong),
    "accuracy": round(acc, 2),
    "bet_bets": bet_bets_total,
    "bet_wins": bet_wins_total,
    "bet_losses": bet_losses_total,
    "bet_stake_total": round(bet_stake_total, 3),
    "bet_return_total": round(bet_return_total, 3),
    "bet_profit_total": round(bet_profit_total, 3),
    "bet_accuracy": round(bet_accuracy, 2),
    "bet_roi": round(bet_roi, 2)
}

accuracy_by_date = load_json_file(accuracy_by_date_path, [])

# Adicionar entrada do dia ao histórico
accuracy_by_date = [entry for entry in accuracy_by_date if entry.get("date") != reference_date]
accuracy_by_date.append(new_entry)
accuracy_by_date.sort(key=lambda x: x.get('date', ''))

print(f"   ✓ {len(accuracy_by_date)} dias com dados")

# === 7️⃣ Calcular estatísticas globais cumulativas ===
total_predictions_cumulative = sum(int(item.get("total_predictions", 0)) for item in accuracy_by_date)
total_correct_cumulative = sum(int(item.get("correct", 0)) for item in accuracy_by_date)
total_wrong_cumulative = total_predictions_cumulative - total_correct_cumulative
accuracy_global = (total_correct_cumulative / total_predictions_cumulative * 100) if total_predictions_cumulative > 0 else 0
bet_bets_cumulative = sum(int(item.get("bet_bets", 0)) for item in accuracy_by_date)
bet_wins_cumulative = sum(int(item.get("bet_wins", 0)) for item in accuracy_by_date)
bet_losses_cumulative = sum(int(item.get("bet_losses", 0)) for item in accuracy_by_date)
bet_stake_cumulative = sum(float(item.get("bet_stake_total", 0)) for item in accuracy_by_date)
bet_return_cumulative = sum(float(item.get("bet_return_total", 0)) for item in accuracy_by_date)
bet_profit_cumulative = sum(float(item.get("bet_profit_total", 0)) for item in accuracy_by_date)
bet_accuracy_global = (bet_wins_cumulative / bet_bets_cumulative * 100) if bet_bets_cumulative > 0 else 0
bet_roi_global = (bet_profit_cumulative / bet_stake_cumulative * 100) if bet_stake_cumulative > 0 else 0

global_stats = {
    "total_predictions": int(total_predictions_cumulative),
    "correct": int(total_correct_cumulative),
    "wrong": int(total_wrong_cumulative),
    "accuracy": round(accuracy_global, 2),
    "bet_bets": int(bet_bets_cumulative),
    "bet_wins": int(bet_wins_cumulative),
    "bet_losses": int(bet_losses_cumulative),
    "bet_stake_total": round(bet_stake_cumulative, 3),
    "bet_return_total": round(bet_return_cumulative, 3),
    "bet_profit_total": round(bet_profit_cumulative, 3),
    "bet_accuracy": round(bet_accuracy_global, 2),
    "bet_roi": round(bet_roi_global, 2),
    "comparison_file": latest_csv_name,
    "comparison_date": comparison_date,
    "runs_processed": int(len(processed_files) + (0 if already_processed else 1)),
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

print(f"   ✓ Taxa de acerto global acumulada: {accuracy_global:.1f}%")
if bet_bets_cumulative > 0:
    print(f"   ✓ Apostas acumuladas: {bet_bets_cumulative} | Ganhas: {bet_wins_cumulative} | Perdidas: {bet_losses_cumulative}")
    print(f"   ✓ Stake acumulado: {bet_stake_cumulative:.3f} | Retorno acumulado: {bet_return_cumulative:.3f} | Lucro acumulado: {bet_profit_cumulative:.3f}")
    print(f"   ✓ Acurácia das apostas acumulada: {bet_accuracy_global:.1f}%")
    print(f"   ✓ ROI das apostas acumulado: {bet_roi_global:.1f}%")

# === 8️⃣ Agregação por superfície ===
accuracy_by_surface = load_json_file(accuracy_by_surface_path, [])
for surface in df_matched['Pred_Torneio'].dropna().unique():
    df_surface = df_matched[df_matched['Pred_Torneio'] == surface]
    total = len(df_surface)
    correct = df_surface['Correct'].sum()
    if not already_processed and total > 0:
        accuracy_by_surface = upsert_count_entry(
            accuracy_by_surface,
            "surface",
            surface,
            int(total),
            int(correct)
        )

accuracy_by_surface = [item for item in accuracy_by_surface if int(item.get("total_predictions", 0)) >= 3]

accuracy_by_surface.sort(key=lambda x: x['accuracy'], reverse=True)

print(f"   ✓ {len(accuracy_by_surface)} torneios/superfícies analisados")

# === 9️⃣ Agregação por faixa de confiança ===
def get_confidence_bucket(conf):
    if conf >= 80:
        return "80-100%"
    elif conf >= 70:
        return "70-79%"
    elif conf >= 60:
        return "60-69%"
    elif conf >= 50:
        return "50-59%"
    else:
        return "<50%"

df_matched['confidence_bucket'] = pd.to_numeric(df_matched['Confiança (%)'], errors='coerce').fillna(0).apply(get_confidence_bucket)

accuracy_by_confidence = load_json_file(accuracy_by_confidence_path, [])
for bucket in ["80-100%", "70-79%", "60-69%", "50-59%", "<50%"]:
    df_bucket = df_matched[df_matched['confidence_bucket'] == bucket]
    total = len(df_bucket)
    if total > 0 and not already_processed:
        correct = df_bucket['Correct'].sum()
        accuracy_by_confidence = upsert_count_entry(
            accuracy_by_confidence,
            "confidence_range",
            bucket,
            int(total),
            int(correct)
        )

bucket_order = {"80-100%": 0, "70-79%": 1, "60-69%": 2, "50-59%": 3, "<50%": 4}
accuracy_by_confidence.sort(key=lambda x: bucket_order.get(x.get("confidence_range"), 999))

print(f"   ✓ Análise por faixa de confiança concluída")

if already_processed:
    print(f"   ↺ {latest_csv_name} já estava processado. Mantido comportamento idempotente para superfícies/confiança.")
else:
    processed_files.add(latest_csv_name)

analytics_state = {
    "processed_comparison_files": sorted(processed_files),
    "last_processed_file": latest_csv_name,
    "last_processed_date": reference_date,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# === 🔟 Salvar JSONs ===
output_files = {
    "global_stats.json": global_stats,
    "accuracy_by_date.json": accuracy_by_date,
    "accuracy_by_surface.json": accuracy_by_surface,
    "accuracy_by_confidence.json": accuracy_by_confidence,
    "analytics_state.json": analytics_state
}

for filename, data in output_files.items():
    output_path = os.path.join(DOCS_DATA_DIR, filename)
    save_json_file(output_path, data)
    print(f"   ✓ {filename}")

# Histórico jogo a jogo (idempotente: reprocessar o mesmo dia substitui as linhas desse dia)
track_record_path = os.path.join(DOCS_DATA_DIR, TRACK_RECORD_FILENAME)
track_record = merge_track_record(load_json_file(track_record_path, []), build_track_rows(df_matched))
with open(track_record_path, "w", encoding="utf-8") as f:
    json.dump(track_record, f, ensure_ascii=False, separators=(",", ":"))
print(f"   ✓ {TRACK_RECORD_FILENAME} ({len(track_record)} jogos)")

print(f"\n✅ Análise concluída! Arquivos salvos em {DOCS_DATA_DIR}")
print(f"🔗 Pronto para deploy no GitHub Pages!")
