# scripts/generate_predictions.py
import datetime
import os
import json
import pandas as pd
from glob import glob
import re

# === CONFIGURAÇÕES ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "previsoes")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "predicts")
STATUS_PATH = os.path.join(DOCS_DATA_DIR, "status.json")

# Cria pastas se não existirem
os.makedirs(DOCS_DATA_DIR, exist_ok=True)

status_atual = None
if os.path.exists(STATUS_PATH):
    with open(STATUS_PATH, encoding="utf-8") as status_file:
        status_atual = json.load(status_file)

if status_atual and status_atual.get("status") == "no_matches":
    output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump([], output_file)
    print(f"⚠️ Estado sem jogos preservado. Site atualizado: {output_json_path}")
    raise SystemExit(0)

# === 1️⃣ Localizar o CSV mais recente de previsões ===
pattern = os.path.join(NOTEBOOKS_DIR, "previsoes_tenis_*.csv")
files = glob(pattern)
if not files:
    output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump([], output_file)
    with open(STATUS_PATH, "w", encoding="utf-8") as status_file:
        json.dump({
            "status": "no_matches",
            "message": "Não há jogos de ténis disponíveis para hoje.",
            "date": datetime.datetime.today().strftime("%Y-%m-%d")
        }, status_file, ensure_ascii=False, indent=4)
    print(f"⚠️ Nenhuma previsão encontrada. Site atualizado sem jogos: {output_json_path}")
    raise SystemExit(0)

date_pattern = re.compile(r"^previsoes_tenis_(\d{4}-\d{2}-\d{2})\.csv$")
dated_files = []
for path in files:
    base = os.path.basename(path)
    match = date_pattern.match(base)
    if not match:
        continue
    date_str = match.group(1)
    dated_files.append((datetime.datetime.strptime(date_str, "%Y-%m-%d"), path))

if not dated_files:
    output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump([], output_file)
    with open(STATUS_PATH, "w", encoding="utf-8") as status_file:
        json.dump({
            "status": "no_matches",
            "message": "Não há jogos de ténis disponíveis para hoje.",
            "date": datetime.datetime.today().strftime("%Y-%m-%d")
        }, status_file, ensure_ascii=False, indent=4)
    print(f"⚠️ Nenhuma previsão encontrada. Site atualizado sem jogos: {output_json_path}")
    raise SystemExit(0)

csv_path = max(dated_files, key=lambda item: item[0])[1]

print(f"📄 Carregando previsões do arquivo: {csv_path}")
df = pd.read_csv(csv_path)

# === 2️⃣ Garantir colunas esperadas ===
expected_cols = [
    "Torneio", "Jogador 1", "Jogador 2", "Vencedor Previsto",
    "Confiança (%)", "ELO Diff", "H2H", "Odd 1", "Odd 2",
    "Superfície", "Valor Aposta", "ROI Esperado (%)"
]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

# === 4️⃣ Salvar como JSON para o dashboard ===
output_json_path = os.path.join(DOCS_DATA_DIR, "predictions.json")
df.to_json(output_json_path, orient="records", force_ascii=False, indent=4)

with open(STATUS_PATH, "w", encoding="utf-8") as status_file:
    json.dump({
        "status": "ready",
        "message": "Previsões disponíveis.",
        "date": csv_path.rsplit("_", 1)[-1].replace(".csv", "")
    }, status_file, ensure_ascii=False, indent=4)

print(f"✅ Previsões exportadas para {output_json_path}")
print(f"🔗 Pronto para deploy no GitHub Pages!")
