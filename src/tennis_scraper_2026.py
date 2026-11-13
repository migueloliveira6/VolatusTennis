import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

LINKS_2026 = os.getenv('LINKS_2026', 'tennisabstract links tourn 26.txt')
BASE_URL_SCRAPER = "https://www.tennisabstract.com"

class TennisAbstractScraper:
    def __init__(self):
        self.base_url = BASE_URL_SCRAPER
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Campos do dataset
        self.columns = [
            'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 
            'tourney_date', 'match_num',
            'winner_name',
            'loser_name',
            'score', 'DRW','best_of', 'round',
            'minutes', 'w_ace', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
            'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
            'winner_rank', 'loser_rank',
            'tournament_type'
        ]
        
        self.all_matches = []
        
    def get_tournament_list_from_file(self, file_path):
        try:
            """Lê os links do ficheiro .txt e retorna lista de torneios"""
            tournaments = []
            with open(file_path, "r") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        tourney_code = url.split("t=")[-1]
                        tourney_name = tourney_code.split("/")[-1].replace("%27", "'")
                        tournaments.append({
                            'tourney_id': tourney_code,
                            'tourney_name': tourney_name,
                            'url': url
                        })
            return tournaments
            
        except Exception as e:
            print(f"Erro ao obter lista de torneios: {e}")
            return []
    
    def extract_match_data(self, soup, tourney_info):
        """Extrai apenas a tabela principal de resultados de simples (id=singles-results)"""
        matches = []

        table = soup.find("table", {"id": "singles-results"})
        if not table:
            print(f"⚠️ Nenhuma tabela 'singles-results' encontrada em {tourney_info['tourney_name']}")
            return matches

        rows = table.find_all("tr")
        if not rows or len(rows) < 2:
            return matches

        # cabeçalho + dados
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 6:
                match_data = self.parse_match_row(cells, tourney_info)
                if match_data:
                    matches.append(match_data)

        return matches

    
    def parse_match_row(self, cells, tourney_info):
        """Parse uma linha de partida baseada na estrutura fornecida"""
        try:
            match_data = {}
            
            # Preencher dados básicos do torneio
            match_data.update({
                'tourney_id': tourney_info['tourney_id'],
                'tourney_name': tourney_info['tourney_name'],
                'match_num': len(self.all_matches) + 1
            })
            
            # Extrair dados específicos da linha
            # Estrutura esperada: Rd | wRk | Winner | lRk | Loser | Score | DR | W:A% | 1stIn | 1st% | 2nd% | BPSvd | L:A% | 1stIn | 1st% | 2nd% | BPSvd | Time
            match_data['round'] = cells[0].get_text(strip=True) if len(cells) > 0 else ''
            match_data['winner_rank'] = cells[1].get_text(strip=True) if len(cells) > 1 else '' 
            match_data['winner_name'] = cells[2].get_text(strip=True) if len(cells) > 2 else ''
            match_data['loser_rank'] = cells[4].get_text(strip=True) if len(cells) > 4 else ''
            match_data['loser_name'] = cells[5].get_text(strip=True) if len(cells) > 5 else ''
            match_data['score'] = cells[6].get_text(strip=True) if len(cells) > 6 else ''
            match_data['DRW'] = cells[7].get_text(strip=True) if len(cells) > 7 else ''
            
            # Estatísticas do vencedor
            if len(cells) > 8:
                match_data['w_ace'] = self._clean_stat(cells[8].get_text(strip=True))
                match_data['w_1stIn'] = self._clean_stat(cells[9].get_text(strip=True))
                match_data['w_1stWon'] = self._clean_stat(cells[10].get_text(strip=True))
                match_data['w_2ndWon'] = self._clean_stat(cells[11].get_text(strip=True))
                match_data['w_bpSaved'] = self._clean_stat(cells[12].get_text(strip=True))
            
            # Estatísticas do perdedor
            if len(cells) > 13:
                match_data['l_ace'] = self._clean_stat(cells[13].get_text(strip=True))
                match_data['l_1stIn'] = self._clean_stat(cells[14].get_text(strip=True))
                match_data['l_1stWon'] = self._clean_stat(cells[15].get_text(strip=True))
                match_data['l_2ndWon'] = self._clean_stat(cells[16].get_text(strip=True))
                match_data['l_bpSaved'] = self._clean_stat(cells[17].get_text(strip=True))
            
            # Tempo de partida
            if len(cells) > 18:
                match_data['minutes'] = self._clean_stat(cells[18].get_text(strip=True))
            
            # Preencher valores padrão para campos adicionais
            self.fill_default_values(match_data, tourney_info)
            
            return match_data
            
        except Exception as e:
            print(f"Erro ao fazer parse da linha: {e}")
            return None
    
    def _clean_stat(self, stat):
        """Limpa valores estatísticos removendo % e convertendo para float"""
        if not stat or stat == '-':
            return ''
        stat = stat.replace('%', '').strip()
        try:
            return float(stat)
        except ValueError:
            return stat
    
    def fill_default_values(self, match_data, tourney_info):
        """Preenche valores padrão para campos obrigatórios"""

        # Dicionário com torneios, datas e superfícies
        tournament_calendar = {
            "Paris": ("20261102", "Hard"),
            #"Basel": ("20261026", "Hard"),
            #"Vienna": ("20261026", "Hard"),
            #"Lyon": ("20261019", "Hard"),
            #"Brussels": ("20261019", "Hard"),
            #"Almaty": ("20261019", "Hard"),
            #"Shanghai-Masters": ("20261007", "Hard"),
            #"Beijing": ("20260930", "Hard"),
            #"Tokyo": ("20260930", "Hard"),
            #"Hangzhou": ("20260923", "Hard"),
            #"Chengdu": ("20260923", "Hard"),
            #"US-Open": ("20260831", "Hard"),
            #"Winston-Salem": ("20260823", "Hard"),
            #"Cincinnati-Masters": ("20260813", "Hard"),
            #"Canada-Masters": ("20260802", "Hard"),
            #"Washington": ("20260727", "Hard"),
            #"Los-Cabos": ("20260727", "Hard"),
            #"Kitzbuhel": ("20260720", "Clay"),
            #"Estoril": ("20260720", "Clay"),
            #"Gstaad": ("20260713", "Clay"),
            #"Bastad": ("20260713", "Clay"),
            #"Umag": ("20260713", "Clay"),
            #"Eastbourne": ("20260622", "Grass"),
            #"Mallorca": ("20260621", "Grass"),
            #"Queen's-Club": ("20260615", "Grass"),
            #"Halle": ("20260615", "Grass"),
            #"Stuttgart": ("20260608", "Grass"),
            #"s-Hertogenbosch": ("20260608", "Grass"),
            #"Geneva": ("20260617", "Clay"),
            #"Hamburg": ("20260517", "Clay"),
            #"Roma-Masters": ("20260506", "Clay"),
            #"Madrid-Masters": ("20260422", "Clay"),
            #"Munich": ("20260413", "Clay"),
            #"Barcelona": ("20260413", "Clay"),
            #"Monte-Carlo-Masters": ("20260405", "Clay"),
            #"Houston": ("20260330", "Clay"),
            #"Marrakech": ("20260330", "Clay"),
            #"Bucharest": ("20260330", "Clay"),
            #"Miami-Masters": ("20260318", "Hard"),
            #"Indian-Wells-Masters": ("20260304", "Hard"),
            #"Dubai": ("20260223", "Hard"),
            #"Acapulco": ("20260223", "Hard"),
            #"Santiago": ("20260223", "Clay"),
            #"Delray-Beach": ("20260216", "Hard"),
            #"Rio-de-Janeiro": ("20260216", "Clay"),
            #"Doha": ("20260216", "Hard"),
            #"Dallas": ("20260209", "Hard"),
            #"Rotterdam": ("20260209", "Hard"),
            #"Buenos-Aires": ("20260209", "Clay"),
            #"Montpellier": ("20260202", "Hard"),
            #"Australian-Open": ("20260119", "Hard"),
            #"Adelaide": ("20260112", "Hard"),
            #"Auckland": ("20260112", "Hard"),
            #"Hong-Kong": ("20260105", "Hard"),
            #"Brisbane": ("20260104", "Hard"),
        }

        # Normalizar nome do torneio para procurar no dicionário
        tourney_name = match_data.get("tourney_name", "")
        date, surface = tournament_calendar.get(
            tourney_name, ("", "Hard")  # default caso não encontre
        )

        tourney_name_lower = tourney_name.lower()
        grand_slam_keywords = [
            "australian-open",
            "roland-garros",
            "french-open",
            "wimbledon",
            "us-open",
            "u.s.-open",
            "usopen",
        ]

        if any(keyword in tourney_name_lower for keyword in grand_slam_keywords):
            tourney_level = "W"
        elif "masters" and "paris" in tourney_name_lower:
            tourney_level = "M"
        else:
            tourney_level = "A"

        # Valores padrão
        defaults = {
            "surface": surface,
            "tourney_date": date,
            "draw_size": 128,
            "tourney_level": tourney_level,
            "best_of": 3,
            "w_df": "",
            "w_svpt": "",
            "w_SvGms": "",
            "w_bpFaced": "",
            "l_df": "",
            "l_svpt": "",
            "l_SvGms": "",
            "l_bpFaced": "",
            "tournament_type": "regular",
        }

        # Preencher campos faltantes
        for key, value in defaults.items():
            if key not in match_data:
                match_data[key] = value

    def scrape_tournament(self, tournament):
        try:
            print(f"🎾 Scraping {tournament['tourney_name']} ...")
            r = self.session.get(tournament['url'], timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            matches = self.extract_match_data(soup, tournament)
            self.all_matches.extend(matches)
            print(f"   ✅ {len(matches)} partidas extraídas")
            time.sleep(2)
        except Exception as e:
            print(f"   ❌ Erro {tournament['tourney_name']}: {e}")
    
    def scrape_all_from_file(self, file_path):
        tournaments = self.get_tournament_list_from_file(file_path)
        print(f"Encontrados {len(tournaments)} torneios no ficheiro")
        for t in tournaments:
            self.scrape_tournament(t)
        return self.all_matches

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.all_matches)
        # garantir todas as colunas no output
        for col in self.columns:
            if col not in df.columns:
                df[col] = ''
        df = df[self.columns]
        df.to_csv(filename, index=False)
        print(f"\n💾 Dados salvos em {filename} ({len(df)} partidas)")
        return df
    
    def save_to_csv(self, filename='tennis_2026_data.csv'):
        """Salva os dados em CSV"""
        if not self.all_matches:
            print("Nenhum dado para salvar")
            return
            
        df = pd.DataFrame(self.all_matches)
        
        # Garantir que todas as colunas estão presentes
        for col in self.columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reordenar colunas
        df = df[self.columns]
        
        # Salvar
        df.to_csv(filename, index=False)
        print(f"Dados salvos em {filename}")
        print(f"Total de partidas: {len(df)}")
        
        return df

# Exemplo de uso
if __name__ == "__main__":
    scraper = TennisAbstractScraper()
    
    # Fazer scraping
    matches = scraper.scrape_all_from_file(LINKS_2026)
    
    # Salvar dados
    df = scraper.save_to_csv('tennis_2026_data_v1.csv')
    
    if df is not None:
        print("\nPrimeiras 5 linhas do dataset:")
        print(df.head())
        
        print(f"\nShape do dataset: {df.shape}")
        print(f"Colunas: {list(df.columns)}")