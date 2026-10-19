import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss, brier_score_loss, mean_absolute_error, r2_score, balanced_accuracy_score
from xgboost import XGBClassifier
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
DATA_PATH = os.getenv('DATA_PATH')

class TennisPredictor:
    def __init__(self):
        """Inicializa todos os atributos necessários"""
        self.model = None
        self.matches = None
        self.player_history = {}  # Histórico de ELO por jogador
        self.h2h_data = {}        # Dados head-to-head
        self.surface_stats = {}   # Estatísticas por superfície
        self.player_profiles = {} # Histórico de ranking/idade por jogador
        self.decision_threshold = 0.5  # Threshold de decisão para escolha binária de vencedor
        self.threshold_profiles = {}  # Perfis de threshold ativos
        self.feature_columns = None  # Colunas usadas no modelo

    def load_data(self, db_path: str | None = None):
        """Carrega os dados das partidas a partir da base de dados SQLite.

        Por defeito usa o ficheiro SQLite `datasets/tennis_data.db` (ver `DB_PATH` env),
        mas pode passar-se um `db_path` alternativo (útil para testes).
        """
        print("Carregando dados de partidas a partir da base de dados SQLite...")
        try:
            from db import get_connection, load_matches_for_training
        except Exception:
            # se a import falhar por caminhos relativos, tentar import local
            from src.db import get_connection, load_matches_for_training

        conn = get_connection(db_path)
        # Carregar e normalizar colunas esperadas pelo pipeline atual
        self.matches = load_matches_for_training(conn)
        conn.close()

        # Verificar colunas essenciais (compatibilidade com o código existente)
        required_cols = ['winner_name', 'loser_name', 'surface', 'tourney_date',
                        'winner_surface_elo', 'loser_surface_elo']
        if not all(col in self.matches.columns for col in required_cols):
            raise ValueError("Dataset na BD não contém todas as colunas necessárias")

        print(f"Dados carregados da BD: {len(self.matches)} partidas")

    def preprocess_data(self):
        """Prepara e processa os dados para treinamento"""
        print("\nPreprocessando dados...")
        
        # Inicializar estruturas de dados
        self.player_history = {}
        self.h2h_data = {}
        self.surface_stats = {}
        self.player_profiles = {}
        self.threshold_profiles = {}
        
        # Filtrar partidas válidas
        valid_matches = self.matches[
            (self.matches['winner_name'].notna()) & 
            (self.matches['loser_name'].notna()) &
            (self.matches['surface'].notna()) &
            (self.matches['winner_surface_elo'].notna()) &
            (self.matches['loser_surface_elo'].notna())
        ].copy()
        
        # Ordenar por data para processamento temporal
        valid_matches = valid_matches.sort_values('tourney_date')

        # Garantir coluna de nível de torneio para engenharia de features.
        if 'tourney_level' not in valid_matches.columns:
            valid_matches['tourney_level'] = 'UNK'
        valid_matches['tourney_level'] = valid_matches['tourney_level'].fillna('UNK').astype(str)
        
        # Construir H2H apenas uma vez antes de extrair features
        self.build_h2h_data(valid_matches)

        rng = np.random.default_rng(42)

        # Processar cada partida
        features = []
        for _, row in valid_matches.iterrows():
            # Processar estatísticas
            self._process_match_stats(row)
            
            # Obter ELOs antes da partida
            winner_elo_pre = self._get_elo_before_match(row['winner_name'], row['surface'], row['tourney_date'])
            loser_elo_pre = self._get_elo_before_match(row['loser_name'], row['surface'], row['tourney_date'])
            
            if pd.isna(winner_elo_pre) or pd.isna(loser_elo_pre):
                continue
            # Uma única linha por partida para evitar leakage entre perspetivas.
            if rng.random() < 0.5:
                features.append(self._extract_features(
                    row, perspective='winner',
                    player_elo=winner_elo_pre,
                    opponent_elo=loser_elo_pre,
                    target=1
                ))
            else:
                features.append(self._extract_features(
                    row, perspective='loser',
                    player_elo=loser_elo_pre,
                    opponent_elo=winner_elo_pre,
                    target=0
                ))
        
        # Criar DataFrame de features
        df = pd.DataFrame([f for f in features if f is not None])
        self.feature_columns = df.drop(['target'], axis=1).columns.tolist()
        
        print(f"\nDistribuição de classes:\n{df['target'].value_counts(normalize=True)}")
        return df

    def _process_match_stats(self, row):
        """Processa estatísticas de uma partida e atualiza as estruturas de dados"""
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        date = row['tourney_date']
        
        # Atualizar histórico de jogadores
        for player, is_winner in [(winner, True), (loser, False)]:
            if player not in self.player_history:
                self.player_history[player] = []
            
            self.player_history[player].append({
                'date': date,
                'surface': surface,
                'elo_after': row['winner_surface_elo'] if is_winner else row['loser_surface_elo'],
                'result': 'win' if is_winner else 'loss'
            })

        # Atualizar perfil (ranking/idade) para previsão e features adicionais
        winner_rank = row.get('winner_rank') if 'winner_rank' in row.index else np.nan
        loser_rank = row.get('loser_rank') if 'loser_rank' in row.index else np.nan
        winner_rank_points = row.get('winner_rank_points') if 'winner_rank_points' in row.index else np.nan
        loser_rank_points = row.get('loser_rank_points') if 'loser_rank_points' in row.index else np.nan
        winner_age = row.get('winner_age') if 'winner_age' in row.index else np.nan
        loser_age = row.get('loser_age') if 'loser_age' in row.index else np.nan

        for player, rank, rank_points, age in [
            (winner, winner_rank, winner_rank_points, winner_age),
            (loser, loser_rank, loser_rank_points, loser_age)
        ]:
            if player not in self.player_profiles:
                self.player_profiles[player] = []
            self.player_profiles[player].append({
                'date': date,
                'rank': rank,
                'rank_points': rank_points,
                'age': age
            })
        # Atualizar estatísticas por superfície
        for player, is_winner in [(winner, True), (loser, False)]:
            if player not in self.surface_stats:
                self.surface_stats[player] = {}
            if surface not in self.surface_stats[player]:
                self.surface_stats[player][surface] = {'wins': 0, 'losses': 0}
            
            if is_winner:
                self.surface_stats[player][surface]['wins'] += 1
            else:
                self.surface_stats[player][surface]['losses'] += 1

    def build_h2h_data(self, matches):
        """Constrói o histórico head-to-head conforme sua estrutura solicitada"""
        print("Construindo histórico H2H...")
        self.h2h_data = {}
        
        for _, row in matches.iterrows():
            winner = row['winner_name']
            loser = row['loser_name']
            date = row['tourney_date']
            
            # Atualizar registro do vencedor
            if winner not in self.h2h_data:
                self.h2h_data[winner] = {}
            if loser not in self.h2h_data[winner]:
                self.h2h_data[winner][loser] = {'w': 0, 'l': 0, 'matches': []}
            self.h2h_data[winner][loser]['w'] += 1
            self.h2h_data[winner][loser]['matches'].append({'date': date, 'result': 'win'})
            
            # Atualizar registro do perdedor
            if loser not in self.h2h_data:
                self.h2h_data[loser] = {}
            if winner not in self.h2h_data[loser]:
                self.h2h_data[loser][winner] = {'w': 0, 'l': 0, 'matches': []}
            self.h2h_data[loser][winner]['l'] += 1
            self.h2h_data[loser][winner]['matches'].append({'date': date, 'result': 'loss'})

    def _format_h2h(self, player1, player2, h2h):
        """Formata o H2H para exibição"""
        total = h2h['w'] + h2h['l']
        if total == 0:
            return "Igual (sem confrontos anteriores)"
        
        percentage = h2h['w'] / total
        return f"{h2h['w']}-{h2h['l']} ({percentage:.0%} para {player1})"

    def _get_elo_before_match(self, player, surface, date):
        """Obtém o ELO do jogador antes da partida atual"""
        # Para a primeira partida do jogador, usar valor inicial (1500 ou outro)
        if player not in self.player_history:
            return 1500
        
        # Obter o ELO mais recente antes da data atual
        player_matches = self.player_history[player]
        surface_matches = [m for m in player_matches if m['surface'] == surface and m['date'] < date]
        
        if not surface_matches:
            return 1500  # Valor inicial se não houver partidas anteriores nessa superfície
        
        # Retornar o ELO após a última partida (que será o ELO antes da atual)
        return sorted(surface_matches, key=lambda x: x['date'])[-1]['elo_after']

    def _update_player_history(self, row):
        """Atualiza o histórico com os ELOs após esta partida"""
        winner = row['winner_name']
        loser = row['loser_name']
        date = row['tourney_date']
        surface = row['surface']
        
        # Atualizar vencedor
        if winner not in self.player_history:
            self.player_history[winner] = []
        self.player_history[winner].append({
            'date': date,
            'surface': surface,
            'elo_before': self._get_elo_before_match(winner, surface, date),
            'elo_after': row['winner_surface_elo'],
            'opponent': loser,
            'result': 'win'
        })
        
        # Atualizar perdedor
        if loser not in self.player_history:
            self.player_history[loser] = []
        self.player_history[loser].append({
            'date': date,
            'surface': surface,
            'elo_before': self._get_elo_before_match(loser, surface, date),
            'elo_after': row['loser_surface_elo'],
            'opponent': winner,
            'result': 'loss'
        })

    def _extract_features(self, row, perspective, player_elo, opponent_elo, target):
        """Extrai features para uma partida com todos os argumentos necessários"""
        if perspective == 'winner':
            player = row['winner_name']
            opponent = row['loser_name']
            player_rank_raw = row.get('winner_rank')
            opponent_rank_raw = row.get('loser_rank')
            player_rank_points_raw = row.get('winner_rank_points')
            opponent_rank_points_raw = row.get('loser_rank_points')
            player_age_raw = row.get('winner_age')
            opponent_age_raw = row.get('loser_age')
        else:
            player = row['loser_name']
            opponent = row['winner_name']
            player_rank_raw = row.get('loser_rank')
            opponent_rank_raw = row.get('winner_rank')
            player_rank_points_raw = row.get('loser_rank_points')
            opponent_rank_points_raw = row.get('winner_rank_points')
            player_age_raw = row.get('loser_age')
            opponent_age_raw = row.get('winner_age')
        
        surface = row['surface']
        tourney_level = row.get('tourney_level', 'UNK')
        date = row['tourney_date']
        
        # Head-to-head até antes desta partida
        h2h = self._get_h2h_stats_before_match(player, opponent, date)
        h2h_total = h2h['w'] + h2h['l']
        h2h_win_rate = h2h['w'] / h2h_total if h2h_total > 0 else 0.5

        player_rank, opponent_rank = self._coalesce_pair(player_rank_raw, opponent_rank_raw)
        player_rank_points, opponent_rank_points = self._coalesce_pair(
            player_rank_points_raw,
            opponent_rank_points_raw
        )
        player_age, opponent_age = self._coalesce_pair(player_age_raw, opponent_age_raw)
        
        # Estatísticas por superfície até antes desta partida
        player_stats = self._get_surface_stats_before(player, surface, date)
        opponent_stats = self._get_surface_stats_before(opponent, surface, date)
        recent_win_diffs = {
            f'win_last_{window}_diff': self._get_recent_wins(player, date, window)
            - self._get_recent_wins(opponent, date, window)
            for window in (3, 5, 10, 25, 50)
        }
        
        return {
            'date': date,
            'player_elo': player_elo,
            'opponent_elo': opponent_elo,
            'elo_diff': player_elo - opponent_elo,
            'player_rank': player_rank,
            'opponent_rank': opponent_rank,
            'rank_diff': self._pairwise_diff(player_rank, opponent_rank),
            'rank_advantage': opponent_rank - player_rank,
            'player_rank_points': player_rank_points,
            'opponent_rank_points': opponent_rank_points,
            'rank_points_diff': self._pairwise_diff(player_rank_points, opponent_rank_points),
            'rank_points_advantage': player_rank_points - opponent_rank_points,
            'player_age': player_age,
            'opponent_age': opponent_age,
            'age_diff': player_age - opponent_age,
            'h2h_win_rate': h2h_win_rate,
            'h2h_diff': 2 * h2h_win_rate - 1,
            'h2h_matches': h2h_total,
            'player_surface_win_rate': player_stats['win_rate'],
            'player_surface_matches': player_stats['total_matches'],
            'opponent_surface_win_rate': opponent_stats['win_rate'],
            'opponent_surface_matches': opponent_stats['total_matches'],
            'surface_win_rate_diff': self._pairwise_diff(
                player_stats['win_rate'], opponent_stats['win_rate']
            ),
            'surface_matches_diff': self._pairwise_diff(
                player_stats['total_matches'], opponent_stats['total_matches']
            ),
            **recent_win_diffs,
            'surface': surface,
            'tourney_level': str(tourney_level) if pd.notna(tourney_level) else 'UNK',
            'target': target
        }

    def _get_h2h_stats_before_match(self, player1, player2, date):
        """Calcula H2H (vitórias e derrotas) até antes de uma data específica"""
        if player1 not in self.h2h_data or player2 not in self.h2h_data[player1]:
            return {'w': 0, 'l': 0}
    
        # Se não houver data especificada, retorna todo o histórico
        if date is None:
            return {
                'w': self.h2h_data[player1][player2]['w'],
                'l': self.h2h_data[player1][player2]['l']
            }
        
        # Filtra partidas anteriores à data especificada
        matches = [m for m in self.h2h_data[player1][player2]['matches'] if m['date'] < date]
        wins = sum(1 for m in matches if m['result'] == 'win')
        losses = sum(1 for m in matches if m['result'] == 'loss')
        
        return {'w': wins, 'l': losses}

    def _get_surface_stats_before(self, player, surface, date):
        """Calcula estatísticas por superfície até antes de uma data"""
        wins = losses = 0
        
        if player not in self.player_history:
            return {'win_rate': 0.5, 'total_matches': 0}
            
        for match in self.player_history[player]:
            if match['date'] < date and match['surface'] == surface:
                if match['result'] == 'win':
                    wins += 1
                else:
                    losses += 1
                    
        total = wins + losses
        return {
            'win_rate': wins / total if total > 0 else 0.5,
            'total_matches': total
        }

    def _get_recent_wins(self, player, date, window):
        """Conta vitórias nos últimos jogos do jogador antes da partida."""
        if player not in self.player_history:
            return 0

        previous_matches = [
            match for match in self.player_history[player]
            if match['date'] < date
        ]
        return sum(match['result'] == 'win' for match in previous_matches[-window:])
    
    def _load_historical_data(self):
        """Carrega apenas os dados necessários para previsões"""
        try:
            data = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_data_xgboost.pkl'))
            self.player_history = data.get('player_history', {})
            self.h2h_data = data.get('h2h_data', {})
            self.surface_stats = data.get('surface_stats', {})
            self.player_profiles = data.get('player_profiles', {})
        except Exception as e:
            print(f"Erro ao carregar dados históricos: {str(e)}")
            self.player_history = {}
            self.h2h_data = {}
            self.surface_stats = {}
            self.player_profiles = {}

    def _coalesce_pair(self, player_value, opponent_value):
        """Preserva NaNs para aproveitar o tratamento nativo do XGBoost."""
        player_clean = np.nan if pd.isna(player_value) else float(player_value)
        opponent_clean = np.nan if pd.isna(opponent_value) else float(opponent_value)
        return player_clean, opponent_clean

    def _pairwise_diff(self, player_value, opponent_value):
        """Calcula uma diferença sem transformar NaNs em valores artificiais."""
        if pd.isna(player_value) or pd.isna(opponent_value):
            return np.nan
        return float(player_value) - float(opponent_value)

    def _get_player_profile_value_before(self, player, field, date, default):
        """Obtém o valor mais recente de um atributo do jogador antes de uma data."""
        if player not in self.player_profiles:
            return np.nan if default is None else float(default)

        valid_values = [
            entry.get(field)
            for entry in self.player_profiles[player]
            if entry.get('date') is not None and entry.get('date') < date and pd.notna(entry.get(field))
        ]
        if not valid_values:
            return np.nan if default is None else float(default)
        return float(valid_values[-1])

    def _get_latest_match_date(self, player1, player2):
        """Obtém a data mais recente em que qualquer um dos jogadores participou"""
        dates = []
        for player in [player1, player2]:
            if player in self.player_history:
                dates.extend([match['date'] for match in self.player_history[player] if match['date']])
        return max(dates) if dates else None

    def train_model(self, df):
        """Treina modelo XGBoost com validação temporal e busca de hiperparâmetros com Optuna"""
        
        # Ordenar por data
        df = df.sort_values('date').reset_index(drop=True)

        # Split temporal global:
        # - 85% inicial para desenvolvimento (train + validação cruzada temporal)
        # - 15% final como teste totalmente holdout
        n = len(df)
        dev_end = int(0.85 * n)
        dev = df.iloc[:dev_end].reset_index(drop=True)
        test = df.iloc[dev_end:].reset_index(drop=True)

        # Features e target
        cols_to_drop = ['date', 'target']
        for col in ['player', 'opponent']:
            if col in df.columns:
                cols_to_drop.append(col)

        def prepare_Xy(dataset, expected_columns=None):
            X = dataset.drop(cols_to_drop, axis=1)
            categorical_cols = [c for c in ['surface', 'tourney_level'] if c in X.columns]
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols)
            if expected_columns is not None:
                X = X.reindex(columns=expected_columns, fill_value=0)
            y = dataset['target']
            return X, y

        X_dev_raw, y_dev = prepare_Xy(dev)
        self.feature_columns = X_dev_raw.columns.tolist()
        X_dev = X_dev_raw.reindex(columns=self.feature_columns, fill_value=0)

        # Busca de hiperparâmetros com validação cruzada temporal
        optuna_trials = int(os.getenv('OPTUNA_TRIALS', '10'))
        print(f"\nIniciando busca de hiperparâmetros com Optuna + TimeSeriesSplit ({optuna_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 1500,
            'scale_pos_weight': 1,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }

        def objective(trial):
            trial_params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.4),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 2)
            }

            tscv = TimeSeriesSplit(n_splits=4)
            fold_losses = []

            for fold_train_idx, fold_val_idx in tscv.split(X_dev):
                X_train_fold = X_dev.iloc[fold_train_idx]
                y_train_fold = y_dev.iloc[fold_train_idx]
                X_val_fold = X_dev.iloc[fold_val_idx]
                y_val_fold = y_dev.iloc[fold_val_idx]

                model = XGBClassifier(**base_params, **trial_params)
                try:
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except TypeError:
                    model.set_params(early_stopping_rounds=50)
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False
                    )

                y_val_proba = model.predict_proba(X_val_fold)[:, 1]
                fold_losses.append(log_loss(y_val_fold, y_val_proba))

            return float(np.mean(fold_losses))

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        try:
            study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        except KeyboardInterrupt:
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                raise
            print(f"\nBusca interrompida pelo utilizador. A usar melhor trial entre {len(completed)} trials concluídos.")

        params = {**base_params, **study.best_params}
        print("Melhores parâmetros encontrados:", study.best_params)
        print(f"Melhor logloss médio de validação temporal (Optuna): {study.best_value:.5f}")

        # Split final dentro do bloco de desenvolvimento para early stopping
        final_train_end = int(0.9 * len(dev))
        final_train = dev.iloc[:final_train_end].reset_index(drop=True)
        final_val = dev.iloc[final_train_end:].reset_index(drop=True)

        X_train_final, y_train_final = prepare_Xy(final_train, expected_columns=self.feature_columns)
        X_val_final, y_val_final = prepare_Xy(final_val, expected_columns=self.feature_columns)

        self.model = XGBClassifier(**params)
        try:
            self.model.fit(X_train_final, y_train_final,
                           eval_set=[(X_val_final, y_val_final)],
                           early_stopping_rounds=50,
                            verbose=50)
        except TypeError:
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(X_train_final, y_train_final,
                           eval_set=[(X_val_final, y_val_final)],
                            verbose=50)
                        
        best_iteration = getattr(self.model, 'best_iteration', None)
        if best_iteration is not None and best_iteration >= 0:
            self.model.set_params(n_estimators=best_iteration + 1)
            print(f"\nMelhor número de estimadores após early stopping: {best_iteration + 1}")
        else:
            print("\nEarly stopping não retornou best_iteration; mantendo n_estimators atual.")

        # Mantém threshold binário fixo para decisão de vencedor,
        # enquanto a otimização do modelo permanece orientada a LogLoss.
        val_proba = self.model.predict_proba(X_val_final)[:, 1]
        threshold_strategy = os.getenv('THRESHOLD_STRATEGY', 'default_050').strip().lower()
        self.threshold_profiles = {
            'default_050': {
                'threshold': 0.5,
                'val_logloss': float(log_loss(y_val_final, val_proba)),
                'val_brier': float(brier_score_loss(y_val_final, val_proba))
            }
        }
        if threshold_strategy not in self.threshold_profiles:
            threshold_strategy = 'default_050'
        self.decision_threshold = self.threshold_profiles[threshold_strategy]['threshold']

        print("\nPerfil de threshold na validação final:")
        print(
            f"- default_050: thr={self.threshold_profiles['default_050']['threshold']:.2f} | "
            f"LogLoss={self.threshold_profiles['default_050']['val_logloss']:.4f} | "
            f"Brier={self.threshold_profiles['default_050']['val_brier']:.4f}"
        )
        print(
            f"Threshold ativo para produção ({threshold_strategy}): "
            f"{self.decision_threshold:.2f}"
        )

        # Avaliação no conjunto de teste
        X_test, y_test = prepare_Xy(test, expected_columns=self.feature_columns)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.decision_threshold).astype(int)
        y_pred_050 = (y_proba >= 0.50).astype(int)
        f1_at_050 = f1_score(y_test, y_pred_050)

        print("\n--- Avaliação Teste (dados mais recentes) ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
        print(f"F1-score @0.50: {f1_at_050:.3f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
        print(f"LogLoss: {log_loss(y_test, y_proba):.3f}")
        print(f"Brier Score: {brier_score_loss(y_test, y_proba):.3f}")
        print("Relatório de classificação:")
        print(classification_report(y_test, y_pred))

        print("\nComparação rápida no teste por perfil de threshold:")
        for profile_name, cfg in self.threshold_profiles.items():
            thr = cfg['threshold']
            y_pred_profile = (y_proba >= thr).astype(int)
            print(
                f"- {profile_name}: thr={thr:.2f} | "
                f"F1={f1_score(y_test, y_pred_profile):.3f} | "
                f"Macro-F1={f1_score(y_test, y_pred_profile, average='macro'):.3f} | "
                f"BalAcc={balanced_accuracy_score(y_test, y_pred_profile):.3f}"
            )

        # Importância das features
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))
    
    def predict_match(self, player1, player2, surface, date=None, tourney_level='UNK'):
        """
        Faz uma previsão para uma partida específica em uma data específica
        """
        # Verificar data
        if date is not None and not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except Exception as e:
                print(f"Erro ao converter data: {e}. Usando data mais recente.")
                date = None

        if date is None:
            date = self.matches['tourney_date'].max()

        # Verificar jogadores
        if player1 not in self.player_history or player2 not in self.player_history:
            print("Um ou ambos os jogadores não foram encontrados nos dados históricos.")
            return None, None, None

        # Obter ELOs
        player1_elo = self._get_elo_before_match(player1, surface, date)
        player2_elo = self._get_elo_before_match(player2, surface, date)

        # Obter H2H
        h2h = self._get_h2h_stats_before_match(player1, player2, date)
        h2h_display = self._format_h2h(player1, player2, h2h)
        h2h_total = h2h['w'] + h2h['l']
        h2h_win_rate = h2h['w'] / h2h_total if h2h_total > 0 else 0.5

        # Stats por superfície
        player1_stats = self._get_surface_stats_before(player1, surface, date)
        player2_stats = self._get_surface_stats_before(player2, surface, date)
        recent_win_diffs = {
            f'win_last_{window}_diff': self._get_recent_wins(player1, date, window)
            - self._get_recent_wins(player2, date, window)
            for window in (3, 5, 10, 25, 50)
        }

        # Ranking e idade (último valor conhecido antes da data)
        player1_rank = self._get_player_profile_value_before(player1, 'rank', date, default=None)
        player2_rank = self._get_player_profile_value_before(player2, 'rank', date, default=None)
        player1_rank_points = self._get_player_profile_value_before(player1, 'rank_points', date, default=None)
        player2_rank_points = self._get_player_profile_value_before(player2, 'rank_points', date, default=None)
        player1_age = self._get_player_profile_value_before(player1, 'age', date, default=None)
        player2_age = self._get_player_profile_value_before(player2, 'age', date, default=None)

        # Features
        features = {
            'elo_diff': player1_elo - player2_elo,
            'player_elo': player1_elo,
            'opponent_elo': player2_elo,
            'player_rank': player1_rank,
            'opponent_rank': player2_rank,
            'rank_diff': self._pairwise_diff(player1_rank, player2_rank),
            'rank_advantage': player2_rank - player1_rank,
            'player_rank_points': player1_rank_points,
            'opponent_rank_points': player2_rank_points,
            'rank_points_diff': self._pairwise_diff(player1_rank_points, player2_rank_points),
            'rank_points_advantage': player1_rank_points - player2_rank_points,
            'player_age': player1_age,
            'opponent_age': player2_age,
            'age_diff': player1_age - player2_age,
            'h2h_win_rate': h2h_win_rate,
            'h2h_diff': 2 * h2h_win_rate - 1,
            'h2h_matches': h2h_total,
            'player_surface_win_rate': player1_stats['win_rate'],
            'player_surface_matches': player1_stats['total_matches'],
            'opponent_surface_win_rate': player2_stats['win_rate'],
            'opponent_surface_matches': player2_stats['total_matches'],
            'surface_win_rate_diff': self._pairwise_diff(
                player1_stats['win_rate'], player2_stats['win_rate']
            ),
            'surface_matches_diff': self._pairwise_diff(
                player1_stats['total_matches'], player2_stats['total_matches']
            ),
            **recent_win_diffs,
            'surface': surface,
            'tourney_level': str(tourney_level) if tourney_level is not None else 'UNK'
        }

        # Criar DataFrame
        df = pd.DataFrame([features])
        df = pd.get_dummies(df, columns=['surface'])

        # Garantir todas as colunas usadas no treino
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Prever probabilidade
        proba = self.model.predict_proba(df)[0]
        prob_player1 = proba[1]  # classe "1" significa player1 venceu

        winner = player1 if prob_player1 >= self.decision_threshold else player2
        confidence = max(prob_player1, 1 - prob_player1)

        # Detalhes
        details = {
            'date': date,
            'surface': surface,
            'player1_elo': player1_elo,
            'player2_elo': player2_elo,
            'elo_diff': features['elo_diff'],
            'h2h': h2h_display,
            'h2h_raw': h2h,
            'player1_rank': player1_rank,
            'player2_rank': player2_rank,
            'rank_advantage': player2_rank - player1_rank,
            'player1_rank_points': player1_rank_points,
            'player2_rank_points': player2_rank_points,
            'rank_points_advantage': player1_rank_points - player2_rank_points,
            'player1_age': player1_age,
            'player2_age': player2_age,
            'age_diff': player1_age - player2_age,
            'decision_threshold': self.decision_threshold,
            'player1_surface_win_rate': player1_stats['win_rate'],
            'player2_surface_win_rate': player2_stats['win_rate'],
            'probability': confidence
        }

        return winner, confidence, details

    
    def save_model(self):
        """Salva o modelo e dados necessários para previsão futura"""
        print("\nSalvando modelo e dados...")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        
        # Salvar componentes do modelo
        joblib.dump(self.model, os.path.join(MODEL_PATH, 'tennis_surface_elo_model_xgboost.pkl'))
        
        # Salvar dados necessários para previsões
        joblib.dump({
            'player_history': self.player_history,
            'h2h_data': self.h2h_data,
            'surface_stats': self.surface_stats,
            'player_profiles': self.player_profiles,
            'decision_threshold': self.decision_threshold,
            'threshold_profiles': self.threshold_profiles,
            'feature_columns': self.feature_columns  # Adicionado para garantir consistência
        }, os.path.join(MODEL_PATH, 'tennis_surface_elo_data_xgboost.pkl'))
        
        print(f"Modelo e dados salvos em {MODEL_PATH}")

    def load_saved_model(self):
        """Carrega um modelo treinado anteriormente"""
        print("Carregando modelo salvo...")
        try:
            self.model = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_model_xgboost.pkl'))
            
            data = joblib.load(os.path.join(MODEL_PATH, 'tennis_surface_elo_data_xgboost.pkl'))
            self.player_history = data.get('player_history', {})
            self.h2h_data = data.get('h2h_data', {})
            self.surface_stats = data.get('surface_stats', {})
            self.player_profiles = data.get('player_profiles', {})
            self.decision_threshold = data.get('decision_threshold', 0.5)
            self.threshold_profiles = data.get('threshold_profiles', {})
            self.feature_columns = data.get('feature_columns')
            
            print("Modelo e dados carregados com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo salvo: {str(e)}")
            return False

    def _get_h2h_win_rate(self, player1, player2):
        """Calcula a taxa de vitória no head-to-head"""
        stats = self.h2h_data.get(player1, {}).get(player2, {'wins': 0, 'losses': 0})
        total = stats['wins'] + stats['losses']
        return stats['wins'] / total if total > 0 else 0.5
    
    def _get_h2h_matches(self, player1, player2):
        """Obtém o número total de partidas head-to-head"""
        stats = self.h2h_data.get(player1, {}).get(player2, {'wins': 0, 'losses': 0})
        return stats['wins'] + stats['losses']
    
    def _get_surface_win_rate(self, player, surface):
        """Obtém a taxa de vitória do jogador em uma superfície"""
        return self.surface_stats.get(player, {}).get(surface, {}).get('win_rate', 0.5)
    
    def _get_surface_matches(self, player, surface):
        """Obtém o número total de partidas do jogador em uma superfície"""
        return self.surface_stats.get(player, {}).get(surface, {}).get('total_matches', 0)

def main():
    predictor = TennisPredictor()
    
    # Verificar se existe modelo treinado
    if all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in [
        'tennis_surface_elo_model_xgboost.pkl',
        'tennis_surface_elo_data_xgboost.pkl'
    ]):
        print("Modelo treinado encontrado. Carregando...")
        if not predictor.load_saved_model():
            print("Falha ao carregar modelo. Treinando novo modelo...")
            predictor.load_data()
            df = predictor.preprocess_data()
            predictor.train_model(df)
            predictor.save_model()
    else:
        print("Nenhum modelo treinado encontrado. Treinando novo modelo...")
        predictor.load_data()
        df = predictor.preprocess_data()
        predictor.train_model(df)
        predictor.save_model()
    
    #predictor.main_menu()

if __name__ == '__main__':
    main()