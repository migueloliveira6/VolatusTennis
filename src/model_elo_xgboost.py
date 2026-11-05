import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss, brier_score_loss, balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit

try:
    from match_policy import add_policy_columns, sort_matches
    from features import FEATURE_GROUPS, FeatureBuilder
    from evaluation import (ATP_LEVELS, apply_calibration, baseline_metrics, expected_calibration_error,
                            fit_calibrator, probability_metrics, reliability_table, select_calibration_method)
except ImportError:
    from src.match_policy import add_policy_columns, sort_matches
    from src.features import FEATURE_GROUPS, FeatureBuilder
    from src.evaluation import (ATP_LEVELS, apply_calibration, baseline_metrics, expected_calibration_error,
                                fit_calibrator, probability_metrics, reliability_table, select_calibration_method)

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH', 'models')

MODEL_FILENAME = 'tennis_surface_elo_model_xgboost.pkl'
DATA_FILENAME = 'tennis_surface_elo_data_xgboost.pkl'
STATE_VERSION = 2  # v2: estado incremental (features.FeatureBuilder) em vez do histórico completo

# Colunas categóricas codificadas com one-hot, iguais no treino e na previsão
CATEGORICAL_COLUMNS = ['surface', 'tourney_level']
# Colunas do DataFrame de features que não entram no modelo
NON_FEATURE_COLUMNS = ['date', 'target', 'match_idx']
QUALIFYING_ROUNDS = ('Q1', 'Q2', 'Q3', 'Q4')


def features_for_groups(groups):
    """Lista de colunas de features pertencentes aos grupos indicados (ver features.FEATURE_GROUPS)."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Grupos de features desconhecidos: {sorted(unknown)}")
    return [col for group in groups for col in FEATURE_GROUPS[group]]


class TennisPredictor:
    def __init__(self):
        """Inicializa todos os atributos necessários"""
        self.model = None
        self.matches = None
        self.feature_builder = FeatureBuilder()  # Estado dos jogadores (ELO, forma, stats, H2H...)
        self.swap_features = None     # Features na perspetiva trocada (avaliação simétrica no treino)
        self.decision_threshold = 0.5  # Threshold de decisão para escolha binária de vencedor
        self.threshold_profiles = {}  # Perfis de threshold ativos
        self.feature_columns = None  # Colunas usadas no modelo
        self.excluded_features = []  # Features retiradas (ablação)
        self.last_match_date = None  # Data do jogo mais recente visto no treino
        self.training_metrics = {}   # Métricas do último treino
        self.calibrator = None       # Calibrador de probabilidades (ver evaluation.py)

    def load_data(self, db_path: str | None = None, start_date: str | None = None, end_date: str | None = None):
        """Carrega os dados das partidas a partir da base de dados SQLite.

        Por defeito usa o ficheiro SQLite `datasets/tennis_data.db` (ver `DB_PATH` env),
        mas pode passar-se um `db_path` alternativo (útil para testes).
        `start_date` (inclusivo) e `end_date` (exclusivo) filtram por `tourney_date` (YYYY-MM-DD).
        """
        print("Carregando dados de partidas a partir da base de dados SQLite...")
        try:
            from db import get_connection, load_matches_for_training
        except Exception:
            # se a import falhar por caminhos relativos, tentar import local
            from src.db import get_connection, load_matches_for_training

        conn = get_connection(db_path)
        self.matches = load_matches_for_training(conn, start_date=start_date, end_date=end_date)
        conn.close()

        required_cols = ['winner_name', 'loser_name', 'surface', 'tourney_date']
        if not all(col in self.matches.columns for col in required_cols):
            raise ValueError("Dataset na BD não contém todas as colunas necessárias")

        print(f"Dados carregados da BD: {len(self.matches)} partidas")

    def preprocess_data(self):
        """Uma passagem cronológica: features com o estado antes de cada jogo, depois atualiza o estado.

        Devolve um DataFrame com uma linha por jogo (perspetiva aleatória) e guarda em
        `self.swap_features` a perspetiva trocada de cada jogo, usada na avaliação simétrica.
        """
        print("\nPreprocessando dados...")
        matches = self.matches[
            self.matches['winner_name'].notna()
            & self.matches['loser_name'].notna()
            & self.matches['surface'].notna()
        ]

        # Política de jogos especiais: W/O e exibições ficam fora do histórico;
        # abandonos entram no histórico mas não são exemplos de treino.
        matches = add_policy_columns(matches)
        excluded = int((~matches['counts_for_history']).sum())
        matches = matches[matches['counts_for_history']]
        print(f"Jogos fora do histórico (W/O, exibições): {excluded} | "
              f"fora do treino (abandonos): {int((~matches['training_target']).sum())}")

        # Ordem cronológica: data do torneio, torneio, ronda, match_num
        matches = sort_matches(matches)
        if 'tourney_level' not in matches.columns:
            matches['tourney_level'] = 'UNK'
        matches['tourney_level'] = matches['tourney_level'].fillna('UNK').astype(str)

        builder = FeatureBuilder()
        rng = np.random.default_rng(42)
        main_rows, swap_rows = [], []

        for match_idx, row in enumerate(matches.to_dict(orient='records')):
            if row['training_target']:
                context = {
                    'surface': row['surface'],
                    'tourney_level': row['tourney_level'],
                    'date': row['tourney_date'],
                    'best_of': row.get('best_of', 3),
                    'is_qualifying': int(str(row.get('round', '')).upper() in QUALIFYING_ROUNDS),
                }
                winner_view = builder.features(
                    row['winner_name'], row['loser_name'], **context,
                    player_rank=row.get('winner_rank'), opponent_rank=row.get('loser_rank'),
                    player_rank_points=row.get('winner_rank_points'), opponent_rank_points=row.get('loser_rank_points'),
                    player_age=row.get('winner_age'), opponent_age=row.get('loser_age'),
                )
                loser_view = builder.features(
                    row['loser_name'], row['winner_name'], **context,
                    player_rank=row.get('loser_rank'), opponent_rank=row.get('winner_rank'),
                    player_rank_points=row.get('loser_rank_points'), opponent_rank_points=row.get('winner_rank_points'),
                    player_age=row.get('loser_age'), opponent_age=row.get('winner_age'),
                )
                # Uma linha de treino por jogo (perspetiva aleatória); a outra só para avaliação simétrica
                if rng.random() < 0.5:
                    main, swap, target = winner_view, loser_view, 1
                else:
                    main, swap, target = loser_view, winner_view, 0
                main_rows.append({'date': row['tourney_date'], 'match_idx': match_idx, **main, 'target': target})
                swap_rows.append({'date': row['tourney_date'], 'match_idx': match_idx, **swap, 'target': 1 - target})
            builder.update(row)

        self.feature_builder = builder
        df = pd.DataFrame(main_rows)
        self.swap_features = pd.DataFrame(swap_rows).set_index('match_idx', drop=False)
        print(f"\nDistribuição de classes:\n{df['target'].value_counts(normalize=True)}")
        return df

    def _format_h2h(self, player1, player2, h2h):
        """Formata o H2H para exibição"""
        total = h2h['w'] + h2h['l']
        if total == 0:
            return "Igual (sem confrontos anteriores)"

        percentage = h2h['w'] / total
        return f"{h2h['w']}-{h2h['l']} ({percentage:.0%} para {player1})"

    def _to_model_matrix(self, df, expected_columns=None):
        """Converte o DataFrame de features na matriz do modelo (one-hot + alinhamento de colunas)."""
        drop = [c for c in NON_FEATURE_COLUMNS + list(self.excluded_features) if c in df.columns]
        X = df.drop(columns=drop)
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols)
        if expected_columns is not None:
            X = X.reindex(columns=expected_columns, fill_value=0)
        return X.astype(float)

    def _symmetric_proba(self, model, block):
        """P(jogador da linha ganha) = média de p(A,B) e 1 − p(B,A), usando a perspetiva trocada."""
        p_main = model.predict_proba(self._to_model_matrix(block, self.feature_columns))[:, 1]
        if self.swap_features is None:
            return p_main
        swap = self.swap_features.loc[block['match_idx'].to_numpy()]
        p_swap = model.predict_proba(self._to_model_matrix(swap, self.feature_columns))[:, 1]
        return (p_main + 1 - p_swap) / 2

    def train_model(self, df, refit_full: bool = True, exclude_features=None, fixed_params=None):
        """Treina modelo XGBoost com validação temporal e busca de hiperparâmetros com Optuna.

        Avalia num holdout temporal (15% mais recentes) com previsões simétricas e calibradas e,
        com `refit_full`, volta a treinar o modelo de produção com todos os dados.
        `exclude_features` retira colunas (ablação); `fixed_params` salta o Optuna e usa esses
        hiperparâmetros. Devolve um dicionário de métricas do holdout.
        """
        self.excluded_features = list(exclude_features or [])

        # Ordenar por data (estável: mantém a ordem cronológica do preprocessamento)
        df = df.sort_values('date', kind='mergesort').reset_index(drop=True)
        self.last_match_date = df['date'].max()

        # Split temporal global:
        # - 85% inicial para desenvolvimento (train + validação cruzada temporal)
        # - 15% final como teste totalmente holdout
        n = len(df)
        dev_end = int(0.85 * n)
        dev = df.iloc[:dev_end].reset_index(drop=True)
        test = df.iloc[dev_end:].reset_index(drop=True)

        # Features e target
        def prepare_Xy(dataset, expected_columns=None):
            return self._to_model_matrix(dataset, expected_columns), dataset['target']

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

        def fit_with_early_stopping(model, X_train, y_train, X_val, y_val, verbose=False):
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=verbose)
            except TypeError:
                model.set_params(early_stopping_rounds=50)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
            return model

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

            fold_losses = []
            for fold_train_idx, fold_val_idx in TimeSeriesSplit(n_splits=4).split(X_dev):
                model = fit_with_early_stopping(
                    XGBClassifier(**base_params, **trial_params),
                    X_dev.iloc[fold_train_idx], y_dev.iloc[fold_train_idx],
                    X_dev.iloc[fold_val_idx], y_dev.iloc[fold_val_idx],
                )
                y_val_proba = model.predict_proba(X_dev.iloc[fold_val_idx])[:, 1]
                fold_losses.append(log_loss(y_dev.iloc[fold_val_idx], y_val_proba))
            return float(np.mean(fold_losses))

        if fixed_params:
            best_params = dict(fixed_params)
            print("Hiperparâmetros fixos (sem Optuna):", best_params)
        else:
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
            best_params = study.best_params
            print("Melhores parâmetros encontrados:", best_params)
            print(f"Melhor logloss médio de validação temporal (Optuna): {study.best_value:.5f}")

        params = {**base_params, **best_params}

        # Split final dentro do bloco de desenvolvimento (por ordem temporal):
        # 80% treino | 10% validação (early stopping) | 10% calibração das probabilidades
        final_train_end = int(0.8 * len(dev))
        final_val_end = int(0.9 * len(dev))
        final_train = dev.iloc[:final_train_end].reset_index(drop=True)
        final_val = dev.iloc[final_train_end:final_val_end].reset_index(drop=True)
        calib = dev.iloc[final_val_end:].reset_index(drop=True)

        X_train_final, y_train_final = prepare_Xy(final_train, expected_columns=self.feature_columns)
        X_val_final, y_val_final = prepare_Xy(final_val, expected_columns=self.feature_columns)

        self.model = fit_with_early_stopping(
            XGBClassifier(**params), X_train_final, y_train_final, X_val_final, y_val_final, verbose=50
        )
        best_iteration = getattr(self.model, 'best_iteration', None)
        if best_iteration is not None and best_iteration >= 0:
            self.model.set_params(n_estimators=best_iteration + 1)
            print(f"\nMelhor número de estimadores após early stopping: {best_iteration + 1}")
        else:
            print("\nEarly stopping não retornou best_iteration; mantendo n_estimators atual.")

        # Threshold binário fixo; o modelo é otimizado para LogLoss.
        val_proba = self._symmetric_proba(self.model, final_val)
        self.threshold_profiles = {
            'default_050': {
                'threshold': 0.5,
                'val_logloss': float(log_loss(y_val_final, val_proba)),
                'val_brier': float(brier_score_loss(y_val_final, val_proba))
            }
        }
        self.decision_threshold = 0.5

        # Calibração: escolhe o método na 2.ª metade do bloco de calibração (ajustado na 1.ª metade)
        # e ajusta-o no bloco inteiro. O teste não entra na escolha.
        y_calib = calib['target']
        p_calib = self._symmetric_proba(self.model, calib)
        half = len(calib) // 2
        calibration_method, calibration_selection = select_calibration_method(
            p_calib[:half], y_calib.iloc[:half], p_calib[half:], y_calib.iloc[half:]
        )
        eval_calibrator = fit_calibrator(calibration_method, p_calib, y_calib)
        print(f"\nCalibração escolhida: {calibration_method} "
              f"(logloss no bloco de seleção: {calibration_selection[calibration_method]['logloss']:.4f})")

        # Avaliação no conjunto de teste (previsões simétricas)
        y_test = test['target']
        y_proba_single = self.model.predict_proba(self._to_model_matrix(test, self.feature_columns))[:, 1]
        y_proba_raw = self._symmetric_proba(self.model, test)
        y_proba = apply_calibration(eval_calibrator, y_proba_raw)
        y_pred = (y_proba >= self.decision_threshold).astype(int)

        # Subconjunto ATP (o que o site prevê) e baselines no mesmo conjunto
        atp_mask = test['tourney_level'].isin(ATP_LEVELS).to_numpy()
        test_atp = test[atp_mask]
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_proba)),
            'logloss': float(log_loss(y_test, y_proba)),
            'brier': float(brier_score_loss(y_test, y_proba)),
            'ece': expected_calibration_error(y_test, y_proba),
            'raw': probability_metrics(y_test, y_proba_raw),
            'single_perspective': probability_metrics(y_test, y_proba_single),
            'baselines': baseline_metrics(test, y_test),
            'atp_only': {
                'model': probability_metrics(y_test[atp_mask], y_proba[atp_mask]),
                'baselines': baseline_metrics(test_atp, y_test[atp_mask]),
            } if atp_mask.sum() > 0 else None,
            'calibration': {
                'method': calibration_method,
                'selection': calibration_selection,
                'reliability_test': reliability_table(y_test, y_proba),
            },
            'excluded_features': self.excluded_features,
            'n_features': len(self.feature_columns),
            'n_train': int(len(final_train)),
            'n_val': int(len(final_val)),
            'n_calib': int(len(calib)),
            'n_test': int(len(test)),
            'test_start': str(test['date'].min()),
            'test_end': str(test['date'].max()),
            'best_params': best_params,
            'tuned_with_optuna': not fixed_params,
        }

        print("\n--- Avaliação Teste (dados mais recentes, previsões simétricas e calibradas) ---")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        print(f"LogLoss: {metrics['logloss']:.4f} (sem calibração: {metrics['raw']['logloss']:.4f}, "
              f"só uma perspetiva: {metrics['single_perspective']['logloss']:.4f})")
        print(f"Brier Score: {metrics['brier']:.4f} (sem calibração: {metrics['raw']['brier']:.4f})")
        print(f"ECE: {metrics['ece']:.4f} (sem calibração: {metrics['raw']['ece']:.4f})")
        print("\nBaselines no mesmo teste:")
        for name, values in metrics['baselines'].items():
            print(f"- {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in values.items() if isinstance(v, float)))
        if metrics['atp_only']:
            m = metrics['atp_only']['model']
            print(f"\nSó ATP ({m['n']} jogos): accuracy={m['accuracy']:.3f} logloss={m['logloss']:.4f} "
                  f"brier={m['brier']:.4f} ece={m['ece']:.4f}")
            for name, values in metrics['atp_only']['baselines'].items():
                print(f"- {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in values.items() if isinstance(v, float)))
        print("Relatório de classificação:")
        print(classification_report(y_test, y_pred))

        # Importância das features
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['top_features'] = feature_importance.head(15).to_dict(orient='records')
            print("\nTop features:")
            print(feature_importance.head(15))

        # Refit de produção com todos os dados (inclui validação e teste, os jogos mais recentes),
        # mantendo o número de árvores encontrado pelo early stopping.
        if refit_full:
            final_estimators = int(self.model.get_params()['n_estimators'])
            X_all, y_all = prepare_Xy(df, expected_columns=self.feature_columns)
            full_params = {**params, 'n_estimators': final_estimators}
            print(f"\nRefit final com 100% dos dados ({len(df)} jogos, {final_estimators} árvores)...")
            self.model = XGBClassifier(**full_params)
            self.model.fit(X_all, y_all, verbose=False)
            # O calibrador de produção usa as previsões fora da amostra do modelo de avaliação
            # (calibração + teste); é uma aproximação, já que o modelo refeito viu mais dados.
            self.calibrator = fit_calibrator(
                calibration_method,
                np.concatenate([p_calib, y_proba_raw]),
                np.concatenate([y_calib.to_numpy(), y_test.to_numpy()]),
            )
        else:
            self.calibrator = eval_calibrator
        metrics['refit_full'] = bool(refit_full)
        metrics['n_estimators'] = int(self.model.get_params()['n_estimators'])

        self.training_metrics = metrics
        return metrics

    def predict_match(self, player1, player2, surface, date=None, tourney_level='UNK', best_of=None,
                      is_qualifying=0):
        """
        Faz uma previsão para uma partida específica em uma data específica.

        `tourney_level` usa os códigos do Sackmann (G, M, A, F, D, C). Sem `best_of`, assume
        5 sets em Grand Slams e 3 nos restantes. A probabilidade é simétrica: média de
        p(jogador1, jogador2) e 1 − p(jogador2, jogador1), depois calibrada.
        """
        builder = self.feature_builder
        if date is not None and not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except Exception as e:
                print(f"Erro ao converter data: {e}. Usando data mais recente.")
                date = None
        if date is None:
            last = builder.last_date or self.last_match_date
            date = pd.Timestamp(last) + pd.Timedelta(days=1) if last is not None else pd.Timestamp.today().normalize()

        if not builder.knows(player1) or not builder.knows(player2):
            print("Um ou ambos os jogadores não foram encontrados nos dados históricos.")
            return None, None, None

        tourney_level = tourney_level if tourney_level is not None else 'UNK'
        if best_of is None:
            best_of = 5 if str(tourney_level).upper() == 'G' else 3
        context = {'surface': surface, 'tourney_level': tourney_level, 'date': date,
                   'best_of': best_of, 'is_qualifying': is_qualifying}
        f12 = builder.features(player1, player2, **context)
        f21 = builder.features(player2, player1, **context)

        X = self._to_model_matrix(pd.DataFrame([f12, f21]), expected_columns=self.feature_columns)
        p12, p21 = self.model.predict_proba(X)[:, 1]
        prob_player1_raw = float((p12 + 1 - p21) / 2)
        prob_player1 = float(apply_calibration(self.calibrator, [prob_player1_raw])[0])

        winner = player1 if prob_player1 >= self.decision_threshold else player2
        confidence = max(prob_player1, 1 - prob_player1)

        h2h = {'w': builder.h2h.get((player1, player2), 0), 'l': builder.h2h.get((player2, player1), 0)}
        details = {
            'date': date,
            'surface': surface,
            'player1_elo': f12['player_elo'],
            'player2_elo': f12['opponent_elo'],
            'elo_diff': f12['elo_diff'],
            'surface_elo_diff': f12['surface_elo_diff'],
            'h2h': self._format_h2h(player1, player2, h2h),
            'h2h_raw': h2h,
            'player1_rank': f12['player_rank'],
            'player2_rank': f12['opponent_rank'],
            'player1_rank_points': f12['player_rank_points'],
            'player2_rank_points': f12['opponent_rank_points'],
            'player1_age': f12['player_age'],
            'player2_age': f12['opponent_age'],
            'player1_surface_win_rate': f12['player_surface_win_rate'],
            'player2_surface_win_rate': f12['opponent_surface_win_rate'],
            'decision_threshold': self.decision_threshold,
            'tourney_level': f12['tourney_level'],
            'best_of': f12['best_of'],
            'probability_raw': max(prob_player1_raw, 1 - prob_player1_raw),
            'calibration': (self.calibrator or {}).get('method', 'none'),
            'probability': confidence
        }

        return winner, confidence, details

    def get_state(self):
        """Estado necessário para previsões (gravado no pkl de dados)."""
        return {
            'state_version': STATE_VERSION,
            'feature_state': self.feature_builder.get_state(),
            'decision_threshold': self.decision_threshold,
            'threshold_profiles': self.threshold_profiles,
            'feature_columns': self.feature_columns,
            'excluded_features': self.excluded_features,
            'last_match_date': self.last_match_date,
            'training_metrics': self.training_metrics,
            'calibrator': self.calibrator,
        }

    def save_model(self, model_dir: str | None = None):
        """Salva o modelo e dados necessários para previsão futura (por defeito em MODEL_PATH)."""
        model_dir = model_dir or MODEL_PATH
        print("\nSalvando modelo e dados...")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model, os.path.join(model_dir, MODEL_FILENAME))
        joblib.dump(self.get_state(), os.path.join(model_dir, DATA_FILENAME))

        print(f"Modelo e dados salvos em {model_dir}")

    def load_saved_model(self, model_dir: str | None = None):
        """Carrega um modelo treinado anteriormente (por defeito de MODEL_PATH)."""
        model_dir = model_dir or MODEL_PATH
        print("Carregando modelo salvo...")
        try:
            data = joblib.load(os.path.join(model_dir, DATA_FILENAME))
            if 'feature_state' not in data:
                print("Pkl de dados num formato antigo (sem estado incremental). "
                      "Treine de novo com: python src/train_model.py")
                return False
            self.model = joblib.load(os.path.join(model_dir, MODEL_FILENAME))
            self.feature_builder = FeatureBuilder(data['feature_state'])
            self.decision_threshold = data.get('decision_threshold', 0.5)
            self.threshold_profiles = data.get('threshold_profiles', {})
            self.feature_columns = data.get('feature_columns')
            self.excluded_features = data.get('excluded_features', [])
            self.last_match_date = data.get('last_match_date')
            self.training_metrics = data.get('training_metrics', {})
            self.calibrator = data.get('calibrator')
            self.swap_features = None

            print("Modelo e dados carregados com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo salvo: {str(e)}")
            return False


def main():
    """Carrega o modelo de produção e mostra um resumo. Para treinar: `python src/train_model.py`."""
    predictor = TennisPredictor()
    if not predictor.load_saved_model():
        print(f"Nenhum modelo válido em '{MODEL_PATH}'. Treine com: python src/train_model.py")
        raise SystemExit(1)

    print(f"Features: {len(predictor.feature_columns or [])} | Jogadores: {len(predictor.feature_builder.players)}")
    print(f"Último jogo no treino: {predictor.last_match_date or 'desconhecido'}")
    if predictor.training_metrics:
        m = predictor.training_metrics
        print(f"Holdout: logloss={m.get('logloss'):.4f} brier={m.get('brier'):.4f} accuracy={m.get('accuracy'):.3f}")


if __name__ == '__main__':
    main()
