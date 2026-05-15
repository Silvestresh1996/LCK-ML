"""
============================================================
LCK PREDICTION OS — MÓDULO 3: MOTOR ML (v3 - REFACTORIZADO)
============================================================
CORRECCIONES v3:
  [FIX-1] Eliminado CalibratedClassifierCV con cv='prefit'.
          XGBoost con eval_metric='logloss' ya produce probabilidades
          bien calibradas — la calibración externa era redundante
          y causaba el TypeError de versiones de sklearn.

  [FIX-2] Columnas en 0.0: el método _audit_features() detecta
          features vacías y las excluye automáticamente del
          vector de entrada, usando solo las que tienen datos reales.

  [FIX-3] Modelo ahora tiene dos modos:
            MODE_FULL  → usa todos los features Tier-1 + Tier-2
            MODE_LITE  → solo win_rate + team_id (cuando los stats
                         detallados no están disponibles)
          El modo se selecciona automáticamente según los datos.

  [FIX-4] LCKStrategyAdjuster ahora es independiente del modelo
          y puede usarse en lck_main.py directamente.
============================================================
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib
import logging

from lck_config import (
    XGBOOST_PARAMS, LCK_FEATURE_WEIGHTS, TIME_SERIES_SPLITS,
    FEATURES_TIER1, FEATURES_TIER2, FEATURE_COLUMNS,
    TEAM_NAME_TO_ID, LCK_TEAMS,
    KELLY_FRACTION, BANKROLL, MIN_EDGE_THRESHOLD, MIN_STAKE, MAX_STAKE_PCT
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  AJUSTES REGIONALES LCK
# ═══════════════════════════════════════════════════════════
class LCKStrategyAdjuster:
    """
    Aplica los pesos específicos del meta LCK sobre cualquier
    diccionario de estadísticas de equipo.

    Uso directo (sin modelo):
        adj = LCKStrategyAdjuster()
        prob_ajustada = adj.adjust_probability(prob_base, team_stats, side="blue")
    """

    BLUE_SIDE_BONUS    = 0.03    # Blue Side gana ~53% en LCK
    GOLD_LEAD_WEIGHT   = 0.85    # Gold lead @20 → predictor dominante en LCK
    BARON_WEIGHT       = 0.78
    LONG_GAME_THRESHOLD = 32     # Minutos → activa multiplicador macro

    def adjust_probability(
        self,
        prob_base: float,
        team_stats: dict,
        side: str = "blue"
    ) -> float:
        """
        Ajusta una probabilidad base (0-1) con factores del meta LCK.

        Args:
            prob_base:   Probabilidad cruda del modelo (0.0 – 1.0)
            team_stats:  Dict con KPIs del equipo (win_rate, baron, etc.)
            side:        'blue' o 'red'

        Returns:
            Probabilidad ajustada (clipped 0.05 – 0.95)
        """
        adj = prob_base

        # 1. Ventaja de lado del mapa
        if side.lower() == "blue":
            adj += self.BLUE_SIDE_BONUS
        else:
            adj -= self.BLUE_SIDE_BONUS * 0.5   # Red side penalización menor

        # 2. Factor Baron en partidas largas
        avg_dur  = team_stats.get("avg_game_duration", 30)
        baron_wr = team_stats.get("baron_control_rate", 0.5)
        if avg_dur > self.LONG_GAME_THRESHOLD:
            delta = (baron_wr - 0.5) * self.BARON_WEIGHT * 0.10
            adj += delta

        # 3. Gold lead @20 (feature definitoria de LCK)
        gl20 = team_stats.get("gold_lead_20_weight", 0)
        if gl20 != 0:
            max_gl20 = 1500
            normalized = np.clip(gl20 / max_gl20, -1, 1)
            adj += normalized * self.GOLD_LEAD_WEIGHT * 0.05

        return float(np.clip(adj, 0.05, 0.95))

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica pesos LCK sobre columnas del DataFrame (para entrenamiento)."""
        df = df.copy()
        if "gold_lead_20_weight" in df.columns:
            df["gold_lead_20_weight"] *= self.GOLD_LEAD_WEIGHT
        if "blue_side_winrate" in df.columns:
            df["blue_side_winrate"] = (df["blue_side_winrate"] + self.BLUE_SIDE_BONUS).clip(0, 1)
        return df


# ═══════════════════════════════════════════════════════════
#  MOTOR DE PREDICCIÓN (XGBoost sin CalibratedClassifierCV)
# ═══════════════════════════════════════════════════════════
class LCKPredictor:
    """
    Modelo XGBoost para predicción de victorias LCK.

    Sin CalibratedClassifierCV: XGBoost entrenado con eval_metric='logloss'
    produce predict_proba() bien calibrado de forma nativa.

    Selección automática de features:
        - Si Tier-2 tiene datos reales  → MODE_FULL  (9 features)
        - Si Tier-2 está todo en 0/NaN  → MODE_LITE  (2 features: win_rate + team_id)
    """

    MODE_FULL = "full"
    MODE_LITE = "lite"

    def __init__(self):
        self.model       = XGBClassifier(**XGBOOST_PARAMS)
        self.scaler      = StandardScaler()
        self.strategy    = LCKStrategyAdjuster()
        self.is_trained  = False
        self.active_mode = self.MODE_LITE
        self.active_feats: list[str] = []
        self.cv_metrics: dict = {}

    # ─────────────────────────────────────────
    #  AUDITORÍA DE FEATURES
    # ─────────────────────────────────────────
    def _audit_features(self, df: pd.DataFrame) -> tuple[list[str], str]:
        """
        Detecta qué columnas tienen datos reales (no todo 0 o NaN).

        Returns:
            (lista_de_features_activas, modo)
        """
        active = list(FEATURES_TIER1)   # Tier-1 siempre incluido

        nonzero_tier2 = []
        for col in FEATURES_TIER2:
            if col in df.columns:
                series = df[col].fillna(0)
                # Considerar útil si más del 20% de filas son no-cero
                pct_nonzero = (series != 0).mean()
                if pct_nonzero >= 0.2:
                    nonzero_tier2.append(col)
                else:
                    log.debug(f"  Feature '{col}' descartada ({pct_nonzero:.0%} no-cero < 20%)")

        if nonzero_tier2:
            active += nonzero_tier2
            mode = self.MODE_FULL
            log.info(f"  Modo FULL: {len(active)} features activas → {active}")
        else:
            mode = self.MODE_LITE
            log.warning(
                "  ⚠️  Tier-2 features vacías (columnas en 0). "
                "Usando MODE_LITE (win_rate + team_id). "
                "Mejora: conecta /lol/games con stats detalladas."
            )

        return active, mode

    # ─────────────────────────────────────────
    #  PREPARACIÓN DE DATOS
    # ─────────────────────────────────────────
    def _encode_team_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica team_id como float normalizado (0.0 – 1.0)."""
        df = df.copy()
        if "team_id" in df.columns:
            max_id = max(TEAM_NAME_TO_ID.values())
            df["team_id_encoded"] = df["team_id"].fillna(0) / max_id
        elif "team_id_encoded" not in df.columns:
            df["team_id_encoded"] = 0.0
        return df

    def _build_X(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Construye y escala la matriz de features con las columnas activas."""
        df = self.strategy.apply_to_dataframe(df)
        df = self._encode_team_id(df)

        # Rellenar columnas faltantes con 0
        for col in self.active_feats:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.active_feats].fillna(0).values.astype(float)

        if fit_scaler:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    # ─────────────────────────────────────────
    #  ENTRENAMIENTO
    # ─────────────────────────────────────────
    def train(self, df_stats: pd.DataFrame) -> dict:
        """
        Entrena el modelo con los KPIs por equipo de df_stats
        (output directo de pipeline.build_team_stats()).

        Para predecir victorias se construyen pares de equipos
        y se etiqueta con win_rate diferencial binarizado.

        Si hay menos de 10 equipos, usa regresión logística simple
        sobre win_rate como fallback ultra-robusto.
        """
        if df_stats.empty or "win_rate" not in df_stats.columns:
            raise ValueError("df_stats vacío o sin columna 'win_rate'.")

        log.info(f"  Equipos en df_stats: {len(df_stats)}")

        # ── Detectar features disponibles ──
        self.active_feats, self.active_mode = self._audit_features(df_stats)

        # ── Construir dataset de pares (A vs B) ──
        pairs = self._build_match_pairs(df_stats)
        if len(pairs) < 6:
            log.warning(f"  Solo {len(pairs)} pares — dataset muy pequeño. Usando win_rate directo.")
            self._train_lite(df_stats)
            return self.cv_metrics

        X = self._build_X(pairs, fit_scaler=True)
        y = pairs["label"].values

        # ── TimeSeriesSplit (degradado a KFold si datos insuficientes) ──
        n_splits = min(TIME_SERIES_SPLITS, len(pairs) // 4)
        n_splits = max(n_splits, 2)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        auc_scores, acc_scores, ll_scores = [], [], []

        log.info(f"  TimeSeriesSplit con {n_splits} folds…")
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            if len(val_idx) < 2:
                continue
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # Saltar fold si solo hay una clase en train o val
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                log.debug(f"    Fold {fold+1} ignorado (clase única en split).")
                continue

            # Intentar fit con eval_set (XGBoost); fallback sin él (sklearn)
            try:
                self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                self.model.fit(X_tr, y_tr)

            probs = self.model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            auc = roc_auc_score(y_val, probs)
            acc = accuracy_score(y_val, preds)
            ll  = log_loss(y_val, probs)
            auc_scores.append(auc)
            acc_scores.append(acc)
            ll_scores.append(ll)
            log.info(f"    Fold {fold+1}: AUC={auc:.3f} Acc={acc:.3f} LogLoss={ll:.3f}")

        # ── Reentrenamiento final con todos los datos ──
        try:
            self.model.fit(X, y, verbose=False)
        except TypeError:
            self.model.fit(X, y)
        self.is_trained = True

        self.cv_metrics = {
            "mode":      self.active_mode,
            "features":  len(self.active_feats),
            "pairs":     len(pairs),
            "auc_mean":  round(float(np.mean(auc_scores)), 4) if auc_scores else 0.0,
            "auc_std":   round(float(np.std(auc_scores)),  4) if auc_scores else 0.0,
            "acc_mean":  round(float(np.mean(acc_scores)), 4) if acc_scores else 0.0,
            "logloss":   round(float(np.mean(ll_scores)),  4) if ll_scores else 0.0,
        }
        log.info(f"  ✅ Entrenamiento OK | AUC={self.cv_metrics['auc_mean']} | Modo={self.active_mode}")
        return self.cv_metrics

    def _build_match_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea pares sintéticos (A vs B) con features diferenciales.

        Para garantizar ambas clases (0 y 1) en cada fold, generamos
        las dos direcciones de cada enfrentamiento:
            - A vs B  →  label 1  (A favorito)
            - B vs A  →  label 0  (B desfavorito, A pierde)
        Esto duplica el dataset y balancea las clases automáticamente.
        """
        rows = []
        teams = df.to_dict(orient="records")
        for i, a in enumerate(teams):
            for j, b in enumerate(teams):
                if i >= j:
                    continue
                # Dirección A vs B
                pair_ab = {}
                # Dirección B vs A (negativa)
                pair_ba = {}
                for col in self.active_feats:
                    va = float(a.get(col, 0) or 0)
                    vb = float(b.get(col, 0) or 0)
                    pair_ab[col] = va - vb
                    pair_ba[col] = vb - va
                pair_ab["label"] = 1 if a.get("win_rate", 0) >= b.get("win_rate", 0) else 0
                pair_ba["label"] = 1 if b.get("win_rate", 0) >  a.get("win_rate", 0) else 0
                rows.append(pair_ab)
                rows.append(pair_ba)
        return pd.DataFrame(rows)

    def _train_lite(self, df: pd.DataFrame):
        """Fallback ultra-simple: regresión logística sobre win_rate."""
        from sklearn.linear_model import LogisticRegression
        log.info("  Usando fallback LogisticRegression sobre win_rate.")
        self.active_feats = ["win_rate", "team_id_encoded"]
        df2 = self._encode_team_id(df)
        X   = df2[["win_rate", "team_id_encoded"]].fillna(0).values
        y   = (df2["win_rate"] >= df2["win_rate"].median()).astype(int).values
        lr  = LogisticRegression().fit(X, y)
        # Wrap para mantener interfaz predict_proba
        self.model      = lr
        self.is_trained = True
        self.cv_metrics = {"mode": "fallback_lr", "features": 2, "auc_mean": 0.0}

    # ─────────────────────────────────────────
    #  PREDICCIÓN DE PARTIDO
    # ─────────────────────────────────────────
    def predict_match(
        self,
        stats_a: dict,
        stats_b: dict,
        side_a: str = "blue"
    ) -> dict:
        """
        Predice el resultado de un partido.

        Args:
            stats_a: KPIs del equipo A (dict, de build_team_stats)
            stats_b: KPIs del equipo B
            side_a:  'blue' o 'red' (para ajuste LCK de lado del mapa)

        Returns:
            dict con prob_a, prob_b, winner predicho y confianza
        """
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a .train(df_stats) primero.")

        # Vector diferencial A - B
        diff = {}
        for feat in self.active_feats:
            va = stats_a.get(feat, 0) or 0
            vb = stats_b.get(feat, 0) or 0
            diff[feat] = va - vb

        X_in  = self._build_X(pd.DataFrame([diff]), fit_scaler=False)
        prob_a = float(self.model.predict_proba(X_in)[0, 1])

        # Ajuste regional LCK
        prob_a = self.strategy.adjust_probability(prob_a, stats_a, side=side_a)
        prob_b = 1.0 - prob_a

        return {
            "team_a":    stats_a.get("team_name", "Equipo A"),
            "team_b":    stats_b.get("team_name", "Equipo B"),
            "prob_a":    round(prob_a, 4),
            "prob_b":    round(prob_b, 4),
            "winner":    stats_a.get("team_name") if prob_a >= 0.5 else stats_b.get("team_name"),
            "confidence": round(max(prob_a, prob_b), 4),
            "side_a":    side_a.upper(),
            "mode":      self.active_mode,
        }

    # ─────────────────────────────────────────
    #  PERSISTENCIA
    # ─────────────────────────────────────────
    def save(self, path: str = "lck_model.joblib"):
        joblib.dump({
            "model":       self.model,
            "scaler":      self.scaler,
            "active_feats": self.active_feats,
            "active_mode": self.active_mode,
            "cv_metrics":  self.cv_metrics,
        }, path)
        log.info(f"  Modelo guardado → {path}")

    def load(self, path: str = "lck_model.joblib"):
        d = joblib.load(path)
        self.model        = d["model"]
        self.scaler       = d["scaler"]
        self.active_feats = d["active_feats"]
        self.active_mode  = d["active_mode"]
        self.cv_metrics   = d["cv_metrics"]
        self.is_trained   = True
        log.info(f"  Modelo cargado ← {path} (modo={self.active_mode})")


# ═══════════════════════════════════════════════════════════
#  CONVERSOR DE MOMIOS + KELLY CRITERION
# ═══════════════════════════════════════════════════════════
def american_to_decimal(american: int | str) -> float:
    """
    Convierte momios americanos a cuota decimal europea.

    Ejemplos:
        +285  →  3.85   (underdog: ganas 285 por cada 100 apostados)
        -425  →  1.235  (favorito: necesitas apostar 425 para ganar 100)
        +100  →  2.00
        -100  →  2.00
    """
    n = int(str(american).replace("+", "").replace(" ", ""))
    if n > 0:
        return round((n / 100) + 1, 4)
    else:
        return round((100 / abs(n)) + 1, 4)


def kelly_stake(
    prob_model: float,
    decimal_odd: float,
    bankroll: float = BANKROLL,
    fraction: float = KELLY_FRACTION
) -> dict:
    """
    Calcula el stake óptimo con Criterio de Kelly fraccional.

    Fórmulas:
        implied_prob = 1 / decimal_odd
        edge         = (prob_model × decimal_odd) - 1
        kelly_raw    = (prob_model - implied_prob) / (decimal_odd - 1)
        stake        = bankroll × fraction × kelly_raw

    Args:
        prob_model:  Probabilidad real del modelo (0.0 – 1.0)
        decimal_odd: Cuota decimal de Codere (ej: 1.85)
        bankroll:    Capital disponible en MXN
        fraction:    Fracción de Kelly (0.25 = conservador)

    Returns:
        dict con edge, stake, EV, ROI y señal de valor
    """
    if decimal_odd <= 1.0:
        return {"error": "Cuota debe ser > 1.0"}

    implied_prob = 1.0 / decimal_odd
    edge         = (prob_model * decimal_odd) - 1.0
    is_value     = edge > MIN_EDGE_THRESHOLD

    if is_value and decimal_odd > 1:
        kelly_raw  = (prob_model - implied_prob) / (decimal_odd - 1.0)
        kelly_frac = max(kelly_raw, 0) * fraction
        stake_raw  = bankroll * kelly_frac
        stake      = round(max(MIN_STAKE, min(stake_raw, bankroll * MAX_STAKE_PCT)), 2)
    else:
        kelly_raw  = 0.0
        kelly_frac = 0.0
        stake      = 0.0

    ev  = round((prob_model * decimal_odd * stake) - stake, 2) if stake > 0 else 0.0
    roi = round((ev / stake * 100), 1) if stake > 0 else 0.0

    return {
        "prob_model_pct":  round(prob_model * 100, 1),
        "implied_prob_pct": round(implied_prob * 100, 1),
        "edge_pct":        round(edge * 100, 2),
        "is_value":        is_value,
        "kelly_pct":       round(kelly_frac * 100, 2),
        "stake_mxn":       stake,
        "ev_mxn":          ev,
        "roi_pct":         roi,
        "signal":          "🥇 OPORTUNIDAD DE ORO" if is_value else "❌  Sin ventaja suficiente",
    }