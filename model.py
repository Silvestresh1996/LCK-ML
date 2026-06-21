"""
============================================================
PREDICTION OS V2 — MOTOR DE PREDICCIÓN (entrenado sobre datos reales)
============================================================
Diferencia clave con la versión anterior:

  ANTES (defectuoso):
    Se generaban pares sintéticos A vs B y se etiquetaban con
    `1 si win_rate(A) >= win_rate(B)`. El modelo "aprendía" que
    gana el de mayor win_rate — algo verdadero POR CONSTRUCCIÓN.
    El AUC resultante era engañoso (tautología).

  AHORA (correcto):
    Se entrena con los RESULTADOS REALES de los partidos
    (winner_id de /lol/matches). Cada fila es un enfrentamiento
    que ocurrió de verdad; la etiqueta es quién ganó realmente.
    Las features son las diferencias de KPIs entre los dos equipos.

Limitación honesta (documentada): los KPIs (win_rate, etc.) se
calculan sobre la misma ventana de partidos que incluye los que
predecimos, así que hay una fuga de información leve. Con datasets
pequeños (decenas de partidos) es un compromiso aceptable y muy
superior a la tautología anterior. Para rigor total habría que
calcular stats "pre-partido" con ventana móvil.

Componentes:
  MatchPredictor   — entrena XGBoost (o LogisticRegression de fallback)
  american_to_decimal / kelly_stake — utilidades de apuestas
============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

import config

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  MOTOR DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════
class MatchPredictor:
    """
    Predice el ganador de un partido a partir de las diferencias
    de KPIs entre los dos equipos, entrenando con resultados reales.

    Flujo:
        pred = MatchPredictor()
        metrics = pred.train(df_stats, df_matches)
        out = pred.predict_match(stats_a, stats_b, side_a="blue")
    """

    MODE_FULL = "full"      # usa Tier-1 + Tier-2 (stats detalladas disponibles)
    MODE_LITE = "lite"      # solo Tier-1 (stats por partida no disponibles)
    MODE_FALLBACK = "fallback_lr"   # muy pocos datos → LogisticRegression

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.active_mode = self.MODE_LITE
        self.active_feats: list[str] = []
        self.cv_metrics: dict = {}

    # ─────────────────────────────────────────
    #  AUDITORÍA DE FEATURES DISPONIBLES
    # ─────────────────────────────────────────
    def _audit_features(self, df_stats: pd.DataFrame) -> tuple[list[str], str]:
        """Decide qué features tienen datos reales y el modo resultante."""
        active = [c for c in config.FEATURES_TIER1 if c in df_stats.columns]

        usable_tier2 = []
        for col in config.FEATURES_TIER2:
            if col in df_stats.columns:
                pct_nonzero = (df_stats[col].fillna(0) != 0).mean()
                if pct_nonzero >= config.TIER2_MIN_NONZERO:
                    usable_tier2.append(col)

        if usable_tier2:
            active += usable_tier2
            mode = self.MODE_FULL
        else:
            mode = self.MODE_LITE
            log.info("  Tier-2 sin datos → modo LITE (solo win_rate y lados).")

        return active, mode

    # ─────────────────────────────────────────
    #  CONSTRUCCIÓN DEL DATASET (resultados reales)
    # ─────────────────────────────────────────
    def _build_dataset(
        self,
        df_stats: pd.DataFrame,
        df_matches: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convierte partidos reales en filas de entrenamiento.

        Para cada partido con ganador conocido:
            features = KPIs(team_a) - KPIs(team_b)
            label    = 1 si ganó team_a, 0 si ganó team_b

        Retorna (X, y) ordenados cronológicamente por begin_at.
        Las filas se quedan SIN escalar (el escalado se hace fuera).
        """
        stats_by_id = {
            int(r["team_id"]): r
            for r in df_stats.to_dict("records")
            if pd.notna(r.get("team_id"))
        }

        df = df_matches.copy()
        if "begin_at" in df.columns:
            df = df.sort_values("begin_at", na_position="last")

        rows, labels = [], []
        for m in df.to_dict("records"):
            ta, tb, w = m.get("team_a_id"), m.get("team_b_id"), m.get("winner_id")
            if ta is None or tb is None or w is None:
                continue
            ta, tb, w = int(ta), int(tb), int(w)
            if ta not in stats_by_id or tb not in stats_by_id:
                continue
            if w not in (ta, tb):
                continue

            sa, sb = stats_by_id[ta], stats_by_id[tb]
            diff = [float(sa.get(f, 0) or 0) - float(sb.get(f, 0) or 0)
                    for f in self.active_feats]
            rows.append(diff)
            labels.append(1 if w == ta else 0)

        return np.array(rows, dtype=float), np.array(labels, dtype=int)

    # ─────────────────────────────────────────
    #  ENTRENAMIENTO
    # ─────────────────────────────────────────
    def train(self, df_stats: pd.DataFrame, df_matches: pd.DataFrame) -> dict:
        """
        Entrena con resultados reales.

        Args:
            df_stats:   KPIs por equipo (de pipeline.build_team_stats)
            df_matches: partidos reales con team_a_id, team_b_id, winner_id,
                        begin_at (de pipeline.get_matches)

        Returns:
            dict de métricas (mode, features, matches, auc_mean, ...)
        """
        if df_stats is None or df_stats.empty or "win_rate" not in df_stats.columns:
            raise ValueError("df_stats vacío o sin columna 'win_rate'.")

        self.active_feats, self.active_mode = self._audit_features(df_stats)

        # Sin partidos reales → no se puede entrenar honestamente: fallback.
        if df_matches is None or df_matches.empty:
            log.warning("  Sin partidos reales — usando fallback sobre win_rate.")
            return self._train_fallback(df_stats)

        X, y = self._build_dataset(df_stats, df_matches)
        n = len(y)
        log.info(f"  {n} partidos reales utilizables | features={self.active_feats}")

        # Pocos datos o una sola clase → fallback robusto.
        if n < config.MIN_MATCHES_FOR_ML or len(np.unique(y)) < 2:
            log.warning(
                f"  Solo {n} partidos (<{config.MIN_MATCHES_FOR_ML}) o clase única "
                "— usando fallback LogisticRegression."
            )
            return self._train_fallback(df_stats)

        # ── Escalado ──
        X_scaled = self.scaler.fit_transform(X)

        # ── CV cronológico ──
        n_splits = max(2, min(config.TIME_SERIES_SPLITS, n // 8))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, lls = [], [], []

        for fold, (tr, val) in enumerate(tscv.split(X_scaled)):
            if len(val) < 2:
                continue
            y_tr, y_val = y[tr], y[val]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue
            mdl = XGBClassifier(**config.XGBOOST_PARAMS)
            mdl.fit(X_scaled[tr], y_tr)
            p = mdl.predict_proba(X_scaled[val])[:, 1]
            aucs.append(roc_auc_score(y_val, p))
            accs.append(accuracy_score(y_val, (p >= 0.5).astype(int)))
            lls.append(log_loss(y_val, p, labels=[0, 1]))
            log.info(f"    Fold {fold+1}: AUC={aucs[-1]:.3f} Acc={accs[-1]:.3f}")

        # ── Modelo final: entrenado sobre el set SIMETRIZADO ──
        # Añadir (B-A, 1-label) elimina cualquier sesgo por el orden en que
        # PandaScore lista a los equipos. Las etiquetas siguen siendo reales.
        X_sym = np.vstack([X, -X])
        y_sym = np.concatenate([y, 1 - y])
        X_sym_scaled = self.scaler.fit_transform(X_sym)
        self.model = XGBClassifier(**config.XGBOOST_PARAMS)
        self.model.fit(X_sym_scaled, y_sym)
        self.is_trained = True

        self.cv_metrics = {
            "mode":     self.active_mode,
            "features": len(self.active_feats),
            "matches":  n,
            "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "auc_std":  round(float(np.std(aucs)), 4) if aucs else None,
            "acc_mean": round(float(np.mean(accs)), 4) if accs else None,
            "logloss":  round(float(np.mean(lls)), 4) if lls else None,
        }
        log.info(f"  ✅ Entrenado | AUC={self.cv_metrics['auc_mean']} | modo={self.active_mode}")
        return self.cv_metrics

    def _train_fallback(self, df_stats: pd.DataFrame) -> dict:
        """
        Fallback cuando no hay suficientes partidos reales: regresión
        logística sobre la diferencia de win_rate. NO produce un AUC
        creíble (datos insuficientes), así que se reporta auc_mean=None.
        """
        self.active_feats = ["win_rate"]
        self.active_mode = self.MODE_FALLBACK

        wr = df_stats["win_rate"].fillna(0.5).to_numpy().reshape(-1, 1)
        # Diferencias contra la media de la liga, como señal mínima.
        X = wr - float(np.mean(wr))
        y = (X[:, 0] >= 0).astype(int)
        self.scaler.fit(X)
        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]   # garantizar dos clases para que LR ajuste
        self.model = LogisticRegression().fit(self.scaler.transform(X), y)
        self.is_trained = True

        self.cv_metrics = {
            "mode": self.MODE_FALLBACK, "features": 1,
            "matches": 0, "auc_mean": None,
        }
        return self.cv_metrics

    # ─────────────────────────────────────────
    #  PREDICCIÓN
    # ─────────────────────────────────────────
    def predict_match(self, stats_a: dict, stats_b: dict, side_a: str = "blue") -> dict:
        """
        Predice la probabilidad de victoria de A.

        side_a aplica un pequeño ajuste heurístico de lado del mapa
        (blue side gana ligeramente más). El ajuste es POST-modelo y
        está documentado como heurístico, no aprendido.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modelo no entrenado. Llama a .train() primero.")

        diff = np.array([[
            float(stats_a.get(f, 0) or 0) - float(stats_b.get(f, 0) or 0)
            for f in self.active_feats
        ]], dtype=float)

        prob_a = float(self.model.predict_proba(self.scaler.transform(diff))[0, 1])

        # Ajuste heurístico de lado del mapa
        if side_a.lower() == "blue":
            prob_a += config.BLUE_SIDE_BONUS
        else:
            prob_a -= config.BLUE_SIDE_BONUS
        prob_a = float(np.clip(prob_a, 0.05, 0.95))

        return {
            "team_a":     stats_a.get("team_name", "Equipo A"),
            "team_b":     stats_b.get("team_name", "Equipo B"),
            "prob_a":     round(prob_a, 4),
            "prob_b":     round(1 - prob_a, 4),
            "winner":     stats_a.get("team_name") if prob_a >= 0.5 else stats_b.get("team_name"),
            "confidence": round(max(prob_a, 1 - prob_a), 4),
            "side_a":     side_a.upper(),
            "mode":       self.active_mode,
        }

    # ─────────────────────────────────────────
    #  PERSISTENCIA
    # ─────────────────────────────────────────
    def save(self, path: str = "model.joblib"):
        joblib.dump({
            "model": self.model, "scaler": self.scaler,
            "active_feats": self.active_feats, "active_mode": self.active_mode,
            "cv_metrics": self.cv_metrics,
        }, path)
        log.info(f"  Modelo guardado → {path}")

    def load(self, path: str = "model.joblib"):
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self.active_feats = d["active_feats"]
        self.active_mode = d["active_mode"]
        self.cv_metrics = d["cv_metrics"]
        self.is_trained = True


# ═══════════════════════════════════════════════════════════
#  CONVERSOR DE MOMIOS + KELLY CRITERION
# ═══════════════════════════════════════════════════════════
def american_to_decimal(american: int | str) -> float:
    """
    Momio americano → cuota decimal.
      +285  →  3.85   |   -425  →  1.235   |   ±100  →  2.00
    """
    n = int(str(american).replace("+", "").replace(" ", ""))
    if n > 0:
        return round((n / 100) + 1, 4)
    return round((100 / abs(n)) + 1, 4)


def kelly_stake(
    prob_model: float,
    decimal_odd: float,
    bankroll: float = config.BANKROLL,
    fraction: float = config.KELLY_FRACTION,
) -> dict:
    """
    Stake óptimo con Criterio de Kelly fraccional.

        implied = 1 / cuota
        edge    = prob_model * cuota - 1
        kelly   = (prob_model - implied) / (cuota - 1)
        stake   = bankroll * fraction * kelly   (acotado a [MIN_STAKE, MAX_STAKE_PCT])

    Retorna un dict con edge, stake, EV, ROI y si es value bet.
    """
    if decimal_odd <= 1.0:
        return {"error": "Cuota debe ser > 1.0", "is_value": False,
                "stake_mxn": 0.0, "ev_mxn": 0.0, "edge_pct": 0.0,
                "implied_prob_pct": 0.0, "kelly_pct": 0.0, "roi_pct": 0.0}

    implied = 1.0 / decimal_odd
    edge = (prob_model * decimal_odd) - 1.0
    is_value = edge > config.MIN_EDGE_THRESHOLD

    if is_value:
        kelly_raw = (prob_model - implied) / (decimal_odd - 1.0)
        kelly_frac = max(kelly_raw, 0) * fraction
        stake = round(max(config.MIN_STAKE,
                          min(bankroll * kelly_frac, bankroll * config.MAX_STAKE_PCT)), 2)
    else:
        kelly_frac = 0.0
        stake = 0.0

    ev = round((prob_model * decimal_odd * stake) - stake, 2) if stake > 0 else 0.0
    roi = round((ev / stake * 100), 1) if stake > 0 else 0.0

    return {
        "prob_model_pct":   round(prob_model * 100, 1),
        "implied_prob_pct": round(implied * 100, 1),
        "edge_pct":         round(edge * 100, 2),
        "is_value":         is_value,
        "kelly_pct":        round(kelly_frac * 100, 2),
        "stake_mxn":        stake,
        "ev_mxn":           ev,
        "roi_pct":          roi,
        "signal":           "🥇 OPORTUNIDAD DE ORO" if is_value else "❌ Sin ventaja suficiente",
    }
