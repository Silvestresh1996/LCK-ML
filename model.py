"""
============================================================
PREDICTION OS V2 — MOTOR DE PREDICCIÓN (Elo + cronológico, sin fuga)
============================================================
Evolución del modelo:

  v1 (defectuoso):  pares sintéticos etiquetados con `win_rate(A)>=win_rate(B)`
                    → tautología, AUC inflado.
  v2 (real):        entrenado sobre resultados reales (winner_id), pero los KPIs
                    se calculaban sobre la misma ventana → fuga de información.
  v3 (este):        SISTEMA ELO entrenado CRONOLÓGICAMENTE.

Cómo funciona (v3):
  1. Se ordenan los partidos por fecha (del más viejo al más nuevo).
  2. Cada equipo arranca con rating Elo base (1500). Se recorren los partidos
     en orden y, ANTES de cada partido, se registran como features el rating y
     la forma reciente que cada equipo tenía *hasta ese momento* (nunca con
     información del futuro → SIN fuga de datos). Después se actualiza el Elo.
  3. Una regresión logística calibra esas features (Δelo, Δforma) hacia una
     probabilidad. La calibración es clave: el *edge* y el Kelly dependen de que
     la probabilidad sea fiable, no solo de acertar el ganador.

Ventajas para apostar:
  • AUC honesto (validación cronológica, sin fuga).
  • Probabilidades calibradas (logística) → cálculo de value bet correcto.
  • Usa solo resultados de partidos: 100% compatible con el plan Free de
    PandaScore (las stats detalladas por partida están bloqueadas en Free).

Componentes:
  MatchPredictor   — Elo + LogisticRegression (fallback simple si hay pocos datos)
  american_to_decimal / kelly_stake — utilidades de apuestas
============================================================
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

import config

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  MOTOR DE PREDICCIÓN ELO
# ═══════════════════════════════════════════════════════════
class MatchPredictor:
    """
    Predice el ganador combinando un rating Elo (fuerza del equipo) con la
    forma reciente, calibrado por regresión logística.

    Flujo:
        pred = MatchPredictor()
        metrics = pred.train(df_stats, df_matches)
        out = pred.predict_match(stats_a, stats_b, side_a="blue")
    """

    MODE_ELO      = "elo"
    MODE_FALLBACK = "fallback_lr"

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.active_mode = self.MODE_ELO
        self.cv_metrics: dict = {}
        # Estado del rating, poblado durante el entrenamiento:
        self.team_elo: dict[int, float] = {}    # {team_id → rating final}
        self.team_form: dict[int, float] = {}   # {team_id → forma reciente 0-1}
        self.team_name: dict[int, str] = {}     # {team_id → nombre} (para display)

    # ─────────────────────────────────────────
    #  ELO
    # ─────────────────────────────────────────
    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        """Probabilidad esperada de que A gane según la fórmula Elo."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _run_elo(self, df_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Recorre los partidos en orden cronológico calculando, ANTES de cada
        partido, las features pre-partido (sin fuga), y actualizando el Elo y
        la forma después. Deja el estado final en self.team_elo / team_form.

        Retorna (X, y) con X = [Δelo, Δforma] por partido y y = ganó A.
        """
        df = df_matches.copy()
        if "begin_at" in df.columns:
            df = df.sort_values("begin_at", na_position="last")

        elo: dict[int, float] = defaultdict(lambda: config.ELO_BASE)
        form: dict[int, list] = defaultdict(list)   # historial de 1/0 por equipo
        win = config.ELO_FORM_WINDOW

        rows, labels = [], []
        for m in df.to_dict("records"):
            a, b, w = m.get("team_a_id"), m.get("team_b_id"), m.get("winner_id")
            if a is None or b is None or w is None:
                continue
            a, b, w = int(a), int(b), int(w)
            if w not in (a, b):
                continue

            ra, rb = elo[a], elo[b]
            fa = float(np.mean(form[a][-win:])) if form[a] else 0.5
            fb = float(np.mean(form[b][-win:])) if form[b] else 0.5

            # Feature pre-partido (información solo del pasado)
            rows.append([ra - rb, fa - fb])
            labels.append(1 if w == a else 0)

            # Actualización Elo tras conocer el resultado
            ea = self._expected(ra, rb)
            sa = 1.0 if w == a else 0.0
            elo[a] = ra + config.ELO_K * (sa - ea)
            elo[b] = rb + config.ELO_K * ((1.0 - sa) - (1.0 - ea))
            form[a].append(sa)
            form[b].append(1.0 - sa)

        self.team_elo = dict(elo)
        self.team_form = {t: (float(np.mean(h[-win:])) if h else 0.5)
                          for t, h in form.items()}
        return np.array(rows, dtype=float), np.array(labels, dtype=int)

    # ─────────────────────────────────────────
    #  ENTRENAMIENTO
    # ─────────────────────────────────────────
    def train(self, df_stats: pd.DataFrame, df_matches: pd.DataFrame) -> dict:
        """
        Entrena el modelo Elo cronológicamente.

        Args:
            df_stats:   KPIs por equipo (para nombres / display).
            df_matches: partidos reales con team_a_id, team_b_id, winner_id,
                        begin_at (de pipeline.get_matches).
        """
        if df_stats is not None and not df_stats.empty and "team_id" in df_stats.columns:
            self.team_name = {
                int(r["team_id"]): r.get("team_name", f"Team_{int(r['team_id'])}")
                for r in df_stats.to_dict("records") if pd.notna(r.get("team_id"))
            }

        if df_matches is None or df_matches.empty:
            log.warning("  Sin partidos reales — fallback sobre win_rate.")
            return self._train_fallback(df_stats)

        X, y = self._run_elo(df_matches)
        n = len(y)
        log.info(f"  {n} partidos reales | Elo calculado para {len(self.team_elo)} equipos")

        if n < config.MIN_MATCHES_FOR_ML or len(np.unique(y)) < 2:
            log.warning(f"  Solo {n} partidos (<{config.MIN_MATCHES_FOR_ML}) — fallback.")
            return self._train_fallback(df_stats)

        # ── CV cronológico (sin fuga: el orden ya es temporal) ──
        n_splits = max(2, min(config.TIME_SERIES_SPLITS, n // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, lls = [], [], []
        for tr, val in tscv.split(X):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[val])) < 2:
                continue
            mdl = LogisticRegression(max_iter=1000)
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[val])[:, 1]
            aucs.append(roc_auc_score(y[val], p))
            accs.append(accuracy_score(y[val], (p >= 0.5).astype(int)))
            lls.append(log_loss(y[val], p, labels=[0, 1]))

        # ── Modelo final sobre todos los partidos ──
        self.model = LogisticRegression(max_iter=1000).fit(X, y)
        self.is_trained = True
        self.active_mode = self.MODE_ELO

        self.cv_metrics = {
            "mode":     self.MODE_ELO,
            "features": 2,
            "matches":  n,
            "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "auc_std":  round(float(np.std(aucs)), 4) if aucs else None,
            "acc_mean": round(float(np.mean(accs)), 4) if accs else None,
            "logloss":  round(float(np.mean(lls)), 4) if lls else None,
        }
        log.info(f"  ✅ Entrenado (Elo) | AUC={self.cv_metrics['auc_mean']} "
                 f"| acc={self.cv_metrics['acc_mean']}")
        return self.cv_metrics

    def _train_fallback(self, df_stats: pd.DataFrame) -> dict:
        """Fallback con muy pocos partidos: logística sobre la diferencia de
        win_rate. Sin AUC creíble (datos insuficientes)."""
        self.active_mode = self.MODE_FALLBACK
        if df_stats is None or df_stats.empty or "win_rate" not in df_stats.columns:
            # Sin nada: modelo trivial 50/50
            self.model = None
            self.is_trained = True
            self.cv_metrics = {"mode": self.MODE_FALLBACK, "features": 0,
                               "matches": 0, "auc_mean": None}
            return self.cv_metrics

        # Asignar un "elo" proporcional al win_rate para poder predecir algo.
        for r in df_stats.to_dict("records"):
            if pd.notna(r.get("team_id")):
                tid = int(r["team_id"])
                self.team_elo[tid] = config.ELO_BASE + (float(r.get("win_rate", 0.5)) - 0.5) * 800
                self.team_form[tid] = float(r.get("win_rate", 0.5))

        wr = df_stats["win_rate"].fillna(0.5).to_numpy()
        X = np.column_stack([(wr - wr.mean()) * 800, np.zeros(len(wr))])
        y = (wr >= np.median(wr)).astype(int)
        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]
        self.model = LogisticRegression(max_iter=1000).fit(X, y)
        self.is_trained = True
        self.cv_metrics = {"mode": self.MODE_FALLBACK, "features": 1,
                           "matches": 0, "auc_mean": None}
        return self.cv_metrics

    # ─────────────────────────────────────────
    #  PREDICCIÓN
    # ─────────────────────────────────────────
    def predict_match(self, stats_a: dict, stats_b: dict, side_a: str = "blue") -> dict:
        """
        Probabilidad de victoria de A usando el Elo y la forma actuales de
        cada equipo, con un pequeño ajuste heurístico de lado del mapa.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a .train() primero.")

        ida = int(stats_a.get("team_id", -1) or -1)
        idb = int(stats_b.get("team_id", -1) or -1)
        ra = self.team_elo.get(ida, config.ELO_BASE)
        rb = self.team_elo.get(idb, config.ELO_BASE)
        fa = self.team_form.get(ida, 0.5)
        fb = self.team_form.get(idb, 0.5)

        if self.model is not None:
            X = np.array([[ra - rb, fa - fb]], dtype=float)
            prob_a = float(self.model.predict_proba(X)[0, 1])
        else:
            # Sin modelo: probabilidad Elo pura
            prob_a = self._expected(ra, rb)

        # Ajuste heurístico de lado (blue gana ligeramente más)
        prob_a += config.BLUE_SIDE_BONUS if side_a.lower() == "blue" else -config.BLUE_SIDE_BONUS
        prob_a = float(np.clip(prob_a, 0.05, 0.95))

        return {
            "team_a":     stats_a.get("team_name", "Equipo A"),
            "team_b":     stats_b.get("team_name", "Equipo B"),
            "prob_a":     round(prob_a, 4),
            "prob_b":     round(1 - prob_a, 4),
            "winner":     stats_a.get("team_name") if prob_a >= 0.5 else stats_b.get("team_name"),
            "confidence": round(max(prob_a, 1 - prob_a), 4),
            "elo_a":      round(ra, 1),
            "elo_b":      round(rb, 1),
            "side_a":     side_a.upper(),
            "mode":       self.active_mode,
        }

    def rating_table(self) -> pd.DataFrame:
        """Tabla de equipos ordenada por Elo (para mostrar/depurar)."""
        rows = [{"team_id": t, "team_name": self.team_name.get(t, f"Team_{t}"),
                 "elo": round(e, 1), "form": round(self.team_form.get(t, 0.5), 3)}
                for t, e in self.team_elo.items()]
        return (pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
                if rows else pd.DataFrame())

    # ─────────────────────────────────────────
    #  PERSISTENCIA
    # ─────────────────────────────────────────
    def save(self, path: str = "model.joblib"):
        joblib.dump({
            "model": self.model, "active_mode": self.active_mode,
            "team_elo": self.team_elo, "team_form": self.team_form,
            "team_name": self.team_name, "cv_metrics": self.cv_metrics,
        }, path)

    def load(self, path: str = "model.joblib"):
        d = joblib.load(path)
        self.model = d["model"]
        self.active_mode = d["active_mode"]
        self.team_elo = d["team_elo"]
        self.team_form = d["team_form"]
        self.team_name = d.get("team_name", {})
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
