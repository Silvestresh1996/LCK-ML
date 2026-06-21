"""
============================================================
PREDICTION OS V2 — MOTOR DE PREDICCIÓN (Elo + KPIs pre-partido, sin fuga)
============================================================
El modelo recorre los partidos en orden cronológico y, ANTES de cada partido,
construye las features con la información que cada equipo tenía HASTA ESE
MOMENTO (nunca con datos del futuro → sin fuga de información):

  • Δelo        — diferencia de rating Elo (fuerza acumulada)
  • Δforma      — diferencia de forma reciente (últimos N partidos)
  • Δoro@15     — diferencia de oro a los 15 min (media histórica pre-partido)
  • Δfirstblood — diferencia en tasa de primera sangre
  • Δfirstdrag  — diferencia en tasa de primer dragón
  • Δbaron      — diferencia en control de barón
  • Δvspm       — diferencia en visión por minuto

Si los datos no traen stats detalladas, el modelo cae a modo "elo" (solo
Δelo y Δforma). Una regresión logística (escalada) calibra las features hacia
una probabilidad fiable — clave para que el edge y el Kelly sean correctos.

  modo "full"  → con stats detalladas (Oracle's Elixir)
  modo "elo"   → solo resultados
  modo "fallback_lr" → muy pocos partidos

Componentes:
  MatchPredictor   — Elo + KPIs pre-partido + LogisticRegression
  american_to_decimal / kelly_stake — utilidades de apuestas
============================================================
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

import config

log = logging.getLogger(__name__)

# KPIs detallados que SÍ aportan señal al modelo.
#
# Nota basada en una ablación rigurosa (LCK 2026, validación cronológica):
#   elo solo ........ AUC 0.719
#   elo + oro@15 .... AUC 0.725  ← mejor
#   elo + forma ..... AUC 0.712
#   las 7 stats ..... AUC 0.622  (sobreajuste: redundantes con el Elo)
# Solo el oro@15 añade señal independiente; el resto (primera sangre, dragón,
# barón, visión, forma) está correlacionado con el Elo y solo mete ruido.
# (Las demás stats siguen disponibles para mostrarse en la GUI, no en el modelo.)
_KPI_SPEC = [
    ("gd15", "a_gd15", "b_gd15"),
]


# ═══════════════════════════════════════════════════════════
#  MOTOR DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════
class MatchPredictor:

    MODE_FULL     = "full"
    MODE_ELO      = "elo"
    MODE_FALLBACK = "fallback_lr"

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.active_mode = self.MODE_ELO
        self.feat_names: list[str] = []
        self.cv_metrics: dict = {}
        # Estado por equipo (poblado al entrenar):
        self.team_elo: dict = {}     # {team_id → Elo final}
        self.team_form: dict = {}    # {team_id → forma reciente 0-1}
        self.team_kpi: dict = {}     # {team_id → {kpi → media histórica}}
        self.team_name: dict = {}    # {team_id → nombre}

    # ─────────────────────────────────────────
    #  ELO
    # ─────────────────────────────────────────
    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    # ─────────────────────────────────────────
    #  CONSTRUCCIÓN CRONOLÓGICA DE FEATURES (sin fuga)
    # ─────────────────────────────────────────
    def _build_chrono(self, df_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Recorre los partidos en orden y arma X (features pre-partido) e y."""
        df = df_matches.copy()
        if "begin_at" in df.columns:
            df = df.sort_values("begin_at", na_position="last")

        has_kpi = all(c in df.columns for a in _KPI_SPEC for c in (a[1], a[2]))
        self.active_mode = self.MODE_FULL if has_kpi else self.MODE_ELO
        # 'form' se calcula y guarda para mostrar, pero NO entra al modelo
        # (la ablación mostró que resta señal).
        self.feat_names = ["elo_diff"] + (
            [f"{k}_diff" for k, _, _ in _KPI_SPEC] if has_kpi else [])

        elo = defaultdict(lambda: config.ELO_BASE)
        form = defaultdict(list)
        kpi_hist = defaultdict(lambda: defaultdict(list))   # {tid: {kpi: [valores]}}
        win = config.ELO_FORM_WINDOW

        X, y = [], []
        for m in df.to_dict("records"):
            a, b, w = m.get("team_a_id"), m.get("team_b_id"), m.get("winner_id")
            if a is None or b is None or w is None or w not in (a, b):
                continue

            ra, rb = elo[a], elo[b]
            feat = [ra - rb]

            if has_kpi:
                for kpi, _, _ in _KPI_SPEC:
                    ha, hb = kpi_hist[a][kpi], kpi_hist[b][kpi]
                    va = float(np.mean(ha)) if ha else 0.0
                    vb = float(np.mean(hb)) if hb else 0.0
                    feat.append(va - vb)

            X.append(feat)
            y.append(1 if w == a else 0)

            # ── actualizar estado tras el partido ──
            ea = self._expected(ra, rb)
            sa = 1.0 if w == a else 0.0
            elo[a] = ra + config.ELO_K * (sa - ea)
            elo[b] = rb + config.ELO_K * ((1.0 - sa) - (1.0 - ea))
            form[a].append(sa)
            form[b].append(1.0 - sa)
            if has_kpi:
                for kpi, ca, cb in _KPI_SPEC:
                    kpi_hist[a][kpi].append(float(m.get(ca, 0) or 0))
                    kpi_hist[b][kpi].append(float(m.get(cb, 0) or 0))

        # ── estado final por equipo (para predecir) ──
        self.team_elo = dict(elo)
        self.team_form = {t: (float(np.mean(h[-win:])) if h else 0.5) for t, h in form.items()}
        self.team_kpi = {
            t: {kpi: (float(np.mean(vals)) if vals else 0.0) for kpi, vals in d.items()}
            for t, d in kpi_hist.items()
        }
        return np.array(X, dtype=float), np.array(y, dtype=int)

    # ─────────────────────────────────────────
    #  ENTRENAMIENTO
    # ─────────────────────────────────────────
    def train(self, df_stats: pd.DataFrame, df_matches: pd.DataFrame) -> dict:
        if df_stats is not None and not df_stats.empty and "team_id" in df_stats.columns:
            self.team_name = {r["team_id"]: r.get("team_name", str(r["team_id"]))
                              for r in df_stats.to_dict("records")}

        if df_matches is None or df_matches.empty:
            log.warning("  Sin partidos reales — fallback sobre win_rate.")
            return self._train_fallback(df_stats)

        X, y = self._build_chrono(df_matches)
        n = len(y)
        log.info(f"  {n} partidos | modo={self.active_mode} | features={self.feat_names}")

        if n < config.MIN_MATCHES_FOR_ML or len(np.unique(y)) < 2:
            log.warning(f"  Solo {n} partidos (<{config.MIN_MATCHES_FOR_ML}) — fallback.")
            return self._train_fallback(df_stats)

        Xs = self.scaler.fit_transform(X)

        # ── CV cronológico ──
        n_splits = max(2, min(config.TIME_SERIES_SPLITS, n // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, lls = [], [], []
        for tr, val in tscv.split(Xs):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[val])) < 2:
                continue
            mdl = LogisticRegression(max_iter=1000)
            mdl.fit(Xs[tr], y[tr])
            p = mdl.predict_proba(Xs[val])[:, 1]
            aucs.append(roc_auc_score(y[val], p))
            accs.append(accuracy_score(y[val], (p >= 0.5).astype(int)))
            lls.append(log_loss(y[val], p, labels=[0, 1]))

        self.model = LogisticRegression(max_iter=1000).fit(Xs, y)
        self.is_trained = True
        self.cv_metrics = {
            "mode":     self.active_mode,
            "features": len(self.feat_names),
            "matches":  n,
            "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "auc_std":  round(float(np.std(aucs)), 4) if aucs else None,
            "acc_mean": round(float(np.mean(accs)), 4) if accs else None,
            "logloss":  round(float(np.mean(lls)), 4) if lls else None,
        }
        log.info(f"  ✅ Entrenado | modo={self.active_mode} | AUC={self.cv_metrics['auc_mean']} "
                 f"| acc={self.cv_metrics['acc_mean']}")
        return self.cv_metrics

    def _train_fallback(self, df_stats: pd.DataFrame) -> dict:
        """Pocos partidos: 'elo' proporcional al win_rate + logística simple."""
        self.active_mode = self.MODE_FALLBACK
        self.feat_names = ["elo_diff"]
        if df_stats is None or df_stats.empty or "win_rate" not in df_stats.columns:
            self.model = None
            self.is_trained = True
            self.cv_metrics = {"mode": self.MODE_FALLBACK, "features": 0, "matches": 0, "auc_mean": None}
            return self.cv_metrics

        for r in df_stats.to_dict("records"):
            tid = r.get("team_id")
            self.team_elo[tid] = config.ELO_BASE + (float(r.get("win_rate", 0.5)) - 0.5) * 800
            self.team_form[tid] = float(r.get("win_rate", 0.5))

        wr = df_stats["win_rate"].fillna(0.5).to_numpy()
        X = ((wr - wr.mean()) * 800).reshape(-1, 1)
        y = (wr >= np.median(wr)).astype(int)
        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]
        self.scaler.fit(X)
        self.model = LogisticRegression(max_iter=1000).fit(self.scaler.transform(X), y)
        self.is_trained = True
        self.cv_metrics = {"mode": self.MODE_FALLBACK, "features": 1, "matches": 0, "auc_mean": None}
        return self.cv_metrics

    # ─────────────────────────────────────────
    #  PREDICCIÓN
    # ─────────────────────────────────────────
    def _feat_vector(self, ida, idb) -> np.ndarray:
        ra = self.team_elo.get(ida, config.ELO_BASE)
        rb = self.team_elo.get(idb, config.ELO_BASE)
        feat = [ra - rb]
        if self.active_mode == self.MODE_FULL:
            ka = self.team_kpi.get(ida, {})
            kb = self.team_kpi.get(idb, {})
            for kpi, _, _ in _KPI_SPEC:
                feat.append(ka.get(kpi, 0.0) - kb.get(kpi, 0.0))
        return np.array([feat], dtype=float)

    def predict_match(self, stats_a: dict, stats_b: dict, side_a: str = "blue") -> dict:
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a .train() primero.")

        ida = stats_a.get("team_id")
        idb = stats_b.get("team_id")
        ra = self.team_elo.get(ida, config.ELO_BASE)
        rb = self.team_elo.get(idb, config.ELO_BASE)

        if self.model is not None:
            X = self.scaler.transform(self._feat_vector(ida, idb))
            prob_a = float(self.model.predict_proba(X)[0, 1])
        else:
            prob_a = self._expected(ra, rb)

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
        rows = [{"team_id": t, "team_name": self.team_name.get(t, str(t)),
                 "elo": round(e, 1), "form": round(self.team_form.get(t, 0.5), 3)}
                for t, e in self.team_elo.items()]
        return (pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
                if rows else pd.DataFrame())

    # ─────────────────────────────────────────
    #  PERSISTENCIA
    # ─────────────────────────────────────────
    def save(self, path: str = "model.joblib"):
        joblib.dump({
            "model": self.model, "scaler": self.scaler, "active_mode": self.active_mode,
            "feat_names": self.feat_names, "team_elo": self.team_elo,
            "team_form": self.team_form, "team_kpi": self.team_kpi,
            "team_name": self.team_name, "cv_metrics": self.cv_metrics,
        }, path)

    def load(self, path: str = "model.joblib"):
        d = joblib.load(path)
        self.model = d["model"]; self.scaler = d["scaler"]
        self.active_mode = d["active_mode"]; self.feat_names = d["feat_names"]
        self.team_elo = d["team_elo"]; self.team_form = d["team_form"]
        self.team_kpi = d.get("team_kpi", {}); self.team_name = d.get("team_name", {})
        self.cv_metrics = d["cv_metrics"]; self.is_trained = True


# ═══════════════════════════════════════════════════════════
#  CONVERSOR DE MOMIOS + KELLY CRITERION
# ═══════════════════════════════════════════════════════════
def american_to_decimal(american: int | str) -> float:
    """Momio americano → cuota decimal.  +285→3.85 | -425→1.235 | ±100→2.00"""
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
