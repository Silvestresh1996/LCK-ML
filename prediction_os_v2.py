"""
╔══════════════════════════════════════════════════════════════╗
║   PREDICTION OS V2  —  Interfaz (rediseño 2026)               ║
║   Stack: customtkinter · matplotlib · scikit-learn           ║
╠══════════════════════════════════════════════════════════════╣
║  USO:   python prediction_os_v2.py   (o doble clic al .bat)  ║
║  Backend (no tocar):                                         ║
║    config.py · oracle_pipeline.py · model.py · bet_tracker.py║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import sys, os, threading, logging
from datetime import datetime
from typing import Callable

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

try:
    import customtkinter as ctk
except ImportError:
    sys.exit("ERROR: Instala customtkinter →  pip install customtkinter")

import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

# ── módulos locales (backend) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config as cfg
    from oracle_pipeline import OraclePipeline
    from model import MatchPredictor, american_to_decimal, kelly_stake
    import bet_tracker
    MODULES_OK = True
except ImportError as _err:
    MODULES_OK = False
    _IMPORT_MSG = str(_err)

# ═══════════════════════════════════════════════════════════════
#  TEMA / PALETA  (índigo + esmeralda sobre pizarra)
# ═══════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

LEAGUES: dict[str, str] = dict(cfg.LEAGUES) if MODULES_OK else {"LCK  — Corea": "LCK"}

C_BG      = "#0E1016"   # fondo de la app
C_SIDE    = "#12141C"   # sidebar
C_PANEL   = "#161A24"   # panel
C_CARD    = "#1B2030"   # tarjeta
C_CARD2   = "#232a3d"   # tarjeta elevada / hover
C_BORDER  = "#2A3145"   # bordes
C_TEXT    = "#EAecF2"   # texto principal
C_MUTED   = "#8A93A8"   # texto apagado
C_ACCENT  = "#6366F1"   # índigo (acción primaria)
C_ACCENT2 = "#818CF8"   # índigo claro (hover)
C_GREEN   = "#34D399"   # valor / positivo
C_GREEN2  = "#10B981"
C_RED     = "#F87171"   # negativo / sin valor
C_BLUE    = "#38BDF8"   # equipo B / azul
C_AMBER   = "#FBBF24"   # destacar
C_WHITE   = "#FFFFFF"
C_DARK    = "#0B0D12"   # texto sobre fondos claros

FONT = "Segoe UI"


# ═══════════════════════════════════════════════════════════════
#  PREDICTION ENGINE  (backend — sin cambios respecto a la versión previa)
# ═══════════════════════════════════════════════════════════════
class PredictionEngine:
    """Envuelve OraclePipeline + MatchPredictor para uso desde la GUI."""

    def __init__(self):
        self.api_key        = cfg.PANDASCORE_API_KEY if MODULES_OK else ""
        self.league_name    = cfg.DEFAULT_LEAGUE_NAME if MODULES_OK else "LCK  — Corea"
        self.league_code    = LEAGUES.get(self.league_name, "LCK")
        self.pipeline       = None
        self.predictor: "MatchPredictor | None" = None
        self.stats_dict: dict  = {}
        self.team_names: list  = []
        self.current_patch: str = ""
        self.model_metrics: dict = {}
        self.df_stats: pd.DataFrame = pd.DataFrame()
        self.bankroll: float   = getattr(cfg, "BANKROLL", 1000.0) if MODULES_OK else 1000.0
        self.kelly_frac: float = getattr(cfg, "KELLY_FRACTION", 0.25) if MODULES_OK else 0.25

    def set_league(self, name: str):
        self.league_name = name
        self.league_code = LEAGUES.get(name, name.split()[0].upper())

    def set_api_key(self, key: str):
        self.api_key = key.strip()

    def fetch_upcoming_matches(self, max_results: int = 40) -> list[dict]:
        """Próximos partidos (PandaScore) cuyos dos equipos están en la liga activa."""
        if not self.api_key or not self.team_names:
            return []
        import re
        import requests

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(s).lower())

        team_by_norm = {norm(n): n for n in self.team_names}
        try:
            s = requests.Session()
            s.headers.update({"Authorization": f"Bearer {self.api_key}",
                              "Accept": "application/json"})
            r = s.get("https://api.pandascore.co/lol/matches",
                      params={"filter[status]": "not_started", "sort": "begin_at",
                              "per_page": 50}, timeout=15)
            data = r.json() if r.ok else []
        except Exception:
            return []

        out, seen = [], set()
        for m in data if isinstance(data, list) else []:
            opps = m.get("opponents", [])
            if len(opps) < 2:
                continue
            na = (opps[0].get("opponent") or {}).get("name")
            nb = (opps[1].get("opponent") or {}).get("name")
            oa = team_by_norm.get(norm(na)) if na else None
            ob = team_by_norm.get(norm(nb)) if nb else None
            if oa and ob and oa != ob and (oa, ob) not in seen:
                seen.add((oa, ob))
                out.append({"name_a": oa, "name_b": ob, "begin_at": m.get("begin_at", "")})
                if len(out) >= max_results:
                    break
        return out

    def load_league_data(self, progress_cb: Callable[[str, float], None] | None = None) -> bool:
        def cb(msg: str, pct: float = 0.0):
            if progress_cb:
                progress_cb(msg, pct)

        if not MODULES_OK:
            cb(f"ERROR: módulo faltante → {_IMPORT_MSG}", 1.0)
            return False

        df_matches = pd.DataFrame()
        df_stats = pd.DataFrame()
        league_code = self.league_code

        try:
            pipeline = OraclePipeline(league_code=league_code)
            cb("Descargando base de datos…", 0.10)
            games = pipeline.load_games(progress_cb=lambda m: cb(m, 0.35))
            self.current_patch = pipeline.current_patch
            cb(f"Parche: {self.current_patch or 'N/A'}", 0.55)
            if not games.empty:
                cb("Calculando KPIs por equipo…", 0.62)
                df_stats = pipeline.build_team_stats(
                    games, min_games=getattr(cfg, "MIN_GAMES_PER_TEAM", 3))
                df_matches = pipeline.build_matches(games)
            self.pipeline = pipeline
        except Exception as exc:
            cb(f"ERROR cargando datos: {exc}", 1.0)
            return False

        if df_stats.empty:
            cb(f"ERROR: sin datos para {league_code}. Revisa tu conexión.", 1.0)
            return False

        cb(f"KPIs listos: {len(df_stats)} equipos", 0.72)
        cb("Entrenando modelo…", 0.82)
        predictor = MatchPredictor()
        try:
            metrics = predictor.train(df_stats, df_matches)
            self.model_metrics = metrics
            cb(f"Modelo OK — AUC: {metrics.get('auc_mean', 'N/A')}  Modo: {metrics.get('mode')}", 0.95)
        except Exception as exc:
            cb(f"Advertencia modelo: {exc}", 0.90)
            predictor.is_trained = False
            self.model_metrics = {}

        self.predictor  = predictor
        self.df_stats   = df_stats
        self.stats_dict = {row["team_name"]: row for row in df_stats.to_dict("records")}
        self.team_names = sorted(self.stats_dict.keys())
        cb("✅  Sistema listo", 1.0)
        return True

    def predict_match(self, name_a: str, name_b: str, side_a: str,
                      odd_a_am: int, odd_b_am: int) -> dict:
        stats_a = self.stats_dict.get(name_a, _fallback_stats(name_a))
        stats_b = self.stats_dict.get(name_b, _fallback_stats(name_b))

        if self.predictor and self.predictor.is_trained:
            pred   = self.predictor.predict_match(stats_a, stats_b, side_a=side_a)
            prob_a = pred["prob_a"]
            prob_b = pred["prob_b"]
            mode   = pred.get("mode", "model")
        else:
            prob_a, prob_b = _fallback_probs(stats_a, stats_b)
            mode = "win_rate"

        dec_a = american_to_decimal(odd_a_am)
        dec_b = american_to_decimal(odd_b_am)
        k_a   = kelly_stake(prob_a, dec_a, bankroll=self.bankroll, fraction=self.kelly_frac)
        k_b   = kelly_stake(prob_b, dec_b, bankroll=self.bankroll, fraction=self.kelly_frac)

        return {
            "name_a": name_a, "name_b": name_b,
            "prob_a": prob_a, "prob_b": prob_b,
            "dec_a": dec_a,   "dec_b": dec_b,
            "odd_a_am": odd_a_am, "odd_b_am": odd_b_am,
            "kelly_a": k_a, "kelly_b": k_b,
            "mode": mode,
            "wr_a": stats_a.get("win_rate", 0.5),
            "wr_b": stats_b.get("win_rate", 0.5),
        }


def _fallback_stats(name: str) -> dict:
    return {"team_name": name, "win_rate": 0.5, "gold_diff_15": 0,
            "baron_control_rate": 0.5, "avg_game_duration": 32,
            "gold_lead_20_weight": 0, "team_id": 0}


def _fallback_probs(a: dict, b: dict) -> tuple[float, float]:
    wa, wb = a.get("win_rate", 0.5), b.get("win_rate", 0.5)
    total  = wa + wb if (wa + wb) > 0 else 1
    pa     = float(np.clip(wa / total, 0.1, 0.9))
    return pa, 1 - pa


# ═══════════════════════════════════════════════════════════════
#  WIDGETS REUTILIZABLES
# ═══════════════════════════════════════════════════════════════
class StatCard(ctk.CTkFrame):
    """Tarjeta de métrica: etiqueta · valor grande · subtítulo."""

    def __init__(self, master, label: str, value: str = "—", sub: str = "",
                 accent: str = C_TEXT, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=14,
                         border_width=1, border_color=C_BORDER, **kw)
        ctk.CTkLabel(self, text=label.upper(), font=(FONT, 10, "bold"),
                     text_color=C_MUTED).pack(padx=16, pady=(14, 0), anchor="w")
        self._vl = ctk.CTkLabel(self, text=value, font=(FONT, 26, "bold"),
                                text_color=accent)
        self._vl.pack(padx=16, pady=(2, 0), anchor="w")
        self._sl = ctk.CTkLabel(self, text=sub, font=(FONT, 10), text_color=C_MUTED)
        self._sl.pack(padx=16, pady=(0, 14), anchor="w")

    def set(self, value: str, sub: str | None = None, accent: str | None = None):
        self._vl.configure(text=value)
        if accent:
            self._vl.configure(text_color=accent)
        if sub is not None:
            self._sl.configure(text=sub)


class ProbBar(ctk.CTkFrame):
    """Barra horizontal de probabilidad A vs B (sin matplotlib, limpia)."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._top = ctk.CTkFrame(self, fg_color="transparent")
        self._top.pack(fill="x", padx=4)
        self._lbl_a = ctk.CTkLabel(self._top, text="—", font=(FONT, 13, "bold"),
                                   text_color=C_ACCENT2)
        self._lbl_a.pack(side="left")
        self._lbl_b = ctk.CTkLabel(self._top, text="—", font=(FONT, 13, "bold"),
                                   text_color=C_BLUE)
        self._lbl_b.pack(side="right")

        self._bar = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10, height=34)
        self._bar.pack(fill="x", padx=4, pady=(6, 4))
        self._bar.pack_propagate(False)
        self._fill_a = ctk.CTkFrame(self._bar, fg_color=C_ACCENT, corner_radius=10)
        self._fill_a.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        self._fill_b = ctk.CTkFrame(self._bar, fg_color=C_BLUE, corner_radius=10)
        self._fill_b.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        self._pct_a = ctk.CTkLabel(self._bar, text="50%", font=(FONT, 12, "bold"),
                                   text_color=C_WHITE, fg_color="transparent")
        self._pct_a.place(relx=0.02, rely=0.5, anchor="w")
        self._pct_b = ctk.CTkLabel(self._bar, text="50%", font=(FONT, 12, "bold"),
                                   text_color=C_WHITE, fg_color="transparent")
        self._pct_b.place(relx=0.98, rely=0.5, anchor="e")

    def reset(self):
        self.update(0.5, "Equipo A", "Equipo B")

    def update(self, prob_a: float, name_a: str, name_b: str):
        p = float(np.clip(prob_a, 0.02, 0.98))
        self._lbl_a.configure(text=f"{name_a[:18]}")
        self._lbl_b.configure(text=f"{name_b[:18]}")
        self._fill_a.place_configure(relwidth=p)
        self._fill_b.place_configure(relx=p, relwidth=1 - p)
        self._pct_a.configure(text=f"{p*100:.0f}%")
        self._pct_b.configure(text=f"{(1-p)*100:.0f}%")


class RankingChart(ctk.CTkFrame):
    """Gráfica de barras horizontales (matplotlib) para ranking por win rate."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=14,
                         border_width=1, border_color=C_BORDER, **kw)
        self._fig = Figure(figsize=(5.2, 3.6), dpi=96, facecolor=C_CARD)
        self._ax  = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().configure(bg=C_CARD, highlightthickness=0)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._placeholder()

    def _style(self):
        ax = self._ax
        ax.set_facecolor(C_CARD)
        self._fig.patch.set_facecolor(C_CARD)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=C_MUTED, labelsize=8, length=0)

    def _placeholder(self):
        self._ax.clear(); self._style()
        self._ax.text(0.5, 0.5, "Carga datos para ver el ranking",
                      ha="center", va="center", color=C_MUTED, fontsize=10)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._canvas.draw()

    def update(self, df: pd.DataFrame):
        if df.empty or "win_rate" not in df.columns:
            self._placeholder(); return
        self._ax.clear(); self._style()
        top = df.nlargest(min(10, len(df)), "win_rate").sort_values("win_rate")
        names  = [str(n)[:16] for n in top["team_name"]]
        values = (top["win_rate"] * 100).tolist()
        colors = [C_GREEN if v >= 60 else C_ACCENT if v >= 45 else C_RED for v in values]
        bars = self._ax.barh(names, values, color=colors, height=0.66, zorder=2)
        self._ax.set_xlim(0, 108)
        self._ax.axvline(50, color=C_BORDER, lw=1, ls="--", alpha=0.7, zorder=1)
        for bar, val in zip(bars, values):
            self._ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                          f"{val:.0f}%", va="center", color=C_MUTED, fontsize=8)
        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()


def _primary_btn(master, text, command, **kw):
    return ctk.CTkButton(master, text=text, command=command,
                         fg_color=C_ACCENT, hover_color=C_ACCENT2, text_color=C_WHITE,
                         font=(FONT, 13, "bold"), corner_radius=10, height=40, **kw)


def _ghost_btn(master, text, command, **kw):
    return ctk.CTkButton(master, text=text, command=command,
                         fg_color=C_CARD2, hover_color=C_BORDER, text_color=C_TEXT,
                         font=(FONT, 12), corner_radius=10, height=38, **kw)


def _combo(master, values, **kw):
    return ctk.CTkComboBox(master, values=values, fg_color=C_PANEL, border_color=C_BORDER,
                           button_color=C_ACCENT, button_hover_color=C_ACCENT2,
                           dropdown_fg_color=C_PANEL, dropdown_hover_color=C_CARD2,
                           font=(FONT, 12), **kw)


def _entry(master, placeholder="", **kw):
    return ctk.CTkEntry(master, placeholder_text=placeholder, fg_color=C_PANEL,
                        border_color=C_BORDER, font=(FONT, 12), **kw)


# ═══════════════════════════════════════════════════════════════
#  FRAME 1 · DASHBOARD
# ═══════════════════════════════════════════════════════════════
class DashboardFrame(ctk.CTkFrame):
    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._build()

    def _build(self):
        _header(self, "Dashboard", "Resumen de la liga y del modelo")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(0, 14))
        row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._c_teams = StatCard(row, "Equipos", "—", "con datos")
        self._c_games = StatCard(row, "Partidos", "—", "reales analizados")
        self._c_auc   = StatCard(row, "Precisión (AUC)", "—", "validación honesta", accent=C_GREEN)
        self._c_patch = StatCard(row, "Parche", "—", "más reciente", accent=C_AMBER)
        for i, c in enumerate([self._c_teams, self._c_games, self._c_auc, self._c_patch]):
            c.grid(row=0, column=i, sticky="nsew", padx=(0, 12 if i < 3 else 0))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        ctk.CTkLabel(left, text="Ranking · win rate", font=(FONT, 12, "bold"),
                     text_color=C_MUTED).pack(anchor="w", pady=(0, 6))
        self._chart = RankingChart(left)
        self._chart.pack(fill="both", expand=True)

        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=14,
                             border_width=1, border_color=C_BORDER, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="ESTADO DEL SISTEMA", font=(FONT, 11, "bold"),
                     text_color=C_MUTED).pack(anchor="w", padx=16, pady=(16, 8))
        self._rows: dict[str, ctk.CTkLabel] = {}
        for key in ["Liga", "Parche", "Modelo", "Features", "Equipos", "Partidos"]:
            r = ctk.CTkFrame(right, fg_color="transparent")
            r.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(r, text=key, font=(FONT, 11), text_color=C_MUTED,
                         width=80, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(r, text="—", font=(FONT, 11, "bold"), text_color=C_TEXT)
            lbl.pack(side="left")
            self._rows[key] = lbl

        ctk.CTkFrame(right, fg_color=C_BORDER, height=1).pack(fill="x", padx=16, pady=10)
        self._tip = ctk.CTkLabel(
            right, text="Consejo: apuesta solo donde el AUC sea alto.\n"
                        "Registra cada apuesta para medir tu ganancia real.",
            font=(FONT, 10), text_color=C_MUTED, justify="left", wraplength=220)
        self._tip.pack(anchor="w", padx=16, pady=(0, 16))

    def refresh(self, engine: PredictionEngine):
        m = engine.model_metrics
        auc = m.get("auc_mean")
        auc_str = f"{auc:.2f}" if isinstance(auc, (int, float)) else "N/A"
        auc_col = C_GREEN if isinstance(auc, (int, float)) and auc >= 0.65 else (
            C_AMBER if isinstance(auc, (int, float)) and auc >= 0.55 else C_RED)
        self._c_teams.set(str(len(engine.team_names)), "con datos")
        self._c_games.set(str(m.get("matches", 0)), "reales analizados")
        self._c_auc.set(auc_str, f"modo {m.get('mode', '?')}", accent=auc_col)
        self._c_patch.set(engine.current_patch or "N/A", "más reciente")
        self._chart.update(engine.df_stats)

        self._rows["Liga"].configure(text=engine.league_name.split("—")[0].strip())
        self._rows["Parche"].configure(text=engine.current_patch or "—")
        self._rows["Modelo"].configure(text=m.get("mode", "—"))
        self._rows["Features"].configure(text=str(m.get("features", "—")))
        self._rows["Equipos"].configure(text=str(len(engine.team_names)))
        self._rows["Partidos"].configure(text=str(m.get("matches", "—")))


# ═══════════════════════════════════════════════════════════════
#  FRAME 2 · MATCH ANALYZER
# ═══════════════════════════════════════════════════════════════
class AnalyzerFrame(ctk.CTkFrame):
    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._result: dict = {}
        self._build()

    def _build(self):
        _header(self, "Analizador de partido", "Calcula la ventaja real frente a la casa")
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # ── Formulario ──
        form = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=14,
                            border_width=1, border_color=C_BORDER, width=320)
        form.pack(side="left", fill="y", padx=(0, 14))
        form.pack_propagate(False)

        def _sec(t):
            ctk.CTkLabel(form, text=t.upper(), font=(FONT, 10, "bold"),
                         text_color=C_MUTED).pack(padx=18, anchor="w", pady=(14, 4))

        ctk.CTkLabel(form, text="Configurar", font=(FONT, 14, "bold"),
                     text_color=C_TEXT).pack(padx=18, pady=(16, 0), anchor="w")

        _sec("Equipo A")
        self._cb_a = _combo(form, ["— carga datos —"], width=284, command=self._auto)
        self._cb_a.pack(padx=18)
        _sec("Equipo B")
        self._cb_b = _combo(form, ["— carga datos —"], width=284, command=self._auto)
        self._cb_b.pack(padx=18)

        _sec("Lado del mapa del Equipo A")
        self._side = ctk.CTkSegmentedButton(
            form, values=["A en Blue", "A en Red"],
            fg_color=C_PANEL, selected_color=C_ACCENT, selected_hover_color=C_ACCENT2,
            unselected_color=C_PANEL, unselected_hover_color=C_CARD2,
            text_color=C_TEXT, font=(FONT, 11, "bold"), command=lambda _v: self._auto())
        self._side.set("A en Blue")
        self._side.pack(padx=18, fill="x")
        ctk.CTkLabel(form, text="Blue side gana ~2% más (se recalcula al cambiar).",
                     font=(FONT, 9), text_color=C_MUTED).pack(padx=18, anchor="w", pady=(3, 0))

        _sec("Momio Equipo A  (ej. -160)")
        self._odd_a = _entry(form, "-160", width=284)
        self._odd_a.pack(padx=18)
        self._odd_a.bind("<KeyRelease>", lambda _e: self._auto())
        _sec("Momio Equipo B  (ej. +140)")
        self._odd_b = _entry(form, "+140", width=284)
        self._odd_b.pack(padx=18)
        self._odd_b.bind("<KeyRelease>", lambda _e: self._auto())

        _primary_btn(form, "⚡  Analizar", self._run, width=284).pack(padx=18, pady=(18, 6))
        self._err = ctk.CTkLabel(form, text="", text_color=C_RED, font=(FONT, 10),
                                 wraplength=280)
        self._err.pack(padx=18)

        # ── Resultados ──
        res = ctk.CTkFrame(body, fg_color="transparent")
        res.pack(side="right", fill="both", expand=True)

        gauge = ctk.CTkFrame(res, fg_color=C_CARD, corner_radius=14,
                             border_width=1, border_color=C_BORDER)
        gauge.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(gauge, text="PROBABILIDAD DE VICTORIA (modelo)", font=(FONT, 10, "bold"),
                     text_color=C_MUTED).pack(anchor="w", padx=16, pady=(14, 2))
        self._bar = ProbBar(gauge)
        self._bar.pack(fill="x", padx=12, pady=(0, 14))

        krow = ctk.CTkFrame(res, fg_color="transparent")
        krow.pack(fill="x", pady=(0, 12))
        krow.grid_columnconfigure((0, 1), weight=1)
        self._kc_a = _PickCard(krow, C_ACCENT2)
        self._kc_a.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._kc_b = _PickCard(krow, C_BLUE)
        self._kc_b.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        bottom = ctk.CTkFrame(res, fg_color=C_CARD, corner_radius=14,
                              border_width=1, border_color=C_BORDER)
        bottom.pack(fill="x")
        inner = ctk.CTkFrame(bottom, fg_color="transparent")
        inner.pack(fill="x", padx=8, pady=10)
        self._reg_btn = ctk.CTkButton(
            inner, text="📒  Registrar apuesta de valor", command=self._register_bet,
            fg_color=C_GREEN2, hover_color=C_GREEN, text_color=C_DARK,
            font=(FONT, 12, "bold"), corner_radius=10, height=38, state="disabled")
        self._reg_btn.pack(side="left", padx=8)
        self._reg_msg = ctk.CTkLabel(inner, text="Analiza un partido para ver oportunidades.",
                                     font=(FONT, 11), text_color=C_MUTED)
        self._reg_msg.pack(side="left", padx=10)

    def refresh_teams(self):
        names = self.engine.team_names or ["— carga datos —"]
        self._cb_a.configure(values=names)
        self._cb_b.configure(values=names)
        if len(names) >= 2:
            self._cb_a.set(names[0]); self._cb_b.set(names[1])

    def _side_value(self) -> str:
        return "blue" if self._side.get() == "A en Blue" else "red"

    def _auto(self, *_):
        """Re-analiza en vivo si ya hubo un primer análisis (incluye cambio de Lado)."""
        if self._result:
            self._run(silent=True)

    def _run(self, silent: bool = False):
        self._err.configure(text="")
        na, nb = self._cb_a.get().strip(), self._cb_b.get().strip()
        if na == nb:
            if not silent:
                self._err.configure(text="Los equipos no pueden ser el mismo.")
            return
        if "carga" in na.lower():
            if not silent:
                self._err.configure(text="Primero carga los datos de la liga.")
            return
        try:
            oa = int((self._odd_a.get().strip() or "-160").replace("+", ""))
            ob = int((self._odd_b.get().strip() or "+140").replace("+", ""))
            if abs(oa) < 100 or abs(ob) < 100:
                raise ValueError
        except ValueError:
            if not silent:
                self._err.configure(text="Momios inválidos. Usa formato americano: -160 o +140")
            return

        r = self.engine.predict_match(na, nb, self._side_value(), oa, ob)
        self._show(r)

    def _show(self, r: dict):
        self._result = r
        self._bar.update(r["prob_a"], r["name_a"], r["name_b"])
        self._kc_a.update(r["name_a"], r["kelly_a"], r["dec_a"], r["odd_a_am"], r["prob_a"])
        self._kc_b.update(r["name_b"], r["kelly_b"], r["dec_b"], r["odd_b_am"], r["prob_b"])

        has_value = r["kelly_a"].get("is_value") or r["kelly_b"].get("is_value")
        self._reg_btn.configure(state="normal" if has_value else "disabled")
        self._reg_msg.configure(
            text="🥇 Hay valor — puedes registrar la apuesta." if has_value
            else "Sin ventaja suficiente en este partido (necesitas edge > 7%).",
            text_color=C_GREEN if has_value else C_MUTED)

    def _register_bet(self):
        r = self._result
        if not r:
            return
        ka, kb = r["kelly_a"], r["kelly_b"]
        if ka.get("is_value") and not kb.get("is_value"):
            pick_a = True
        elif kb.get("is_value") and not ka.get("is_value"):
            pick_a = False
        else:
            pick_a = ka.get("edge_pct", -99) >= kb.get("edge_pct", -99)

        k    = ka if pick_a else kb
        pick = r["name_a"] if pick_a else r["name_b"]
        prob = r["prob_a"] if pick_a else r["prob_b"]
        odd  = r["odd_a_am"] if pick_a else r["odd_b_am"]
        dec  = r["dec_a"] if pick_a else r["dec_b"]
        side_a = self._side_value()
        lado = side_a if pick_a else ("red" if side_a == "blue" else "blue")

        bet_tracker.add_bet(
            liga=self.engine.league_name.split("—")[0].strip(),
            partido=f"{r['name_a']} vs {r['name_b']}",
            pick=pick, lado=lado, prob_modelo=prob, momio=odd,
            cuota=dec, edge_pct=k.get("edge_pct", 0), stake_mxn=k.get("stake_mxn", 0))
        self._reg_msg.configure(text=f"✓ Registrada: {pick} (${k.get('stake_mxn', 0):,.0f} MXN)",
                                text_color=C_GREEN)
        self._reg_btn.configure(state="disabled", text="Registrada ✓")


class _PickCard(ctk.CTkFrame):
    """Tarjeta de resultado Kelly por equipo."""

    def __init__(self, master, color, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=14,
                         border_width=1, border_color=C_BORDER, **kw)
        self._color = color
        self._name = ctk.CTkLabel(self, text="—", font=(FONT, 12, "bold"), text_color=color)
        self._name.pack(padx=16, pady=(14, 0), anchor="w")
        self._sig = ctk.CTkLabel(self, text="—", font=(FONT, 16, "bold"), text_color=C_MUTED)
        self._sig.pack(padx=16, pady=(2, 0), anchor="w")
        self._detail = ctk.CTkLabel(self, text="—", font=(FONT, 11), text_color=C_MUTED,
                                    justify="left")
        self._detail.pack(padx=16, pady=(6, 14), anchor="w")

    def update(self, name, k, dec, odd_am, prob):
        self._name.configure(text=name[:18])
        is_v = k.get("is_value", False)
        self._sig.configure(text="🥇 VALOR" if is_v else "Sin ventaja",
                            text_color=C_GREEN if is_v else C_MUTED)
        edge = k.get("edge_pct", 0); impl = k.get("implied_prob_pct", 0)
        stake = k.get("stake_mxn", 0); ev = k.get("ev_mxn", 0)
        if is_v:
            txt = (f"Modelo {prob*100:.0f}%  ·  casa {impl:.0f}%\n"
                   f"Edge {edge:+.1f}%   ·   cuota {dec:.2f} ({odd_am:+d})\n"
                   f"Apostar  ${stake:,.0f}   ·   EV  +${ev:,.0f} MXN")
        else:
            txt = (f"Modelo {prob*100:.0f}%  ·  casa {impl:.0f}%\n"
                   f"Edge {edge:+.1f}%   (necesitas > 7%)\n"
                   f"Cuota {dec:.2f} ({odd_am:+d})")
        self._detail.configure(text=txt)


# ═══════════════════════════════════════════════════════════════
#  FRAME 3 · VALUE BET SCANNER
# ═══════════════════════════════════════════════════════════════
class ValueBetsFrame(ctk.CTkFrame):
    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._rows: list[dict] = []
        self._build()

    def _build(self):
        _header(self, "Scanner de value bets",
                "Carga la jornada, escribe las cuotas de Codere y escanea")

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 12))
        _primary_btn(bar, "📅  Cargar partidos del día", self._load_today).pack(side="left", padx=(0, 8))
        _ghost_btn(bar, "+ Agregar manual", self._add_dialog).pack(side="left", padx=(0, 8))
        ctk.CTkButton(bar, text="🔍  Escanear todo", command=self._scan,
                      fg_color=C_GREEN2, hover_color=C_GREEN, text_color=C_DARK,
                      font=(FONT, 13, "bold"), corner_radius=10, height=40
                      ).pack(side="left", padx=(0, 8))
        _ghost_btn(bar, "Limpiar", self._clear).pack(side="left")

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C_CARD, corner_radius=14)
        self._scroll.pack(fill="both", expand=True)
        hdr = ctk.CTkFrame(self._scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        for t, w in [("Partido", 175), ("Momio A", 76), ("Momio B", 76),
                     ("Prob.", 64), ("Edge", 70), ("Stake", 78), ("Señal", 120)]:
            ctk.CTkLabel(hdr, text=t, font=(FONT, 10, "bold"), text_color=C_MUTED,
                         width=w, anchor="w").pack(side="left", padx=4)
        ctk.CTkFrame(self._scroll, fg_color=C_BORDER, height=1).pack(fill="x", padx=10)
        self._rf = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._rf.pack(fill="both", expand=True)
        self._sum = ctk.CTkLabel(self, text="", font=(FONT, 12), text_color=C_MUTED)
        self._sum.pack(pady=8)

    def _add_dialog(self):
        if not self.engine.team_names:
            messagebox.showwarning("Sin datos", "Carga primero los datos de la liga.")
            return
        _MatchDialog(self, self.engine.team_names, self._receive)

    def _receive(self, data: dict):
        self._rows.append(data); self._render()

    def _load_today(self):
        if not self.engine.team_names:
            messagebox.showwarning("Sin datos", "Carga primero los datos de la liga.")
            return
        fx = self.engine.fetch_upcoming_matches()
        if not fx:
            messagebox.showinfo(
                "Sin partidos próximos",
                "No encontré partidos próximos para esta liga.\n\n"
                "Puede que no haya jornada ahora, o que falte tu API Key de PandaScore "
                "en Configuración (se usa SOLO para el calendario; los datos y el modelo "
                "son de Oracle's Elixir).")
            return
        existing = {(r["name_a"], r["name_b"]) for r in self._rows}
        added = 0
        for f in fx:
            if (f["name_a"], f["name_b"]) in existing:
                continue
            self._rows.append({"name_a": f["name_a"], "name_b": f["name_b"],
                               "odd_a": None, "odd_b": None, "side": "blue"})
            added += 1
        self._render()
        self._sum.configure(text=f"Cargué {added} partido(s). Escribe los momios de Codere "
                                 f"y pulsa «Escanear todo».")

    def _remove(self, row):
        if row in self._rows:
            self._rows.remove(row)
        self._render()

    @staticmethod
    def _parse(txt):
        try:
            v = int(str(txt).replace("+", "").replace(" ", ""))
            return v if abs(v) >= 100 else None
        except (ValueError, TypeError):
            return None

    def _scan(self):
        if not self._rows:
            messagebox.showinfo("Sin partidos", "Agrega o carga partidos primero.")
            return
        for row in self._rows:
            ea, eb = row.get("_ea"), row.get("_eb")
            if ea is not None and eb is not None:
                row["odd_a"] = self._parse(ea.get())
                row["odd_b"] = self._parse(eb.get())
        for row in self._rows:
            row.pop("result", None); row.pop("error", None)
            if row.get("odd_a") is None or row.get("odd_b") is None:
                row["error"] = "Momios inválidos (ej: -150 / +130)"
                continue
            try:
                row["result"] = self.engine.predict_match(
                    row["name_a"], row["name_b"], row.get("side", "blue"),
                    row["odd_a"], row["odd_b"])
            except Exception as e:
                row["error"] = str(e)
        self._render()

    def _render(self):
        for w in self._rf.winfo_children():
            w.destroy()
        tot_stake = tot_ev = n_val = 0
        for row in self._rows:
            r = row.get("result")
            line = ctk.CTkFrame(self._rf, fg_color=C_PANEL, corner_radius=10)
            line.pack(fill="x", padx=10, pady=3)

            def _l(text, color=C_TEXT, w=None):
                kw = dict(font=(FONT, 11), text_color=color, anchor="w")
                if w:
                    kw["width"] = w
                ctk.CTkLabel(line, text=text, **kw).pack(side="left", padx=4, pady=6)

            _l(f"{row['name_a']} vs {row['name_b']}"[:24], w=175)
            ea = _entry(line, "-150", width=70, justify="center")
            if row.get("odd_a") is not None:
                ea.insert(0, f"{row['odd_a']:+d}")
            ea.pack(side="left", padx=4, pady=6)
            eb = _entry(line, "+130", width=70, justify="center")
            if row.get("odd_b") is not None:
                eb.insert(0, f"{row['odd_b']:+d}")
            eb.pack(side="left", padx=4, pady=6)
            row["_ea"], row["_eb"] = ea, eb
            ctk.CTkButton(line, text="✕", width=26, height=26, fg_color=C_CARD,
                          hover_color=C_RED, text_color=C_MUTED,
                          command=lambda rr=row: self._remove(rr)).pack(side="right", padx=6)

            if r:
                ka, kb = r["kelly_a"], r["kelly_b"]
                a_better = ka.get("edge_pct", -999) >= kb.get("edge_pct", -999)
                bk = ka if a_better else kb
                bp = r["prob_a"] if a_better else r["prob_b"]
                edge = bk.get("edge_pct", 0); stake = bk.get("stake_mxn", 0)
                ev = bk.get("ev_mxn", 0); is_v = bk.get("is_value", False)
                _l(f"{bp*100:.0f}%", w=64)
                _l(f"{edge:+.1f}%", color=(C_GREEN if edge > 0 else C_RED), w=70)
                _l(f"${stake:,.0f}" if is_v else "—", w=78)
                _l("🥇 VALOR" if is_v else "Sin ventaja",
                   color=(C_GREEN if is_v else C_MUTED), w=120)
                if is_v:
                    n_val += 1; tot_stake += stake; tot_ev += ev
            elif row.get("error"):
                _l(row["error"], color=C_RED)

        if any(r.get("result") for r in self._rows):
            pct = tot_stake / self.engine.bankroll * 100 if self.engine.bankroll > 0 else 0
            self._sum.configure(
                text=f"Oportunidades: {n_val}/{len(self._rows)}   ·   "
                     f"Stake total: ${tot_stake:,.0f} MXN ({pct:.1f}%)   ·   "
                     f"EV esperado: +${tot_ev:,.0f} MXN")

    def _clear(self):
        self._rows.clear()
        for w in self._rf.winfo_children():
            w.destroy()
        self._sum.configure(text="")


class _MatchDialog(ctk.CTkToplevel):
    def __init__(self, master, team_names, callback):
        super().__init__(master)
        self.title("Agregar partido")
        self.geometry("400x340")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)
        self.grab_set()
        self._cb = callback

        ctk.CTkLabel(self, text="Nuevo partido", font=(FONT, 15, "bold"),
                     text_color=C_TEXT).pack(pady=(20, 14))

        def _row(label, w):
            f = ctk.CTkFrame(self, fg_color="transparent")
            f.pack(fill="x", padx=26, pady=5)
            ctk.CTkLabel(f, text=label, font=(FONT, 11), text_color=C_MUTED,
                         width=90, anchor="w").pack(side="left")
            wd = w(f); wd.pack(side="left", fill="x", expand=True)
            return wd

        self._a = _row("Equipo A:", lambda f: _combo(f, team_names))
        if team_names:
            self._a.set(team_names[0])
        self._b = _row("Equipo B:", lambda f: _combo(f, team_names))
        if len(team_names) > 1:
            self._b.set(team_names[1])
        self._oa = _row("Momio A:", lambda f: _entry(f, "-160"))
        self._ob = _row("Momio B:", lambda f: _entry(f, "+140"))
        self._err = ctk.CTkLabel(self, text="", text_color=C_RED, font=(FONT, 10))
        self._err.pack()
        _primary_btn(self, "Agregar", self._submit, width=160).pack(pady=16)

    def _submit(self):
        na, nb = self._a.get().strip(), self._b.get().strip()
        if na == nb:
            self._err.configure(text="Los equipos no pueden ser iguales."); return
        try:
            oa = int((self._oa.get() or "-160").replace("+", ""))
            ob = int((self._ob.get() or "+140").replace("+", ""))
            if abs(oa) < 100 or abs(ob) < 100:
                raise ValueError
        except ValueError:
            self._err.configure(text="Momios inválidos. Usa: -160 o +140"); return
        self._cb({"name_a": na, "name_b": nb, "odd_a": oa, "odd_b": ob, "side": "blue"})
        self.destroy()


# ═══════════════════════════════════════════════════════════════
#  FRAME 4 · REGISTRO DE APUESTAS
# ═══════════════════════════════════════════════════════════════
class RegistroFrame(ctk.CTkFrame):
    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._build()

    def _build(self):
        _header(self, "Registro de apuestas",
                "Marca cada apuesta como ganada o perdida para medir tu ganancia real")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(0, 14))
        row.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self._c_bets = StatCard(row, "Apuestas", "0", "registradas")
        self._c_win  = StatCard(row, "Aciertos", "—", "de resueltas")
        self._c_prof = StatCard(row, "Ganancia neta", "$0", "MXN", accent=C_GREEN)
        self._c_yld  = StatCard(row, "Yield", "—", "ganancia / arriesgado", accent=C_GREEN)
        self._c_pend = StatCard(row, "Pendientes", "0", "sin resultado", accent=C_AMBER)
        for i, c in enumerate([self._c_bets, self._c_win, self._c_prof, self._c_yld, self._c_pend]):
            c.grid(row=0, column=i, sticky="nsew", padx=(0, 10 if i < 4 else 0))

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C_CARD, corner_radius=14)
        self._scroll.pack(fill="both", expand=True)
        hdr = ctk.CTkFrame(self._scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        for t, w in [("Fecha", 92), ("Partido", 180), ("Apuesta", 130), ("Stake", 70),
                     ("Resultado", 196), ("Ganancia", 86), ("", 30)]:
            ctk.CTkLabel(hdr, text=t, font=(FONT, 10, "bold"), text_color=C_MUTED,
                         width=w, anchor="w").pack(side="left", padx=4)
        ctk.CTkFrame(self._scroll, fg_color=C_BORDER, height=1).pack(fill="x", padx=10)
        self._rf = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._rf.pack(fill="both", expand=True)
        self._empty = ctk.CTkLabel(self._rf, text="Aún no hay apuestas. Regístralas desde el Analizador.",
                                   font=(FONT, 11), text_color=C_MUTED)

    def refresh(self):
        for w in self._rf.winfo_children():
            w.destroy()
        s = bet_tracker.summary()
        self._c_bets.set(str(s["n_total"]), "registradas")
        self._c_win.set(f"{s['n_won']}/{s['n_settled']}" if s["n_settled"] else "—",
                        f"{s['win_rate']*100:.0f}% acierto" if s["n_settled"] else "sin resueltas")
        col = C_GREEN if s["profit"] >= 0 else C_RED
        self._c_prof.set(f"${s['profit']:+,.0f}", "MXN", accent=col)
        self._c_yld.set(f"{s['yield_pct']:+.1f}%" if s["n_settled"] else "—",
                        "ganancia / arriesgado", accent=col if s["n_settled"] else C_TEXT)
        self._c_pend.set(str(s["n_pending"]), "sin resultado")

        bets = bet_tracker.all_bets()
        if not bets:
            self._empty.pack(pady=30); return
        for b in bets:
            self._render_row(b)

    def _render_row(self, b: dict):
        estado = b.get("estado", "pendiente")
        line = ctk.CTkFrame(self._rf, fg_color=C_PANEL, corner_radius=10)
        line.pack(fill="x", padx=10, pady=3)

        def _l(text, w, color=C_TEXT):
            ctk.CTkLabel(line, text=text, font=(FONT, 11), text_color=color,
                         width=w, anchor="w").pack(side="left", padx=4, pady=7)

        _l(b.get("fecha", "")[:10], 92, C_MUTED)
        _l(b.get("partido", "")[:24], 180)
        _l(f"{b.get('pick', '')[:10]}  {b.get('cuota', '')}", 130, C_ACCENT2)
        _l(f"${float(b.get('stake_mxn', 0) or 0):,.0f}", 70)

        seg = ctk.CTkFrame(line, fg_color="transparent", width=196)
        seg.pack(side="left", padx=2)
        for est, txt, col in [("ganada", "✓ Ganó", C_GREEN), ("perdida", "✗ Perdió", C_RED),
                              ("pendiente", "Pend.", C_MUTED)]:
            active = (estado == est)
            ctk.CTkButton(seg, text=txt, width=58, height=26,
                          font=(FONT, 10, "bold" if active else "normal"),
                          fg_color=col if active else C_CARD,
                          text_color=C_DARK if active else C_MUTED, hover_color=col,
                          command=lambda i=b["id"], e=est: self._set(i, e)).pack(side="left", padx=1)

        gan = float(b.get("ganancia_mxn", 0) or 0)
        gcol = C_GREEN if gan > 0 else (C_RED if gan < 0 else C_MUTED)
        _l(f"${gan:+,.0f}" if estado != "pendiente" else "—", 86, gcol)
        ctk.CTkButton(line, text="✕", width=26, height=26, fg_color=C_CARD,
                      hover_color=C_RED, text_color=C_MUTED,
                      command=lambda i=b["id"]: self._del(i)).pack(side="left", padx=2)

    def _set(self, bet_id, estado):
        bet_tracker.set_estado(bet_id, estado); self.refresh()

    def _del(self, bet_id):
        bet_tracker.delete_bet(bet_id); self.refresh()


# ═══════════════════════════════════════════════════════════════
#  FRAME 5 · CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════
class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, engine: PredictionEngine, on_league_change: Callable, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._on_league_change = on_league_change
        self._build()

    def _build(self):
        _header(self, "Configuración", "Bankroll, Kelly, liga y API key")
        card = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=14,
                            border_width=1, border_color=C_BORDER)
        card.pack(fill="x")

        def _sec(t):
            ctk.CTkLabel(card, text=t, font=(FONT, 12, "bold"), text_color=C_ACCENT2
                         ).pack(anchor="w", padx=20, pady=(18, 6))

        def _div():
            ctk.CTkFrame(card, fg_color=C_BORDER, height=1).pack(fill="x", padx=20, pady=6)

        # Bankroll y Kelly
        _sec("💰  Bankroll y Kelly")
        f = ctk.CTkFrame(card, fg_color="transparent")
        f.pack(fill="x", padx=20)
        ctk.CTkLabel(f, text="Bankroll (MXN):", font=(FONT, 11), text_color=C_MUTED,
                     width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(f, textvariable=self._br_var, width=140, fg_color=C_PANEL,
                     border_color=C_BORDER).pack(side="left")
        self._kelly_lbl = ctk.CTkLabel(card, text=f"Fracción Kelly: {self._kf():.0f}%",
                                       font=(FONT, 11), text_color=C_MUTED)
        self._kelly_lbl.pack(anchor="w", padx=20, pady=(12, 2))
        self._slider = ctk.CTkSlider(card, from_=10, to=50, number_of_steps=8,
                                     fg_color=C_BORDER, progress_color=C_ACCENT,
                                     button_color=C_ACCENT, button_hover_color=C_ACCENT2,
                                     command=self._upd_kelly)
        self._slider.set(self.engine.kelly_frac * 100)
        self._slider.pack(fill="x", padx=20, pady=(0, 6))

        _div()
        # Liga
        _sec("🏆  Liga activa")
        self._cb_league = _combo(card, list(LEAGUES.keys()), width=300,
                                 command=lambda c: self.engine.set_league(c))
        self._cb_league.set(self.engine.league_name)
        self._cb_league.pack(anchor="w", padx=20)

        _div()
        # API key
        _sec("🔑  API Key de PandaScore  (opcional)")
        ctk.CTkLabel(card, text="Solo para «Cargar partidos del día» (calendario). "
                                "Los datos del modelo son de Oracle's Elixir, gratis.",
                     font=(FONT, 10), text_color=C_MUTED, wraplength=560,
                     justify="left").pack(anchor="w", padx=20, pady=(0, 6))
        self._e_key = ctk.CTkEntry(card, width=420, show="*", fg_color=C_PANEL,
                                   border_color=C_BORDER, placeholder_text="Pega tu API Key")
        self._e_key.pack(anchor="w", padx=20, pady=(0, 6))
        if MODULES_OK and cfg.PANDASCORE_API_KEY:
            self._e_key.insert(0, cfg.PANDASCORE_API_KEY)
        _ghost_btn(card, "Aplicar API Key", self._apply_key, width=160).pack(anchor="w", padx=20)

        _primary_btn(card, "💾  Guardar configuración", self._save, width=240
                     ).pack(anchor="w", padx=20, pady=(18, 8))
        self._status = ctk.CTkLabel(card, text="", font=(FONT, 11), text_color=C_GREEN)
        self._status.pack(anchor="w", padx=20, pady=(0, 16))

    @property
    def _br_var(self):
        if not hasattr(self, "_bankroll_var"):
            self._bankroll_var = tk.StringVar(value=str(int(self.engine.bankroll)))
        return self._bankroll_var

    def _kf(self):
        return self.engine.kelly_frac * 100

    def _upd_kelly(self, val):
        self._kelly_lbl.configure(text=f"Fracción Kelly: {float(val):.0f}%")
        self.engine.kelly_frac = float(val) / 100

    def _apply_key(self):
        key = self._e_key.get().strip()
        if not key:
            return
        self.engine.set_api_key(key)
        try:
            cfg.save_api_key(key)
            cfg.PANDASCORE_API_KEY = key
            self._status.configure(text="API Key guardada.", text_color=C_GREEN)
        except Exception as exc:
            self._status.configure(text=f"Aplicada (no se pudo guardar: {exc})", text_color=C_AMBER)

    def _save(self):
        try:
            br = float(self._br_var.get().replace(",", "").replace("$", ""))
            self.engine.bankroll = br
            self.engine.kelly_frac = self._slider.get() / 100
            name = self._cb_league.get()
            self.engine.set_league(name)
            self._status.configure(
                text=f"✅ Guardado — Bankroll ${br:,.0f} · Kelly {self.engine.kelly_frac*100:.0f}% "
                     f"· Liga {name.split('—')[0].strip()}",
                text_color=C_GREEN)
            self._on_league_change(name)
        except ValueError:
            self._status.configure(text="Bankroll inválido.", text_color=C_RED)


# ═══════════════════════════════════════════════════════════════
#  HELPER de encabezado de página
# ═══════════════════════════════════════════════════════════════
def _header(master, title, subtitle):
    ctk.CTkLabel(master, text=title, font=(FONT, 22, "bold"), text_color=C_TEXT
                 ).pack(anchor="w", pady=(0, 2))
    ctk.CTkLabel(master, text=subtitle, font=(FONT, 12), text_color=C_MUTED
                 ).pack(anchor="w", pady=(0, 16))


# ═══════════════════════════════════════════════════════════════
#  APP PRINCIPAL
# ═══════════════════════════════════════════════════════════════
class PredictionOSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Prediction OS · eSports Value Betting")
        self.geometry("1240x760")
        self.minsize(1040, 660)
        self.configure(fg_color=C_BG)

        self.engine = PredictionEngine()
        self._loading = False

        self._build_sidebar()
        self._build_content()
        self._build_statusbar()
        self.after(500, self._start_load)

    # ── Sidebar ──
    def _build_sidebar(self):
        side = ctk.CTkFrame(self, fg_color=C_SIDE, width=216, corner_radius=0)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        logo = ctk.CTkFrame(side, fg_color="transparent", height=72)
        logo.pack(fill="x", pady=(6, 0))
        logo.pack_propagate(False)
        ctk.CTkLabel(logo, text="⚡ PREDICTION OS", font=(FONT, 15, "bold"),
                     text_color=C_WHITE).pack(anchor="w", padx=20, pady=(18, 0))
        ctk.CTkLabel(logo, text="value betting · LoL", font=(FONT, 10),
                     text_color=C_MUTED).pack(anchor="w", padx=20)

        ctk.CTkFrame(side, fg_color=C_BORDER, height=1).pack(fill="x", padx=14, pady=8)

        nav = [
            ("dashboard", "📊   Dashboard",  self._go_dashboard),
            ("analyzer",  "⚔️   Analizador",  self._go_analyzer),
            ("valuebets", "💰   Value Bets",  self._go_valuebets),
            ("registro",  "📒   Registro",    self._go_registro),
            ("settings",  "⚙️   Configuración", self._go_settings),
        ]
        self._nav: dict[str, ctk.CTkButton] = {}
        for key, label, cmd in nav:
            b = ctk.CTkButton(side, text=label, anchor="w", font=(FONT, 13),
                              fg_color="transparent", text_color=C_MUTED,
                              hover_color=C_CARD, height=44, corner_radius=10, command=cmd)
            b.pack(fill="x", padx=12, pady=2)
            self._nav[key] = b

        ctk.CTkFrame(side, fg_color="transparent").pack(expand=True)
        ctk.CTkFrame(side, fg_color=C_BORDER, height=1).pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(side, text="LIGA", font=(FONT, 9, "bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=18, pady=(2, 2))
        self._league_cb = _combo(side, list(LEAGUES.keys()), height=34,
                                 command=self._on_sidebar_league)
        self._league_cb.set(self.engine.league_name)
        self._league_cb.pack(fill="x", padx=12, pady=(0, 8))

        self._load_btn = _ghost_btn(side, "⟳  Cargar / Actualizar", self._start_load)
        self._load_btn.pack(fill="x", padx=12, pady=(0, 8))

        dot = ctk.CTkFrame(side, fg_color="transparent", height=26)
        dot.pack(fill="x", padx=18, pady=(0, 12))
        self._dot = ctk.CTkLabel(dot, text="●", font=(FONT, 14), text_color=C_MUTED)
        self._dot.pack(side="left")
        self._dot_lbl = ctk.CTkLabel(dot, text=" Sin datos", font=(FONT, 10), text_color=C_MUTED)
        self._dot_lbl.pack(side="left")

    def _build_content(self):
        self._content = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True, padx=24, pady=18)
        self._frames = {
            "dashboard": DashboardFrame(self._content, self.engine),
            "analyzer":  AnalyzerFrame(self._content, self.engine),
            "valuebets": ValueBetsFrame(self._content, self.engine),
            "registro":  RegistroFrame(self._content, self.engine),
            "settings":  SettingsFrame(self._content, self.engine,
                                       on_league_change=self._on_sidebar_league),
        }
        for f in self._frames.values():
            f.pack(fill="both", expand=True)
        self._show("dashboard")

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color=C_SIDE, height=26, corner_radius=0)
        bar.pack(side="bottom", fill="x")
        self._status = ctk.CTkLabel(bar, text="Iniciando…", font=(FONT, 9), text_color=C_MUTED)
        self._status.pack(side="left", padx=14)
        ctk.CTkLabel(bar, text=f"Prediction OS · {datetime.now().year}", font=(FONT, 9),
                     text_color=C_MUTED).pack(side="right", padx=14)
        self._progress = ctk.CTkProgressBar(bar, mode="determinate", fg_color=C_BORDER,
                                            progress_color=C_ACCENT, height=6, width=220)
        self._progress.set(0)
        self._progress.pack(side="right", padx=(0, 12), pady=10)

    # ── Navegación ──
    def _show(self, name: str):
        for k, f in self._frames.items():
            (f.pack if k == name else f.pack_forget)(**({"fill": "both", "expand": True} if k == name else {}))
        for k, b in self._nav.items():
            if k == name:
                b.configure(fg_color=C_ACCENT, text_color=C_WHITE)
            else:
                b.configure(fg_color="transparent", text_color=C_MUTED)

    def _go_dashboard(self): self._show("dashboard")
    def _go_analyzer(self):  self._show("analyzer")
    def _go_valuebets(self): self._show("valuebets")
    def _go_registro(self):  self._frames["registro"].refresh(); self._show("registro")
    def _go_settings(self):  self._show("settings")

    # ── Carga de datos ──
    def _on_sidebar_league(self, name: str):
        self.engine.set_league(name)
        if self._league_cb.get() != name:
            self._league_cb.set(name)
        self._status.configure(text=f"Liga: {name.split('—')[0].strip()} — pulsa Cargar para actualizar")

    def _start_load(self):
        if self._loading:
            return
        self._loading = True
        self._load_btn.configure(state="disabled", text="Cargando…")
        self._dot.configure(text_color=C_AMBER)
        self._dot_lbl.configure(text=" Cargando…")
        self._progress.set(0)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        ok = self.engine.load_league_data(
            progress_cb=lambda m, p: self.after(0, self._progress_cb, m, p))
        self.after(0, self._done, ok)

    def _progress_cb(self, msg, pct):
        self._status.configure(text=msg)
        self._progress.set(max(0, min(1, pct)))

    def _done(self, ok: bool):
        self._loading = False
        self._load_btn.configure(state="normal", text="⟳  Cargar / Actualizar")
        if ok:
            self._dot.configure(text_color=C_GREEN)
            self._dot_lbl.configure(text=f" {len(self.engine.team_names)} equipos")
            self._frames["dashboard"].refresh(self.engine)
            self._frames["analyzer"].refresh_teams()
        else:
            self._dot.configure(text_color=C_RED)
            self._dot_lbl.configure(text=" Error")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not MODULES_OK:
        root = ctk.CTk()
        root.title("Error de inicio")
        root.geometry("520x200")
        ctk.CTkLabel(root, text=f"Faltan módulos:\n\n{_IMPORT_MSG}\n\n"
                                "Asegúrate de tener config.py, oracle_pipeline.py, model.py "
                                "y bet_tracker.py en el mismo directorio.",
                     font=(FONT, 12), justify="center", wraplength=480).pack(expand=True)
        root.mainloop()
        sys.exit(1)

    app = PredictionOSApp()
    app.mainloop()
