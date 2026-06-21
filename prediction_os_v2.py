"""
╔══════════════════════════════════════════════════════════════╗
║        PREDICTION OS V2.0  —  Dashboard GUI                  ║
║        Stack: customtkinter · matplotlib · xgboost           ║
╠══════════════════════════════════════════════════════════════╣
║  INSTALACIÓN:                                                ║
║    pip install customtkinter matplotlib                      ║
║    pip install xgboost scikit-learn pandas numpy requests    ║
║                                                              ║
║  USO:                                                        ║
║    python prediction_os_v2.py                                ║
║                                                              ║
║  ARCHIVOS REQUERIDOS (mismo directorio):                     ║
║    config.py  universal_pipeline.py  model.py               ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import sys, os, threading, logging
from datetime import datetime
from typing import Callable

# ── silenciar logs internos del pipeline en la GUI ──
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── GUI imports ──
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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── módulos locales ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config as cfg
    from oracle_pipeline import OraclePipeline
    from model import MatchPredictor, american_to_decimal, kelly_stake
    MODULES_OK = True
except ImportError as _err:
    MODULES_OK = False
    _IMPORT_MSG = str(_err)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Selector de ligas con sus IDs en PandaScore (desde config central)
LEAGUES: dict[str, int] = dict(cfg.LEAGUES) if MODULES_OK else {
    "LCK  — Korea":       293,
    "LPL  — China":       290,
    "LEC  — Europe":     4197,
    "LCS  — N. America": 4198,
}

# Paleta de colores
C_GOLD   = "#C89B3C"   # Oro League of Legends
C_GOLD2  = "#F0C060"   # Oro claro para hover
C_BG     = "#141414"   # Fondo principal
C_SIDE   = "#1a1a1a"   # Sidebar
C_PANEL  = "#1e1e1e"   # Panel secundario
C_CARD   = "#242424"   # Tarjeta
C_BORDER = "#333333"   # Borde
C_TEXT   = "#e8e8e8"   # Texto principal
C_MUTED  = "#888888"   # Texto apagado
C_GREEN  = "#2ecc71"   # Valor positivo
C_RED    = "#e74c3c"   # Sin valor / peligro
C_BLUE   = "#3498db"   # Equipo B
C_WHITE  = "#ffffff"
C_DARK2  = "#0d0d0d"


# ═══════════════════════════════════════════════════════════════
#  PREDICTION ENGINE  (backend thread-safe)
# ═══════════════════════════════════════════════════════════════
class PredictionEngine:
    """
    Envuelve UniversalPipeline + MatchPredictor para el uso desde la GUI.
    Todos los métodos pesados están pensados para correr en hilos.
    """

    def __init__(self):
        self.api_key        = cfg.PANDASCORE_API_KEY if MODULES_OK else ""
        self.league_id      = cfg.DEFAULT_LEAGUE_ID if MODULES_OK else 293
        self.league_name    = "LCK — Korea"
        self.pipeline       = None
        self.predictor: "MatchPredictor | None" = None
        self.stats_dict: dict  = {}          # {team_name → stats dict}
        self.team_names: list  = []          # lista ordenada de nombres
        self.current_patch: str = ""
        self.model_metrics: dict = {}
        self.df_stats: pd.DataFrame = pd.DataFrame()
        self.bankroll: float   = getattr(cfg, "BANKROLL", 1000.0) if MODULES_OK else 1000.0
        self.kelly_frac: float = getattr(cfg, "KELLY_FRACTION", 0.25) if MODULES_OK else 0.25

    def set_league(self, name: str):
        self.league_id   = LEAGUES.get(name, 293)
        self.league_name = name

    def set_api_key(self, key: str):
        """Actualiza la API key en tiempo de ejecución sin reiniciar."""
        self.api_key = key.strip()

    # ─────────────────────────────────────────
    #  Pipeline completo (corre en thread)
    # ─────────────────────────────────────────
    def load_league_data(self, progress_cb: Callable[[str, float], None] | None = None) -> bool:
        """
        Flujo completo usando OraclePipeline + MatchPredictor:
          1. Descarga (con caché) la base de Oracle's Elixir de la liga
          2. Calcula KPIs por equipo (incluye oro@15 real)
          3. Entrena el modelo Elo + oro@15 sobre los partidos reales

        Datos reales o error claro. NUNCA datos falsos: apostarías sobre
        números inventados. Oracle's Elixir es gratis y no requiere API key.

        progress_cb(mensaje, fracción 0-1) se llama en cada paso.
        """
        def cb(msg: str, pct: float = 0.0):
            if progress_cb:
                progress_cb(msg, pct)

        if not MODULES_OK:
            cb(f"ERROR: módulo faltante → {_IMPORT_MSG}", 1.0)
            return False

        df_matches = pd.DataFrame()
        df_stats = pd.DataFrame()
        league_code = self.league_name.split()[0].upper()   # "LCK — Korea" → "LCK"

        try:
            pipeline = OraclePipeline(league_code=league_code)

            # ── 1. Descarga + carga (con caché) ──
            cb("Descargando base de datos…", 0.10)
            games = pipeline.load_games(progress_cb=lambda m: cb(m, 0.35))
            self.current_patch = pipeline.current_patch
            cb(f"Parche: {self.current_patch or 'N/A'}", 0.55)

            # ── 2. KPIs + matches ──
            if not games.empty:
                cb("Calculando KPIs por equipo…", 0.62)
                df_stats = pipeline.build_team_stats(
                    games, min_games=getattr(cfg, "MIN_GAMES_PER_TEAM", 3)
                )
                df_matches = pipeline.build_matches(games)
            self.pipeline = pipeline
        except Exception as exc:
            cb(f"ERROR cargando datos: {exc}", 1.0)
            return False

        if df_stats.empty:
            cb(f"ERROR: sin datos para {league_code}. Revisa tu conexión.", 1.0)
            return False

        cb(f"KPIs listos: {len(df_stats)} equipos", 0.72)

        # ── 3. Entrenamiento (Elo + oro@15, sobre resultados reales) ──
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

        # ── Guardar resultados ──
        self.predictor  = predictor
        self.df_stats   = df_stats
        self.stats_dict = {
            row["team_name"]: row
            for row in df_stats.to_dict("records")
        }
        self.team_names = sorted(self.stats_dict.keys())

        cb("✅  Sistema listo", 1.0)
        return True

    # ─────────────────────────────────────────
    #  Predicción individual
    # ─────────────────────────────────────────
    def predict_match(
        self,
        name_a: str, name_b: str,
        side_a: str,
        odd_a_am: int, odd_b_am: int
    ) -> dict:
        """Corre predicción + Kelly. Retorna dict con todos los resultados."""
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

class KPICard(ctk.CTkFrame):
    """Tarjeta de métrica: título · valor grande · subtítulo."""

    def __init__(self, master, label: str, value: str = "—",
                 sub: str = "", accent: bool = False, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=12, **kw)
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 11),
                     text_color=C_MUTED).pack(padx=14, pady=(14, 2), anchor="w")
        self._vl = ctk.CTkLabel(
            self, text=value,
            font=("Segoe UI", 22, "bold"),
            text_color=C_GOLD if accent else C_TEXT
        )
        self._vl.pack(padx=14, anchor="w")
        self._sl = ctk.CTkLabel(self, text=sub, font=("Segoe UI", 10),
                                text_color=C_MUTED)
        self._sl.pack(padx=14, pady=(0, 14), anchor="w")

    def update_value(self, value: str, sub: str = ""):
        self._vl.configure(text=value)
        if sub:
            self._sl.configure(text=sub)


class RankingChart(ctk.CTkFrame):
    """Gráfica horizontal de barras (matplotlib) para ranking de equipos."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=12, **kw)
        self._fig = Figure(figsize=(5.2, 3.4), dpi=96, facecolor=C_CARD)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(C_CARD)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().configure(bg=C_CARD)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self._draw_placeholder()

    def _style_ax(self):
        ax = self._ax
        ax.set_facecolor(C_CARD)
        self._fig.patch.set_facecolor(C_CARD)
        for spine in ax.spines.values():
            spine.set_color(C_BORDER)
        ax.tick_params(colors=C_MUTED, labelsize=8)

    def _draw_placeholder(self):
        self._ax.clear(); self._style_ax()
        self._ax.text(0.5, 0.5, "Carga datos para ver el ranking",
                      ha="center", va="center", color=C_MUTED, fontsize=10)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._canvas.draw()

    def update(self, df: pd.DataFrame):
        if df.empty or "win_rate" not in df.columns:
            self._draw_placeholder(); return
        self._ax.clear(); self._style_ax()

        top = df.nlargest(min(9, len(df)), "win_rate").sort_values("win_rate")
        names  = [str(n)[:14] for n in top["team_name"]]
        values = (top["win_rate"] * 100).tolist()
        colors = [C_GOLD if v >= 65 else C_BLUE if v >= 45 else C_RED for v in values]

        bars = self._ax.barh(names, values, color=colors, height=0.62, zorder=2)
        self._ax.set_xlim(0, 108)
        self._ax.axvline(50, color=C_BORDER, lw=1, ls="--", alpha=0.6, zorder=1)

        for bar, val in zip(bars, values):
            self._ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", color=C_MUTED, fontsize=7.5
            )
        self._ax.set_xlabel("Win Rate (%)", color=C_MUTED, fontsize=8)
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()


class ProbabilityGauge(ctk.CTkFrame):
    """
    Semicírculo gauge matplotlib que muestra la probabilidad
    de victoria del Equipo A (0 – 100 %).
    """

    def __init__(self, master, **kw):
        super().__init__(master, fg_color=C_CARD, corner_radius=12, **kw)
        self._fig = Figure(figsize=(4.6, 2.4), dpi=96, facecolor=C_CARD)
        self._ax  = self._fig.add_subplot(111, aspect="equal")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.reset()

    def reset(self):
        self.render_gauge(0.5, "—", "—")

    def update(self, prob_a: float, name_a: str, name_b: str):
        self.render_gauge(prob_a, name_a, name_b)

    def render_gauge(self, prob_a: float = 0.5, name_a: str = "—", name_b: str = "—"):
        ax = self._ax
        ax.clear()
        ax.set_facecolor(C_CARD)
        self._fig.patch.set_facecolor(C_CARD)

        # ── Fondo gris (semicírculo completo) ──
        theta = np.linspace(np.pi, 0, 200)
        r_outer, r_inner = 1.0, 0.58
        xs_o = np.cos(theta) * r_outer
        ys_o = np.sin(theta) * r_outer
        xs_i = np.cos(theta[::-1]) * r_inner
        ys_i = np.sin(theta[::-1]) * r_inner
        ax.fill(
            np.concatenate([xs_o, xs_i]),
            np.concatenate([ys_o, ys_i]),   
            color=C_BORDER, zorder=1
        )

        # ── Arco de Team A (izq → prob_a) ──
        angle_a = np.pi * (1 - prob_a)           # ángulo final del arco A
        theta_a = np.linspace(np.pi, angle_a, 200)
        ax.fill(
            np.concatenate([np.cos(theta_a) * r_outer, np.cos(theta_a[::-1]) * r_inner]),
            np.concatenate([np.sin(theta_a) * r_outer, np.sin(theta_a[::-1]) * r_inner]),
            color=C_GOLD, zorder=2, alpha=0.92
        )

        # ── Arco de Team B (prob_a → der) ──
        theta_b = np.linspace(angle_a, 0, 200)
        ax.fill(
            np.concatenate([np.cos(theta_b) * r_outer, np.cos(theta_b[::-1]) * r_inner]),
            np.concatenate([np.sin(theta_b) * r_outer, np.sin(theta_b[::-1]) * r_inner]),
            color=C_BLUE, zorder=2, alpha=0.85
        )

        # ── Aguja ──
        needle_angle = np.pi * (1 - prob_a)
        ax.annotate(
            "", xy=(np.cos(needle_angle) * 0.8, np.sin(needle_angle) * 0.8),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C_WHITE, lw=2.5)
        )
        ax.add_patch(mpatches.Circle((0, 0), 0.07, color=C_WHITE, zorder=5))

        # ── Texto central ──
        ax.text(0, -0.12, f"{prob_a * 100:.1f}%",
                ha="center", va="center", color=C_WHITE,
                fontsize=15, fontweight="bold", zorder=6)
        ax.text(-1.0, -0.22, name_a[:10], ha="center", color=C_GOLD, fontsize=8)
        ax.text(+1.0, -0.22, name_b[:10], ha="center", color=C_BLUE, fontsize=8)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.4, 1.1)
        ax.axis("off")
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()


# ═══════════════════════════════════════════════════════════════
#  FRAME 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
class DashboardFrame(ctk.CTkFrame):

    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Dashboard", font=("Segoe UI", 20, "bold"),
                     text_color=C_TEXT).pack(anchor="w", pady=(0, 14))

        # ── KPI cards ──
        kpi_row = ctk.CTkFrame(self, fg_color="transparent")
        kpi_row.pack(fill="x", pady=(0, 14))
        kpi_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._c_teams   = KPICard(kpi_row, "Equipos cargados",     "—", "Sin datos")
        self._c_matches = KPICard(kpi_row, "Partidas analizadas",   "—", "últimas 100")
        self._c_auc     = KPICard(kpi_row, "AUC del Modelo",        "—", "TimeSeriesSplit CV", accent=True)
        self._c_patch   = KPICard(kpi_row, "Parche Sincronizado",   "—", "Sync dinámico", accent=True)

        for col, w in enumerate([self._c_teams, self._c_matches, self._c_auc, self._c_patch]):
            w.grid(row=0, column=col, sticky="nsew", padx=(0, 10 if col < 3 else 0))

        # ── Chart + info row ──
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # Ranking chart
        left = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(left, text="Ranking de equipos · Win Rate",
                     font=("Segoe UI", 12), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(12, 0))
        self._chart = RankingChart(left)
        self._chart.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        # Status info
        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12, width=200)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Estado del sistema",
                     font=("Segoe UI", 12), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(12, 6))

        self._rows: dict[str, ctk.CTkLabel] = {}
        for key in ["Liga activa", "Parche", "Modo ML", "Features", "Equipos", "Partidas"]:
            row = ctk.CTkFrame(right, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(row, text=f"{key}:", font=("Segoe UI", 11),
                         text_color=C_MUTED, width=80, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(row, text="—", font=("Segoe UI", 11), text_color=C_TEXT)
            lbl.pack(side="left")
            self._rows[key] = lbl

    def refresh(self, engine: PredictionEngine):
        m = engine.model_metrics
        n_teams = len(engine.team_names)
        n_rows  = len(engine.df_stats)

        auc = m.get("auc_mean")
        auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"
        n_matches = m.get("matches", 0)

        self._c_teams.update_value(str(n_teams), "equipos con datos")
        self._c_matches.update_value(str(n_matches), "partidos reales")
        self._c_auc.update_value(
            auc_str,
            f"modo {m.get('mode', '?')}"
        )
        self._c_patch.update_value(engine.current_patch or "N/A", "parche activo")
        self._chart.update(engine.df_stats)

        ln = next((k for k, v in LEAGUES.items() if v == engine.league_id), "—")
        self._rows["Liga activa"].configure(text=ln.split("—")[0].strip())
        self._rows["Parche"].configure(text=engine.current_patch or "—")
        self._rows["Modo ML"].configure(text=m.get("mode", "—"))
        self._rows["Features"].configure(text=str(m.get("features", "—")))
        self._rows["Equipos"].configure(text=str(n_teams))
        self._rows["Partidas"].configure(text=str(n_rows))


# ═══════════════════════════════════════════════════════════════
#  FRAME 2: MATCH ANALYZER
# ═══════════════════════════════════════════════════════════════
class MatchAnalyzerFrame(ctk.CTkFrame):

    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._result: dict = {}
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Match Analyzer",
                     font=("Segoe UI", 20, "bold"), text_color=C_TEXT
                     ).pack(anchor="w", pady=(0, 14))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # ── Left: input form ──
        form = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12, width=320)
        form.pack(side="left", fill="y", padx=(0, 12))
        form.pack_propagate(False)

        ctk.CTkLabel(form, text="Configurar partido",
                     font=("Segoe UI", 13, "bold"), text_color=C_TEXT
                     ).pack(padx=18, pady=(16, 12), anchor="w")

        # Team selectors
        def _section(text): 
            ctk.CTkLabel(form, text=text, font=("Segoe UI", 10),
                         text_color=C_MUTED).pack(padx=18, anchor="w", pady=(8, 2))

        _section("EQUIPO A  (Blue Side por defecto)")
        self._cb_a = ctk.CTkComboBox(form, values=["— carga datos primero —"],
                                      width=280, fg_color=C_PANEL, border_color=C_BORDER,
                                      button_color=C_GOLD, dropdown_fg_color=C_PANEL)
        self._cb_a.pack(padx=18, pady=(0, 4))

        _section("EQUIPO B  (Red Side por defecto)")
        self._cb_b = ctk.CTkComboBox(form, values=["— carga datos primero —"],
                                      width=280, fg_color=C_PANEL, border_color=C_BORDER,
                                      button_color=C_GOLD, dropdown_fg_color=C_PANEL)
        self._cb_b.pack(padx=18, pady=(0, 4))

        # Side toggle
        _section("LADO DEL MAPA")
        side_row = ctk.CTkFrame(form, fg_color="transparent")
        side_row.pack(padx=18, fill="x", pady=(0, 4))
        self._side_var = tk.StringVar(value="blue")
        ctk.CTkRadioButton(side_row, text="Blue Side (A)",
                           variable=self._side_var, value="blue",
                           fg_color=C_GOLD, border_color=C_GOLD
                           ).pack(side="left", padx=(0, 16))
        ctk.CTkRadioButton(side_row, text="Red Side (A)",
                           variable=self._side_var, value="red",
                           fg_color=C_RED, border_color=C_RED
                           ).pack(side="left")

        # Odds inputs
        _section("MOMIO AMERICANO — EQUIPO A  (ej. -425 o +170)")
        self._odd_a = ctk.CTkEntry(form, placeholder_text="-425",
                                    width=280, fg_color=C_PANEL, border_color=C_BORDER)
        self._odd_a.pack(padx=18)

        _section("MOMIO AMERICANO — EQUIPO B  (ej. +285)")
        self._odd_b = ctk.CTkEntry(form, placeholder_text="+285",
                                    width=280, fg_color=C_PANEL, border_color=C_BORDER)
        self._odd_b.pack(padx=18)

        # Analyze button
        self._btn_analyze = ctk.CTkButton(
            form, text="⚡  Analizar partido",
            fg_color=C_GOLD, text_color=C_DARK2, hover_color=C_GOLD2,
            font=("Segoe UI", 13, "bold"), height=42, width=280,
            command=self._run_analysis
        )
        self._btn_analyze.pack(padx=18, pady=20)

        # Error label
        self._err_lbl = ctk.CTkLabel(form, text="", text_color=C_RED,
                                      font=("Segoe UI", 10), wraplength=270)
        self._err_lbl.pack(padx=18)

        # ── Right: results ──
        results = ctk.CTkFrame(body, fg_color="transparent")
        results.pack(side="right", fill="both", expand=True)

        # Gauge
        gauge_frame = ctk.CTkFrame(results, fg_color=C_CARD, corner_radius=12)
        gauge_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(gauge_frame, text="Probabilidad de victoria · ajuste LCK",
                     font=("Segoe UI", 12), text_color=C_MUTED
                     ).pack(anchor="w", padx=14, pady=(12, 0))
        self._gauge = ProbabilityGauge(gauge_frame)
        self._gauge.pack(fill="x", padx=6, pady=(0, 8))

        # Kelly cards
        kelly_frame = ctk.CTkFrame(results, fg_color="transparent")
        kelly_frame.pack(fill="x", pady=(0, 10))
        kelly_frame.grid_columnconfigure((0, 1), weight=1)

        self._kelly_a = _KellyCard(kelly_frame, side="a")
        self._kelly_a.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._kelly_b = _KellyCard(kelly_frame, side="b")
        self._kelly_b.grid(row=0, column=1, sticky="nsew")

        # Metrics row
        self._metrics_frame = ctk.CTkFrame(results, fg_color=C_CARD, corner_radius=12)
        self._metrics_frame.pack(fill="x")
        self._metrics_lbl = ctk.CTkLabel(
            self._metrics_frame, text="Ingresa los datos y pulsa Analizar",
            font=("Segoe UI", 11), text_color=C_MUTED
        )
        self._metrics_lbl.pack(padx=14, pady=14)

    def refresh_teams(self):
        names = self.engine.team_names or ["— carga datos primero —"]
        self._cb_a.configure(values=names)
        self._cb_b.configure(values=names)
        if len(names) >= 2:
            self._cb_a.set(names[0])
            self._cb_b.set(names[1])

    def _run_analysis(self):
        self._err_lbl.configure(text="")
        name_a = self._cb_a.get().strip()
        name_b = self._cb_b.get().strip()

        if name_a == name_b:
            self._err_lbl.configure(text="Los equipos no pueden ser el mismo.")
            return
        if "carga" in name_a.lower():
            self._err_lbl.configure(text="Primero carga los datos de la liga.")
            return

        try:
            raw_a = self._odd_a.get().strip() or "-200"
            raw_b = self._odd_b.get().strip() or "+160"
            odd_a = int(raw_a.replace("+", ""))
            odd_b = int(raw_b.replace("+", ""))
            if abs(odd_a) < 100 or abs(odd_b) < 100:
                raise ValueError
        except ValueError:
            self._err_lbl.configure(text="Momios inválidos. Usa formato americano: -425 o +285")
            return

        side = self._side_var.get()

        self._btn_analyze.configure(state="disabled", text="Analizando…")
        result = self.engine.predict_match(name_a, name_b, side, odd_a, odd_b)
        self._show_result(result)
        self._btn_analyze.configure(state="normal", text="⚡  Analizar partido")

    def _show_result(self, r: dict):
        self._result = r
        prob_a = r["prob_a"]
        name_a = r["name_a"]
        name_b = r["name_b"]

        self._gauge.update(prob_a, name_a, name_b)
        self._kelly_a.update(name_a, r["kelly_a"], r["dec_a"], r["odd_a_am"], r["prob_a"])
        self._kelly_b.update(name_b, r["kelly_b"], r["dec_b"], r["odd_b_am"], r["prob_b"])

        # Metrics summary
        for w in self._metrics_frame.winfo_children():
            w.destroy()

        vals = [
            ("Modelo",        r.get("mode", "—")),
            ("Win Rate A",    f"{r['wr_a']*100:.0f}%"),
            ("Win Rate B",    f"{r['wr_b']*100:.0f}%"),
            ("Cuota dec. A",  f"{r['dec_a']:.3f}"),
            ("Cuota dec. B",  f"{r['dec_b']:.3f}"),
        ]
        for i, (k, v) in enumerate(vals):
            col = ctk.CTkFrame(self._metrics_frame, fg_color="transparent")
            col.grid(row=0, column=i, padx=12, pady=10, sticky="nsew")
            self._metrics_frame.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(col, text=k, font=("Segoe UI", 9),
                         text_color=C_MUTED).pack()
            ctk.CTkLabel(col, text=v, font=("Segoe UI", 12, "bold"),
                         text_color=C_TEXT).pack()


class _KellyCard(ctk.CTkFrame):
    """Tarjeta de resultado Kelly para un equipo."""

    def __init__(self, master, side: str, **kw):
        color = C_GOLD if side == "a" else C_BLUE
        super().__init__(master, fg_color=C_CARD, corner_radius=12, **kw)
        self._color = color
        ctk.CTkLabel(self, text="—", font=("Segoe UI", 11, "bold"),
                     text_color=color).pack(padx=14, pady=(14, 2), anchor="w")
        self._signal = ctk.CTkLabel(self, text="—", font=("Segoe UI", 18, "bold"),
                                     text_color=C_MUTED)
        self._signal.pack(padx=14, anchor="w")
        self._detail = ctk.CTkLabel(self, text="—\n—\n—",
                                     font=("Segoe UI", 10), text_color=C_MUTED,
                                     justify="left")
        self._detail.pack(padx=14, pady=(4, 14), anchor="w")
        self._name_lbl = self.winfo_children()[0]

    def update(self, name: str, k: dict, dec: float, odd_am: int, prob: float):
        self._name_lbl.configure(text=name[:16])
        is_val = k.get("is_value", False)
        signal = "🥇 OPORTUNIDAD" if is_val else "❌ Sin ventaja"
        color  = C_GREEN if is_val else C_RED
        self._signal.configure(text=signal, text_color=color)

        stake = k.get("stake_mxn", 0)
        ev    = k.get("ev_mxn", 0)
        edge  = k.get("edge_pct", 0)
        impl  = k.get("implied_prob_pct", 0)

        detail = (
            f"Prob modelo: {prob*100:.1f}%   Prob casera: {impl:.1f}%\n"
            f"Edge: {edge:+.1f}%   Cuota: {odd_am:+d}  ({dec:.3f})\n"
            f"Stake Kelly: ${stake:,.0f} MXN   EV: +${ev:,.0f}"
            if is_val else
            f"Prob modelo: {prob*100:.1f}%   Prob casera: {impl:.1f}%\n"
            f"Edge: {edge:+.1f}%   (necesitas > 7%)\n"
            f"Cuota: {odd_am:+d}  ({dec:.3f})"
        )
        self._detail.configure(text=detail)


# ═══════════════════════════════════════════════════════════════
#  FRAME 3: VALUE BET SCANNER
# ═══════════════════════════════════════════════════════════════
class ValueBetsFrame(ctk.CTkFrame):
    """
    Permite ingresar múltiples partidos con sus momios
    y escanea todos en busca de value bets.
    """

    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._rows: list[dict] = []
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Value Bet Scanner",
                     font=("Segoe UI", 20, "bold"), text_color=C_TEXT
                     ).pack(anchor="w", pady=(0, 6))
        ctk.CTkLabel(self, text="Agrega los partidos de la jornada con sus momios. El sistema marcará las oportunidades de valor.",
                     font=("Segoe UI", 11), text_color=C_MUTED
                     ).pack(anchor="w", pady=(0, 12))

        # Toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 10))
        ctk.CTkButton(toolbar, text="+ Agregar partido",
                      fg_color=C_GOLD, text_color=C_DARK2, hover_color=C_GOLD2,
                      font=("Segoe UI", 12, "bold"), height=36,
                      command=self._add_match_dialog
                      ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(toolbar, text="🔍  Escanear todo",
                      fg_color=C_BLUE, hover_color="#2980b9",
                      font=("Segoe UI", 12, "bold"), height=36,
                      command=self._scan_all
                      ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(toolbar, text="Limpiar",
                      fg_color=C_CARD, hover_color=C_BORDER,
                      font=("Segoe UI", 11), height=36,
                      command=self._clear
                      ).pack(side="left")

        # Scrollable results area
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C_CARD, corner_radius=12)
        self._scroll.pack(fill="both", expand=True)

        # Column headers
        hdr = ctk.CTkFrame(self._scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        for text, w in [("Partido", 200), ("Prob. Modelo", 110), ("Momio", 90),
                         ("Edge", 80), ("Stake MXN", 90), ("Señal", 150)]:
            ctk.CTkLabel(hdr, text=text, font=("Segoe UI", 10, "bold"),
                         text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=4)

        ctk.CTkFrame(self._scroll, fg_color=C_BORDER, height=1).pack(fill="x", padx=10)

        self._results_frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._results_frame.pack(fill="both", expand=True)

        self._summary_lbl = ctk.CTkLabel(self, text="", font=("Segoe UI", 11),
                                          text_color=C_MUTED)
        self._summary_lbl.pack(pady=8)

    def _add_match_dialog(self):
        if not self.engine.team_names:
            messagebox.showwarning("Sin datos", "Carga primero los datos de la liga.")
            return
        _MatchInputDialog(self, self.engine.team_names, callback=self._receive_match)

    def _receive_match(self, data: dict):
        self._rows.append(data)
        self._render_rows()

    def _scan_all(self):
        if not self._rows:
            messagebox.showinfo("Sin partidos", "Agrega al menos un partido primero.")
            return
        for i, row in enumerate(self._rows):
            try:
                result = self.engine.predict_match(
                    row["name_a"], row["name_b"],
                    row.get("side", "blue"),
                    row["odd_a"], row["odd_b"]
                )
                self._rows[i]["result"] = result
            except Exception as e:
                self._rows[i]["error"] = str(e)
        self._render_rows()

    def _render_rows(self):
        for w in self._results_frame.winfo_children():
            w.destroy()

        total_stake = total_ev = n_value = 0

        for row in self._rows:
            r = row.get("result")
            line = ctk.CTkFrame(self._results_frame, fg_color=C_PANEL, corner_radius=8)
            line.pack(fill="x", padx=10, pady=3)

            def _lbl(parent, text, color=C_TEXT, w=None):
                kw = dict(font=("Segoe UI", 11), text_color=color, anchor="w")
                if w:
                    kw["width"] = w
                ctk.CTkLabel(parent, text=text, **kw).pack(side="left", padx=6, pady=8)

            match_str = f"{row['name_a']} vs {row['name_b']}"
            _lbl(line, match_str[:26], w=200)

            if r:
                ka, kb = r["kelly_a"], r["kelly_b"]
                # Pick the side with higher edge
                best_k   = ka if ka.get("edge_pct", -999) >= kb.get("edge_pct", -999) else kb
                best_prob = r["prob_a"] if ka.get("edge_pct", -999) >= kb.get("edge_pct", -999) else r["prob_b"]
                best_name = row["name_a"] if ka.get("edge_pct", -999) >= kb.get("edge_pct", -999) else row["name_b"]
                best_odd  = row["odd_a"] if best_name == row["name_a"] else row["odd_b"]

                edge  = best_k.get("edge_pct", 0)
                stake = best_k.get("stake_mxn", 0)
                ev    = best_k.get("ev_mxn", 0)
                is_v  = best_k.get("is_value", False)

                _lbl(line, f"{best_prob*100:.1f}%", w=110)
                _lbl(line, f"{best_odd:+d}", w=90)
                ec = C_GREEN if edge > 0 else C_RED
                _lbl(line, f"{edge:+.1f}%", color=ec, w=80)
                _lbl(line, f"${stake:,.0f}" if is_v else "—", w=90)

                sig = "🥇 OPORTUNIDAD DE ORO" if is_v else "❌ Sin ventaja"
                sc  = C_GREEN if is_v else C_MUTED
                _lbl(line, sig, color=sc, w=150)

                if is_v:
                    n_value += 1
                    total_stake += stake
                    total_ev    += ev
            else:
                _lbl(line, "—", w=110)
                _lbl(line, "—", w=90)
                err = row.get("error", "Pulsa Escanear")
                _lbl(line, err, color=C_MUTED)

        # Summary
        if self._rows:
            pct = total_stake / self.engine.bankroll * 100 if self.engine.bankroll > 0 else 0
            self._summary_lbl.configure(
                text=f"Oportunidades: {n_value}/{len(self._rows)}  ·  "
                     f"Stake total: ${total_stake:,.0f} MXN ({pct:.1f}%)  ·  "
                     f"EV esperado: +${total_ev:,.0f} MXN"
            )

    def _clear(self):
        self._rows.clear()
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._summary_lbl.configure(text="")


class _MatchInputDialog(ctk.CTkToplevel):
    """Ventana modal para ingresar un partido + momios."""

    def __init__(self, master, team_names: list, callback: Callable):
        super().__init__(master)
        self.title("Agregar partido")
        self.geometry("400x360")
        self.resizable(False, False)
        self.grab_set()
        self._callback = callback
        self._names    = team_names

        ctk.CTkLabel(self, text="Nuevo partido",
                     font=("Segoe UI", 14, "bold")).pack(pady=(18, 12))

        def _row(label, widget_fn):
            f = ctk.CTkFrame(self, fg_color="transparent")
            f.pack(fill="x", padx=24, pady=4)
            ctk.CTkLabel(f, text=label, font=("Segoe UI", 10),
                         text_color=C_MUTED, width=100, anchor="w").pack(side="left")
            w = widget_fn(f)
            w.pack(side="left", fill="x", expand=True)
            return w

        self._cb_a = _row("Equipo A:", lambda f: ctk.CTkComboBox(
            f, values=team_names, fg_color=C_PANEL, border_color=C_BORDER,
            button_color=C_GOLD, dropdown_fg_color=C_PANEL))
        if team_names: self._cb_a.set(team_names[0])

        self._cb_b = _row("Equipo B:", lambda f: ctk.CTkComboBox(
            f, values=team_names, fg_color=C_PANEL, border_color=C_BORDER,
            button_color=C_GOLD, dropdown_fg_color=C_PANEL))
        if len(team_names) > 1: self._cb_b.set(team_names[1])

        self._e_odd_a = _row("Momio A:", lambda f: ctk.CTkEntry(
            f, placeholder_text="-425", fg_color=C_PANEL, border_color=C_BORDER))
        self._e_odd_b = _row("Momio B:", lambda f: ctk.CTkEntry(
            f, placeholder_text="+285", fg_color=C_PANEL, border_color=C_BORDER))

        self._err = ctk.CTkLabel(self, text="", text_color=C_RED, font=("Segoe UI", 10))
        self._err.pack()

        ctk.CTkButton(self, text="Agregar", fg_color=C_GOLD, text_color=C_DARK2,
                      hover_color=C_GOLD2, font=("Segoe UI", 12, "bold"),
                      command=self._submit).pack(pady=16)

    def _submit(self):
        na = self._cb_a.get().strip()
        nb = self._cb_b.get().strip()
        if na == nb:
            self._err.configure(text="Los equipos no pueden ser iguales.")
            return
        try:
            oa = int(self._e_odd_a.get().replace("+", "") or "-200")
            ob = int(self._e_odd_b.get().replace("+", "") or "+160")
            if abs(oa) < 100 or abs(ob) < 100:
                raise ValueError
        except ValueError:
            self._err.configure(text="Momios inválidos. Usa: -425 o +285")
            return
        self._callback({"name_a": na, "name_b": nb, "odd_a": oa, "odd_b": ob, "side": "blue"})
        self.destroy()


# ═══════════════════════════════════════════════════════════════
#  FRAME 4: SETTINGS
# ═══════════════════════════════════════════════════════════════
class SettingsFrame(ctk.CTkFrame):

    def __init__(self, master, engine: PredictionEngine,
                 on_league_change: Callable, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._on_league_change = on_league_change
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Configuración",
                     font=("Segoe UI", 20, "bold"), text_color=C_TEXT
                     ).pack(anchor="w", pady=(0, 14))

        card = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=12)
        card.pack(fill="x", pady=(0, 12))

        def _section(parent, title):
            ctk.CTkLabel(parent, text=title, font=("Segoe UI", 11, "bold"),
                         text_color=C_GOLD).pack(anchor="w", padx=18, pady=(16, 6))

        def _field(parent, label, widget):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=4)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=C_MUTED,
                         width=160, anchor="w").pack(side="left")
            widget(row)

        # ── API ──
        _section(card, "🔑  Conexión API")
        self._e_key = ctk.CTkEntry(card, width=380, show="*",
                                    fg_color=C_PANEL, border_color=C_BORDER,
                                    placeholder_text="Pega tu PandaScore API Key aquí")
        self._e_key.pack(padx=18, pady=(0, 4))
        if MODULES_OK:
            self._e_key.insert(0, cfg.PANDASCORE_API_KEY)

        ctk.CTkButton(card, text="Aplicar API Key",
                      fg_color=C_PANEL, hover_color=C_BORDER,
                      font=("Segoe UI", 11), command=self._apply_key
                      ).pack(padx=18, pady=(0, 10), anchor="w")

        ctk.CTkFrame(card, fg_color=C_BORDER, height=1).pack(fill="x", padx=18)

        # ── Liga ──
        _section(card, "🏆  Liga activa")
        self._cb_league = ctk.CTkComboBox(
            card, values=list(LEAGUES.keys()), width=280,
            fg_color=C_PANEL, border_color=C_BORDER,
            button_color=C_GOLD, dropdown_fg_color=C_PANEL,
            command=self._on_league_select
        )
        self._cb_league.set(list(LEAGUES.keys())[0])
        self._cb_league.pack(padx=18, pady=(0, 14))

        ctk.CTkFrame(card, fg_color=C_BORDER, height=1).pack(fill="x", padx=18)

        # ── Bankroll ──
        _section(card, "💰  Bankroll y Kelly")

        _field(card, "Bankroll (MXN):", lambda p: ctk.CTkEntry(
            p, textvariable=self._br_var, width=120,
            fg_color=C_PANEL, border_color=C_BORDER
        ).pack(side="left"))

        _field(card, f"Fracción Kelly: {self._kf_pct():.0f}%", lambda p: None)
        self._slider = ctk.CTkSlider(card, from_=10, to=50, number_of_steps=8,
                                      fg_color=C_BORDER, progress_color=C_GOLD,
                                      button_color=C_GOLD, button_hover_color=C_GOLD2,
                                      command=self._update_kelly_label)
        self._slider.set(self.engine.kelly_frac * 100)
        self._slider.pack(padx=18, fill="x", pady=(0, 4))
        self._kelly_lbl = ctk.CTkLabel(card, text=f"Kelly al {self._kf_pct():.0f}%",
                                        font=("Segoe UI", 10), text_color=C_MUTED)
        self._kelly_lbl.pack(padx=18, anchor="w", pady=(0, 14))

        ctk.CTkButton(card, text="💾  Guardar configuración",
                      fg_color=C_GOLD, text_color=C_DARK2, hover_color=C_GOLD2,
                      font=("Segoe UI", 12, "bold"), height=38,
                      command=self._save
                      ).pack(padx=18, pady=12, anchor="w")

        self._status = ctk.CTkLabel(card, text="", font=("Segoe UI", 10),
                                     text_color=C_GREEN)
        self._status.pack(padx=18, pady=(0, 12), anchor="w")

    def _br_var_init(self):
        v = tk.StringVar(value=str(int(self.engine.bankroll)))
        return v

    # Lazy init to avoid Tkinter-before-mainloop issues
    @property
    def _br_var(self):
        if not hasattr(self, "_bankroll_var"):
            self._bankroll_var = tk.StringVar(value=str(int(self.engine.bankroll)))
        return self._bankroll_var

    def _kf_pct(self): return self.engine.kelly_frac * 100

    def _apply_key(self):
        key = self._e_key.get().strip()
        if key:
            self.engine.set_api_key(key)
            try:
                cfg.save_api_key(key)
                cfg.PANDASCORE_API_KEY = key
                self._status.configure(
                    text="API Key guardada. Pulsa Cargar para actualizar.",
                    text_color=C_GREEN)
            except Exception as exc:
                self._status.configure(
                    text=f"API Key aplicada (no se pudo guardar: {exc})",
                    text_color=C_GOLD)

    def _on_league_select(self, choice: str):
        self.engine.set_league(choice)

    def _update_kelly_label(self, val):
        pct = float(val)
        self._kelly_lbl.configure(text=f"Kelly al {pct:.0f}%")
        self.engine.kelly_frac = pct / 100

    def _save(self):
        try:
            br = float(self._br_var.get().replace(",", "").replace("$", ""))
            self.engine.bankroll   = br
            self.engine.kelly_frac = self._slider.get() / 100
            league_name = self._cb_league.get()
            self.engine.set_league(league_name)
            self._status.configure(
                text=f"✅ Guardado — Bankroll: ${br:,.0f} MXN  Kelly: {self.engine.kelly_frac*100:.0f}%",
                text_color=C_GREEN
            )
            self._on_league_change(league_name)
        except ValueError:
            self._status.configure(text="Bankroll inválido.", text_color=C_RED)


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════
class PredictionOSApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Prediction OS V2.0  —  eSports Analytics Dashboard")
        self.geometry("1180x720")
        self.minsize(980, 620)
        self.configure(fg_color=C_BG)

        self.engine = PredictionEngine()
        self._active_frame: str = "dashboard"
        self._loading = False

        self._build_sidebar()
        self._build_content()
        self._build_statusbar()

        # Cargar datos al inicio (hilo de fondo)
        self.after(600, self._start_load)

    # ─────────────────────────────────────────
    #  Layout
    # ─────────────────────────────────────────
    def _build_sidebar(self):
        side = ctk.CTkFrame(self, fg_color=C_SIDE, width=200, corner_radius=0)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        # Logo / titulo
        logo = ctk.CTkFrame(side, fg_color=C_DARK2, height=64, corner_radius=0)
        logo.pack(fill="x")
        logo.pack_propagate(False)
        ctk.CTkLabel(logo, text="⚡ PREDICTION OS",
                     font=("Segoe UI", 13, "bold"), text_color=C_GOLD
                     ).pack(expand=True)
        ctk.CTkLabel(logo, text="V2.0",
                     font=("Segoe UI", 9), text_color=C_MUTED
                     ).place(relx=1.0, rely=1.0, x=-10, y=-6, anchor="se")

        ctk.CTkFrame(side, fg_color=C_BORDER, height=1).pack(fill="x")

        # Nav buttons
        nav_items = [
            ("dashboard",  "📊  Dashboard",     self._go_dashboard),
            ("analyzer",   "⚔️   Match Analyzer", self._go_analyzer),
            ("valuebets",  "💰  Value Bets",     self._go_valuebets),
            ("settings",   "⚙️   Configuración",  self._go_settings),
        ]
        self._nav_btns: dict[str, ctk.CTkButton] = {}
        ctk.CTkFrame(side, fg_color="transparent", height=10).pack()

        for key, label, cmd in nav_items:
            btn = ctk.CTkButton(
                side, text=label,
                anchor="w", font=("Segoe UI", 12),
                fg_color=C_GOLD if key == "dashboard" else "transparent",
                text_color=C_DARK2 if key == "dashboard" else C_TEXT,
                hover_color=C_CARD,
                height=42, corner_radius=8,
                command=cmd
            )
            btn.pack(fill="x", padx=10, pady=3)
            self._nav_btns[key] = btn

        # Separator
        ctk.CTkFrame(side, fg_color="transparent").pack(expand=True)
        ctk.CTkFrame(side, fg_color=C_BORDER, height=1).pack(fill="x")

        # Load button in sidebar
        self._load_btn = ctk.CTkButton(
            side, text="⟳  Cargar / Actualizar",
            fg_color=C_PANEL, hover_color=C_BORDER,
            font=("Segoe UI", 11), height=38,
            command=self._start_load
        )
        self._load_btn.pack(fill="x", padx=10, pady=8)

        # League selector compacto
        self._league_cb = ctk.CTkComboBox(
            side, values=list(LEAGUES.keys()),
            fg_color=C_PANEL, border_color=C_BORDER,
            button_color=C_GOLD, dropdown_fg_color=C_PANEL,
            font=("Segoe UI", 10), height=32,
            command=self._on_sidebar_league_change
        )
        self._league_cb.set(list(LEAGUES.keys())[0])
        self._league_cb.pack(fill="x", padx=10, pady=(0, 10))

        # Status dot
        self._dot_frame = ctk.CTkFrame(side, fg_color="transparent", height=28)
        self._dot_frame.pack(fill="x", padx=10, pady=(0, 14))
        self._dot = ctk.CTkLabel(self._dot_frame, text="●", font=("Segoe UI", 14),
                                  text_color=C_MUTED)
        self._dot.pack(side="left")
        self._dot_lbl = ctk.CTkLabel(self._dot_frame, text=" Sin datos",
                                      font=("Segoe UI", 10), text_color=C_MUTED)
        self._dot_lbl.pack(side="left")

    def _build_content(self):
        self._content = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True, padx=20, pady=16)

        self._frames: dict[str, ctk.CTkFrame] = {
            "dashboard": DashboardFrame(self._content, self.engine),
            "analyzer":  MatchAnalyzerFrame(self._content, self.engine),
            "valuebets": ValueBetsFrame(self._content, self.engine),
            "settings":  SettingsFrame(self._content, self.engine,
                                        on_league_change=self._on_sidebar_league_change),
        }
        for f in self._frames.values():
            f.pack(fill="both", expand=True)

        self._show("dashboard")

    def _build_statusbar(self):
        self._statusbar = ctk.CTkFrame(self, fg_color=C_DARK2, height=24, corner_radius=0)
        self._statusbar.pack(side="bottom", fill="x")
        self._status_lbl = ctk.CTkLabel(
            self._statusbar, text="Iniciando sistema…",
            font=("Segoe UI", 9), text_color=C_MUTED
        )
        self._status_lbl.pack(side="left", padx=12)
        ctk.CTkLabel(
            self._statusbar,
            text=f"Prediction OS V2.0  ·  {datetime.now().year}",
            font=("Segoe UI", 9), text_color=C_MUTED
        ).pack(side="right", padx=12)

        self._progress = ctk.CTkProgressBar(
            self._statusbar, mode="determinate",
            fg_color=C_BORDER, progress_color=C_GOLD,
            height=6, width=200
        )
        self._progress.set(0)
        self._progress.pack(side="right", padx=(0, 12), pady=9)

    # ─────────────────────────────────────────
    #  Navigation
    # ─────────────────────────────────────────
    def _show(self, name: str):
        for k, f in self._frames.items():
            if k == name:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()
        self._active_frame = name

        for k, btn in self._nav_btns.items():
            if k == name:
                btn.configure(fg_color=C_GOLD, text_color=C_DARK2)
            else:
                btn.configure(fg_color="transparent", text_color=C_TEXT)

    def _go_dashboard(self):  self._show("dashboard")
    def _go_analyzer(self):   self._show("analyzer")
    def _go_valuebets(self):  self._show("valuebets")
    def _go_settings(self):   self._show("settings")

    # ─────────────────────────────────────────
    #  Data loading (background thread)
    # ─────────────────────────────────────────
    def _on_sidebar_league_change(self, name: str):
        self.engine.set_league(name)
        self._status_lbl.configure(
            text=f"Liga cambiada a {name.split('—')[0].strip()} — pulsa Cargar para actualizar"
        )

    def _start_load(self):
        if self._loading:
            return
        self._loading = True
        self._load_btn.configure(state="disabled", text="Cargando…")
        self._dot.configure(text_color=C_GOLD)
        self._dot_lbl.configure(text=" Cargando…")
        self._progress.set(0)

        thread = threading.Thread(target=self._load_worker, daemon=True)
        thread.start()

    def _load_worker(self):
        def progress(msg: str, pct: float):
            self.after(0, self._update_progress, msg, pct)

        ok = self.engine.load_league_data(progress_cb=progress)
        self.after(0, self._on_load_done, ok)

    def _update_progress(self, msg: str, pct: float):
        self._status_lbl.configure(text=msg)
        self._progress.set(max(0, min(1, pct)))

    def _on_load_done(self, ok: bool):
        self._loading = False
        self._load_btn.configure(state="normal", text="⟳  Cargar / Actualizar")

        if ok:
            self._dot.configure(text_color=C_GREEN)
            self._dot_lbl.configure(text=f" {len(self.engine.team_names)} equipos")
            self._frames["dashboard"].refresh(self.engine)
            self._frames["analyzer"].refresh_teams()
        else:
            self._dot.configure(text_color=C_RED)
            self._dot_lbl.configure(text=" Error de conexión")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not MODULES_OK:
        # Si faltan módulos, mostrar ventana de error mínima
        root = ctk.CTk()
        root.title("Error de inicio")
        root.geometry("500x200")
        ctk.CTkLabel(
            root,
            text=f"No se encontraron los módulos requeridos:\n\n{_IMPORT_MSG}\n\n"
                  "Asegúrate de que config.py, universal_pipeline.py y\n"
                  "model.py estén en el mismo directorio.",
            font=("Segoe UI", 12), justify="center", wraplength=460
        ).pack(expand=True)
        root.mainloop()
        sys.exit(1)

    app = PredictionOSApp()
    app.mainloop()
