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
    import bet_tracker
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
        self.league_name    = cfg.DEFAULT_LEAGUE_NAME if MODULES_OK else "LCK  — Corea"
        self.league_code    = LEAGUES.get(self.league_name, "LCK")
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
        self.league_name = name
        self.league_code = LEAGUES.get(name, name.split()[0].upper())

    def set_api_key(self, key: str):
        """Actualiza la API key en tiempo de ejecución sin reiniciar."""
        self.api_key = key.strip()

    def fetch_upcoming_matches(self, max_results: int = 40) -> list[dict]:
        """
        Trae los PRÓXIMOS partidos (calendario) y deja solo aquellos cuyos dos
        equipos pertenecen a la liga activa, emparejando por nombre con los
        equipos de Oracle's Elixir. Usa PandaScore (status not_started).

        Requiere API Key de PandaScore (solo para el calendario; los datos y el
        modelo siguen siendo de Oracle's Elixir). Sin key o sin partidos → [].
        """
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
        league_code = self.league_code   # código de Oracle's Elixir (ej. "LCK")

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

        self._rows["Liga activa"].configure(text=engine.league_name.split("—")[0].strip())
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

        # Registrar apuesta
        reg_row = ctk.CTkFrame(results, fg_color="transparent")
        reg_row.pack(fill="x", pady=(8, 0))
        self._btn_reg = ctk.CTkButton(
            reg_row, text="📒  Registrar apuesta de valor",
            fg_color=C_PANEL, hover_color=C_BORDER, font=("Segoe UI", 12),
            height=36, state="disabled", command=self._register_bet)
        self._btn_reg.pack(side="left")
        self._reg_msg = ctk.CTkLabel(reg_row, text="", font=("Segoe UI", 10), text_color=C_GREEN)
        self._reg_msg.pack(side="left", padx=12)

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

        # Habilitar registro solo si hay un lado con valor
        has_value = r["kelly_a"].get("is_value") or r["kelly_b"].get("is_value")
        self._reg_msg.configure(text="")
        self._btn_reg.configure(
            state="normal" if has_value else "disabled",
            text="📒  Registrar apuesta de valor" if has_value else "Sin valor para registrar")

    def _register_bet(self):
        r = self._result
        if not r:
            return
        ka, kb = r["kelly_a"], r["kelly_b"]
        # Elige el lado con valor (o el de mayor edge)
        pick_a = ka.get("is_value") and ka.get("edge_pct", -99) >= kb.get("edge_pct", -99)
        if not ka.get("is_value") and kb.get("is_value"):
            pick_a = False
        elif ka.get("is_value") and not kb.get("is_value"):
            pick_a = True

        k    = ka if pick_a else kb
        pick = r["name_a"] if pick_a else r["name_b"]
        prob = r["prob_a"] if pick_a else r["prob_b"]
        odd  = r["odd_a_am"] if pick_a else r["odd_b_am"]
        dec  = r["dec_a"] if pick_a else r["dec_b"]
        side_a = self._side_var.get()           # lado del equipo A
        lado = side_a if pick_a else ("red" if side_a == "blue" else "blue")

        bet_tracker.add_bet(
            liga=self.engine.league_name.split("—")[0].strip(),
            partido=f"{r['name_a']} vs {r['name_b']}",
            pick=pick, lado=lado, prob_modelo=prob, momio=odd,
            cuota=dec, edge_pct=k.get("edge_pct", 0), stake_mxn=k.get("stake_mxn", 0),
        )
        self._reg_msg.configure(text=f"✓ Registrada: {pick} (${k.get('stake_mxn',0):,.0f})")
        self._btn_reg.configure(state="disabled", text="Registrada ✓")


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
        ctk.CTkButton(toolbar, text="📅  Cargar partidos del día",
                      fg_color=C_GOLD, text_color=C_DARK2, hover_color=C_GOLD2,
                      font=("Segoe UI", 12, "bold"), height=36,
                      command=self._load_today
                      ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(toolbar, text="+ Agregar manual",
                      fg_color=C_CARD, hover_color=C_BORDER,
                      font=("Segoe UI", 12), height=36,
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
        for text, w in [("Partido", 170), ("Momio A", 78), ("Momio B", 78),
                         ("Prob.", 70), ("Edge", 70), ("Stake", 80), ("Señal", 130)]:
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

    def _load_today(self):
        if not self.engine.team_names:
            messagebox.showwarning("Sin datos", "Carga primero los datos de la liga.")
            return
        fixtures = self.engine.fetch_upcoming_matches()
        if not fixtures:
            messagebox.showinfo(
                "Sin partidos próximos",
                "No encontré partidos próximos para esta liga.\n\n"
                "Puede ser que no haya jornada programada ahora, o que falte tu "
                "API Key de PandaScore en Configuración (se usa SOLO para traer "
                "el calendario; los datos y el modelo son de Oracle's Elixir)."
            )
            return
        existing = {(r["name_a"], r["name_b"]) for r in self._rows}
        added = 0
        for fx in fixtures:
            if (fx["name_a"], fx["name_b"]) in existing:
                continue
            self._rows.append({"name_a": fx["name_a"], "name_b": fx["name_b"],
                               "odd_a": None, "odd_b": None, "side": "blue"})
            added += 1
        self._render_rows()
        self._summary_lbl.configure(
            text=f"Cargué {added} partido(s). Escribe los momios de Codere en cada "
                 f"fila y pulsa «Escanear todo».")

    def _remove_row(self, row: dict):
        if row in self._rows:
            self._rows.remove(row)
        self._render_rows()

    @staticmethod
    def _parse_odd(txt: str):
        try:
            v = int(str(txt).replace("+", "").replace(" ", ""))
            return v if abs(v) >= 100 else None
        except (ValueError, TypeError):
            return None

    def _scan_all(self):
        if not self._rows:
            messagebox.showinfo("Sin partidos", "Agrega al menos un partido primero.")
            return
        # Leer los momios escritos en las casillas de cada fila
        for row in self._rows:
            ea, eb = row.get("_ea"), row.get("_eb")
            if ea is not None and eb is not None:
                row["odd_a"] = self._parse_odd(ea.get())
                row["odd_b"] = self._parse_odd(eb.get())
        for row in self._rows:
            row.pop("result", None)
            row.pop("error", None)
            if row.get("odd_a") is None or row.get("odd_b") is None:
                row["error"] = "Momios inválidos (ej: -150 / +130)"
                continue
            try:
                row["result"] = self.engine.predict_match(
                    row["name_a"], row["name_b"], row.get("side", "blue"),
                    row["odd_a"], row["odd_b"])
            except Exception as e:
                row["error"] = str(e)
        self._render_rows()

    def _render_rows(self):
        for w in self._results_frame.winfo_children():
            w.destroy()

        total_stake = total_ev = n_value = 0

        for row in self._rows:
            r = row.get("result")
            line = ctk.CTkFrame(self._results_frame, fg_color=C_PANEL, corner_radius=8)
            line.pack(fill="x", padx=10, pady=3)

            def _lbl(text, color=C_TEXT, w=None):
                kw = dict(font=("Segoe UI", 11), text_color=color, anchor="w")
                if w:
                    kw["width"] = w
                ctk.CTkLabel(line, text=text, **kw).pack(side="left", padx=4, pady=6)

            _lbl(f"{row['name_a']} vs {row['name_b']}"[:24], w=170)

            # Casillas de momio (siempre editables — escribe las de Codere)
            ea = ctk.CTkEntry(line, width=70, placeholder_text="-150", justify="center",
                              fg_color=C_PANEL, border_color=C_BORDER)
            if row.get("odd_a") is not None:
                ea.insert(0, f"{row['odd_a']:+d}")
            ea.pack(side="left", padx=4, pady=6)
            eb = ctk.CTkEntry(line, width=70, placeholder_text="+130", justify="center",
                              fg_color=C_PANEL, border_color=C_BORDER)
            if row.get("odd_b") is not None:
                eb.insert(0, f"{row['odd_b']:+d}")
            eb.pack(side="left", padx=4, pady=6)
            row["_ea"], row["_eb"] = ea, eb

            ctk.CTkButton(line, text="✕", width=26, height=26, fg_color=C_CARD,
                          hover_color=C_RED, text_color=C_MUTED,
                          command=lambda rr=row: self._remove_row(rr)).pack(side="right", padx=6)

            if r:
                ka, kb = r["kelly_a"], r["kelly_b"]
                a_better  = ka.get("edge_pct", -999) >= kb.get("edge_pct", -999)
                best_k    = ka if a_better else kb
                best_prob = r["prob_a"] if a_better else r["prob_b"]
                edge  = best_k.get("edge_pct", 0)
                stake = best_k.get("stake_mxn", 0)
                ev    = best_k.get("ev_mxn", 0)
                is_v  = best_k.get("is_value", False)

                _lbl(f"{best_prob*100:.0f}%", w=70)
                _lbl(f"{edge:+.1f}%", color=(C_GREEN if edge > 0 else C_RED), w=70)
                _lbl(f"${stake:,.0f}" if is_v else "—", w=80)
                _lbl("🥇 VALOR" if is_v else "Sin ventaja",
                     color=(C_GREEN if is_v else C_MUTED), w=130)

                if is_v:
                    n_value += 1
                    total_stake += stake
                    total_ev    += ev
            elif row.get("error"):
                _lbl(row["error"], color=C_RED)

        # Summary (solo tras escanear)
        if any(row.get("result") for row in self._rows):
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
#  FRAME: REGISTRO DE APUESTAS
# ═══════════════════════════════════════════════════════════════
class BetLogFrame(ctk.CTkFrame):
    """Historial de apuestas: marca resultados y mide tu rendimiento real."""

    def __init__(self, master, engine: PredictionEngine, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.engine = engine
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Registro de apuestas",
                     font=("Segoe UI", 20, "bold"), text_color=C_TEXT
                     ).pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(self, text="Marca cada apuesta como ganada o perdida para medir si el "
                                "sistema te da ganancias de verdad.",
                     font=("Segoe UI", 11), text_color=C_MUTED).pack(anchor="w", pady=(0, 12))

        # ── Resumen (tarjetas) ──
        self._sumrow = ctk.CTkFrame(self, fg_color="transparent")
        self._sumrow.pack(fill="x", pady=(0, 12))
        self._sumrow.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self._c_bets    = KPICard(self._sumrow, "Apuestas",        "0", "registradas")
        self._c_win     = KPICard(self._sumrow, "Aciertos",        "—", "de resueltas")
        self._c_profit  = KPICard(self._sumrow, "Ganancia neta",   "$0", "MXN", accent=True)
        self._c_yield   = KPICard(self._sumrow, "Yield",           "—", "ganancia / arriesgado", accent=True)
        self._c_pending = KPICard(self._sumrow, "Pendientes",      "0", "sin resultado")
        for i, c in enumerate([self._c_bets, self._c_win, self._c_profit, self._c_yield, self._c_pending]):
            c.grid(row=0, column=i, sticky="nsew", padx=(0, 8 if i < 4 else 0))

        # ── Tabla ──
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C_CARD, corner_radius=12)
        self._scroll.pack(fill="both", expand=True)

        hdr = ctk.CTkFrame(self._scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        for text, w in [("Fecha", 96), ("Partido", 190), ("Apuesta", 120), ("Cuota", 60),
                        ("Stake", 70), ("Resultado", 200), ("Ganancia", 90), ("", 30)]:
            ctk.CTkLabel(hdr, text=text, font=("Segoe UI", 10, "bold"),
                         text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=4)
        ctk.CTkFrame(self._scroll, fg_color=C_BORDER, height=1).pack(fill="x", padx=10)

        self._rows_frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._rows_frame.pack(fill="both", expand=True)

        self._empty_lbl = ctk.CTkLabel(self._rows_frame,
                                       text="Aún no hay apuestas. Regístralas desde Match Analyzer.",
                                       font=("Segoe UI", 11), text_color=C_MUTED)

    def refresh(self):
        for w in self._rows_frame.winfo_children():
            w.destroy()

        bets = bet_tracker.all_bets()
        s = bet_tracker.summary()
        self._c_bets.update_value(str(s["n_total"]), "registradas")
        self._c_win.update_value(
            f"{s['n_won']}/{s['n_settled']}" if s["n_settled"] else "—",
            f"{s['win_rate']*100:.0f}% acierto" if s["n_settled"] else "sin resueltas")
        col_p = C_GREEN if s["profit"] >= 0 else C_RED
        self._c_profit.update_value(f"${s['profit']:+,.0f}", "MXN")
        self._c_profit._vl.configure(text_color=col_p)
        self._c_yield.update_value(f"{s['yield_pct']:+.1f}%" if s["n_settled"] else "—",
                                   "ganancia / arriesgado")
        self._c_yield._vl.configure(text_color=col_p if s["n_settled"] else C_TEXT)
        self._c_pending.update_value(str(s["n_pending"]), "sin resultado")

        if not bets:
            self._empty_lbl.pack(pady=30)
            return

        for b in bets:
            self._render_row(b)

    def _render_row(self, b: dict):
        estado = b.get("estado", "pendiente")
        line = ctk.CTkFrame(self._rows_frame, fg_color=C_PANEL, corner_radius=8)
        line.pack(fill="x", padx=10, pady=3)

        def lbl(text, w, color=C_TEXT):
            ctk.CTkLabel(line, text=text, font=("Segoe UI", 11), text_color=color,
                         width=w, anchor="w").pack(side="left", padx=4, pady=7)

        lbl(b.get("fecha", "")[:10], 96, C_MUTED)
        lbl(b.get("partido", "")[:24], 190)
        lbl(f"{b.get('pick','')[:10]} {b.get('cuota','')}", 120, C_GOLD)
        lbl(b.get("cuota", ""), 60)
        lbl(f"${float(b.get('stake_mxn',0)):,.0f}", 70)

        # Botones de resultado
        seg = ctk.CTkFrame(line, fg_color="transparent", width=200)
        seg.pack(side="left", padx=4)
        for est, txt, col in [("ganada", "✓ Ganó", C_GREEN),
                              ("perdida", "✗ Perdió", C_RED),
                              ("pendiente", "Pend.", C_MUTED)]:
            active = (estado == est)
            ctk.CTkButton(
                seg, text=txt, width=58, height=26,
                font=("Segoe UI", 10, "bold" if active else "normal"),
                fg_color=col if active else C_CARD,
                text_color=C_DARK2 if active else C_MUTED,
                hover_color=col,
                command=lambda i=b["id"], e=est: self._set(i, e),
            ).pack(side="left", padx=1)

        gan = float(b.get("ganancia_mxn", 0) or 0)
        gcol = C_GREEN if gan > 0 else (C_RED if gan < 0 else C_MUTED)
        lbl(f"${gan:+,.0f}" if estado != "pendiente" else "—", 90, gcol)

        ctk.CTkButton(line, text="✕", width=26, height=26, fg_color=C_CARD,
                      hover_color=C_RED, text_color=C_MUTED,
                      command=lambda i=b["id"]: self._delete(i)).pack(side="left", padx=2)

    def _set(self, bet_id, estado):
        bet_tracker.set_estado(bet_id, estado)
        self.refresh()

    def _delete(self, bet_id):
        bet_tracker.delete_bet(bet_id)
        self.refresh()


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
            ("registro",   "📒  Registro",       self._go_registro),
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
            "registro":  BetLogFrame(self._content, self.engine),
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
    def _go_registro(self):   self._frames["registro"].refresh(); self._show("registro")
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
