"""
============================================================
PREDICTION OS V2 — CONFIGURACIÓN CENTRAL (multi-liga)
============================================================
Este archivo NO contiene secretos y puede subirse a git.

La API key de PandaScore se resuelve, en orden:
  1. Variable de entorno  PANDASCORE_API_KEY
  2. Archivo local        secrets_local.py  (gitignored)

Para configurarla (elige una):
  • PowerShell:   $env:PANDASCORE_API_KEY = "tu_key"
  • o crea secrets_local.py con:  PANDASCORE_API_KEY = "tu_key"
============================================================
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


# ─────────────────────────────────────────────
#  🔑  API KEY (sin hardcode, segura para .exe)
# ─────────────────────────────────────────────
def _app_dir() -> str:
    """Directorio de la app: junto al .exe si está empaquetado, o al fuente."""
    if getattr(sys, "frozen", False):          # ejecutable PyInstaller
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# Archivo de texto plano (una línea) junto a la app. NUNCA se empaqueta
# dentro del .exe, así que la llave no queda extraíble del binario.
KEY_FILE = os.path.join(_app_dir(), "api_key.txt")


def get_api_key() -> str:
    """
    Resuelve la API key en orden:
      1. Variable de entorno PANDASCORE_API_KEY
      2. Archivo api_key.txt junto a la app
      3. secrets_local.py (solo en modo desarrollo)
    """
    key = os.getenv("PANDASCORE_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
        except OSError:
            pass
    try:
        import secrets_local  # type: ignore
        return str(getattr(secrets_local, "PANDASCORE_API_KEY", "")).strip()
    except ImportError:
        return ""


def save_api_key(key: str) -> None:
    """Guarda la API key en api_key.txt para que persista entre sesiones."""
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())


PANDASCORE_API_KEY  = get_api_key()
PANDASCORE_BASE_URL = "https://api.pandascore.co"

CURRENT_YEAR = datetime.now().year   # Dinámico — nunca hardcoded


# ─────────────────────────────────────────────
#  🏆  LIGAS SOPORTADAS  (nombre visible → código de liga en Oracle's Elixir)
#      El primer token del nombre DEBE ser el código de Oracle's Elixir.
#      Todas viven en el mismo CSV anual: cambiar de liga no re-descarga nada.
# ─────────────────────────────────────────────
LEAGUES: dict[str, str] = {
    "LCK  — Corea":          "LCK",
    "LPL  — China":          "LPL",
    "LEC  — Europa":         "LEC",
    "LCS  — Norteamérica":   "LCS",
    "LCP  — Asia-Pacífico":  "LCP",
    "LJL  — Japón":          "LJL",
    "CBLOL — Brasil":        "CBLOL",
    "VCS  — Vietnam":        "VCS",
    "LFL  — Francia":        "LFL",
    "EM  — EMEA Masters":    "EM",
    "GLOBAL — Internacional (MSI/Worlds)": "GLOBAL",
}
DEFAULT_LEAGUE_NAME = "LCK  — Corea"

# Compatibilidad con universal_pipeline.py (cliente PandaScore, no usado por defecto)
DEFAULT_LEAGUE_ID = 293


# ─────────────────────────────────────────────
#  📊  FEATURES DEL MODELO
#      El modelo entrena sobre DIFERENCIAS (A - B) de estas columnas.
#      Tier-1: siempre disponibles desde /lol/matches.
#      Tier-2: requieren stats por partida (/lol/games); pueden venir en 0.
# ─────────────────────────────────────────────
FEATURES_TIER1 = [
    "win_rate",
    "blue_side_winrate",
    "red_side_winrate",
    "baron_control_rate",
]
FEATURES_TIER2 = [
    "gold_diff_15",
    "first_blood_rate",
    "first_dragon_rate",
    "vspm",
    "gold_lead_20_weight",
]
FEATURE_COLUMNS = FEATURES_TIER1 + FEATURES_TIER2

# Una feature Tier-2 se considera utilizable si al menos este % de filas
# tiene valores no-cero. Si ninguna lo cumple, el modelo corre en modo "lite".
TIER2_MIN_NONZERO = 0.20


# ─────────────────────────────────────────────
#  🤖  PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":      150,
    "max_depth":         3,        # Bajo: el dataset es pequeño (evita overfit)
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}

# CV cronológico (TimeSeriesSplit). Se reduce solo si hay pocos partidos.
TIME_SERIES_SPLITS = 4

# Nº mínimo de partidos reales para entrenar el modelo.
# Por debajo, se usa el fallback (regresión logística sobre win_rate).
MIN_MATCHES_FOR_ML = 20

# Equipos con menos de estas partidas se descartan al calcular KPIs.
MIN_GAMES_PER_TEAM = 3

# ─────────────────────────────────────────────
#  ♟️  SISTEMA DE RATING ELO (entrenamiento cronológico, sin fuga de datos)
# ─────────────────────────────────────────────
ELO_BASE        = 1500.0   # rating inicial de cada equipo
ELO_K           = 30.0     # velocidad de ajuste tras cada partido
ELO_FORM_WINDOW = 5        # nº de partidos recientes para la "forma" reciente


# ─────────────────────────────────────────────
#  🗺️  AJUSTE DE LADO DEL MAPA (heurístico, post-modelo)
#      Blue side gana ~52-53% histórico en ligas mayores.
# ─────────────────────────────────────────────
BLUE_SIDE_BONUS = 0.02   # +2% prob al equipo en blue side


# ─────────────────────────────────────────────
#  💰  BANKROLL Y KELLY
# ─────────────────────────────────────────────
BANKROLL           = 1_000.0    # MXN — capital inicial
KELLY_FRACTION     = 0.25       # Kelly fraccional conservador (25%)
MIN_EDGE_THRESHOLD = 0.07       # Ventaja mínima para considerar value bet (7%)
MIN_STAKE          = 20.0       # Apuesta mínima en MXN
MAX_STAKE_PCT      = 0.10       # Nunca apostar más del 10% del bankroll
