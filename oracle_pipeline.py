"""
============================================================
PREDICTION OS V2 — DATA PIPELINE: Oracle's Elixir (gratis, detallado)
============================================================
Fuente de datos: Oracle's Elixir (oracleselixir.com), CSVs públicos con stats
detalladas por partido para todas las ligas pro de LoL. Gratis, sin API key.

Por qué Oracle's Elixir en vez de PandaScore:
  - Incluye las stats por partido (oro@15, primera sangre, dragón, barones,
    visión) que el plan Free de PandaScore BLOQUEA (HTTP 403).
  - Más partidos (LCK 2026: ~349 juegos vs. ~100 en PandaScore).
  - No requiere API key.

Flujo:
  pipe   = OraclePipeline(league_code="LCK")
  games  = pipe.load_games(progress_cb)      # filas por (juego, equipo)
  matches= pipe.build_matches(games)         # 1 fila por juego (para el modelo)
  stats  = pipe.build_team_stats(games)      # KPIs por equipo (para la GUI)

Los datos se descargan una vez y se cachean localmente; se re-descargan solo si
el caché tiene más de `cache_hours` horas.
============================================================
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional, Callable

import requests
import numpy as np
import pandas as pd

import config

log = logging.getLogger("OraclePipeline")

# Carpeta pública de Google Drive donde Oracle's Elixir publica los CSV anuales.
_OE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"}

# Ligas domésticas mayores (cada equipo acumula muchos partidos aquí).
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "LCP", "LJL", "CBLOL", "VCS"}

# Torneos INTERNACIONALES: son los que enlazan regiones (un coreano vs. un
# chino solo pasa aquí). Sin ellos, el Elo de cada región quedaría sin escala
# común. Detectados en los datos 2026 de Oracle's Elixir:
#   EWC = Esports World Cup, FST = First Stand, Asia Master, AC.
INTERNATIONAL_LEAGUES = {"EWC", "FST", "Asia Master", "AC"}

# Modo GLOBAL = ligas mayores + torneos internacionales (excluye ligas menores
# y académicas). Da un Elo entre regiones calibrado por los enfrentamientos
# internacionales reales.
GLOBAL_LEAGUES = MAJOR_LEAGUES | INTERNATIONAL_LEAGUES

# Columnas que necesitamos del CSV (165 en total — leemos solo estas por rapidez).
_USECOLS = [
    "gameid", "league", "year", "date", "participantid", "side", "result",
    "teamname", "teamid", "golddiffat15", "firstblood", "firstdragon",
    "barons", "gamelength", "vspm", "patch", "datacompleteness",
]


class OraclePipeline:
    def __init__(self, league_code: str = "LCK",
                 year: Optional[int] = None, cache_hours: float = 12.0):
        self.league_code = league_code.strip().upper()
        self.year = int(year) if year else config.CURRENT_YEAR
        self.cache_hours = cache_hours
        self.current_patch: str = ""
        self._cache_path = os.path.join(config._app_dir(), f"oe_{self.year}.csv")

    # ───────────────────────────────────────────────────────────
    #  DESCARGA + CACHÉ
    # ───────────────────────────────────────────────────────────
    def _resolve_file_id(self) -> Optional[str]:
        """Busca el ID del CSV del año en la carpeta pública de Drive."""
        try:
            t = requests.get(
                f"https://drive.google.com/drive/folders/{_OE_FOLDER_ID}",
                headers=_UA, timeout=30,
            ).text
        except requests.RequestException as e:
            log.error(f"No se pudo abrir la carpeta de Drive: {e}")
            return None

        name = f"{self.year}_LoL_esports_match_data_from_OraclesElixir.csv"
        for m in re.finditer(re.escape(name), t):
            ids = re.findall(r'"([-\w]{28,44})"', t[max(0, m.start() - 120):m.start()])
            if ids:
                return ids[-1]
        log.error(f"No se encontró el archivo {name} en la carpeta de Drive.")
        return None

    def _cache_fresh(self) -> bool:
        if not os.path.exists(self._cache_path):
            return False
        age_h = (time.time() - os.path.getmtime(self._cache_path)) / 3600.0
        return age_h < self.cache_hours

    def _download(self, progress_cb: Optional[Callable] = None) -> bool:
        """Descarga el CSV del año a caché. Retorna True si hay archivo usable."""
        if self._cache_fresh():
            log.info(f"Caché reciente: {self._cache_path}")
            return True

        if progress_cb:
            progress_cb("Buscando base de datos de Oracle's Elixir…")
        fid = self._resolve_file_id()
        if not fid:
            return os.path.exists(self._cache_path)   # usa caché viejo si existe

        url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
        if progress_cb:
            progress_cb(f"Descargando datos {self.year} (~45 MB)…")
        try:
            r = requests.get(url, headers=_UA, timeout=180)
            if r.ok and len(r.content) > 100_000 and r.content[:7] == b"gameid,":
                with open(self._cache_path, "wb") as f:
                    f.write(r.content)
                log.info(f"Descargado {len(r.content)//(1024*1024)} MB → {self._cache_path}")
                return True
            log.error(f"Descarga inválida (status {r.status_code}, {len(r.content)} bytes).")
        except requests.RequestException as e:
            log.error(f"Error de descarga: {e}")
        return os.path.exists(self._cache_path)

    # ───────────────────────────────────────────────────────────
    #  CARGA Y FILTRADO
    # ───────────────────────────────────────────────────────────
    def load_games(self, progress_cb: Optional[Callable] = None) -> pd.DataFrame:
        """
        Devuelve las filas de EQUIPO (participantid 100/200) de la liga,
        ordenadas por fecha. Una fila = (juego, equipo) con sus stats.
        """
        if not self._download(progress_cb):
            log.error("Sin datos de Oracle's Elixir.")
            return pd.DataFrame()

        if progress_cb:
            progress_cb("Cargando base de datos…")
        try:
            df = pd.read_csv(self._cache_path, usecols=_USECOLS, low_memory=False)
        except (OSError, ValueError) as e:
            log.error(f"No se pudo leer el CSV: {e}")
            return pd.DataFrame()

        df = df[df["participantid"].isin([100, 200])].copy()
        if self.league_code == "GLOBAL":
            # Modo internacional: ligas mayores + torneos que enlazan regiones.
            df = df[df["league"].isin(GLOBAL_LEAGUES)].copy()
        else:
            # Modo liga: solo esa liga.
            df = df[df["league"] == self.league_code].copy()
        if df.empty:
            log.warning(f"Sin filas para {self.league_code} en {self.year}.")
            return df

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # Parche más reciente visto
        patches = [str(p) for p in df["patch"].dropna().unique() if str(p)[:1].isdigit()]
        if patches:
            self.current_patch = max(patches, key=lambda p: [int(x) for x in re.findall(r"\d+", p)])

        log.info(f"{self.league_code} {self.year}: {df['gameid'].nunique()} juegos, "
                 f"{len(df)} filas de equipo")
        return df

    # ───────────────────────────────────────────────────────────
    #  CONSTRUCCIÓN DE MATCHES (para el modelo)
    # ───────────────────────────────────────────────────────────
    def build_matches(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Una fila por juego con ambos equipos (a=blue, b=red), el ganador y las
        stats por equipo de ESE juego. El modelo usa esto para construir, en
        orden cronológico, medias móviles pre-partido (sin fuga de datos).
        """
        if games.empty:
            return pd.DataFrame()

        rows = []
        for gid, g in games.groupby("gameid", sort=False):
            if len(g) != 2:
                continue
            blue = g[g["side"] == "Blue"]
            red = g[g["side"] == "Red"]
            if blue.empty or red.empty:
                continue
            a = blue.iloc[0]   # equipo blue
            b = red.iloc[0]    # equipo red
            winner = a["teamid"] if a["result"] == 1 else b["teamid"]
            rows.append({
                "gameid": gid,
                "begin_at": a["date"],
                "team_a_id": a["teamid"], "team_a_name": a["teamname"],
                "team_b_id": b["teamid"], "team_b_name": b["teamname"],
                "winner_id": winner,
                # stats de este juego para cada equipo
                "a_gd15": _f(a["golddiffat15"]), "b_gd15": _f(b["golddiffat15"]),
                "a_fb": _f(a["firstblood"]),     "b_fb": _f(b["firstblood"]),
                "a_fd": _f(a["firstdragon"]),    "b_fd": _f(b["firstdragon"]),
                "a_baron": _f(a["barons"]),      "b_baron": _f(b["barons"]),
                "a_vspm": _f(a["vspm"]),         "b_vspm": _f(b["vspm"]),
                "a_dur": _f(a["gamelength"]) / 60.0, "b_dur": _f(b["gamelength"]) / 60.0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("begin_at").reset_index(drop=True)
        return df

    # ───────────────────────────────────────────────────────────
    #  KPIs POR EQUIPO (para la GUI / display)
    # ───────────────────────────────────────────────────────────
    def build_team_stats(self, games: pd.DataFrame,
                         min_games: int = 3, progress_cb=None) -> pd.DataFrame:
        if games.empty:
            return pd.DataFrame()

        # En modo GLOBAL, mostrar solo equipos de ligas mayores (para que el
        # selector no tenga cientos de equipos de ligas menores).
        global_major = None
        if self.league_code == "GLOBAL":
            home = games.groupby("teamid")["league"].agg(
                lambda s: s.value_counts().idxmax())
            global_major = set(home[home.isin(MAJOR_LEAGUES)].index)

        rows = []
        for tid, g in games.groupby("teamid"):
            n = len(g)
            if n < min_games:
                continue
            if global_major is not None and tid not in global_major:
                continue
            blue = g[g["side"] == "Blue"]
            red = g[g["side"] == "Red"]
            rows.append({
                "team_id": tid,
                "team_name": g["teamname"].mode().iloc[0] if not g["teamname"].mode().empty else str(tid),
                "games_played": n,
                "win_rate": round(g["result"].mean(), 4),
                "blue_side_winrate": round(blue["result"].mean(), 4) if len(blue) else 0.5,
                "red_side_winrate": round(red["result"].mean(), 4) if len(red) else 0.5,
                "gold_diff_15": round(g["golddiffat15"].mean(), 1),
                "first_blood_rate": round(g["firstblood"].mean(), 4),
                "first_dragon_rate": round(g["firstdragon"].mean(), 4),
                "baron_control_rate": round((g["barons"] > 0).mean(), 4),
                "vspm": round(g["vspm"].mean(), 3),
                "avg_game_duration": round(g["gamelength"].mean() / 60.0, 1),
                "gold_lead_20_weight": round(g["golddiffat15"].mean() * 1.4, 1),
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("win_rate", ascending=False).reset_index(drop=True)
        log.info(f"KPIs por equipo: {len(result)} equipos")
        return result


def _f(v, default: float = 0.0) -> float:
    """Convierte a float seguro (NaN/None → default)."""
    try:
        x = float(v)
        return x if x == x else default   # x!=x detecta NaN
    except (TypeError, ValueError):
        return default


# ───────────────────────────────────────────────────────────────
#  Diagnóstico rápido
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipe = OraclePipeline(league_code="LCK")
    games = pipe.load_games(progress_cb=lambda m: print("  ·", m))
    if games.empty:
        print("Sin datos.")
    else:
        matches = pipe.build_matches(games)
        stats = pipe.build_team_stats(games)
        print(f"\nParche: {pipe.current_patch} | juegos: {len(matches)} | equipos: {len(stats)}\n")
        for _, r in stats.head(10).iterrows():
            print(f"  {r['team_name']:20} WR={r['win_rate']*100:4.0f}%  "
                  f"GD@15={r['gold_diff_15']:+6.0f}  FB={r['first_blood_rate']*100:3.0f}%  "
                  f"Baron={r['baron_control_rate']*100:3.0f}%  vspm={r['vspm']:.1f}")
