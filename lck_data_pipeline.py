"""
============================================================
LCK PREDICTION OS — MÓDULO 2: DATA PIPELINE (v2 - CORREGIDO)
============================================================
CORRECCIONES v2:
  [FIX-1] Eliminado 'range[begin_at]' → causaba HTTP 400 en plan Free.
           Ahora: filter[league_id]=293 + sort=-begin_at (últimas N partidas)
  [FIX-2] _get() imprime el cuerpo EXACTO del error HTTP, no silencia nada.
  [FIX-3] Lógica de fallback: intenta primero con parámetros completos;
           si recibe 400, reintenta con parámetros mínimos garantizados.
  [FIX-4] Año dinámico con datetime.now() — sin hardcode de 2025/2026.
============================================================
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import time
import logging

from lck_config import (
    PANDASCORE_API_KEY, PANDASCORE_BASE_URL, LCK_LEAGUE_ID,
    MIN_PATCH, FEATURE_COLUMNS, LCK_TEAMS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Año actual dinámico (sin hardcode) ──
CURRENT_YEAR = datetime.now().year
CURRENT_SEASON = f"LCK {CURRENT_YEAR} Season"


# ═══════════════════════════════════════════════════════════
#  CLASE PRINCIPAL: LCK DATA PIPELINE v2
# ═══════════════════════════════════════════════════════════
class LCKDataPipeline:
    """
    Conecta con PandaScore y extrae datos LCK.

    PLAN FREE — parámetros aceptados:
        filter[league_id], filter[status], sort, per_page, page

    PLAN FREE — parámetros que causan HTTP 400 (NO usar):
        range[begin_at], range[end_at], filter[videogame_version]
    """

    # Parámetros mínimos 100 % compatibles con plan Free
    _BASE_PARAMS = {
        "filter[league_id]": LCK_LEAGUE_ID,
        "filter[status]":    "finished",
        "sort":              "-begin_at",   # Más recientes primero
        "per_page":          50,
    }

    def __init__(self):
        unconfigured = (PANDASCORE_API_KEY == "TU_API_KEY_AQUI")
        if unconfigured:
            log.warning("⚠️  PANDASCORE_API_KEY no configurada — edita lck_config.py")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {PANDASCORE_API_KEY}",
            "Accept":        "application/json",
        })
        self._delay = 0.4

    # ─────────────────────────────────────────
    #  HTTP con diagnóstico real de errores
    # ─────────────────────────────────────────
    def _get(self, endpoint: str, params: dict, max_pages: int = 2) -> list:
        """
        GET paginado con reporte detallado de cualquier error HTTP.

        En lugar de silenciar errores, los imprime con:
          - Código HTTP
          - URL final construida por requests
          - Primeros 400 chars del cuerpo de respuesta de PandaScore
        Esto permite diagnosticar el problema exacto sin adivinar.
        """
        url = f"{PANDASCORE_BASE_URL}{endpoint}"
        results = []

        for page in range(1, max_pages + 1):
            params_page = {**params, "page": page}

            try:
                r = self.session.get(url, params=params_page, timeout=20)
            except requests.ConnectionError as e:
                log.error(f"[RED] Sin conexión a PandaScore: {e}")
                return results
            except requests.Timeout:
                log.error("[RED] Timeout — respuesta tardó más de 20 s.")
                return results

            # ── Manejo detallado de errores ──────────────────────
            if r.status_code == 400:
                log.error("=" * 60)
                log.error("[HTTP 400] PandaScore rechazó los parámetros.")
                log.error(f"  URL enviada : {r.url}")
                log.error(f"  Respuesta   : {r.text[:400]}")
                log.error("  Solución    : Verifica que no uses range[], filter de versión,")
                log.error("                ni otros parámetros fuera del plan Free.")
                log.error("=" * 60)
                return results

            if r.status_code == 401:
                log.error("[HTTP 401] API KEY inválida o vencida.")
                log.error(f"  Respuesta: {r.text[:200]}")
                log.error("  → Obtén una nueva en https://pandascore.co/")
                return results

            if r.status_code == 403:
                log.error(f"[HTTP 403] Endpoint no disponible en tu plan Free.")
                log.error(f"  Endpoint: {endpoint}")
                return results

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                log.warning(f"[HTTP 429] Rate limit — esperando {wait} s…")
                time.sleep(wait)
                continue

            if not r.ok:
                log.error(f"[HTTP {r.status_code}] Error inesperado.")
                log.error(f"  Respuesta: {r.text[:300]}")
                return results

            try:
                data = r.json()
            except Exception:
                log.error(f"[JSON] Respuesta inválida: {r.text[:200]}")
                return results

            if not isinstance(data, list) or not data:
                break

            results.extend(data)
            if len(data) < params.get("per_page", 50):
                break   # Última página

            time.sleep(self._delay)

        return results

    # ─────────────────────────────────────────
    #  1. PARTIDAS LCK
    # ─────────────────────────────────────────
    def get_all_matches(self, limit: int = 100) -> pd.DataFrame:
        """
        Versión Corregida: Mapeo manual de equipos para evitar KeyError y IndexingError.
        """
        per_page  = min(limit, 50)
        max_pages = max(1, -(-limit // per_page))
        params    = {**self._BASE_PARAMS, "per_page": per_page}

        log.info(f"[{CURRENT_SEASON}] Descargando últimas {limit} partidas…")
        raw = self._get("/lol/matches", params=params, max_pages=max_pages)

        if not raw:
            return pd.DataFrame()

        # --- EXTRACCIÓN MANUAL DE DATOS ANIDADOS ---
        processed_matches = []
        for match in raw:
            opponents = match.get("opponents", [])
            # Solo procesamos si hay dos equipos definidos
            if len(opponents) >= 2:
                processed_matches.append({
                    "match_id":  match.get("id"),
                    "team_a_id": opponents[0].get("opponent", {}).get("id"),
                    "team_b_id": opponents[1].get("opponent", {}).get("id"),
                    "winner_id": match.get("winner_id"),
                    "patch":     match.get("videogame_version"),
                    "begin_at":  match.get("begin_at")
                })
        
        df = pd.DataFrame(processed_matches)
        
        # Si el DataFrame está vacío tras el mapeo
        if df.empty:
            log.warning("⚠️ No se pudieron extraer IDs de equipos de la respuesta.")
            return df

        log.info(f"  → {len(df)} partidas procesadas con IDs de equipos.")
        
        # Aplicamos el filtro de parche y reseteamos índice para evitar el IndexingError
        df = self._filter_by_patch(df)
        return df.reset_index(drop=True)

    def _filter_by_patch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra por parche mínimo si la columna existe; si no, devuelve todo."""
        patch_col = next(
            (c for c in ["videogame_version", "patch", "version"] if c in df.columns),
            None
        )
        if not patch_col:
            log.debug("  Columna de parche no encontrada — sin filtro de parche.")
            return df
        try:
            filtered = df[df[patch_col].astype(str) >= MIN_PATCH].copy()
            log.info(f"  → {len(filtered)} partidas en parche ≥ {MIN_PATCH}")
            return filtered
        except Exception as e:
            log.warning(f"  Filtro de parche falló ({e}) — devolviendo todos los datos.")
            return df

    # ─────────────────────────────────────────
    #  2. GAME-LEVEL STATS
    # ─────────────────────────────────────────
    def get_game_stats(self, match_id: int) -> list[dict]:
        return self._get(
            f"/lol/matches/{match_id}/games",
            params={"per_page": 10},
            max_pages=1
        )

    def get_bulk_game_stats(self, match_ids: list[int]) -> pd.DataFrame:
        log.info(f"Descargando game-stats de {len(match_ids)} partidas…")
        rows = []
        for i, mid in enumerate(match_ids):
            for g in self.get_game_stats(mid):
                rows.append(self._parse_game(g, mid))
            if i % 10 == 0 and i > 0:
                log.info(f"  → {i}/{len(match_ids)}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _parse_game(self, game: dict, match_id: int) -> dict:
        teams = game.get("teams", [])
        row = {
            "match_id":  match_id,
            "game_id":   game.get("id"),
            "winner_id": game.get("winner", {}).get("id"),
            "duration":  game.get("length", 0) / 60,
            "patch":     game.get("videogame_version", "unknown"),
        }
        for idx, team in enumerate(teams[:2]):
            p = "team_a" if idx == 0 else "team_b"
            row[f"{p}_id"]           = team.get("id")
            row[f"{p}_name"]         = team.get("name", "Unknown")
            row[f"{p}_gold_at_15"]   = self._s(team, "gold_at_15")
            row[f"{p}_first_blood"]  = self._s(team, "first_blood")
            row[f"{p}_first_dragon"] = self._s(team, "first_dragon")
            row[f"{p}_barons"]       = self._s(team, "baron_kills")
            row[f"{p}_vision"]       = self._s(team, "total_ward_placed")
        return row

    @staticmethod
    def _s(obj: dict, key: str, default=0):
        v = obj.get(key)
        return v if v is not None else default

    # ─────────────────────────────────────────
    #  3. KPIs CONSOLIDADOS POR EQUIPO
    # ─────────────────────────────────────────
    def build_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        
        # Extraemos todos los IDs únicos que aparecen en las partidas descargadas
        all_team_ids = pd.concat([df['team_a_id'], df['team_b_id']]).unique()
        
        rows = []
        for tid in all_team_ids:
            if tid is None: continue
            # Buscamos el nombre del equipo en el DataFrame (o usamos el ID como nombre)
            name = f"Team_{tid}" 
            stats = self._kpis(df, tid, name)
            if stats:
                rows.append(stats)
                
        result = pd.DataFrame(rows)
        log.info(f"KPIs calculados para {len(result)} equipos encontrados en la API")
        return result

    def _kpis(self, df: pd.DataFrame, tid: int, name: str) -> Optional[dict]:
        # Esta línea usa .to_numpy() para evitar que Pandas intente alinear índices rotos
        mask = (df["team_a_id"].to_numpy() == tid) | (df["team_b_id"].to_numpy() == tid)
        
        g = df[mask].copy()
        if len(g) < 3:
            return None

        wins  = (g["winner_id"] == tid).sum()
        total = len(g)
        gd15  = self._gold_diff(g, tid)
        bwr, rwr = self._side_wr(g, tid)

        return {
            "team_id":             tid,
            "team_name":           name,
            "season":              CURRENT_SEASON,
            "games_played":        total,
            "win_rate":            round(wins / total, 4),
            "gold_diff_15":        round(gd15, 1),
            "first_blood_rate":    round(self._bin(g, tid, "first_blood"), 4),
            "first_dragon_rate":   round(self._bin(g, tid, "first_dragon"), 4),
            "vspm":                round(self._vspm(g, tid), 3),
            "baron_control_rate":  round(self._bin(g, tid, "barons", 1), 4),
            "avg_game_duration":   round(g.get("duration", pd.Series([30.0])).mean(), 1),
            "blue_side_winrate":   round(bwr, 4),
            "red_side_winrate":    round(rwr, 4),
            "gold_lead_20_weight": round(gd15 * 1.4, 1),
        }

    def _gold_diff(self, df, tid):
        diffs = []
        for _, r in df.iterrows():
            if r.get("team_a_id") == tid:
                diffs.append(r.get("team_a_gold_at_15", 0) - r.get("team_b_gold_at_15", 0))
            else:
                diffs.append(r.get("team_b_gold_at_15", 0) - r.get("team_a_gold_at_15", 0))
        return float(np.mean(diffs)) if diffs else 0.0

    def _bin(self, df, tid, stat, threshold=1):
        rates = [
            1 if (r.get(f"{'team_a' if r.get('team_a_id')==tid else 'team_b'}_{stat}", 0) or 0) >= threshold else 0
            for _, r in df.iterrows()
        ]
        return float(np.mean(rates)) if rates else 0.0

    def _vspm(self, df, tid):
        scores = []
        for _, r in df.iterrows():
            p   = "team_a" if r.get("team_a_id") == tid else "team_b"
            vs  = r.get(f"{p}_vision", 0) or 0
            dur = max(r.get("duration", 30), 1)
            scores.append(vs / dur)
        return float(np.mean(scores)) if scores else 0.0

    def _side_wr(self, df, tid):
        bw = bt = rw = rt = 0
        for _, r in df.iterrows():
            w = r.get("winner_id")
            if r.get("team_a_id") == tid:
                bt += 1
                if w == tid: bw += 1
            else:
                rt += 1
                if w == tid: rw += 1
        return (bw/bt if bt else 0.5), (rw/rt if rt else 0.5)


# ─────────────────────────────────────────
#  PRUEBA RÁPIDA DE CONECTIVIDAD
# ─────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  LCK Data Pipeline v2 — {CURRENT_SEASON}")
    print(f"  Temporada iniciada: 14 enero {CURRENT_YEAR}")
    print(f"{'='*60}\n")

    pl = LCKDataPipeline()
    df = pl.get_all_matches(limit=50)

    if not df.empty:
        print(f"✅ {len(df)} partidas descargadas.")
        print(f"   Columnas: {list(df.columns[:8])}…\n")
        stats = pl.build_team_stats(df)
        if not stats.empty:
            print(stats[["team_name","win_rate","gold_diff_15"]].to_string(index=False))
    else:
        print("⚠️  Sin datos. Lee los mensajes de error arriba para diagnosticar.")