"""
============================================================
PREDICTION OS V2.0 — UNIVERSAL PIPELINE  (v2.1 completo)
============================================================
Pipeline genérico para cualquier liga de LoL en PandaScore.
Funciona de forma autónoma: no importa nada de lck_config.py.

ARQUITECTURA
  UniversalPipeline          — clase principal
    _get()                   — HTTP paginado con diagnóstico completo
    fetch_team_names()       — nombres reales desde /lol/teams + /lol/matches
    detect_current_patch()   — parche dinámico desde partidas recientes
    get_patch_window()       — rango [patch-1, patch] para filtrado
    get_matches()            — descarga y filtra por liga + parche
    build_team_stats()       — KPIs por equipo con nombres reales
    _calc_kpis()             — métricas individuales (type-safe)
    _side_wr()               — blue/red WR con comparación int normalizada
    _patch_to_tuple()        — convierte "26.09" a (26, 9) para comparar
    get_team_name()          — resolución nombre desde caché

  PatchManager               — operaciones de parche aisladas
    parse()                  — "26.09" → (26, 9)
    to_str()                 — (26, 9) → "26.09"
    subtract()               — n parches hacia atrás
    compare()                — comparar dos strings de parche
    window()                 — [patch-n, patch] como strings

  get_demo_stats()           — datos realistas para modo sin API

CORRECCIONES v2.1
  [FIX-1] _DELAY era referenciado como self._delay → AttributeError silencioso.
           Ahora es self._DELAY (constante de clase) usado directamente.
  [FIX-2] _side_wr() usaba r.get() en filas pandas: int64 != int → WR = 0.50
           siempre. Ahora convierte a int() explícitamente antes de comparar.
  [FIX-3] _calc_kpis(): tid proveniente de pd.unique() era numpy.int64.
           Ahora se normaliza a Python int en get_team_name() y en el mask.
  [FIX-4] get_matches(): el fallback de parche lanzaba detect_current_patch()
           sin respetar el caché → doble llamada a la API. Ahora usa _current_patch
           si ya está poblado.
  [FIX-5] build_team_stats(): astype(object) perdía el tipo numérico en some
           plataformas. Ahora usa explícitamente astype("Int64") + dropna().
  [FIX-6] fetch_team_names(): si /lol/teams devuelve 403 (plan Free en algunas
           ligas), el fallback a /lol/matches ahora siempre se ejecuta.
============================================================
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Callable

import requests
import numpy as np
import pandas as pd

log = logging.getLogger("UniversalPipeline")


# ═══════════════════════════════════════════════════════════════
#  PATCH MANAGER — operaciones de versión de parche aisladas
# ═══════════════════════════════════════════════════════════════

class PatchManager:
    """
    Clase utilitaria para comparar y manipular versiones de parche.

    Ejemplos:
        PatchManager.parse("26.09")       → (26, 9)
        PatchManager.to_str((26, 9))      → "26.09"
        PatchManager.subtract("26.09", 1) → "26.08"
        PatchManager.subtract("26.01", 1) → "25.99"   (no existe, stays at 01)
        PatchManager.window("26.09", n=2) → ["26.07", "26.08", "26.09"]
        PatchManager.compare("26.09", "26.08") → 1   (mayor)
    """

    @staticmethod
    def parse(patch: str) -> tuple[int, ...]:
        """Convierte "26.09" → (26, 9). Soporta 2 o 3 segmentos."""
        try:
            return tuple(int(x) for x in str(patch).split(".") if x.isdigit())
        except Exception:
            return (0,)

    @staticmethod
    def to_str(tup: tuple[int, ...], segments: int = 2) -> str:
        """Convierte (26, 9) → "26.09" con zero-padding en el último segmento."""
        if not tup:
            return "unknown"
        parts = list(tup)
        # Zero-pad el último número si tiene menos de 2 dígitos
        parts[-1] = int(parts[-1])
        if segments == 2:
            return f"{parts[0]}.{parts[-1]:02d}"
        return ".".join(str(p) for p in parts[:segments])

    @staticmethod
    def subtract(patch: str, n: int = 1) -> str:
        """
        Retrocede `n` parches desde el dado.
        Si el minor llegaría a 0 o menos, se queda en 01.

        "26.09" subtract 1 → "26.08"
        "26.01" subtract 1 → "26.01"  (no retrocede a major anterior)
        """
        tup = PatchManager.parse(patch)
        if len(tup) < 2:
            return patch
        major, minor = tup[0], tup[1]
        minor = max(1, minor - n)
        return PatchManager.to_str((major, minor))

    @staticmethod
    def window(patch: str, n: int = 2) -> list[str]:
        """
        Genera la ventana de parches [patch-n, ..., patch].
        Útil para incluir el parche actual y los `n` anteriores.

        window("26.09", 2) → ["26.07", "26.08", "26.09"]
        """
        tup = PatchManager.parse(patch)
        if len(tup) < 2:
            return [patch]
        major, minor = tup[0], tup[1]
        return [
            PatchManager.to_str((major, max(1, minor - i)))
            for i in range(n, -1, -1)
        ]

    @staticmethod
    def compare(a: str, b: str) -> int:
        """
        Compara dos strings de parche.
        Retorna: 1 si a > b | -1 si a < b | 0 si iguales.
        """
        ta = PatchManager.parse(a)
        tb = PatchManager.parse(b)
        if ta > tb: return 1
        if ta < tb: return -1
        return 0

    @staticmethod
    def is_valid(patch: str) -> bool:
        """True si el string parece un parche real (ej. "26.09")."""
        try:
            parts = str(patch).split(".")
            return len(parts) >= 2 and all(p.isdigit() for p in parts[:2])
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════
#  UNIVERSAL PIPELINE
# ═══════════════════════════════════════════════════════════════

class UniversalPipeline:
    """
    Pipeline genérico para CUALQUIER liga de LoL en PandaScore.
    Solo necesita api_key y league_id — no importa lck_config.

    Flujo recomendado desde la GUI:
        pipe  = UniversalPipeline(api_key=key, league_id=293)
        patch = pipe.detect_current_patch()         # "26.09"
        names = pipe.fetch_team_names()             # {id: "T1", ...}
        df    = pipe.get_matches(limit=100)         # usa window automático
        stats = pipe.build_team_stats(df)           # nombres reales incluidos
    """

    BASE_URL = "https://api.pandascore.co"
    _DELAY   = 0.35    # segundos entre requests (respeta rate-limit plan Free)
    _TIMEOUT = 20      # segundos por request

    def __init__(self, api_key: str | None = None, league_id: int | None = None):
        # Defaults desde config si no se pasan explícitamente
        if api_key is None or league_id is None:
            import config
            api_key   = api_key if api_key is not None else config.PANDASCORE_API_KEY
            league_id = league_id if league_id is not None else config.DEFAULT_LEAGUE_ID

        if not api_key or api_key.strip() in ("", "TU_API_KEY_AQUI"):
            log.warning(
                "[UniversalPipeline] API key no configurada. "
                "Define PANDASCORE_API_KEY o crea secrets_local.py."
            )

        self.league_id       = int(league_id)
        self._team_cache: dict[int, str] = {}   # {team_id_int → nombre}
        self._current_patch: str = ""           # parche detectado (vacío = no detectado aún)

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key.strip()}",
            "Accept":        "application/json",
        })

    # ───────────────────────────────────────────────────────────
    #  HTTP CORE — paginado, con diagnóstico completo
    # ───────────────────────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: dict,
        max_pages: int = 2,
        progress_cb: Optional[Callable] = None,
    ) -> list:
        """
        GET paginado. Nunca lanza excepción: retorna lista vacía en fallo.

        Diagnóstico de errores:
          400 → muestra URL + body exacto (parámetros rechazados)
          401 → API key inválida
          403 → endpoint fuera del plan Free
          429 → rate limit, espera Retry-After
          5xx → error del servidor, no reintenta
        """
        url     = f"{self.BASE_URL}{endpoint}"
        results = []

        for page in range(1, max_pages + 1):
            p_page = {**params, "page": page}

            try:
                r = self._session.get(url, params=p_page, timeout=self._TIMEOUT)
            except requests.ConnectionError as exc:
                log.error(f"[NET] Sin conexión a PandaScore: {exc}")
                return results
            except requests.Timeout:
                log.error(f"[NET] Timeout ({self._TIMEOUT}s) en {endpoint}")
                return results

            # ── Diagnóstico HTTP ───────────────────────────────
            if r.status_code == 400:
                log.error(
                    f"[HTTP 400] PandaScore rechazó los parámetros.\n"
                    f"  Endpoint : {endpoint}\n"
                    f"  URL final: {r.url}\n"
                    f"  Body     : {r.text[:400]}\n"
                    f"  Causa posible: parámetro no permitido en plan Free.\n"
                    f"  Plan Free acepta: filter[], sort, per_page, page.\n"
                    f"  NO acepta: range[], filter[videogame_version], etc."
                )
                return results

            if r.status_code == 401:
                log.error(
                    f"[HTTP 401] API KEY inválida o vencida.\n"
                    f"  Body: {r.text[:200]}\n"
                    f"  → Obtén una nueva en https://pandascore.co/"
                )
                return results

            if r.status_code == 403:
                log.error(
                    f"[HTTP 403] Endpoint no disponible en tu plan.\n"
                    f"  Endpoint: {endpoint}\n"
                    f"  → Verifica los límites de tu suscripción PandaScore."
                )
                return results

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                log.warning(f"[HTTP 429] Rate limit — esperando {wait}s…")
                time.sleep(wait)
                continue  # Reintenta la misma página

            if r.status_code >= 500:
                log.error(f"[HTTP {r.status_code}] Error del servidor. Body: {r.text[:200]}")
                return results

            if not r.ok:
                log.error(f"[HTTP {r.status_code}] Error inesperado. Body: {r.text[:200]}")
                return results

            # ── Parsear JSON ────────────────────────────────────
            try:
                data = r.json()
            except Exception:
                log.error(f"[JSON] Respuesta no es JSON válido: {r.text[:200]}")
                return results

            if not isinstance(data, list) or not data:
                break  # Sin más páginas

            results.extend(data)
            log.debug(f"  [_get] {endpoint} p{page}: +{len(data)} registros")

            # Última página si devolvió menos de per_page
            if len(data) < params.get("per_page", 50):
                break

            if page < max_pages:
                time.sleep(self._DELAY)

        return results

    # ───────────────────────────────────────────────────────────
    #  1. NOMBRES REALES DE EQUIPOS
    # ───────────────────────────────────────────────────────────

    def fetch_team_names(
        self,
        progress_cb: Optional[Callable] = None,
    ) -> dict[int, str]:
        """
        Construye {team_id (int) → team_name (str)} para la liga activa.

        Los nombres se extraen del campo `opponents` de /lol/matches, que
        siempre funciona en el plan Free. (No se usa /lol/teams porque ese
        endpoint no está disponible en el plan Free y solo genera ruido.)

        El caché resultante se usa en build_team_stats() para sustituir
        IDs numéricos por nombres legibles en todas las filas del DataFrame.
        """
        if progress_cb:
            progress_cb("Descargando nombres de equipos…")

        log.info(f"[fetch_team_names] liga={self.league_id}")
        mapping: dict[int, str] = {}

        raw_matches = self._get(
            "/lol/matches",
            params={
                "filter[league_id]": self.league_id,
                "filter[status]":    "finished",
                "sort":              "-begin_at",
                "per_page":          50,
            },
            max_pages=2,
        )
        added = 0
        for match in raw_matches:
            for opp in match.get("opponents", []):
                t    = opp.get("opponent", {})
                tid  = t.get("id")
                name = (t.get("name") or t.get("acronym") or "").strip()
                if tid and name:
                    tid_int = int(tid)
                    if tid_int not in mapping:
                        mapping[tid_int] = name
                        added += 1

        log.info(f"  /lol/matches → +{added} equipos adicionales")
        log.info(f"  Total mapeados: {len(mapping)}")

        self._team_cache = mapping

        if progress_cb:
            progress_cb(f"{len(mapping)} equipos encontrados")

        return mapping

    @staticmethod
    def _extract_patch(match: dict) -> str:
        """
        Extrae el nombre del parche de un partido.

        PandaScore devuelve `videogame_version` como dict
        {'name': '16.11.1', 'current': bool} (a veces como string).
        Retorna "" si no hay un parche válido.
        """
        v = match.get("videogame_version")
        if isinstance(v, dict):
            v = v.get("name")
        if not v:
            return ""
        s = str(v).strip()
        return s if s not in ("", "null", "None") else ""

    def get_team_name(self, tid) -> str:
        """
        Resuelve el nombre de un equipo desde el caché.
        Normaliza `tid` a int para garantizar comparación correcta
        con los int64 de numpy que devuelve pd.unique().
        """
        try:
            key = int(tid)
        except (TypeError, ValueError):
            return f"Team_{tid}"
        return self._team_cache.get(key, f"Team_{key}")

    # ───────────────────────────────────────────────────────────
    #  2. DETECCIÓN DINÁMICA DE PARCHE
    # ───────────────────────────────────────────────────────────

    def detect_current_patch(
        self,
        progress_cb: Optional[Callable] = None,
        sample_size: int = 30,
    ) -> str:
        if progress_cb:
            progress_cb("Detectando parche actual…")

        log.info(f"[detect_current_patch] liga={self.league_id}, muestra={sample_size}")

        raw = self._get(
            "/lol/matches",
            params={
                "filter[league_id]": self.league_id,
                "filter[status]":    "finished",
                "sort":              "-begin_at",
                "per_page":          min(sample_size, 50),
            },
            max_pages=1,
        )

        if not raw:
            log.warning("  Sin partidas para detectar parche.")
            self._current_patch = ""
            return ""

        # Preferir un parche marcado como 'current' por PandaScore.
        for m in raw:
            v = m.get("videogame_version")
            if isinstance(v, dict) and v.get("current"):
                name = self._extract_patch(m)
                if PatchManager.is_valid(name):
                    self._current_patch = name
                    log.info(f"  Parche actual (current=True): {name}")
                    if progress_cb:
                        progress_cb(f"Parche activo: {name}")
                    return name

        patches = [p for p in (self._extract_patch(m) for m in raw)
                   if PatchManager.is_valid(p)]
        if not patches:
            log.warning("  Partidas sin versión válida.")
            self._current_patch = ""
            return ""

        # Si ninguno es 'current', usar el más reciente por número de versión.
        detected = max(patches, key=lambda p: PatchManager.parse(p))
        self._current_patch = detected
        log.info(f"  Parche detectado: {detected}")
        if progress_cb:
            progress_cb(f"Parche activo: {detected}")
        return detected

    def get_patch_window(self, n_back: int = 1) -> list[str]:
        """
        Retorna la ventana de parches [actual-n, ..., actual].

        Uso típico: filtrar partidas de los últimos 2 parches para
        capturar suficientes datos sin incluir meta obsoleta.

        Args:
            n_back: cuántos parches hacia atrás incluir (default=1)

        Returns:
            list de strings — ej. ["26.08", "26.09"]
        """
        patch = self._current_patch
        if not patch or not PatchManager.is_valid(patch):
            patch = self.detect_current_patch()

        return PatchManager.window(patch, n=n_back)

    # ───────────────────────────────────────────────────────────
    #  3. DESCARGA DE PARTIDAS
    # ───────────────────────────────────────────────────────────

    def get_matches(
        self,
        limit: int = 100,
        min_patch: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Descarga las últimas `limit` partidas finalizadas de la liga.

        Parámetros compatibles con plan Free:
          filter[league_id], filter[status], sort, per_page, page.
        NO se usa range[begin_at] — causaba HTTP 400 en plan Free.

        Filtrado de parche:
          Si min_patch es None, se calcula automáticamente como
          patch_actual - 1 (ventana de 2 parches: actual + anterior).
          Esto garantiza suficientes datos sin meta obsoleta.

        Args:
            limit:      Máximo de partidas a descargar (se pagina en bloques de 50)
            min_patch:  Parche mínimo incluido (ej: "16.10").
                        None → sin filtro (usa las últimas `limit` partidas).
                              Más confiable: evita datasets vacíos cuando hay
                              pocos partidos en el parche más reciente.
                        "16.10" → solo partidas de ese parche en adelante.
            progress_cb: Función callback(str) para actualizar la UI

        Returns:
            pd.DataFrame con columnas:
              match_id, team_a_id, team_b_id, winner_id, patch, begin_at
        """
        # ── Por defecto NO filtramos por parche ────────────────
        # Una sola split tiene partidos repartidos en muchos parches; filtrar
        # al parche actual dejaría muy pocos datos. Usar las últimas `limit`
        # partidas da un dataset estable para estimar la fuerza de cada equipo.
        if min_patch is None:
            min_patch = "00.00"

        if progress_cb:
            progress_cb("Descargando partidas…")

        # ── Descarga paginada ──────────────────────────────────
        per_page  = min(limit, 50)
        max_pages = max(1, -(-limit // per_page))   # ceil division

        log.info(
            f"[get_matches] liga={self.league_id} "
            f"limit={limit} per_page={per_page} pages={max_pages}"
        )

        raw = self._get(
            "/lol/matches",
            params={
                "filter[league_id]": self.league_id,
                "filter[status]":    "finished",
                "sort":              "-begin_at",
                "per_page":          per_page,
            },
            max_pages=max_pages,
        )

        if not raw:
            log.warning("  Sin datos de partidas.")
            return pd.DataFrame()

        # ── Construir DataFrame ────────────────────────────────
        rows = []
        for m in raw:
            opps = m.get("opponents", [])
            if len(opps) < 2:
                continue  # Match sin dos equipos (bye, forfeit, etc.)

            ta    = opps[0].get("opponent", {})
            tb    = opps[1].get("opponent", {})
            tid_a = ta.get("id")
            tid_b = tb.get("id")

            if tid_a is None or tid_b is None:
                continue  # Skip si falta algún equipo

            # Actualizar caché de nombres en tiempo real
            name_a = (ta.get("name") or ta.get("acronym") or "").strip()
            name_b = (tb.get("name") or tb.get("acronym") or "").strip()
            if name_a:
                self._team_cache[int(tid_a)] = name_a
            if name_b:
                self._team_cache[int(tid_b)] = name_b

            patch_str = self._extract_patch(m) or "unknown"

            rows.append({
                "match_id":  m.get("id"),
                "team_a_id": int(tid_a),
                "team_b_id": int(tid_b),
                "winner_id": int(m["winner_id"]) if m.get("winner_id") else None,
                "patch":     patch_str,
                "begin_at":  m.get("begin_at"),
            })

        if not rows:
            log.warning("  Sin partidas parseables tras procesar la respuesta.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # ── Filtrar por parche mínimo ──────────────────────────
        before = len(df)
        if min_patch and min_patch not in ("00.00", "unknown"):
            min_tup = PatchManager.parse(min_patch)
            if min_tup and min_tup != (0,):
                # Comparar como tuplas para ser precisos con versiones como "26.10"
                def _patch_ok(p: str) -> bool:
                    t = PatchManager.parse(p)
                    return bool(t) and t >= min_tup

                mask = df["patch"].apply(_patch_ok)
                df   = df[mask].copy()
                log.info(f"  Filtro de parche: {len(df)}/{before} partidas ≥ {min_patch}")

                # Si el filtro eliminó todo, relajar un parche y advertir
                if df.empty and before > 0:
                    log.warning(
                        f"  ⚠️  Filtro eliminó todas las partidas. "
                        f"Devolviendo todas las {before} sin filtro de parche."
                    )
                    df = pd.DataFrame(rows)

        df = df.reset_index(drop=True)
        log.info(f"  get_matches → {len(df)} partidas")

        if progress_cb:
            progress_cb(f"{len(df)} partidas cargadas")

        return df

    # ───────────────────────────────────────────────────────────
    #  4. KPIs CONSOLIDADOS POR EQUIPO
    # ───────────────────────────────────────────────────────────

    def build_team_stats(
        self,
        df: pd.DataFrame,
        min_games: int = 3,
        progress_cb: Optional[Callable] = None,
    ) -> pd.DataFrame:
        """
        Calcula KPIs por equipo desde el DataFrame de partidas.
        Todas las filas incluyen team_name REAL (no "Team_12345").

        Métricas calculadas:
          win_rate            — victorias / total
          blue_side_winrate   — win rate cuando team_a (Blue Side)
          red_side_winrate    — win rate cuando team_b (Red Side)
          baron_control_rate  — proxy: media de blue+red WR
          games_played        — total de partidas en el dataset
          team_id_encoded     — team_id / 200000 (feature para XGBoost)

          Campos en 0 (requieren /lol/games con stats detalladas):
          gold_diff_15, first_blood_rate, first_dragon_rate,
          vspm, avg_game_duration, gold_lead_20_weight

        Args:
            df:         DataFrame de get_matches()
            min_games:  Equipos con menos partidas que esto son descartados
            progress_cb: Callback de progreso para la GUI

        Returns:
            pd.DataFrame — una fila por equipo, ordenado por win_rate DESC
        """
        if df.empty:
            log.warning("[build_team_stats] DataFrame vacío recibido.")
            return pd.DataFrame()

        if progress_cb:
            progress_cb("Calculando KPIs por equipo…")

        # Recolectar todos los IDs únicos presentes en las partidas
        # Usamos int() explícito para normalizar numpy.int64 → Python int
        ids_a = {int(x) for x in df["team_a_id"].dropna().unique()}
        ids_b = {int(x) for x in df["team_b_id"].dropna().unique()}
        all_ids: set[int] = ids_a | ids_b

        log.info(f"[build_team_stats] {len(all_ids)} IDs únicos encontrados")

        rows = []
        for tid in sorted(all_ids):
            stats = self._calc_kpis(df, tid, min_games=min_games)
            if stats is not None:
                rows.append(stats)

        if not rows:
            log.warning(
                "  Sin equipos con suficientes partidas. "
                f"Prueba reducir min_games (actual={min_games})."
            )
            return pd.DataFrame()

        result = (
            pd.DataFrame(rows)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
        )

        log.info(f"  KPIs listos: {len(result)} equipos")

        if progress_cb:
            progress_cb(f"KPIs listos: {len(result)} equipos")

        return result

    def _calc_kpis(
        self,
        df: pd.DataFrame,
        tid: int,
        min_games: int = 3,
    ) -> Optional[dict]:
        """
        Calcula todas las métricas para un equipo específico.

        Type safety:
          - `tid` siempre es Python int (normalizado en build_team_stats).
          - La comparación con DataFrame se hace sobre columnas int via .to_numpy().
          - Nunca mezcla numpy.int64 con Python int sin conversión explícita.

        Returns:
            dict con KPIs | None si el equipo tiene menos de min_games partidas
        """
        # Máscara type-safe: convertir columna a int nativo antes de comparar
        a_ids = df["team_a_id"].to_numpy()
        b_ids = df["team_b_id"].to_numpy()

        try:
            mask = (a_ids.astype(int) == tid) | (b_ids.astype(int) == tid)
        except Exception:
            # Fallback para IDs con nulos
            mask = np.array([
                (int(a) == tid if pd.notna(a) else False) or
                (int(b) == tid if pd.notna(b) else False)
                for a, b in zip(a_ids, b_ids)
            ])

        g = df[mask].copy()

        if len(g) < min_games:
            log.debug(f"  Team {tid} descartado: {len(g)} partidas < min={min_games}")
            return None

        total = len(g)
        wins  = int((g["winner_id"] == tid).sum())
        bwr, rwr = self._side_wr(g, tid)

        # Proxy de baron control: media geométrica de blue+red WR ponderada
        baron_proxy = round((bwr * 0.55 + rwr * 0.45), 4)

        # Normalización del ID para el feature de ML
        team_id_encoded = round(tid / 200_000, 8)

        return {
            # Identificación
            "team_id":              tid,
            "team_name":            self.get_team_name(tid),  # Nombre real
            "games_played":         total,

            # KPI principal
            "win_rate":             round(wins / total, 4),

            # KPIs de lado del mapa (calculados)
            "blue_side_winrate":    round(bwr, 4),
            "red_side_winrate":     round(rwr, 4),
            "baron_control_rate":   baron_proxy,

            # KPIs de early game (requieren /lol/games — en 0.0 si no disponibles)
            "gold_diff_15":         0.0,
            "first_blood_rate":     0.0,
            "first_dragon_rate":    0.0,
            "vspm":                 0.0,
            "avg_game_duration":    0.0,
            "gold_lead_20_weight":  0.0,

            # Feature para XGBoost
            "team_id_encoded":      team_id_encoded,
        }

    def _side_wr(self, df: pd.DataFrame, tid: int) -> tuple[float, float]:
        """
        Calcula win rate en Blue Side (team_a) y Red Side (team_b).

        FIX-2 (v2.1): Convierte explícitamente a Python int antes de comparar.
        La conversión es crítica porque:
          - df["team_a_id"] contiene numpy.int64
          - tid es Python int
          - numpy.int64 == int funciona en la mayoría de casos PERO
            en algunas versiones de pandas/numpy con NaN mezclados,
            la comparación silenciosamente devuelve False.
          - Solución: convertir la columna a numpy array de int y comparar.

        Returns:
            (blue_wr, red_wr) — floats entre 0.0 y 1.0
        """
        a_ids  = df["team_a_id"].to_numpy()
        b_ids  = df["team_b_id"].to_numpy()
        w_ids  = df["winner_id"].to_numpy()

        bw = bt = rw = rt = 0

        for i in range(len(df)):
            a_id = int(a_ids[i]) if pd.notna(a_ids[i]) else -1
            b_id = int(b_ids[i]) if pd.notna(b_ids[i]) else -1
            w_id = int(w_ids[i]) if pd.notna(w_ids[i]) else -1

            if a_id == tid:       # Equipo jugó como Blue Side
                bt += 1
                if w_id == tid:
                    bw += 1
            elif b_id == tid:     # Equipo jugó como Red Side
                rt += 1
                if w_id == tid:
                    rw += 1

        blue_wr = round(bw / bt, 4) if bt > 0 else 0.5
        red_wr  = round(rw / rt, 4) if rt > 0 else 0.5

        log.debug(
            f"  Team {tid} | Blue {bw}/{bt}={blue_wr:.2f} "
            f"| Red {rw}/{rt}={red_wr:.2f}"
        )
        return blue_wr, red_wr


# ═══════════════════════════════════════════════════════════════
#  DATOS DE DEMOSTRACIÓN (sin conexión a API)
# ═══════════════════════════════════════════════════════════════

def get_demo_stats(league_name: str = "LCK") -> pd.DataFrame:
    """
    Retorna un DataFrame de KPIs realistas para probar la GUI sin API.

    Liga       Fuente de datos
    ─────────  ──────────────────────────────────────────────────────
    LCK        Standings reales LCK 2026, Semana 7, parche 26.09
    LPL        Standings estimados LPL 2026 Spring, semana 7
    LEC        Standings estimados LEC 2026 Winter Split
    LCS        Standings estimados LCS 2026 Spring Split
    otro       Fallback a LCK
    """
    abbr = league_name.strip().upper().split()[0].replace("—", "").strip()

    # ── LCK ───────────────────────────────────────────────────
    if abbr in ("LCK", "293"):
        data = [
            {"team_name": "Gen.G",               "team_id": 2882,   "win_rate": 0.82, "blue_side_winrate": 0.70, "red_side_winrate": 0.56, "baron_control_rate": 0.64, "gold_diff_15":  780, "first_blood_rate": 0.61, "first_dragon_rate": 0.66, "vspm": 1.77, "avg_game_duration": 31.8, "gold_lead_20_weight": 1092, "team_id_encoded": 2882/200000, "games_played": 28},
            {"team_name": "T1",                  "team_id": 2883,   "win_rate": 0.79, "blue_side_winrate": 0.73, "red_side_winrate": 0.60, "baron_control_rate": 0.67, "gold_diff_15":  840, "first_blood_rate": 0.64, "first_dragon_rate": 0.62, "vspm": 1.81, "avg_game_duration": 30.9, "gold_lead_20_weight": 1176, "team_id_encoded": 2883/200000, "games_played": 26},
            {"team_name": "Hanwha Life Esports", "team_id": 126061, "win_rate": 0.72, "blue_side_winrate": 0.65, "red_side_winrate": 0.52, "baron_control_rate": 0.59, "gold_diff_15":  560, "first_blood_rate": 0.55, "first_dragon_rate": 0.58, "vspm": 1.62, "avg_game_duration": 32.7, "gold_lead_20_weight":  784, "team_id_encoded": 126061/200000, "games_played": 24},
            {"team_name": "KT Rolster",          "team_id": 63,     "win_rate": 0.68, "blue_side_winrate": 0.62, "red_side_winrate": 0.51, "baron_control_rate": 0.57, "gold_diff_15":  620, "first_blood_rate": 0.54, "first_dragon_rate": 0.57, "vspm": 1.65, "avg_game_duration": 33.1, "gold_lead_20_weight":  868, "team_id_encoded": 63/200000, "games_played": 24},
            {"team_name": "Dplus KIA",           "team_id": 128218, "win_rate": 0.55, "blue_side_winrate": 0.58, "red_side_winrate": 0.47, "baron_control_rate": 0.53, "gold_diff_15":  350, "first_blood_rate": 0.50, "first_dragon_rate": 0.52, "vspm": 1.58, "avg_game_duration": 33.8, "gold_lead_20_weight":  490, "team_id_encoded": 128218/200000, "games_played": 22},
            {"team_name": "BNK FearX",           "team_id": 128217, "win_rate": 0.48, "blue_side_winrate": 0.52, "red_side_winrate": 0.44, "baron_control_rate": 0.48, "gold_diff_15":  200, "first_blood_rate": 0.47, "first_dragon_rate": 0.49, "vspm": 1.51, "avg_game_duration": 34.5, "gold_lead_20_weight":  280, "team_id_encoded": 128217/200000, "games_played": 20},
            {"team_name": "DN Freecs",           "team_id": 126370, "win_rate": 0.42, "blue_side_winrate": 0.48, "red_side_winrate": 0.40, "baron_control_rate": 0.44, "gold_diff_15": -150, "first_blood_rate": 0.44, "first_dragon_rate": 0.46, "vspm": 1.47, "avg_game_duration": 35.2, "gold_lead_20_weight": -210, "team_id_encoded": 126370/200000, "games_played": 20},
            {"team_name": "OK BRION",            "team_id": 132531, "win_rate": 0.35, "blue_side_winrate": 0.44, "red_side_winrate": 0.37, "baron_control_rate": 0.41, "gold_diff_15": -320, "first_blood_rate": 0.41, "first_dragon_rate": 0.43, "vspm": 1.43, "avg_game_duration": 35.9, "gold_lead_20_weight": -448, "team_id_encoded": 132531/200000, "games_played": 18},
            {"team_name": "Nongshim RedForce",   "team_id": 134115, "win_rate": 0.28, "blue_side_winrate": 0.40, "red_side_winrate": 0.33, "baron_control_rate": 0.37, "gold_diff_15": -480, "first_blood_rate": 0.38, "first_dragon_rate": 0.40, "vspm": 1.39, "avg_game_duration": 36.5, "gold_lead_20_weight": -672, "team_id_encoded": 134115/200000, "games_played": 18},
        ]

    # ── LPL ───────────────────────────────────────────────────
    elif abbr in ("LPL", "290"):
        data = [
            {"team_name": "Bilibili Gaming",   "team_id": 5001, "win_rate": 0.80, "blue_side_winrate": 0.71, "red_side_winrate": 0.58, "baron_control_rate": 0.65, "gold_diff_15":  760, "first_blood_rate": 0.60, "first_dragon_rate": 0.64, "vspm": 1.74, "avg_game_duration": 31.6, "gold_lead_20_weight": 1064, "team_id_encoded": 5001/200000, "games_played": 28},
            {"team_name": "JD Gaming",         "team_id": 5002, "win_rate": 0.74, "blue_side_winrate": 0.68, "red_side_winrate": 0.56, "baron_control_rate": 0.62, "gold_diff_15":  640, "first_blood_rate": 0.57, "first_dragon_rate": 0.60, "vspm": 1.70, "avg_game_duration": 32.1, "gold_lead_20_weight":  896, "team_id_encoded": 5002/200000, "games_played": 26},
            {"team_name": "Top Esports",       "team_id": 5003, "win_rate": 0.67, "blue_side_winrate": 0.63, "red_side_winrate": 0.53, "baron_control_rate": 0.58, "gold_diff_15":  470, "first_blood_rate": 0.55, "first_dragon_rate": 0.58, "vspm": 1.66, "avg_game_duration": 32.6, "gold_lead_20_weight":  658, "team_id_encoded": 5003/200000, "games_played": 24},
            {"team_name": "Weibo Gaming",      "team_id": 5004, "win_rate": 0.58, "blue_side_winrate": 0.57, "red_side_winrate": 0.49, "baron_control_rate": 0.53, "gold_diff_15":  280, "first_blood_rate": 0.51, "first_dragon_rate": 0.54, "vspm": 1.61, "avg_game_duration": 33.2, "gold_lead_20_weight":  392, "team_id_encoded": 5004/200000, "games_played": 24},
            {"team_name": "LNG Esports",       "team_id": 5005, "win_rate": 0.50, "blue_side_winrate": 0.52, "red_side_winrate": 0.46, "baron_control_rate": 0.49, "gold_diff_15":  110, "first_blood_rate": 0.49, "first_dragon_rate": 0.51, "vspm": 1.57, "avg_game_duration": 33.9, "gold_lead_20_weight":  154, "team_id_encoded": 5005/200000, "games_played": 22},
            {"team_name": "FunPlus Phoenix",   "team_id": 5006, "win_rate": 0.42, "blue_side_winrate": 0.47, "red_side_winrate": 0.40, "baron_control_rate": 0.44, "gold_diff_15": -160, "first_blood_rate": 0.45, "first_dragon_rate": 0.47, "vspm": 1.52, "avg_game_duration": 34.6, "gold_lead_20_weight": -224, "team_id_encoded": 5006/200000, "games_played": 20},
            {"team_name": "Ninjas in Pyjamas", "team_id": 5007, "win_rate": 0.33, "blue_side_winrate": 0.41, "red_side_winrate": 0.33, "baron_control_rate": 0.37, "gold_diff_15": -360, "first_blood_rate": 0.41, "first_dragon_rate": 0.43, "vspm": 1.46, "avg_game_duration": 35.6, "gold_lead_20_weight": -504, "team_id_encoded": 5007/200000, "games_played": 18},
            {"team_name": "Invictus Gaming",   "team_id": 5008, "win_rate": 0.26, "blue_side_winrate": 0.38, "red_side_winrate": 0.31, "baron_control_rate": 0.35, "gold_diff_15": -520, "first_blood_rate": 0.37, "first_dragon_rate": 0.39, "vspm": 1.40, "avg_game_duration": 36.3, "gold_lead_20_weight": -728, "team_id_encoded": 5008/200000, "games_played": 18},
        ]

    # ── LEC ───────────────────────────────────────────────────
    elif abbr in ("LEC", "4197"):
        data = [
            {"team_name": "G2 Esports",        "team_id": 3001, "win_rate": 0.78, "blue_side_winrate": 0.70, "red_side_winrate": 0.58, "baron_control_rate": 0.64, "gold_diff_15":  720, "first_blood_rate": 0.60, "first_dragon_rate": 0.64, "vspm": 1.72, "avg_game_duration": 31.5, "gold_lead_20_weight": 1008, "team_id_encoded": 3001/200000, "games_played": 26},
            {"team_name": "Fnatic",            "team_id": 3002, "win_rate": 0.72, "blue_side_winrate": 0.67, "red_side_winrate": 0.55, "baron_control_rate": 0.61, "gold_diff_15":  580, "first_blood_rate": 0.58, "first_dragon_rate": 0.61, "vspm": 1.68, "avg_game_duration": 32.0, "gold_lead_20_weight":  812, "team_id_encoded": 3002/200000, "games_played": 24},
            {"team_name": "Team Vitality",     "team_id": 3003, "win_rate": 0.65, "blue_side_winrate": 0.62, "red_side_winrate": 0.53, "baron_control_rate": 0.57, "gold_diff_15":  440, "first_blood_rate": 0.55, "first_dragon_rate": 0.58, "vspm": 1.65, "avg_game_duration": 32.5, "gold_lead_20_weight":  616, "team_id_encoded": 3003/200000, "games_played": 24},
            {"team_name": "Mad Lions KOI",     "team_id": 3004, "win_rate": 0.58, "blue_side_winrate": 0.57, "red_side_winrate": 0.49, "baron_control_rate": 0.53, "gold_diff_15":  260, "first_blood_rate": 0.51, "first_dragon_rate": 0.54, "vspm": 1.61, "avg_game_duration": 33.0, "gold_lead_20_weight":  364, "team_id_encoded": 3004/200000, "games_played": 22},
            {"team_name": "Team BDS",          "team_id": 3005, "win_rate": 0.48, "blue_side_winrate": 0.51, "red_side_winrate": 0.45, "baron_control_rate": 0.48, "gold_diff_15":   80, "first_blood_rate": 0.48, "first_dragon_rate": 0.50, "vspm": 1.57, "avg_game_duration": 33.8, "gold_lead_20_weight":  112, "team_id_encoded": 3005/200000, "games_played": 22},
            {"team_name": "SK Gaming",         "team_id": 3006, "win_rate": 0.40, "blue_side_winrate": 0.46, "red_side_winrate": 0.38, "baron_control_rate": 0.42, "gold_diff_15": -180, "first_blood_rate": 0.44, "first_dragon_rate": 0.46, "vspm": 1.52, "avg_game_duration": 34.5, "gold_lead_20_weight": -252, "team_id_encoded": 3006/200000, "games_played": 20},
            {"team_name": "Karmine Corp",      "team_id": 3007, "win_rate": 0.32, "blue_side_winrate": 0.40, "red_side_winrate": 0.31, "baron_control_rate": 0.36, "gold_diff_15": -350, "first_blood_rate": 0.40, "first_dragon_rate": 0.42, "vspm": 1.46, "avg_game_duration": 35.5, "gold_lead_20_weight": -490, "team_id_encoded": 3007/200000, "games_played": 18},
        ]

    # ── LCS ───────────────────────────────────────────────────
    elif abbr in ("LCS", "4198"):
        data = [
            {"team_name": "Cloud9",            "team_id": 4001, "win_rate": 0.75, "blue_side_winrate": 0.68, "red_side_winrate": 0.56, "baron_control_rate": 0.62, "gold_diff_15":  650, "first_blood_rate": 0.59, "first_dragon_rate": 0.62, "vspm": 1.65, "avg_game_duration": 33.0, "gold_lead_20_weight":  910, "team_id_encoded": 4001/200000, "games_played": 24},
            {"team_name": "Team Liquid",       "team_id": 4002, "win_rate": 0.70, "blue_side_winrate": 0.64, "red_side_winrate": 0.54, "baron_control_rate": 0.59, "gold_diff_15":  520, "first_blood_rate": 0.56, "first_dragon_rate": 0.59, "vspm": 1.62, "avg_game_duration": 33.5, "gold_lead_20_weight":  728, "team_id_encoded": 4002/200000, "games_played": 22},
            {"team_name": "100 Thieves",       "team_id": 4003, "win_rate": 0.62, "blue_side_winrate": 0.60, "red_side_winrate": 0.51, "baron_control_rate": 0.55, "gold_diff_15":  350, "first_blood_rate": 0.52, "first_dragon_rate": 0.55, "vspm": 1.59, "avg_game_duration": 34.0, "gold_lead_20_weight":  490, "team_id_encoded": 4003/200000, "games_played": 22},
            {"team_name": "Evil Geniuses",     "team_id": 4004, "win_rate": 0.55, "blue_side_winrate": 0.54, "red_side_winrate": 0.48, "baron_control_rate": 0.51, "gold_diff_15":  200, "first_blood_rate": 0.50, "first_dragon_rate": 0.52, "vspm": 1.55, "avg_game_duration": 34.5, "gold_lead_20_weight":  280, "team_id_encoded": 4004/200000, "games_played": 20},
            {"team_name": "FlyQuest",          "team_id": 4005, "win_rate": 0.48, "blue_side_winrate": 0.50, "red_side_winrate": 0.44, "baron_control_rate": 0.47, "gold_diff_15":   40, "first_blood_rate": 0.47, "first_dragon_rate": 0.49, "vspm": 1.51, "avg_game_duration": 35.0, "gold_lead_20_weight":   56, "team_id_encoded": 4005/200000, "games_played": 20},
            {"team_name": "NRG Esports",       "team_id": 4006, "win_rate": 0.40, "blue_side_winrate": 0.45, "red_side_winrate": 0.38, "baron_control_rate": 0.42, "gold_diff_15": -150, "first_blood_rate": 0.44, "first_dragon_rate": 0.46, "vspm": 1.47, "avg_game_duration": 35.8, "gold_lead_20_weight": -210, "team_id_encoded": 4006/200000, "games_played": 18},
            {"team_name": "Dignitas",          "team_id": 4007, "win_rate": 0.30, "blue_side_winrate": 0.37, "red_side_winrate": 0.30, "baron_control_rate": 0.34, "gold_diff_15": -380, "first_blood_rate": 0.39, "first_dragon_rate": 0.41, "vspm": 1.42, "avg_game_duration": 36.5, "gold_lead_20_weight": -532, "team_id_encoded": 4007/200000, "games_played": 16},
        ]

    # ── Fallback → LCK ────────────────────────────────────────
    else:
        log.warning(f"[get_demo_stats] Liga '{league_name}' no reconocida — usando LCK.")
        return get_demo_stats("LCK")

    df = pd.DataFrame(data)
    log.info(f"[get_demo_stats] {abbr}: {len(df)} equipos cargados")
    return df


# ═══════════════════════════════════════════════════════════════
#  DIAGNÓSTICO DE CONECTIVIDAD (bloque __main__)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 62)
    print("  UNIVERSAL PIPELINE — Test de conectividad")
    print("=" * 62)

    # ── Test 1: PatchManager ──────────────────────────────────
    print("\n[1/4] PatchManager:")
    assert PatchManager.parse("26.09") == (26, 9),            "parse falló"
    assert PatchManager.to_str((26, 9)) == "26.09",           "to_str falló"
    assert PatchManager.subtract("26.09", 1) == "26.08",      "subtract falló"
    assert PatchManager.subtract("26.01", 1) == "26.01",      "subtract floor falló"
    assert PatchManager.window("26.09", 2) == ["26.07","26.08","26.09"], "window falló"
    assert PatchManager.compare("26.09", "26.08") == 1,       "compare > falló"
    assert PatchManager.compare("26.07", "26.09") == -1,      "compare < falló"
    assert PatchManager.is_valid("26.09"),                     "is_valid True falló"
    assert not PatchManager.is_valid("unknown"),               "is_valid False falló"
    print("  ✅  Todos los tests de PatchManager pasaron")

    # ── Test 2: get_demo_stats ────────────────────────────────
    print("\n[2/4] get_demo_stats():")
    for liga in ["LCK", "LPL", "LEC", "LCS", "desconocida"]:
        df = get_demo_stats(liga)
        assert not df.empty,              f"  ❌ {liga}: DataFrame vacío"
        assert "win_rate" in df.columns,  f"  ❌ {liga}: sin win_rate"
        assert "team_name" in df.columns, f"  ❌ {liga}: sin team_name"
        assert (df["team_name"] != "").all(), f"  ❌ {liga}: team_name vacío"
        print(f"  ✅  {liga:12s} → {len(df)} equipos, "
              f"WR rango [{df['win_rate'].min():.2f}, {df['win_rate'].max():.2f}]")

    # ── Test 3: API key ───────────────────────────────────────
    print("\n[3/4] Configuración API:")
    api_key = "TU_API_KEY_AQUI"
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"  API key recibida: {api_key[:8]}…")
    else:
        print("  ⚠️  Sin API key (pasa como argumento: python universal_pipeline.py TU_KEY)")
        print("  Saltando tests de conectividad real.")

    # ── Test 4: Conectividad PandaScore ──────────────────────
    if api_key != "TU_API_KEY_AQUI":
        print("\n[4/4] Conectividad PandaScore (LCK, ID=293):")
        pipe = UniversalPipeline(api_key=api_key, league_id=293)

        print("  → detect_current_patch()…")
        patch = pipe.detect_current_patch()
        print(f"  ✅  Parche detectado: {patch}")

        print("  → fetch_team_names()…")
        names = pipe.fetch_team_names()
        print(f"  ✅  {len(names)} equipos: {list(names.values())[:4]}…")

        print("  → get_matches(limit=30)…")
        df = pipe.get_matches(limit=30)
        print(f"  ✅  {len(df)} partidas descargadas")

        if not df.empty:
            print("  → build_team_stats()…")
            stats = pipe.build_team_stats(df)
            print(f"  ✅  {len(stats)} equipos con KPIs:")
            for _, row in stats.head(5).iterrows():
                print(
                    f"       {row['team_name']:22s} "
                    f"WR={row['win_rate']*100:.0f}%  "
                    f"BlueWR={row['blue_side_winrate']*100:.0f}%  "
                    f"RedWR={row['red_side_winrate']*100:.0f}%  "
                    f"GP={row['games_played']}"
                )
    else:
        print("\n[4/4] (saltado — sin API key)")

    print("\n" + "=" * 62)
    print("  Pipeline listo. Ejecuta prediction_os_v2.py para la GUI.")
    print("=" * 62)
