"""
============================================================
LCK PREDICTION OS — MÓDULO 4: RUNNER INTERACTIVO (v3)
============================================================
FLUJO COMPLETO:
  1. Conecta con tu pipeline (lck_data_pipeline.py) para
     obtener KPIs reales de la LCK CL 2026.
  2. Entrena el modelo con esos KPIs.
  3. Te pregunta los partidos de mañana uno a uno.
  4. Acepta momios en formato AMERICANO (+285, -425) y
     los convierte a decimal automáticamente.
  5. Calcula la probabilidad del modelo + ajuste LCK.
  6. Aplica Kelly al 25% sobre tu bankroll de $1,000 MXN.
  7. Muestra la señal: Oportunidad de Oro o Sin Ventaja.

USO:
    python lck_main.py

DEPENDENCIAS:
    pip install xgboost scikit-learn pandas numpy requests joblib
============================================================
"""

import sys
import logging
from datetime import datetime

# ── Configurar logging ANTES de cualquier import interno ──
logging.basicConfig(
    level=logging.WARNING,          # Silenciar DEBUG/INFO del pipeline en modo interactivo
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Imports del sistema ──
from lck_config import (
    BANKROLL, KELLY_FRACTION, LCK_TEAMS, TEAM_NAME_TO_ID,
    CURRENT_YEAR, MIN_EDGE_THRESHOLD
)
from lck_data_pipeline import LCKDataPipeline   # ← Tu pipeline, sin modificar
from lck_ml_model import (
    LCKPredictor, american_to_decimal, kelly_stake
)


# ═══════════════════════════════════════════════════════════
#  HELPERS DE UI (consola limpia)
# ═══════════════════════════════════════════════════════════
RESET  = "\033[0m"
BOLD   = "\033[1m"
GOLD   = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
LINE   = "─" * 56


def header():
    print(f"\n{BOLD}{'═'*56}{RESET}")
    print(f"{BOLD}  ⚡ LCK PREDICTION OS  ·  {CURRENT_YEAR}{RESET}")
    print(f"{DIM}  Motor cuantitativo · Parche 26.09 · Kelly 25%{RESET}")
    print(f"{BOLD}{'═'*56}{RESET}\n")


def section(title: str):
    print(f"\n{CYAN}{LINE}{RESET}")
    print(f"{CYAN}  {title}{RESET}")
    print(f"{CYAN}{LINE}{RESET}")


def ask(prompt: str, default: str = "") -> str:
    """Input con soporte de valor por defecto y manejo de Ctrl+C."""
    try:
        val = input(f"  {prompt}").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Saliendo del sistema. ¡Buena suerte!\n")
        sys.exit(0)


def pick_team(prompt: str, teams: list[str]) -> str:
    """Muestra lista numerada y permite elegir por número o nombre."""
    print(f"\n  {prompt}")
    for i, t in enumerate(teams, 1):
        print(f"    {DIM}{i:>2}.{RESET} {t}")
    while True:
        choice = ask("Tu elección (número o nombre): ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(teams):
                return teams[idx]
        else:
            matches = [t for t in teams if choice.lower() in t.lower()]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                print(f"  Ambiguo — coincide con: {matches}. Sé más específico.")
        print(f"  {RED}Opción inválida. Intenta de nuevo.{RESET}")


def parse_american_odds(raw: str) -> int | None:
    """Parsea momio americano desde string. Acepta: +285, -425, 285, -425."""
    cleaned = raw.replace(" ", "").replace(",", "")
    try:
        val = int(cleaned)
        # Validar rango razonable de momios
        if abs(val) < 100:
            print(f"  {RED}Momio inválido ({val}). Los momios americanos son ≥ 100 (ej: +150, -200).{RESET}")
            return None
        return val
    except ValueError:
        print(f"  {RED}Formato inválido. Usa: +285 o -425{RESET}")
        return None


def show_bet_analysis(
    team_a: str, team_b: str,
    prob_a: float, odd_a_american: int,
    prob_b: float, odd_b_american: int,
    side_a: str, bankroll: float
):
    """Imprime el análisis completo de un partido."""
    dec_a = american_to_decimal(odd_a_american)
    dec_b = american_to_decimal(odd_b_american)

    k_a = kelly_stake(prob_a, dec_a, bankroll)
    k_b = kelly_stake(prob_b, dec_b, bankroll)

    print(f"\n  {BOLD}{team_a}{RESET}  vs  {BOLD}{team_b}{RESET}")
    print(f"  Lado: {team_a} = {side_a.upper()} | {team_b} = {'RED' if side_a=='blue' else 'BLUE'}\n")

    # Tabla comparativa
    print(f"  {'':30} {CYAN}{team_a:>10}{RESET}   {CYAN}{team_b:>10}{RESET}")
    print(f"  {'Prob. Modelo':30} {prob_a*100:>9.1f}%   {prob_b*100:>9.1f}%")
    print(f"  {'Momio Americano':30} {odd_a_american:>+10}   {odd_b_american:>+10}")
    print(f"  {'Cuota Decimal':30} {dec_a:>10.3f}   {dec_b:>10.3f}")
    print(f"  {'Prob. Implícita casera':30} {k_a['implied_prob_pct']:>9.1f}%   {k_b['implied_prob_pct']:>9.1f}%")
    print(f"  {'Ventaja (Edge)':30} {k_a['edge_pct']:>+9.1f}%   {k_b['edge_pct']:>+9.1f}%")

    print(f"\n  {LINE}")

    # Análisis por equipo
    for team, k, odd_am, dec in [
        (team_a, k_a, odd_a_american, dec_a),
        (team_b, k_b, odd_b_american, dec_b),
    ]:
        if k["is_value"]:
            print(f"\n  {GOLD}{k['signal']}{RESET}")
            print(f"  Apostar por:       {BOLD}{team}{RESET}")
            print(f"  Momio:             {odd_am:+d}  ({dec:.3f} decimal)")
            print(f"  Kelly %:           {k['kelly_pct']:.2f}% del bankroll")
            print(f"  {GREEN}Stake recomendado: ${k['stake_mxn']:,.2f} MXN{RESET}")
            print(f"  EV esperado:       +${k['ev_mxn']:,.2f} MXN  (+{k['roi_pct']:.1f}% ROI)")
        else:
            print(f"\n  {RED}{k['signal']}{RESET}  ←  {team}")
            print(f"  Edge: {k['edge_pct']:+.1f}%  (necesitas > {MIN_EDGE_THRESHOLD*100:.0f}%)")

    print(f"\n  {LINE}")


# ═══════════════════════════════════════════════════════════
#  CARGA DE DATOS Y ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════
def load_and_train(bankroll: float) -> tuple[dict, LCKPredictor]:
    """
    Conecta con PandaScore, calcula KPIs y entrena el modelo.

    Si PandaScore falla, ofrece continuar con datos de demostración
    en lugar de saltar al modo demo silenciosamente.

    Returns:
        (stats_por_equipo, predictor_entrenado)
    """
    section(f"[1/2] Cargando datos LCK {CURRENT_YEAR}…")

    pipeline  = LCKDataPipeline()
    df_matches = pd.DataFrame()

    try:
        logging.getLogger().setLevel(logging.INFO)   # Activar logs durante la carga
        df_matches = pipeline.get_all_matches(limit=100)
        logging.getLogger().setLevel(logging.WARNING) # Silenciar de nuevo tras la carga
    except Exception as e:
        print(f"\n  {RED}Error al conectar con PandaScore:{RESET} {e}")

    if df_matches is not None and not df_matches.empty:
        df_stats = pipeline.build_team_stats(df_matches)
        print(f"  ✅ {len(df_matches)} partidas | {len(df_stats)} equipos con KPIs")
    else:
        print(f"\n  {RED}⚠️  Sin datos de PandaScore.{RESET}")
        print("  Revisa tu API KEY en lck_config.py y la conectividad.")
        use_demo = ask("  ¿Continuar con datos de demostración? (s/n): ", default="s")
        if use_demo.lower() != "s":
            print("  Saliendo. Configura tu API KEY e intenta de nuevo.")
            sys.exit(0)
        df_stats = _demo_stats()
        print(f"  ✅ Usando stats de demostración ({len(df_stats)} equipos).")

    # Convertir a dict para acceso rápido por nombre de equipo
    stats_dict = {
        row["team_name"]: row
        for row in df_stats.to_dict(orient="records")
    }

    section("[2/2] Entrenando modelo XGBoost…")
    predictor = LCKPredictor()
    try:
        metrics = predictor.train(df_stats)
        print(f"  ✅ Modelo OK | Modo={metrics.get('mode','?')} | "
              f"AUC={metrics.get('auc_mean', '—')} | "
              f"Features={metrics.get('features', '—')}")
        if metrics.get("mode") == "lite":
            print(f"  {DIM}  → Las columnas Tier-2 (gold_diff_15, etc.) están en 0.{RESET}")
            print(f"  {DIM}  → Para activarlas, integra /lol/games con stats por partida.{RESET}")
    except Exception as e:
        print(f"  {RED}Error en entrenamiento: {e}{RESET}")
        print("  El sistema continuará con estimaciones por win_rate.")
        predictor.is_trained = False

    return stats_dict, predictor


# ═══════════════════════════════════════════════════════════
#  LOOP PRINCIPAL: PARTIDOS + MOMIOS
# ═══════════════════════════════════════════════════════════
def run_match_session(stats_dict: dict, predictor: LCKPredictor, bankroll: float):
    """
    Sesión interactiva: el usuario ingresa partidos y momios americanos.
    """
    team_names  = sorted(stats_dict.keys())
    total_stake = 0.0
    total_ev    = 0.0
    n_matches   = 0
    n_value     = 0

    while True:
        section(f"Partido #{n_matches + 1} — Ingresa los equipos")

        # ── Selección de equipos ──
        team_a_name = pick_team("¿Qué equipo juega en BLUE SIDE (o primero)?", team_names)
        remaining   = [t for t in team_names if t != team_a_name]
        team_b_name = pick_team("¿Cuál es su rival (RED SIDE)?", remaining)

        stats_a = stats_dict.get(team_a_name, _fallback_stats(team_a_name))
        stats_b = stats_dict.get(team_b_name, _fallback_stats(team_b_name))

        # ── Momio americano del equipo A ──
        print(f"\n  Ahora ingresa los momios de Codere (formato americano).")
        print(f"  Ejemplos: {DIM}favorito -425 | underdog +285 | parejo +100{RESET}")

        odd_a_american = None
        while odd_a_american is None:
            raw = ask(f"  Momio americano de {BOLD}{team_a_name}{RESET}: ")
            odd_a_american = parse_american_odds(raw)

        odd_b_american = None
        while odd_b_american is None:
            raw = ask(f"  Momio americano de {BOLD}{team_b_name}{RESET}: ")
            odd_b_american = parse_american_odds(raw)

        # ── Predicción del modelo ──
        if predictor.is_trained:
            try:
                pred = predictor.predict_match(stats_a, stats_b, side_a="blue")
                prob_a = pred["prob_a"]
                prob_b = pred["prob_b"]
            except Exception as e:
                log.warning(f"Error en predicción: {e}. Usando win_rate.")
                prob_a, prob_b = _fallback_probs(stats_a, stats_b)
        else:
            prob_a, prob_b = _fallback_probs(stats_a, stats_b)

        # ── Mostrar análisis completo ──
        show_bet_analysis(
            team_a_name, team_b_name,
            prob_a, odd_a_american,
            prob_b, odd_b_american,
            side_a="blue", bankroll=bankroll
        )

        # Acumular resumen de sesión
        n_matches += 1
        dec_a = american_to_decimal(odd_a_american)
        dec_b = american_to_decimal(odd_b_american)
        for prob, dec in [(prob_a, dec_a), (prob_b, dec_b)]:
            k = kelly_stake(prob, dec, bankroll)
            if k["is_value"]:
                total_stake += k["stake_mxn"]
                total_ev    += k["ev_mxn"]
                n_value     += 1

        # ── ¿Otro partido? ──
        more = ask("\n  ¿Analizar otro partido? (s/n): ", default="s")
        if more.lower() != "s":
            break

    # ── Resumen final de la sesión ──
    section(f"RESUMEN DE LA JORNADA — {datetime.now().strftime('%d/%m/%Y')}")
    print(f"  Partidos analizados:    {n_matches}")
    print(f"  Oportunidades de valor: {n_value}")
    print(f"  Bankroll:              ${bankroll:,.2f} MXN")
    print(f"  Stake total sugerido:  ${total_stake:,.2f} MXN  ({total_stake/bankroll*100:.1f}% del capital)")
    print(f"  EV total esperado:    +${total_ev:,.2f} MXN")
    if total_stake > 0:
        pct = total_stake / bankroll * 100
        risk = "🟢 Conservador" if pct < 5 else ("🟡 Moderado" if pct < 10 else "🔴 Alto")
        print(f"  Nivel de riesgo:        {risk}")
    print(f"\n  {DIM}Apostá con responsabilidad. El modelo es una herramienta,{RESET}")
    print(f"  {DIM}no una garantía de ganancia.{RESET}\n")


# ═══════════════════════════════════════════════════════════
#  UTILIDADES INTERNAS
# ═══════════════════════════════════════════════════════════
def _fallback_probs(stats_a: dict, stats_b: dict) -> tuple[float, float]:
    """Probabilidades simples por win_rate cuando el modelo no está disponible."""
    import numpy as np
    wr_a = float(stats_a.get("win_rate", 0.5))
    wr_b = float(stats_b.get("win_rate", 0.5))
    total = wr_a + wr_b
    prob_a = (wr_a / total) if total > 0 else 0.5
    return round(float(np.clip(prob_a, 0.1, 0.9)), 4), round(1 - float(np.clip(prob_a, 0.1, 0.9)), 4)


def _fallback_stats(team_name: str) -> dict:
    """Stats neutras para equipos sin datos en el pipeline."""
    return {
        "team_name":           team_name,
        "team_id":             TEAM_NAME_TO_ID.get(team_name, 0),
        "win_rate":            0.5,
        "gold_diff_15":        0.0,
        "first_blood_rate":    0.5,
        "first_dragon_rate":   0.5,
        "vspm":                1.5,
        "baron_control_rate":  0.5,
        "avg_game_duration":   32.0,
        "blue_side_winrate":   0.5,
        "red_side_winrate":    0.5,
        "gold_lead_20_weight": 0.0,
    }


def _demo_stats() -> "pd.DataFrame":
    """Stats de demostración realistas LCK 2026 Semana 7 (parche 26.09)."""
    import pandas as pd
    return pd.DataFrame([
        # Fuente: resultados acumulados Liquipedia LCK 2026 Rounds 1-2
        {"team_name": "Gen.G",               "team_id": 2882,   "win_rate": 0.82, "gold_diff_15":  780, "baron_control_rate": 0.71, "vspm": 1.77, "first_blood_rate": 0.61, "first_dragon_rate": 0.66, "avg_game_duration": 31.8, "blue_side_winrate": 0.70, "red_side_winrate": 0.56, "gold_lead_20_weight": 1092},
        {"team_name": "T1",                  "team_id": 2883,   "win_rate": 0.79, "gold_diff_15":  840, "baron_control_rate": 0.69, "vspm": 1.81, "first_blood_rate": 0.64, "first_dragon_rate": 0.62, "avg_game_duration": 30.9, "blue_side_winrate": 0.73, "red_side_winrate": 0.60, "gold_lead_20_weight": 1176},
        {"team_name": "Hanwha Life Esports", "team_id": 126061, "win_rate": 0.72, "gold_diff_15":  560, "baron_control_rate": 0.63, "vspm": 1.62, "first_blood_rate": 0.55, "first_dragon_rate": 0.58, "avg_game_duration": 32.7, "blue_side_winrate": 0.65, "red_side_winrate": 0.52, "gold_lead_20_weight":  784},
        {"team_name": "KT Rolster",          "team_id": 63,     "win_rate": 0.68, "gold_diff_15":  620, "baron_control_rate": 0.61, "vspm": 1.65, "first_blood_rate": 0.54, "first_dragon_rate": 0.57, "avg_game_duration": 33.1, "blue_side_winrate": 0.62, "red_side_winrate": 0.51, "gold_lead_20_weight":  868},
        {"team_name": "Dplus KIA",           "team_id": 128218, "win_rate": 0.55, "gold_diff_15":  350, "baron_control_rate": 0.55, "vspm": 1.58, "first_blood_rate": 0.50, "first_dragon_rate": 0.52, "avg_game_duration": 33.8, "blue_side_winrate": 0.58, "red_side_winrate": 0.47, "gold_lead_20_weight":  490},
        {"team_name": "BNK FearX",           "team_id": 128217, "win_rate": 0.48, "gold_diff_15":  200, "baron_control_rate": 0.50, "vspm": 1.51, "first_blood_rate": 0.47, "first_dragon_rate": 0.49, "avg_game_duration": 34.5, "blue_side_winrate": 0.52, "red_side_winrate": 0.44, "gold_lead_20_weight":  280},
        {"team_name": "DN Freecs",           "team_id": 126370, "win_rate": 0.42, "gold_diff_15": -150, "baron_control_rate": 0.46, "vspm": 1.47, "first_blood_rate": 0.44, "first_dragon_rate": 0.46, "avg_game_duration": 35.2, "blue_side_winrate": 0.48, "red_side_winrate": 0.40, "gold_lead_20_weight": -210},
        {"team_name": "OK BRION",            "team_id": 132531, "win_rate": 0.35, "gold_diff_15": -320, "baron_control_rate": 0.42, "vspm": 1.43, "first_blood_rate": 0.41, "first_dragon_rate": 0.43, "avg_game_duration": 35.9, "blue_side_winrate": 0.44, "red_side_winrate": 0.37, "gold_lead_20_weight": -448},
        {"team_name": "Nongshim RedForce",   "team_id": 134115, "win_rate": 0.28, "gold_diff_15": -480, "baron_control_rate": 0.38, "vspm": 1.39, "first_blood_rate": 0.38, "first_dragon_rate": 0.40, "avg_game_duration": 36.5, "blue_side_winrate": 0.40, "red_side_winrate": 0.33, "gold_lead_20_weight": -672},
    ])


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas no instalado. Ejecuta: pip install pandas")
    sys.exit(1)


if __name__ == "__main__":
    header()

    # Bankroll configurable al inicio
    print(f"  Bankroll por defecto: ${BANKROLL:,.0f} MXN")
    raw_br = ask(f"  ¿Usar otro monto? (Enter para ${BANKROLL:,.0f}): ")
    try:
        bankroll = float(raw_br.replace(",", "").replace("$", "")) if raw_br else BANKROLL
    except ValueError:
        bankroll = BANKROLL
    print(f"  ✅ Bankroll: ${bankroll:,.2f} MXN  |  Kelly: {KELLY_FRACTION*100:.0f}%  |  Edge mínimo: {MIN_EDGE_THRESHOLD*100:.0f}%\n")

    # Cargar pipeline y entrenar
    stats_dict, predictor = load_and_train(bankroll)

    # Sesión de análisis de partidos
    run_match_session(stats_dict, predictor, bankroll)