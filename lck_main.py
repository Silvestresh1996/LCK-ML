"""
============================================================
PREDICTION OS V2 — CLI INTERACTIVO (terminal)
============================================================
Versión de terminal del sistema. La GUI equivalente es
prediction_os_v2.py.

FLUJO:
  1. Descarga partidas reales de la liga (PandaScore) y calcula KPIs.
  2. Entrena el modelo sobre resultados reales.
  3. Pregunta los partidos uno a uno con momios AMERICANOS (+285, -425).
  4. Calcula probabilidad del modelo + Kelly y muestra value bets.

USO:
    python lck_main.py            # LCK por defecto
    python lck_main.py 290        # otra liga por ID (290=LPL, 4197=LEC, 4198=LCS)

Requiere la API key en la variable de entorno PANDASCORE_API_KEY
o en secrets_local.py. Sin datos de API, usa modo demostración.
============================================================
"""

import sys
import logging

import pandas as pd

import config
from universal_pipeline import UniversalPipeline
from model import MatchPredictor, american_to_decimal, kelly_stake

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Colores de consola ──
RESET, BOLD, GOLD, GREEN, RED, CYAN, DIM = (
    "\033[0m", "\033[1m", "\033[93m", "\033[92m", "\033[91m", "\033[96m", "\033[2m"
)
LINE = "─" * 56


def header(league_name: str):
    print(f"\n{BOLD}{'═'*56}{RESET}")
    print(f"{BOLD}  ⚡ PREDICTION OS V2 · {league_name}  ·  {config.CURRENT_YEAR}{RESET}")
    print(f"{DIM}  Modelo entrenado sobre resultados reales · Kelly {config.KELLY_FRACTION*100:.0f}%{RESET}")
    print(f"{BOLD}{'═'*56}{RESET}\n")


def ask(prompt: str, default: str = "") -> str:
    try:
        val = input(f"  {prompt}").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Saliendo. ¡Suerte!\n")
        sys.exit(0)


def pick_team(prompt: str, teams: list[str]) -> str:
    print(f"\n  {prompt}")
    for i, t in enumerate(teams, 1):
        print(f"    {DIM}{i:>2}.{RESET} {t}")
    while True:
        choice = ask("Tu elección (número o nombre): ")
        if choice.isdigit() and 1 <= int(choice) <= len(teams):
            return teams[int(choice) - 1]
        matches = [t for t in teams if choice.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguo: {matches}. Sé más específico.")
        else:
            print(f"  {RED}Opción inválida.{RESET}")


def parse_american(raw: str):
    try:
        val = int(raw.replace(" ", "").replace(",", "").replace("+", ""))
        if abs(val) < 100:
            print(f"  {RED}Momio inválido. Los momios americanos son ≥ 100.{RESET}")
            return None
        return val
    except ValueError:
        print(f"  {RED}Formato inválido. Usa +285 o -425.{RESET}")
        return None


def load_and_train(league_id: int) -> tuple[dict, MatchPredictor, str]:
    """Descarga datos reales, calcula KPIs y entrena. Sin datos → aborta."""
    league_name = next((n for n, i in config.LEAGUES.items() if i == league_id), f"Liga {league_id}")
    print(f"{CYAN}{LINE}\n  Cargando datos de {league_name}…\n{CYAN}{LINE}{RESET}")

    if not config.PANDASCORE_API_KEY:
        print(f"  {RED}Falta la API Key. Define PANDASCORE_API_KEY o secrets_local.py.{RESET}")
        sys.exit(1)

    df_matches, df_stats = pd.DataFrame(), pd.DataFrame()
    try:
        pipe = UniversalPipeline(league_id=league_id)
        pipe.detect_current_patch()
        pipe.fetch_team_names()
        df_matches = pipe.get_matches(limit=100)
        if not df_matches.empty:
            df_stats = pipe.build_team_stats(df_matches, min_games=config.MIN_GAMES_PER_TEAM)
    except Exception as e:
        print(f"  {RED}Error de conexión: {e}{RESET}")
        sys.exit(1)

    if df_stats.empty:
        print(f"  {RED}La API no devolvió datos utilizables. Revisa tu API Key, "
              f"tu plan o la liga.{RESET}")
        sys.exit(1)

    predictor = MatchPredictor()
    metrics = predictor.train(df_stats, df_matches)
    auc = metrics.get("auc_mean")
    auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"
    print(f"  ✅ Modelo: modo={metrics.get('mode')} | AUC={auc_str} | "
          f"partidos reales={metrics.get('matches', 0)}")

    stats_dict = {row["team_name"]: row for row in df_stats.to_dict("records")}
    return stats_dict, predictor, league_name


def analyze_match(team_a, team_b, stats_dict, predictor):
    sa = stats_dict[team_a]
    sb = stats_dict[team_b]

    print(f"\n  Momios de Codere (americano). Ej: {DIM}favorito -425 | underdog +285{RESET}")
    oa = None
    while oa is None:
        oa = parse_american(ask(f"  Momio de {BOLD}{team_a}{RESET} (blue): "))
    ob = None
    while ob is None:
        ob = parse_american(ask(f"  Momio de {BOLD}{team_b}{RESET} (red): "))

    pred = predictor.predict_match(sa, sb, side_a="blue")
    prob_a, prob_b = pred["prob_a"], pred["prob_b"]
    dec_a, dec_b = american_to_decimal(oa), american_to_decimal(ob)
    k_a = kelly_stake(prob_a, dec_a, config.BANKROLL, config.KELLY_FRACTION)
    k_b = kelly_stake(prob_b, dec_b, config.BANKROLL, config.KELLY_FRACTION)

    print(f"\n  {'':24}{CYAN}{team_a[:12]:>12}{RESET}  {CYAN}{team_b[:12]:>12}{RESET}")
    print(f"  {'Prob. modelo':24}{prob_a*100:>11.1f}%  {prob_b*100:>11.1f}%")
    print(f"  {'Cuota decimal':24}{dec_a:>12.3f}  {dec_b:>12.3f}")
    print(f"  {'Edge':24}{k_a['edge_pct']:>+11.1f}%  {k_b['edge_pct']:>+11.1f}%")
    print(f"\n  {LINE}")

    found = False
    for team, k, dec in [(team_a, k_a, dec_a), (team_b, k_b, dec_b)]:
        if k["is_value"]:
            found = True
            print(f"\n  {GOLD}{k['signal']}{RESET} → {BOLD}{team}{RESET}")
            print(f"  Cuota {dec:.3f} | Stake {GREEN}${k['stake_mxn']:,.2f} MXN{RESET} "
                  f"| EV +${k['ev_mxn']:,.2f} ({k['roi_pct']:.1f}% ROI)")
    if not found:
        print(f"\n  {RED}Sin value bets — necesitas edge > {config.MIN_EDGE_THRESHOLD*100:.0f}%.{RESET}")
    print(f"\n  {LINE}")


def main():
    league_id = config.DEFAULT_LEAGUE_ID
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        league_id = int(sys.argv[1])

    stats_dict, predictor, league_name = load_and_train(league_id)
    header(league_name)
    teams = sorted(stats_dict.keys())

    while True:
        team_a = pick_team("Equipo en BLUE SIDE:", teams)
        team_b = pick_team("Rival (RED SIDE):", [t for t in teams if t != team_a])
        analyze_match(team_a, team_b, stats_dict, predictor)
        if ask("\n  ¿Otro partido? (s/n): ", default="s").lower() != "s":
            break

    print(f"\n  {DIM}Apuesta con responsabilidad. El modelo es una herramienta,{RESET}")
    print(f"  {DIM}no una garantía de ganancia.{RESET}\n")


if __name__ == "__main__":
    main()
