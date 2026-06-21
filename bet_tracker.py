"""
============================================================
PREDICTION OS V2 — REGISTRO DE APUESTAS
============================================================
Lleva un historial local de las apuestas que marcas, para medir si el
sistema REALMENTE te da ganancias (no basta con que el modelo "acierte":
lo que importa es el dinero a largo plazo).

Guarda en bets_log.csv (junto a la app, gitignored). Cada apuesta tiene:
  fecha, liga, partido, pick (equipo apostado), prob. del modelo, momio,
  cuota, edge, stake, estado (pendiente/ganada/perdida) y la ganancia.

La ganancia se calcula al marcar el resultado:
  ganada  → +stake * (cuota - 1)
  perdida → -stake
============================================================
"""

from __future__ import annotations

import os
import csv
from datetime import datetime

import config

PENDING, WON, LOST = "pendiente", "ganada", "perdida"

_FIELDS = [
    "id", "fecha", "liga", "partido", "pick", "lado",
    "prob_modelo", "momio", "cuota", "edge_pct",
    "stake_mxn", "estado", "ganancia_mxn",
]
_PATH = os.path.join(config._app_dir(), "bets_log.csv")


def _read() -> list[dict]:
    if not os.path.exists(_PATH):
        return []
    try:
        with open(_PATH, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _write(rows: list[dict]) -> None:
    with open(_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _FIELDS})


def add_bet(
    liga: str, partido: str, pick: str, lado: str,
    prob_modelo: float, momio: int, cuota: float, edge_pct: float,
    stake_mxn: float,
) -> dict:
    """Registra una apuesta nueva en estado 'pendiente'."""
    rows = _read()
    new_id = (max((int(r["id"]) for r in rows), default=0) + 1)
    bet = {
        "id": str(new_id),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "liga": liga,
        "partido": partido,
        "pick": pick,
        "lado": lado,
        "prob_modelo": f"{prob_modelo*100:.1f}",
        "momio": f"{momio:+d}",
        "cuota": f"{cuota:.3f}",
        "edge_pct": f"{edge_pct:+.1f}",
        "stake_mxn": f"{stake_mxn:.2f}",
        "estado": PENDING,
        "ganancia_mxn": "0.00",
    }
    rows.append(bet)
    _write(rows)
    return bet


def set_estado(bet_id: str, estado: str) -> None:
    """Marca el resultado de una apuesta y recalcula la ganancia."""
    rows = _read()
    for r in rows:
        if r["id"] == str(bet_id):
            r["estado"] = estado
            stake = float(r.get("stake_mxn", 0) or 0)
            cuota = float(r.get("cuota", 0) or 0)
            if estado == WON:
                r["ganancia_mxn"] = f"{stake * (cuota - 1):.2f}"
            elif estado == LOST:
                r["ganancia_mxn"] = f"{-stake:.2f}"
            else:
                r["ganancia_mxn"] = "0.00"
            break
    _write(rows)


def delete_bet(bet_id: str) -> None:
    _write([r for r in _read() if r["id"] != str(bet_id)])


def all_bets() -> list[dict]:
    """Todas las apuestas, de la más reciente a la más antigua."""
    return list(reversed(_read()))


def summary() -> dict:
    """Métricas agregadas del historial."""
    rows = _read()
    settled = [r for r in rows if r["estado"] in (WON, LOST)]
    won = [r for r in settled if r["estado"] == WON]
    staked_settled = sum(float(r.get("stake_mxn", 0) or 0) for r in settled)
    profit = sum(float(r.get("ganancia_mxn", 0) or 0) for r in settled)
    staked_all = sum(float(r.get("stake_mxn", 0) or 0) for r in rows)

    return {
        "n_total":     len(rows),
        "n_settled":   len(settled),
        "n_pending":   len(rows) - len(settled),
        "n_won":       len(won),
        "win_rate":    (len(won) / len(settled)) if settled else 0.0,
        "staked_all":  staked_all,
        "staked_settled": staked_settled,
        "profit":      profit,
        # Yield = ganancia / dinero arriesgado en apuestas resueltas
        "yield_pct":   (profit / staked_settled * 100) if staked_settled else 0.0,
    }
