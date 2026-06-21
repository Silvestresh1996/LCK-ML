# Prediction OS V2 — Value betting para esports de LoL

Sistema que descarga partidas de ligas de League of Legends (PandaScore),
entrena un modelo sobre **resultados reales** y detecta *value bets*
comparando la probabilidad del modelo contra los momios de la casa,
con gestión de stake por Criterio de Kelly fraccional.

## Estructura

| Archivo | Rol |
|---|---|
| `config.py` | Configuración central: ligas, features, parámetros del modelo, bankroll/Kelly. **Sin secretos.** |
| `secrets_local.py` | Tu API key de PandaScore. **Gitignored — no se sube.** |
| `universal_pipeline.py` | Capa de datos: descarga partidas, detecta parche, calcula KPIs por equipo. Multi-liga. |
| `model.py` | Modelo de predicción (XGBoost) entrenado sobre resultados reales + utilidades de momios/Kelly. |
| `prediction_os_v2.py` | **GUI** (dashboard, analizador de partidos, scanner de value bets, ajustes). |
| `lck_main.py` | **CLI** equivalente para terminal. |

## Configurar la API key

Elige una opción:

```powershell
# Opción A — variable de entorno (recomendado)
$env:PANDASCORE_API_KEY = "tu_key"

# Opción B — archivo local (ya creado en secrets_local.py)
```

## Uso

```bash
pip install customtkinter matplotlib xgboost scikit-learn pandas numpy requests joblib

python prediction_os_v2.py     # interfaz gráfica
python lck_main.py             # terminal (LCK por defecto)
python lck_main.py 290         # otra liga: 290=LPL, 4197=LEC, 4198=LCS
```

Si la API no devuelve datos, el sistema usa **datos de demostración** para
seguir siendo usable sin conexión.

## Sobre el modelo (importante)

El modelo entrena con los **resultados reales** de los partidos descargados:
cada partido es una fila, las features son las diferencias de KPIs entre los
dos equipos y la etiqueta es quién ganó realmente. El AUC reportado (validación
cronológica) es honesto, típicamente en el rango **0.55–0.70** para datasets
pequeños de una sola liga — no esperes valores cercanos a 1.0.

**Limitaciones honestas:**
- Los KPIs se calculan sobre la misma ventana que se predice (fuga leve de
  información, aceptable a esta escala).
- Las features de *early game* (`gold_diff_15`, `vspm`, baron…) requieren el
  endpoint de stats por partida; mientras no se integren, el modelo corre en
  modo `lite` (solo win rate y lados del mapa).
- Con menos de ~20 partidos reales se usa un fallback simple sobre win rate.

> Apuesta con responsabilidad. El modelo es una herramienta de apoyo, no una
> garantía de ganancia.
