# Prediction OS V2 — Value betting para esports de LoL

Descarga partidas **reales** de ligas de League of Legends (PandaScore),
entrena un modelo sobre los **resultados reales** de esas partidas y detecta
*value bets* comparando la probabilidad del modelo contra los momios de la casa,
con gestión de stake por Criterio de Kelly fraccional.

> **Solo datos reales.** La app no usa datos de demostración: si no hay conexión
> o la API no responde, se detiene con un error claro. Nunca verás números
> inventados sobre los que podrías apostar por error.

## Cómo ejecutar

**Opción 1 — doble clic (recomendada):**
Abre **`Prediction OS.bat`**. Lanza la interfaz gráfica sin consola.

**Opción 2 — terminal:**
```powershell
python prediction_os_v2.py     # interfaz gráfica
python lck_main.py             # versión de terminal (LCK)
python lck_main.py 290         # otra liga: 290=LPL, 4197=LEC, 4198=LCS
```

Primera instalación de dependencias:
```powershell
pip install customtkinter matplotlib xgboost scikit-learn pandas numpy requests joblib
```

## API key (cómo se cuida tu llave)

La llave **nunca** está escrita en el código ni se sube a git. Se resuelve así:
1. Variable de entorno `PANDASCORE_API_KEY`, o
2. Archivo `api_key.txt` junto a la app (gitignored), o
3. `secrets_local.py` (solo en desarrollo, gitignored).

La forma más fácil: abre la app → pestaña **Configuración** → pega tu API key →
**Aplicar**. Se guarda en `api_key.txt` y no la vuelves a pedir.

## Estructura

| Archivo | Rol |
|---|---|
| `config.py` | Configuración central: ligas, features, parámetros del modelo, bankroll/Kelly, manejo de la API key. **Sin secretos.** |
| `universal_pipeline.py` | Capa de datos: descarga partidas reales, detecta parche, calcula KPIs por equipo. Multi-liga. |
| `model.py` | Modelo (XGBoost) entrenado sobre resultados reales + utilidades de momios/Kelly. |
| `prediction_os_v2.py` | **GUI** (dashboard, analizador, scanner de value bets, configuración). |
| `lck_main.py` | **CLI** de terminal. |
| `Prediction OS.bat` | Lanzador de doble clic. |
| `build_exe.ps1` | Construye un `.exe` con PyInstaller (ver nota abajo). |

## Sobre el `.exe`

`build_exe.ps1` genera `dist\PredictionOS.exe`. **Importante:** si tienes
**Smart App Control** activado en Windows, bloqueará cualquier `.exe` sin firma
digital (incluido este). Como Smart App Control solo se puede *apagar* (no se
puede volver a encender sin reinstalar Windows), la recomendación es **usar el
lanzador `Prediction OS.bat`** en lugar del `.exe`. El resultado es el mismo:
doble clic y se abre la app.

## Sobre el modelo (léelo)

El modelo entrena con los **resultados reales** de los partidos descargados:
cada partido es una fila, las features son diferencias de KPIs entre los dos
equipos y la etiqueta es quién ganó de verdad.

- El AUC reportado (validación cronológica) ronda **0.80–0.85** con datos reales
  de una liga activa. **Ojo:** es algo optimista porque los KPIs se calculan
  sobre la misma ventana de partidos que se predice (fuga leve de información).
  El rendimiento real sobre partidos futuros será algo menor.
- Las features de *early game* (`gold_diff_15`, `vspm`, baron) requieren el
  endpoint de stats por partida, aún no integrado → el modelo corre en modo
  `lite` (win rate + lados del mapa + proxy de baron). Integrarlo es la mejora
  pendiente con más potencial.

> Apuesta con responsabilidad. El modelo es una herramienta de apoyo: te ayuda a
> encontrar momios con valor, pero **no garantiza ganancias**. Las casas son muy
> eficientes; el margen real, si existe, es pequeño y exige disciplina con el
> bankroll y el Kelly fraccionado.
