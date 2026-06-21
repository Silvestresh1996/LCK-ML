# ⚡ Prediction OS V2 — Value betting para esports de LoL

**Motor cuantitativo de predicción de esports.**

Descarga datos **reales** de partidas de League of Legends desde
[Oracle's Elixir](https://oracleselixir.com) (gratis, sin API key), entrena un
modelo de rating **Elo + oro@15** sobre los resultados reales y detecta
*value bets* comparando la probabilidad del modelo contra los momios de la casa,
con gestión de stake por Criterio de Kelly fraccional.

> **Solo datos reales.** La app no usa datos de demostración: si no hay conexión
> o no se pueden cargar los datos, se detiene con un error claro. Nunca verás
> números inventados sobre los que podrías apostar por error.

## 🚀 Cómo ejecutar

**Opción 1 — doble clic (recomendada):**
Abre **`Prediction OS.bat`**. Lanza la interfaz gráfica sin consola.

**Opción 2 — terminal:**
```powershell
python prediction_os_v2.py     # interfaz gráfica
python lck_main.py             # versión de terminal (LCK)
python lck_main.py LPL         # otra liga: LPL, LEC, LCS
```

Primera instalación de dependencias:
```powershell
pip install customtkinter matplotlib scikit-learn pandas numpy requests joblib
```

## 🌐 Datos: Oracle's Elixir (gratis, sin API key)

La app descarga el CSV anual de Oracle's Elixir (datos públicos de todas las
ligas pro de LoL) desde Google Drive, **se cachea localmente** (`oe_<año>.csv`)
y se re-descarga solo cuando el caché envejece (>12 h). **No necesitas API key.**

## 📁 Estructura del proyecto

| Archivo | Rol |
|---|---|
| `config.py` | Configuración central: ligas, parámetros del modelo (Elo/Kelly), bankroll. |
| `oracle_pipeline.py` | **Capa de datos**: descarga/cachea Oracle's Elixir, calcula KPIs por equipo y arma los partidos. |
| `model.py` | Modelo Elo + oro@15 (validación cronológica) + utilidades de momios/Kelly. |
| `prediction_os_v2.py` | **GUI**: dashboard, analizador, scanner de value bets, configuración. |
| `lck_main.py` | **CLI** de terminal. |
| `Prediction OS.bat` | Lanzador de doble clic. |
| `build_exe.ps1` | Construye un `.exe` con PyInstaller (ver nota). |
| `universal_pipeline.py` | Cliente de PandaScore (no usado por defecto; para un futuro plan de pago). |

## 📦 Sobre el `.exe`

`build_exe.ps1` genera `dist\PredictionOS.exe`. **Importante:** si tienes
**Smart App Control** activado en Windows, bloqueará cualquier `.exe` sin firma
digital. Como solo se puede *apagar* (no se reactiva sin reinstalar Windows), la
recomendación es **usar `Prediction OS.bat`** en lugar del `.exe`.

## 📈 Sobre el modelo (léelo)

El modelo usa un **rating Elo entrenado cronológicamente** más la **diferencia
de oro a los 15 minutos**, ambos calculados *antes* de cada partido (nunca con
información del futuro → **sin fuga de datos**). Una regresión logística calibra
eso hacia una probabilidad fiable, clave para que el edge y el Kelly sean
correctos. Busca **Valor Esperado (+EV)**: si la probabilidad del modelo supera
a la implícita de la casa, genera señal con stake por Kelly.

- **AUC honesto ≈ 0.72** (validación cronológica, LCK 2026, 349 partidos).
- **Por qué solo Elo + oro@15:** una ablación rigurosa mostró que el resto de
  stats (primera sangre, dragón, barón, visión, forma reciente) están muy
  correlacionadas con el Elo y solo metían ruido (con las 7 juntas el AUC caía a
  0.62). El oro@15 es la única que añade señal independiente. *Menos es más.*
- Las demás stats sí se calculan y se muestran en la GUI (para tu criterio),
  aunque no entren al modelo.

> Apuesta con responsabilidad. El modelo es una herramienta de apoyo: te ayuda a
> encontrar momios con valor, pero **no garantiza ganancias**. Las casas son muy
> eficientes; el margen real, si existe, es pequeño y exige disciplina con el
> bankroll y el Kelly fraccionado.

---
Desarrollado por **Jorge Silvestre Medeles Medina** | Estudiante de Ingeniería en Ciencia de Datos.
Datos de [Oracle's Elixir](https://oracleselixir.com) (Tim Sevenhuysen).
