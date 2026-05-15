# ⚡ LCK Prediction OS · 2026
**Motor cuantitativo de predicción de Esports basado en XGBoost**

Este sistema automatiza la extracción de datos de la liga coreana de League of Legends (LCK CL) y utiliza Machine Learning para encontrar discrepancias de valor en los momios de las casas de apuestas.

## 🚀 Características principales
* **ETL Pipeline:** Extracción automática de las últimas 100 partidas vía PandaScore API.
* **ML Engine:** Modelo de clasificación XGBoost entrenado con KPIs específicos del parche actual (26.09).
* **Risk Management:** Implementación del **Criterio de Kelly (25%)** para optimizar el crecimiento del bankroll y minimizar el riesgo de ruina.
* **Betting Intelligence:** Interfaz de consola que calcula la ventaja (Edge) real sobre momios americanos.

## 📁 Estructura del Proyecto
* `lck_data_pipeline.py`: Gestor de conexión con la API y limpieza de datos.
* `lck_ml_model.py`: Entrenamiento del modelo y validación cruzada.
* `lck_main.py`: Punto de entrada interactivo para el usuario.
* `.gitignore`: Protección de credenciales y archivos temporales.

## 🛠️ Instalación y Uso
1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install pandas xgboost scikit-learn requests

   📈 Metodología
El sistema no solo predice quién ganará, sino que busca Valor Esperado (+EV). Si la probabilidad del modelo es significativamente mayor a la probabilidad implícita de la casa de apuestas, el sistema genera una señal de entrada con un Stake calculado proporcionalmente a la confianza del modelo.

Desarrollado por Jorge Silvestre Medeles Medina | Estudiante de Ingeniería en Ciencia de Datos.
