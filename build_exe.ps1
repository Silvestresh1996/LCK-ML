# ============================================================
#  Build del ejecutable de Prediction OS V2 (Windows)
# ============================================================
#  Uso:   powershell -ExecutionPolicy Bypass -File build_exe.ps1
#  Salida: dist\PredictionOS.exe
#
#  Notas:
#   - --exclude-module secrets_local  → la API key NUNCA entra al .exe.
#     El .exe lee la llave de api_key.txt (junto al .exe) o de la
#     variable de entorno PANDASCORE_API_KEY, o se pega en Configuración.
#   - --windowed → sin ventana de consola (es una app gráfica).
# ============================================================

pyinstaller --noconfirm --onefile --windowed --name "PredictionOS" `
    --collect-all customtkinter `
    --collect-submodules sklearn `
    --collect-data xgboost --collect-binaries xgboost `
    --exclude-module secrets_local `
    prediction_os_v2.py

Write-Host ""
Write-Host "Listo. Ejecutable en: dist\PredictionOS.exe" -ForegroundColor Green
Write-Host "Pon tu API key en Configuracion la primera vez (se guarda en api_key.txt)." -ForegroundColor Yellow
