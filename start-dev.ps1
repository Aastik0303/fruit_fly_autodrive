# Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173) in two new windows.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File D:\project_flywire\start-dev.ps1

$root = $PSScriptRoot

Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; .\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000"
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root\frontend'; npm run dev"

Write-Host 'Backend  : http://127.0.0.1:8000/docs'
Write-Host 'Frontend : http://localhost:5173  (ready after the backend has loaded, ~10 s)'
