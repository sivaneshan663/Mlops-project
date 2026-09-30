$prototypePython = Join-Path $PSScriptRoot '..\..\work\prototype-env\Scripts\python.exe'
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) { $prototypePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
$mlflowPython = Join-Path $PSScriptRoot '..\..\work\mlflow-ui-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $prototypePython)) { throw 'Missing Python environment. See README.md.' }
$env:OLLAMA_MODEL='qwen3:1.7b'
$env:EMBEDDING_MODEL='all-minilm'
Set-Location -LiteralPath $PSScriptRoot
if ((Test-Path -LiteralPath $mlflowPython) -and -not (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $mlflowPython -ArgumentList @('-m','mlflow','ui','--backend-store-uri','.\mlruns','--host','127.0.0.1','--port','5000') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
}
if ((Test-Path -LiteralPath (Join-Path $PSScriptRoot 'artifacts\sft_manifest.json')) -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\work\base-qwen06'))) {
    $adapterListener = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
    if (-not $adapterListener) {
        Start-Process -FilePath $prototypePython -ArgumentList @('-m','uvicorn','adapter_service:app','--host','127.0.0.1','--port','8001') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
    }
}
$appListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($appListener) { Write-Output 'A service is already running on port 8000. Open http://127.0.0.1:8000.' }
else { & $prototypePython -m uvicorn app:app --host 127.0.0.1 --port 8000 }
