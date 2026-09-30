$mlflowPython = Join-Path $PSScriptRoot '..\..\work\mlflow-ui-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mlflowPython)) {
    throw 'The MLflow UI environment is missing. See README.md.'
}

Set-Location -LiteralPath $PSScriptRoot
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Output 'MLflow is already running at http://127.0.0.1:5000.'
} else {
    & $mlflowPython -m mlflow ui --backend-store-uri '.\mlruns' --host 127.0.0.1 --port 5000
}
