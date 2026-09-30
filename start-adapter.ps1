$prototypePython = Join-Path $PSScriptRoot '..\..\work\prototype-env\Scripts\python.exe'
Set-Location -LiteralPath $PSScriptRoot
& $prototypePython -m uvicorn adapter_service:app --host 127.0.0.1 --port 8001
