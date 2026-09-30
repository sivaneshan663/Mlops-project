$prototypePython = Join-Path $PSScriptRoot '..\..\work\prototype-env\Scripts\python.exe'
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) { $prototypePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $prototypePython)) { throw 'The project Python environment is missing. See README.md for setup.' }
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:OLLAMA_MODEL) { $env:OLLAMA_MODEL = 'qwen3:1.7b' }
if (-not $env:EMBEDDING_MODEL) { $env:EMBEDDING_MODEL = 'all-minilm' }
& $prototypePython -m uvicorn app:app --host 127.0.0.1 --port 8000
