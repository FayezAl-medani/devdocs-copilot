$env:PATH = "C:\Users\ASUS\.local\bin;" + $env:PATH
Write-Host "Running ruff check..." -ForegroundColor Cyan
uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running ruff format check..." -ForegroundColor Cyan
uv run ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running pyright..." -ForegroundColor Cyan
uv run pyright
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running pytest..." -ForegroundColor Cyan
uv run pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All checks passed successfully!" -ForegroundColor Green
