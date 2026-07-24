# LandIQ - convenience launcher
# Usage:
#   .\run.ps1 api        -> start FastAPI on port 8000
#   .\run.ps1 dashboard  -> start Streamlit dashboard on port 8501
#   .\run.ps1 train      -> retrain all ML models
#   .\run.ps1 seed       -> seed the DB (requires Postgres running)
#   .\run.ps1 test       -> run all tests

param([string]$Command = "api")

$PYTHON = ".\venv311\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "venv311 not found. Run: python -m venv venv311 && .\venv311\Scripts\pip install -r requirements.txt"
    exit 1
}

switch ($Command) {
    "api" {
        Write-Host "[LandIQ] Starting FastAPI API on http://localhost:8000/docs" -ForegroundColor Cyan
        & $PYTHON -m uvicorn api.main:app --reload --port 8000
    }
    "dashboard" {
        Write-Host "[LandIQ] Starting Streamlit dashboard on http://localhost:8501" -ForegroundColor Cyan
        & $PYTHON -m streamlit run dashboard/app.py
    }
    "train" {
        Write-Host "[LandIQ] Training all ML models..." -ForegroundColor Cyan
        & $PYTHON -m scripts.train_models
    }
    "seed" {
        Write-Host "[LandIQ] Seeding areas into DB..." -ForegroundColor Cyan
        & $PYTHON -m scripts.seed_areas
    }
    "test" {
        Write-Host "[LandIQ] Running test suite..." -ForegroundColor Cyan
        & $PYTHON -m pytest tests/ -v
    }
    default {
        Write-Host "Unknown command: $Command"
        Write-Host "Usage: .\run.ps1 [api|dashboard|train|seed|test]"
        exit 1
    }
}
