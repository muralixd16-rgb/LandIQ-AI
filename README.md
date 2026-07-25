# LandIQ — Smart Real Estate Investment Advisor

A two-sided platform that tells buyers which undervalued Telangana areas will
appreciate the most based on government development plans, and tells sellers
whether to hold or sell their property based on the same signals.

> *"You can't afford Hitech City — so where should you invest today that
> becomes the next Hitech City?"*


## 🚀 Live Demo

🌐 **Dashboard:** https://landiq-dashboard.onrender.com/

⚡ **API:** https://landiq-api.onrender.com/

📚 **API Documentation:** https://landiq-api.onrender.com/docs


## Architecture — Postgres Only Stack

| Layer              | Technology                              |
|--------------------|----------------------------------------|
| API                | FastAPI + Uvicorn                       |
| Database           | PostgreSQL 16 + PostGIS                 |
| Async Jobs         | FastAPI `BackgroundTasks` (no Redis)    |
| Vector Search      | PostgreSQL Full-Text Search (no Qdrant) |
| ML — Clustering    | scikit-learn K-Means                    |
| ML — Classification| scikit-learn Random Forest              |
| ML — Prediction    | XGBoost Regressor + SHAP                |
| NLP                | spaCy `en_core_web_sm`                  |
| Geocoding          | GeoPy + Nominatim + local gazetteer     |
| PDF Parsing        | PyMuPDF                                 |
| Dashboard          | Streamlit + Folium                      |
| Container          | Docker Compose (2 services only)        |

## Quick Start (Docker — recommended)

```bash
cp .env.example .env
docker compose up --build
```

- **API + Swagger docs:** http://localhost:8000/docs
- **Streamlit dashboard:** http://localhost:8501

The API auto-trains all models and seeds 35 Hyderabad localities on first startup.

## Quick Start (local, no Docker)

Requires Python 3.11+ and a running Postgres/PostGIS instance.

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train all ML models (saves to models/)
python -m scripts.train_models

# Seed area data
python -m scripts.seed_areas

# Run API
uvicorn api.main:app --reload --port 8000

# Run dashboard (separate terminal)
streamlit run dashboard/app.py --server.port 8501
```

## API Endpoints

| Method | Endpoint                  | Description                                      |
|--------|---------------------------|--------------------------------------------------|
| POST   | `/profile`                | Submit investor profile → segment + strategy     |
| GET    | `/profile/{id}`           | Retrieve saved profile                           |
| POST   | `/recommend/buyer`        | Top-N area shortlist ranked by predicted ROI     |
| POST   | `/recommend/seller`       | Hold/sell recommendation for a property          |
| POST   | `/predict/price`          | Price appreciation prediction for any area       |
| POST   | `/upload-plan`            | Upload government PDF → background NLP ingestion |
| GET    | `/areas`                  | List all tracked areas (filter by zone)          |
| GET    | `/areas/{id}/report`      | Full investment report for an area               |
| GET    | `/heatmap`                | GeoJSON for Folium map rendering                 |
| POST   | `/report/generate`        | Downloadable investment report                   |

## Example — Buyer Flow

```bash
# 1. Create profile
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Murali",
    "monthly_income": 65000,
    "max_budget": 2800000,
    "investment_horizon": "mid",
    "risk_appetite": "moderate",
    "preferred_asset_type": "plot",
    "target_roi_percent": 80
  }'
# → Returns: segment_label, segment_strategy, profile id

# 2. Get recommendations
curl -X POST http://localhost:8000/recommend/buyer \
  -H "Content-Type: application/json" \
  -d '{"profile_id": 1, "top_n": 5}'
# → Returns: 5 ranked areas with ROI predictions and scorecards
```

## Example — Seller Flow

```bash
curl -X POST http://localhost:8000/recommend/seller \
  -H "Content-Type: application/json" \
  -d '{
    "area_name": "Kompally",
    "property_size_sqyd": 200,
    "current_estimated_value": 4800000
  }'
# → Returns: HOLD/SELL recommendation with predicted future values
```

## Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
landiq/
├── api/
│   ├── main.py          # FastAPI app + all routes
│   └── schemas.py       # Pydantic request/response models
├── ml/
│   ├── segmentation.py  # K-Means investor clustering
│   ├── classifier.py    # Random Forest zone classifier
│   ├── predictor.py     # XGBoost price predictor
│   └── explainer.py     # SHAP explanations
├── pipeline/
│   ├── extractor.py     # spaCy NLP extraction
│   └── ingest.py        # Background PDF ingestion (FastAPI BackgroundTasks)
├── db/
│   ├── models.py        # SQLAlchemy + PostGIS models
│   ├── session.py       # DB engine + session management
│   └── spatial.py       # PostGIS radius queries + Postgres FTS
├── dashboard/
│   ├── app.py           # Streamlit multi-page dashboard
│   └── map.py           # Folium heatmap builder
├── scripts/
│   ├── seed_areas.py    # Seed 35 Hyderabad localities
│   ├── train_models.py  # Train all 3 ML models
│   └── train_segmenter.py
├── tests/
│   ├── test_api.py
│   ├── test_ml.py
│   └── test_pipeline.py
├── models/              # Saved .joblib model artifacts
├── docker-compose.yml   # Postgres + API + Dashboard only
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Why No Redis / Qdrant?

- **No Redis**: Background PDF ingestion runs via FastAPI's built-in `BackgroundTasks`,
  keeping the system synchronous-safe and removing a separate broker process.
- **No Qdrant**: Document chunk search uses PostgreSQL Full-Text Search with a GIN
  trigram index — no external vector DB, no embedding API keys, no extra container.

This makes `docker compose up` a **2-service deployment**: Postgres and the App.
