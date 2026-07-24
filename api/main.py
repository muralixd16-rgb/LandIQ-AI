"""
LandIQ — Smart Real Estate Investment Advisor
Main FastAPI application.

Endpoints:
    POST /profile              — Investor profiling + K-Means segmentation
    GET  /profile/{id}         — Retrieve saved profile
    POST /recommend/buyer      — Ranked area shortlist for a buyer profile
    POST /recommend/seller     — Hold/sell recommendation for a seller
    POST /predict/price        — Price appreciation prediction for any area
    POST /upload-plan          — Upload government PDF → background NLP ingestion
    GET  /areas                — List all tracked areas
    GET  /areas/{id}/report    — Full investment report for an area
    GET  /heatmap              — GeoJSON for map rendering
    POST /report/generate      — Generate downloadable PDF investment report

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""
import os
import io
import joblib
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File,
    BackgroundTasks, Query
)
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.session import get_db, init_db
from db.models import (
    InvestorProfile, Area, PricePrediction, DevelopmentProject,
    InvestmentHorizon, RiskAppetite, AssetType, ZoneLabel,
)
from db.spatial import full_text_search, heatmap_geojson
from api.schemas import (
    InvestorProfileCreate, InvestorProfileResponse,
    BuyerRecommendationRequest, BuyerRecommendationResponse,
    SellerRecommendationRequest, SellerRecommendationResponse,
    AreaScorecard, AreaDetail, PricePredictRequest, PricePredictResponse,
    AreaListItem,
)
from ml.segmentation import InvestorSegmenter, SEGMENT_PROFILES

SEGMENTER_PATH  = os.path.join("models", "segmenter.joblib")
CLASSIFIER_PATH = os.path.join("models", "classifier.joblib")
PREDICTOR_PATH  = os.path.join("models", "predictor.joblib")

_segmenter  = None
_classifier = None
_predictor  = None


def _load_models():
    global _segmenter, _classifier, _predictor
    if os.path.exists(SEGMENTER_PATH):
        _segmenter = InvestorSegmenter.load(SEGMENTER_PATH)
        print("[startup] Loaded investor segmenter.")
    if os.path.exists(CLASSIFIER_PATH):
        _classifier = joblib.load(CLASSIFIER_PATH)
        print("[startup] Loaded zone classifier.")
    if os.path.exists(PREDICTOR_PATH):
        _predictor = joblib.load(PREDICTOR_PATH)
        print("[startup] Loaded price predictor.")

from sqlalchemy.orm import Session


def _train_all_models(db: Session):
    """Train all models from scratch using DB data if they don't exist. Fallback to synthetic data if DB not ready."""
    from ml.segmentation import InvestorSegmenter, load_training_data_from_db as load_inv, generate_synthetic_training_data as gen_inv
    from ml.classifier  import ZoneClassifier,    load_training_data_from_db as load_zone, generate_synthetic_training_data as gen_zone
    from ml.predictor   import PricePredictor,    load_training_data_from_db as load_price, generate_synthetic_training_data as gen_price
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(SEGMENTER_PATH):
        try:
            df = load_inv(db)
            print(f"[startup] Loaded {len(df)} real investor profiles from DB for training segmenter.")
        except Exception as e:
            print(f"[startup] Fallback to synthetic data for segmenter training: {e}")
            df = gen_inv(800)
        seg = InvestorSegmenter(n_clusters=4)
        seg.fit(df)
        seg.save(SEGMENTER_PATH)
        print("[startup] Trained + saved segmenter.")

    if not os.path.exists(CLASSIFIER_PATH):
        try:
            df = load_zone(db)
            print(f"[startup] Loaded {len(df)} real areas from DB for training classifier.")
        except Exception as e:
            print(f"[startup] Fallback to synthetic data for classifier training: {e}")
            df = gen_zone(800)
        clf = ZoneClassifier(n_estimators=200)
        clf.fit(df)
        clf.save(CLASSIFIER_PATH)
        print("[startup] Trained + saved classifier.")

    if not os.path.exists(PREDICTOR_PATH):
        try:
            df = load_price(db)
            print(f"[startup] Loaded {len(df)} real areas from DB for training predictor.")
        except Exception as e:
            print(f"[startup] Fallback to synthetic data for predictor training: {e}")
            df = gen_price(1000)
        pred = PricePredictor()
        pred.fit(df)
        pred.save(PREDICTOR_PATH)
        print("[startup] Trained + saved predictor.")


def _seed_if_empty(db: Session):
    try:
        from db.models import Area
        if db.query(Area).count() == 0:
            from scripts.seed_areas import seed_areas
            count = seed_areas(db)
            print(f"[startup] Seeded {count} areas into the database.")

        from scripts.seed_areas import seed_profiles
        p_count = seed_profiles(db)
        if p_count > 0:
            print(f"[startup] Seeded {p_count} investor profiles into the database.")
    except Exception as exc:
        # Gracefully skip seeding when running against SQLite (tests) or
        # when PostGIS tables haven't been created yet.
        print(f"[startup] Skipping database seed (DB not ready): {exc}")


def _run_initial_predictions(db: Session):
    """Generate initial PricePrediction rows for all areas if missing."""
    if _predictor is None or _classifier is None:
        return
    try:
        from db.models import Area, PricePrediction
        areas = db.query(Area).all()
        for area in areas:
            if db.query(PricePrediction).filter(PricePrediction.area_id == area.id).first():
                continue
            area_dict = _area_to_dict(area)
            zone = _classifier.predict_one(area_dict)
            area_dict["zone_label"] = zone
            preds = _predictor.predict_one(area_dict)
            db.add(PricePrediction(
                area_id=area.id,
                predicted_appreciation_1yr=preds["1yr"],
                predicted_appreciation_3yr=preds["3yr"],
                predicted_appreciation_5yr=preds["5yr"],
                confidence_lower=preds["lower"],
                confidence_upper=preds["upper"],
                model_version="xgb-v1",
            ))
        db.commit()
    except Exception as exc:
        # Skip initial predictions when DB schema is incomplete (e.g. SQLite tests).
        print(f"[startup] Skipping initial predictions (DB not ready): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables (PostGIS extensions are created in the DB container)
    init_db()

    # Seed first, then train models using database data!
    from db.session import SessionLocal
    db = SessionLocal()
    try:
        _seed_if_empty(db)
        _train_all_models(db)
        _load_models()
        _run_initial_predictions(db)
    except Exception as exc:
        print(f"[startup] Error in lifespan startup tasks: {exc}")
    finally:
        db.close()

    yield


app = FastAPI(
    title="LandIQ API",
    description=(
        "Smart Real Estate Investment Advisor for Telangana — "
        "buyer/seller intelligence powered by government development "
        "plan analysis and ML price prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Dependency helpers ────────────────────────────────────────────────────── #

def get_segmenter() -> InvestorSegmenter:
    if _segmenter is None:
        raise HTTPException(503, detail="Segmentation model not loaded.")
    return _segmenter

def get_classifier():
    if _classifier is None:
        raise HTTPException(503, detail="Zone classifier not loaded.")
    return _classifier

def get_predictor():
    if _predictor is None:
        raise HTTPException(503, detail="Price predictor not loaded.")
    return _predictor

def _area_to_dict(area: Area) -> dict:
    return {
        "current_price_sqft":        area.current_price_sqft or 0,
        "price_cagr_3yr":            area.price_cagr_3yr or 0,
        "population_growth_rate":    area.population_growth_rate or 0,
        "distance_to_city_center_km": area.distance_to_city_center_km or 0,
        "distance_to_metro_km":      area.distance_to_metro_km or 0,
        "distance_to_highway_km":    area.distance_to_highway_km or 0,
        "development_index":         area.development_index or 0,
        "num_upcoming_projects":     area.num_upcoming_projects or 0,
    }

def _latest_prediction(area: Area) -> Optional[PricePrediction]:
    if area.predictions:
        return sorted(area.predictions, key=lambda p: p.generated_at, reverse=True)[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "LandIQ API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "dashboard": "http://localhost:8501",
    }


# ── Investor Profiling ────────────────────────────────────────────────────── #

@app.post("/profile", response_model=InvestorProfileResponse, tags=["Investor Profiling"])
def create_profile(
    payload: InvestorProfileCreate,
    db: Session = Depends(get_db),
    segmenter: InvestorSegmenter = Depends(get_segmenter),
):
    """Submit investor profile → get segment label + strategy."""
    segment_label, strategy = segmenter.predict_one({
        "monthly_income":    payload.monthly_income,
        "max_budget":        payload.max_budget,
        "investment_horizon": payload.investment_horizon.value,
        "risk_appetite":     payload.risk_appetite.value,
        "target_roi_percent": payload.target_roi_percent or 50.0,
    })
    profile = InvestorProfile(
        name=payload.name,
        monthly_income=payload.monthly_income,
        max_budget=payload.max_budget,
        investment_horizon=InvestmentHorizon(payload.investment_horizon.value),
        risk_appetite=RiskAppetite(payload.risk_appetite.value),
        preferred_asset_type=AssetType(payload.preferred_asset_type.value),
        target_roi_percent=payload.target_roi_percent,
        segment_label=segment_label,
    )
    db.add(profile); db.commit(); db.refresh(profile)
    return InvestorProfileResponse(
        id=profile.id, name=profile.name,
        segment_label=segment_label, segment_strategy=strategy,
        monthly_income=profile.monthly_income, max_budget=profile.max_budget,
        investment_horizon=profile.investment_horizon.value,
        risk_appetite=profile.risk_appetite.value,
    )


@app.get("/profile/{profile_id}", response_model=InvestorProfileResponse, tags=["Investor Profiling"])
def get_profile(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(InvestorProfile).filter(InvestorProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(404, detail="Profile not found")
    strategy = next(
        (s["strategy"] for s in SEGMENT_PROFILES if s["label"] == profile.segment_label),
        "Strategy unavailable",
    )
    return InvestorProfileResponse(
        id=profile.id, name=profile.name,
        segment_label=profile.segment_label, segment_strategy=strategy,
        monthly_income=profile.monthly_income, max_budget=profile.max_budget,
        investment_horizon=profile.investment_horizon.value,
        risk_appetite=profile.risk_appetite.value,
    )


# ── Buyer Recommendation ──────────────────────────────────────────────────── #

@app.post("/recommend/buyer", response_model=BuyerRecommendationResponse, tags=["Recommendations"])
def recommend_buyer(
    payload: BuyerRecommendationRequest,
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
    classifier=Depends(get_classifier),
):
    """
    Given a saved investor profile, return the top N areas ranked by
    predicted ROI at the investor's target horizon.
    """
    profile = db.query(InvestorProfile).filter(InvestorProfile.id == payload.profile_id).first()
    if not profile:
        raise HTTPException(404, detail="Profile not found")

    areas = db.query(Area).all()
    if not areas:
        raise HTTPException(503, detail="No area data available. Seed the database first.")

    # Map horizon to prediction column
    horizon_map = {
        InvestmentHorizon.SHORT: "1yr",
        InvestmentHorizon.MID:   "3yr",
        InvestmentHorizon.LONG:  "5yr",
    }
    horizon = horizon_map.get(profile.investment_horizon, "3yr")

    # Budget filter: area price per sqft * 200 sqyd should fit budget
    # (rough 200 sqyd plot as baseline)
    budget_filter_price = profile.max_budget / (200 * 9)    # sqyd → sqft, ÷ 9

    scored = []
    for area in areas:
        if (area.current_price_sqft or 0) > budget_filter_price * 1.3:
            continue   # area too expensive even at moderate size
        area_dict = _area_to_dict(area)
        area_dict["zone_label"] = area.zone_label.value if area.zone_label else "stable"
        preds = predictor.predict_one(area_dict)
        roi = preds.get(horizon, preds["3yr"])
        scored.append((area, preds, roi))

    # Sort by predicted ROI descending
    scored.sort(key=lambda x: x[2], reverse=True)
    top_n = scored[:payload.top_n]

    scorecards = []
    for area, preds, roi in top_n:
        pred = _latest_prediction(area)
        zone = area.zone_label.value if area.zone_label else "stable"
        risk = _zone_to_risk(zone)
        scorecards.append(AreaScorecard(
            area_id=area.id,
            area_name=area.name,
            district=area.district or "",
            zone_label=zone,
            current_price_sqft=area.current_price_sqft or 0,
            development_index=area.development_index or 0,
            predicted_roi_1yr=preds["1yr"],
            predicted_roi_3yr=preds["3yr"],
            predicted_roi_5yr=preds["5yr"],
            confidence_lower=preds["lower"],
            confidence_upper=preds["upper"],
            risk_level=risk,
            key_driver=_key_driver(area),
            num_upcoming_projects=area.num_upcoming_projects or 0,
        ))

    return BuyerRecommendationResponse(
        profile_id=profile.id,
        segment_label=profile.segment_label or "",
        target_horizon=horizon,
        areas=scorecards,
    )


# ── Seller Recommendation ─────────────────────────────────────────────────── #

@app.post("/recommend/seller", response_model=SellerRecommendationResponse, tags=["Recommendations"])
def recommend_seller(
    payload: SellerRecommendationRequest,
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
):
    """
    Given a property location + size, estimate current value and provide
    a hold/sell timing recommendation.
    """
    area = db.query(Area).filter(
        Area.name.ilike(f"%{payload.area_name}%")
    ).first()
    if not area:
        raise HTTPException(404, detail=f"Area '{payload.area_name}' not found")

    sqft = payload.property_size_sqyd * 9      # 1 sqyd = 9 sqft
    price_sqft = area.current_price_sqft or 5000
    current_value = payload.current_estimated_value or (sqft * price_sqft)

    area_dict = _area_to_dict(area)
    area_dict["zone_label"] = area.zone_label.value if area.zone_label else "stable"
    preds = predictor.predict_one(area_dict)

    v1 = current_value * (1 + preds["1yr"] / 100)
    v3 = current_value * (1 + preds["3yr"] / 100)
    v5 = current_value * (1 + preds["5yr"] / 100)

    zone = area.zone_label.value if area.zone_label else "stable"
    recommendation, reasoning = _sell_or_hold(zone, preds, area)

    return SellerRecommendationResponse(
        area_name=area.name,
        zone_label=zone,
        current_estimated_value=round(current_value, 0),
        predicted_value_1yr=round(v1, 0),
        predicted_value_3yr=round(v3, 0),
        predicted_value_5yr=round(v5, 0),
        predicted_roi_3yr=preds["3yr"],
        recommendation=recommendation,
        reasoning=reasoning,
    )


# ── Price Prediction ──────────────────────────────────────────────────────── #

@app.post("/predict/price", response_model=PricePredictResponse, tags=["Predictions"])
def predict_price(
    payload: PricePredictRequest,
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
    classifier=Depends(get_classifier),
):
    """Predict price appreciation at 1yr/3yr/5yr for a given area name."""
    area = db.query(Area).filter(
        Area.name.ilike(f"%{payload.area_name}%")
    ).first()
    if not area:
        raise HTTPException(404, detail=f"Area '{payload.area_name}' not found")

    area_dict = _area_to_dict(area)
    zone = classifier.predict_one(area_dict)
    area_dict["zone_label"] = zone
    preds = predictor.predict_one(area_dict)

    try:
        from ml.explainer import explain_prediction
        explanation = explain_prediction(predictor, area_dict, horizon="3yr")
    except Exception:
        explanation = {"summary": "SHAP explanation unavailable.", "contributions": []}

    return PricePredictResponse(
        area_name=area.name,
        zone_label=zone,
        predicted_appreciation_1yr=preds["1yr"],
        predicted_appreciation_3yr=preds["3yr"],
        predicted_appreciation_5yr=preds["5yr"],
        confidence_lower=preds["lower"],
        confidence_upper=preds["upper"],
        explanation_summary=explanation.get("summary", ""),
        top_contributors=explanation.get("contributions", []),
    )


# ── Document Upload ───────────────────────────────────────────────────────── #

@app.post("/upload-plan", tags=["Document Pipeline"])
async def upload_plan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a government PDF. Starts an in-process background task that:
    extracts text chunks, runs spaCy NLP, geocodes locations,
    saves DevelopmentProject rows, and updates area development scores.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    database_url = os.environ.get("DATABASE_URL", "postgresql://landiq:landiq@localhost:5432/landiq")

    from pipeline.ingest import process_document
    background_tasks.add_task(process_document, file.filename, file_bytes, database_url)

    return {
        "status": "accepted",
        "filename": file.filename,
        "message": "Document is being processed in the background. "
                   "Area scores will update automatically when extraction completes.",
    }


# ── Areas ─────────────────────────────────────────────────────────────────── #

@app.get("/areas", tags=["Areas"])
def list_areas(
    zone: Optional[str] = Query(None, description="Filter by zone label"),
    min_dev_index: float = Query(0, description="Minimum development index"),
    db: Session = Depends(get_db),
):
    """List all tracked areas with zone, dev index, price, and predictions."""
    query = db.query(Area)
    if zone:
        try:
            query = query.filter(Area.zone_label == ZoneLabel(zone))
        except ValueError:
            raise HTTPException(400, detail=f"Invalid zone: {zone}")
    if min_dev_index > 0:
        query = query.filter(Area.development_index >= min_dev_index)

    areas = query.order_by(Area.development_index.desc()).all()
    result = []
    for area in areas:
        pred = _latest_prediction(area)
        result.append(AreaListItem(
            id=area.id,
            name=area.name,
            district=area.district or "",
            zone_label=area.zone_label.value if area.zone_label else "stable",
            current_price_sqft=area.current_price_sqft or 0,
            development_index=area.development_index or 0,
            predicted_roi_3yr=pred.predicted_appreciation_3yr if pred else None,
            num_upcoming_projects=area.num_upcoming_projects or 0,
        ))
    return {"count": len(result), "areas": result}


@app.get("/areas/{area_id}/report", response_model=AreaDetail, tags=["Areas"])
def area_report(
    area_id: int,
    search_query: Optional[str] = Query(None, description="Search document chunks"),
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
    classifier=Depends(get_classifier),
):
    """Full investment report for a specific area."""
    area = db.query(Area).filter(Area.id == area_id).first()
    if not area:
        raise HTTPException(404, detail="Area not found")

    area_dict = _area_to_dict(area)
    zone = classifier.predict_one(area_dict)
    area_dict["zone_label"] = zone
    preds = predictor.predict_one(area_dict)

    try:
        from ml.explainer import explain_prediction
        expl = explain_prediction(predictor, area_dict, horizon="3yr")
        expl_summary = expl.get("summary", "")
    except Exception:
        expl_summary = ""

    # Nearby projects
    nearby_projects = db.query(DevelopmentProject).all()[:10]   # simplified; full PostGIS query in spatial.py
    projects_info = [
        {
            "name": p.project_name,
            "type": p.project_type,
            "budget_crore": p.budget_crore,
            "completion_year": p.estimated_completion_year,
        }
        for p in nearby_projects
    ]

    # Document FTS
    doc_snippets = []
    query_text = search_query or area.name
    chunks = full_text_search(db, query_text, limit=5)
    doc_snippets = [
        {"source": c.source_document, "excerpt": c.content[:300]}
        for c in chunks
    ]

    return AreaDetail(
        id=area.id,
        name=area.name,
        district=area.district or "",
        zone_label=zone,
        current_price_sqft=area.current_price_sqft or 0,
        development_index=area.development_index or 0,
        price_cagr_3yr=area.price_cagr_3yr or 0,
        distance_to_metro_km=area.distance_to_metro_km or 0,
        distance_to_highway_km=area.distance_to_highway_km or 0,
        num_upcoming_projects=area.num_upcoming_projects or 0,
        predicted_roi_1yr=preds["1yr"],
        predicted_roi_3yr=preds["3yr"],
        predicted_roi_5yr=preds["5yr"],
        confidence_lower=preds["lower"],
        confidence_upper=preds["upper"],
        explanation_summary=expl_summary,
        risk_level=_zone_to_risk(zone),
        nearby_projects=projects_info,
        document_excerpts=doc_snippets,
    )


@app.get("/heatmap", tags=["Map"])
def heatmap(db: Session = Depends(get_db)):
    """GeoJSON FeatureCollection for Folium map rendering."""
    return heatmap_geojson(db)


# ── PDF Report Generator ──────────────────────────────────────────────────── #

@app.post("/report/generate", tags=["Reports"])
def generate_report(
    payload: BuyerRecommendationRequest,
    db: Session = Depends(get_db),
    predictor=Depends(get_predictor),
    classifier=Depends(get_classifier),
):
    """Generate a downloadable PDF investment report for top 3 areas."""
    profile = db.query(InvestorProfile).filter(InvestorProfile.id == payload.profile_id).first()
    if not profile:
        raise HTTPException(404, detail="Profile not found")

    areas = db.query(Area).order_by(Area.development_index.desc()).limit(3).all()

    lines = [
        f"LandIQ Investment Report",
        f"========================",
        f"Investor: {profile.name or 'Anonymous'}",
        f"Segment:  {profile.segment_label or 'N/A'}",
        f"Budget:   ₹{profile.max_budget:,.0f}",
        f"Horizon:  {profile.investment_horizon.value}",
        f"",
        f"Top 3 Recommended Areas",
        f"-----------------------",
    ]
    for i, area in enumerate(areas, 1):
        area_dict = _area_to_dict(area)
        area_dict["zone_label"] = area.zone_label.value if area.zone_label else "stable"
        preds = predictor.predict_one(area_dict)
        lines += [
            f"",
            f"{i}. {area.name} ({area.district})",
            f"   Zone:              {area.zone_label.value if area.zone_label else 'N/A'}",
            f"   Current price:     ₹{area.current_price_sqft:,.0f}/sqft",
            f"   Dev Index:         {area.development_index:.1f}/100",
            f"   Predicted ROI 1yr: +{preds['1yr']:.1f}%",
            f"   Predicted ROI 3yr: +{preds['3yr']:.1f}%",
            f"   Predicted ROI 5yr: +{preds['5yr']:.1f}%",
        ]

    content = "\n".join(lines).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=landiq_report.txt"},
    )


# ── Helper functions ──────────────────────────────────────────────────────── #

def _zone_to_risk(zone: str) -> str:
    return {
        "rapid_development": "Medium-High",
        "emerging":          "Medium",
        "stable":            "Low-Medium",
        "saturated":         "Low",
    }.get(zone, "Medium")


def _key_driver(area: Area) -> str:
    if (area.distance_to_metro_km or 99) < 3:
        return "Metro proximity"
    if (area.distance_to_highway_km or 99) < 2:
        return "Highway corridor"
    if (area.development_index or 0) > 70:
        return "High development density"
    if (area.price_cagr_3yr or 0) > 15:
        return "Strong historical CAGR"
    return "Infrastructure growth corridor"


def _sell_or_hold(zone: str, preds: dict, area: Area) -> tuple[str, str]:
    roi_3yr = preds.get("3yr", 0)
    zone_label = zone or "stable"

    if zone_label == "rapid_development" and roi_3yr >= 30:
        return "HOLD — Strong upside", (
            f"Area is in Rapid Development zone with {roi_3yr:.0f}% predicted 3yr appreciation. "
            f"Development index: {area.development_index:.0f}/100. "
            f"Selling now would mean missing significant future gains."
        )
    elif zone_label == "saturated" or roi_3yr < 10:
        return "SELL — Limited upside", (
            f"Area shows only {roi_3yr:.0f}% predicted 3yr appreciation in a {zone_label.replace('_',' ')} zone. "
            f"Capital can be redeployed to higher-growth localities."
        )
    elif zone_label == "emerging" and roi_3yr >= 20:
        return "HOLD — Emerging corridor", (
            f"Emerging zone with {roi_3yr:.0f}% predicted 3yr ROI. "
            f"Hold for at least 3–5 years to capture infrastructure appreciation."
        )
    else:
        return "WAIT — Monitor for 6 months", (
            f"Mixed signals: {roi_3yr:.0f}% predicted 3yr appreciation in {zone_label.replace('_', ' ')} zone. "
            f"Watch for upcoming infrastructure announcements before deciding."
        )


# Expose init_db for test patching
def init_db_patched():
    init_db()
