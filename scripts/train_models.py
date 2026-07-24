"""
Train all three LandIQ ML models from scratch using real data from the database.
If database is empty, seeds it first. Fallback to synthetic data if DB is unavailable.

Run with:
    python -m scripts.train_models

Saves to models/:
    segmenter.joblib   — K-Means investor segmentation
    classifier.joblib  — Random Forest zone classifier
    predictor.joblib   — XGBoost price predictor
"""
import os
from db.session import SessionLocal
from scripts.seed_areas import seed_areas, seed_profiles
from ml.segmentation import InvestorSegmenter, load_training_data_from_db as load_investors, generate_synthetic_training_data as gen_investor
from ml.classifier import ZoneClassifier, load_training_data_from_db as load_zones, generate_synthetic_training_data as gen_zones
from ml.predictor import PricePredictor, load_training_data_from_db as load_prices, generate_synthetic_training_data as gen_prices

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    db = None

    try:
        db = SessionLocal()
        # Seed if empty so training data exists
        print("\n=== Checking database seed status ===")
        a_seeded = seed_areas(db)
        p_seeded = seed_profiles(db)
        if a_seeded > 0 or p_seeded > 0:
            print(f"  Database seeded: {a_seeded} areas, {p_seeded} investor profiles.")
        else:
            print("  Database already seeded.")
    except Exception as e:
        print(f"  Warning: Database connection or seeding failed ({e}). Proceeding with synthetic fallbacks.")
        db = None

    try:
        # ── 1. Investor Segmentation (K-Means) ──────────────────────────────── #
        print("\n=== Training investor segmenter (K-Means) ===")
        df_investors = None
        if db is not None:
            try:
                df_investors = load_investors(db)
                print(f"  Loaded {len(df_investors)} real investor profiles from database.")
            except Exception as e:
                print(f"  Warning: Could not load real profiles from DB ({e}). Falling back to synthetic.")
        
        if df_investors is None:
            print("  Generating synthetic training data.")
            df_investors = gen_investor(n=800)

        segmenter = InvestorSegmenter(n_clusters=4)
        segmenter.fit(df_investors)
        segmenter.save("models/segmenter.joblib")
        test_profile = {
            "monthly_income": 65000, "max_budget": 2800000,
            "investment_horizon": "mid", "risk_appetite": "moderate",
            "target_roi_percent": 80,
        }
        lbl, strat = segmenter.predict_one(test_profile)
        print(f"  Test -> segment: {lbl}")

        # ── 2. Zone Classifier (Random Forest) ──────────────────────────────── #
        print("\n=== Training zone classifier (Random Forest) ===")
        df_zones = None
        if db is not None:
            try:
                df_zones = load_zones(db)
                print(f"  Loaded {len(df_zones)} real areas from database.")
            except Exception as e:
                print(f"  Warning: Could not load real zones from DB ({e}). Falling back to synthetic.")
        
        if df_zones is None:
            print("  Generating synthetic training data.")
            df_zones = gen_zones(n=800)

        classifier = ZoneClassifier(n_estimators=200)
        classifier.fit(df_zones)
        classifier.save("models/classifier.joblib")
        test_area = {
            "current_price_sqft": 5400, "price_cagr_3yr": 16.3,
            "population_growth_rate": 6.1, "distance_to_city_center_km": 22.0,
            "distance_to_metro_km": 7.0, "distance_to_highway_km": 1.0,
            "development_index": 75.0, "num_upcoming_projects": 5,
        }
        zone = classifier.predict_one(test_area)
        print(f"  Test area -> zone: {zone}")
        print(f"  Feature importances: {classifier.feature_importances()}")

        # ── 3. Price Predictor (XGBoost) ────────────────────────────────────── #
        print("\n=== Training price predictor (XGBoost) ===")
        df_prices = None
        if db is not None:
            try:
                df_prices = load_prices(db)
                print(f"  Loaded {len(df_prices)} real areas from database.")
            except Exception as e:
                print(f"  Warning: Could not load real prices from DB ({e}). Falling back to synthetic.")
        
        if df_prices is None:
            print("  Generating synthetic training data.")
            df_prices = gen_prices(n=1000)

        predictor = PricePredictor()
        predictor.fit(df_prices)
        predictor.save("models/predictor.joblib")
        test_area_pred = {**test_area, "zone_label": zone}
        result = predictor.predict_one(test_area_pred)
        print(f"  Test area -> prediction: {result}")

        print("\nAll models saved to models/")

    finally:
        if db is not None:
            db.close()
