"""
Standalone training entrypoint for the investor segmentation model.

Run with:
    python -m scripts.train_segmenter

This is kept separate from ml/segmentation.py so that the InvestorSegmenter
class is always pickled with module path "ml.segmentation" (not "__main__"),
which is required for joblib.load() to work correctly when the model is
loaded later inside the FastAPI app.
"""
import os
from db.session import SessionLocal
from scripts.seed_areas import seed_profiles
from ml.segmentation import InvestorSegmenter, load_training_data_from_db, generate_synthetic_training_data

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    db = None
    try:
        db = SessionLocal()
        # Seed if empty
        seed_profiles(db)
    except Exception as e:
        print(f"Warning: Database connection or seeding failed ({e}). Proceeding with synthetic fallback.")
        db = None

    try:
        df = None
        if db is not None:
            try:
                df = load_training_data_from_db(db)
                print(f"Loaded {len(df)} real investor profiles from database.")
            except Exception as e:
                print(f"Warning: Could not load profiles from DB ({e}). Falling back to synthetic.")

        if df is None:
            print("Generating synthetic training data.")
            df = generate_synthetic_training_data(n=600)

        segmenter = InvestorSegmenter(n_clusters=4)
        segmenter.fit(df)
        segmenter.save("models/segmenter.joblib")

        test_profile = {
            "monthly_income": 65000,
            "max_budget": 2800000,
            "investment_horizon": "mid",
            "risk_appetite": "moderate",
            "target_roi_percent": 80,
        }
        label, strategy = segmenter.predict_one(test_profile)
        print(f"\nTest profile -> segment: {label}")
        print(f"Strategy: {strategy}")
    finally:
        if db is not None:
            db.close()
