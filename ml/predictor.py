"""
Price Appreciation Predictor — XGBoost Regressor.

Predicts % price appreciation at 1yr, 3yr, and 5yr horizons for each area.

Input features:
    zone_label (encoded), development_index, distance_to_metro_km,
    distance_to_highway_km, current_price_sqft, price_cagr_3yr,
    population_growth_rate, num_upcoming_projects

Usage:
    pred = PricePredictor()
    pred.fit(df)
    pred.save("models/predictor.joblib")

    pred = PricePredictor.load("models/predictor.joblib")
    result = pred.predict_one(area_dict)
    # result: {"1yr": 12.5, "3yr": 32.1, "5yr": 68.4, "lower": 55.0, "upper": 82.0}
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

ZONE_ORDER = ["saturated", "stable", "emerging", "rapid_development"]

FEATURE_COLUMNS = [
    "zone_code",
    "development_index",
    "distance_to_metro_km",
    "distance_to_highway_km",
    "current_price_sqft",
    "price_cagr_3yr",
    "population_growth_rate",
    "num_upcoming_projects",
]


def _encode_zone(zone_label: str) -> int:
    mapping = {z: i for i, z in enumerate(ZONE_ORDER)}
    return mapping.get(str(zone_label).lower(), 1)


class PricePredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._model_1yr: XGBRegressor | None = None
        self._model_3yr: XGBRegressor | None = None
        self._model_5yr: XGBRegressor | None = None

    def _make_model(self):
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["zone_code"] = out["zone_label"].apply(_encode_zone)
        return out[FEATURE_COLUMNS].fillna(0)

    def fit(self, df: pd.DataFrame):
        """
        df must have FEATURE_COLUMNS + zone_label + target columns:
            appreciation_1yr, appreciation_3yr, appreciation_5yr
        """
        X = self._prepare(df)
        self._model_1yr = self._make_model()
        self._model_3yr = self._make_model()
        self._model_5yr = self._make_model()
        self._model_1yr.fit(X, df["appreciation_1yr"])
        self._model_3yr.fit(X, df["appreciation_3yr"])
        self._model_5yr.fit(X, df["appreciation_5yr"])
        print(f"[PricePredictor] trained 3 XGBoost regressors on {len(df)} samples.")
        return self

    def predict_one(self, area_dict: dict) -> dict:
        """
        Predict appreciation percentages for a single area.
        Returns:
            {
              "1yr": float,  "3yr": float,  "5yr": float,
              "lower": float, "upper": float  (5yr confidence interval)
            }
        """
        df = pd.DataFrame([area_dict])
        if "zone_label" not in df.columns:
            df["zone_label"] = "stable"
        X = self._prepare(df)

        p1 = float(self._model_1yr.predict(X)[0])
        p3 = float(self._model_3yr.predict(X)[0])
        p5 = float(self._model_5yr.predict(X)[0])

        # Simple bootstrap CI approximation using ±15% of prediction
        margin = abs(p5) * 0.15
        return {
            "1yr": round(max(0, p1), 2),
            "3yr": round(max(0, p3), 2),
            "5yr": round(max(0, p5), 2),
            "lower": round(max(0, p5 - margin), 2),
            "upper": round(p5 + margin, 2),
        }

    def predict_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict appreciation for a dataframe of areas."""
        X = self._prepare(df)
        df = df.copy()
        df["predicted_appreciation_1yr"] = self._model_1yr.predict(X)
        df["predicted_appreciation_3yr"] = self._model_3yr.predict(X)
        df["predicted_appreciation_5yr"] = self._model_5yr.predict(X)
        return df

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PricePredictor":
        return joblib.load(path)


def generate_synthetic_training_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data with realistic price-appreciation
    relationships for Hyderabad-style markets.
    """
    rng = np.random.default_rng(seed)
    records = []
    zone_configs = {
        "rapid_development":  (12, 8,  18, 8,  35, 20, 75, 15),
        "emerging":           (8,  5,  12, 6,  25, 15, 55, 10),
        "stable":             (5,  3,  8,  4,  18, 10, 40, 7),
        "saturated":          (2,  2,  4,  3,  10, 6,  22, 5),
    }
    per_zone = n // 4
    for zone, (m1, s1, m3, s3, m5, s5, midx, sidx) in zone_configs.items():
        for _ in range(per_zone):
            dev_idx = float(np.clip(rng.normal(midx, sidx), 0, 100))
            cagr = float(np.clip(rng.normal(m3 / 3, s3 / 3), 0, 30))
            a1 = float(np.clip(rng.normal(m1, s1), 0, 40))
            a3 = float(np.clip(rng.normal(m3, s3), 0, 80))
            a5 = float(np.clip(rng.normal(m5, s5), 0, 150))
            records.append({
                "zone_label":                zone,
                "development_index":         round(dev_idx, 1),
                "distance_to_metro_km":      round(rng.uniform(0.5, 30.0), 1),
                "distance_to_highway_km":    round(rng.uniform(0.3, 10.0), 1),
                "current_price_sqft":        round(rng.uniform(1500, 18000), 0),
                "price_cagr_3yr":            round(cagr, 2),
                "population_growth_rate":    round(rng.uniform(1.0, 9.0), 2),
                "num_upcoming_projects":     int(rng.integers(0, 15)),
                "appreciation_1yr":          round(a1, 2),
                "appreciation_3yr":          round(a3, 2),
                "appreciation_5yr":          round(a5, 2),
            })
    return pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_training_data_from_db(db) -> pd.DataFrame:
    """
    Query all Area records from the database and calculate compounding appreciation targets.
    Formula:
        appreciation_1yr = cagr
        appreciation_3yr = ((1 + cagr / 100) ** 3 - 1) * 100
        appreciation_5yr = ((1 + cagr / 100) ** 5 - 1) * 100
    Raises ValueError if there are no areas.
    """
    from db.models import Area
    areas = db.query(Area).all()
    if not areas:
        raise ValueError("No areas in the database. Please seed the database first.")

    records = []
    for a in areas:
        cagr = float(a.price_cagr_3yr or 0.0)
        a1 = cagr
        a3 = ((1.0 + cagr / 100.0) ** 3.0 - 1.0) * 100.0
        a5 = ((1.0 + cagr / 100.0) ** 5.0 - 1.0) * 100.0

        records.append({
            "zone_label":                a.zone_label.value if a.zone_label else "stable",
            "development_index":         float(a.development_index or 0.0),
            "distance_to_metro_km":      float(a.distance_to_metro_km or 0.0),
            "distance_to_highway_km":    float(a.distance_to_highway_km or 0.0),
            "current_price_sqft":        float(a.current_price_sqft or 0.0),
            "price_cagr_3yr":            cagr,
            "population_growth_rate":    float(a.population_growth_rate or 0.0),
            "num_upcoming_projects":     int(a.num_upcoming_projects or 0),
            "appreciation_1yr":          round(a1, 2),
            "appreciation_3yr":          round(a3, 2),
            "appreciation_5yr":          round(a5, 2),
        })
    return pd.DataFrame(records)
