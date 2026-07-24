"""
Area Zone Classifier — Random Forest.

Classifies each locality into one of four development zones:
    rapid_development | emerging | stable | saturated

Input features:
    current_price_sqft, price_cagr_3yr, population_growth_rate,
    distance_to_city_center_km, distance_to_metro_km, distance_to_highway_km,
    development_index, num_upcoming_projects

Usage:
    clf = ZoneClassifier()
    clf.fit(df)
    clf.save("models/classifier.joblib")

    clf = ZoneClassifier.load("models/classifier.joblib")
    label = clf.predict_one(area_dict)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

ZONE_LABELS = ["saturated", "stable", "emerging", "rapid_development"]

FEATURE_COLUMNS = [
    "current_price_sqft",
    "price_cagr_3yr",
    "population_growth_rate",
    "distance_to_city_center_km",
    "distance_to_metro_km",
    "distance_to_highway_km",
    "development_index",
    "num_upcoming_projects",
]


class ZoneClassifier:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            max_depth=8,
        )
        self.label_encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame):
        """
        Train the zone classifier.
        df must have FEATURE_COLUMNS + 'zone_label' column.
        """
        X = df[FEATURE_COLUMNS].fillna(0)
        y = self.label_encoder.fit_transform(df["zone_label"])
        self.model.fit(X, y)
        scores = cross_val_score(self.model, X, y, cv=min(5, len(df) // 4), scoring="accuracy")
        print(f"[ZoneClassifier] trained on {len(df)} samples. "
              f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        return self

    def predict_one(self, area_dict: dict) -> str:
        """Predict zone label for a single area feature dict."""
        df = pd.DataFrame([area_dict]).reindex(columns=FEATURE_COLUMNS).fillna(0)
        raw = self.model.predict(df)[0]
        return self.label_encoder.inverse_transform([raw])[0]

    def predict_proba_one(self, area_dict: dict) -> dict[str, float]:
        """Return class probabilities as {zone_label: probability}."""
        df = pd.DataFrame([area_dict]).reindex(columns=FEATURE_COLUMNS).fillna(0)
        proba = self.model.predict_proba(df)[0]
        classes = self.label_encoder.inverse_transform(self.model.classes_)
        return dict(zip(classes, proba.tolist()))

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist()))

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ZoneClassifier":
        return joblib.load(path)


def generate_synthetic_training_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic area profiles with realistic distributions
    matching Hyderabad's market patterns.
    """
    rng = np.random.default_rng(seed)
    records = []
    segment_configs = [
        # (zone, price_range, cagr_range, dev_idx_range, metro_range)
        ("rapid_development",  (6000, 15000),  (12, 22), (70, 100), (0.5, 5.0)),
        ("emerging",           (2500, 7000),   (8, 18),  (40, 74),  (4.0, 12.0)),
        ("stable",             (4000, 9000),   (4, 10),  (20, 39),  (2.0, 8.0)),
        ("saturated",          (8000, 20000),  (2, 6),   (0, 19),   (0.5, 4.0)),
    ]
    per_segment = n // 4
    for zone, (p_lo, p_hi), (c_lo, c_hi), (d_lo, d_hi), (m_lo, m_hi) in segment_configs:
        for _ in range(per_segment):
            price = rng.uniform(p_lo, p_hi)
            records.append({
                "current_price_sqft":        round(price, 0),
                "price_cagr_3yr":            round(rng.uniform(c_lo, c_hi), 2),
                "population_growth_rate":    round(rng.uniform(1.0, 9.0), 2),
                "distance_to_city_center_km": round(rng.uniform(3.0, 70.0), 1),
                "distance_to_metro_km":      round(rng.uniform(m_lo, m_hi), 1),
                "distance_to_highway_km":    round(rng.uniform(0.3, 8.0), 1),
                "development_index":         round(rng.uniform(d_lo, d_hi), 1),
                "num_upcoming_projects":     int(rng.integers(0, 12)),
                "zone_label":                zone,
            })
    return pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_training_data_from_db(db) -> pd.DataFrame:
    """
    Query all Area records from the database and format as a DataFrame for ZoneClassifier.
    Raises ValueError if there are no areas.
    """
    from db.models import Area
    areas = db.query(Area).all()
    if not areas:
        raise ValueError("No areas in the database. Please seed the database first.")

    records = []
    for a in areas:
        records.append({
            "current_price_sqft":        float(a.current_price_sqft or 0.0),
            "price_cagr_3yr":            float(a.price_cagr_3yr or 0.0),
            "population_growth_rate":    float(a.population_growth_rate or 0.0),
            "distance_to_city_center_km": float(a.distance_to_city_center_km or 0.0),
            "distance_to_metro_km":      float(a.distance_to_metro_km or 0.0),
            "distance_to_highway_km":    float(a.distance_to_highway_km or 0.0),
            "development_index":         float(a.development_index or 0.0),
            "num_upcoming_projects":     int(a.num_upcoming_projects or 0),
            "zone_label":                a.zone_label.value if a.zone_label else "stable",
        })
    return pd.DataFrame(records)
