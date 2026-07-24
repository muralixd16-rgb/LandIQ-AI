"""
Investor Segmentation Engine — K-Means clustering.

Groups investors into 4 segments based on income, budget, horizon,
risk appetite, and target ROI. Each segment maps to a recommendation
strategy used downstream by the matching engine.

Usage:
    from ml.segmentation import InvestorSegmenter
    segmenter = InvestorSegmenter()
    segmenter.fit(training_dataframe)
    segmenter.save("models/segmenter.joblib")

    # later, at inference time
    segmenter = InvestorSegmenter.load("models/segmenter.joblib")
    label, strategy = segmenter.predict_one(profile_dict)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Encoding maps for categorical fields
HORIZON_MAP = {"short": 1, "mid": 2, "long": 3}
RISK_MAP = {"conservative": 1, "moderate": 2, "aggressive": 3}

# Human-readable segment names + strategy, keyed by centroid budget rank (0=lowest ... 3=highest)
SEGMENT_PROFILES = [
    {
        "label": "budget_buyer",
        "strategy": "Long-term hold in emerging outskirts. Target plots/farm land. "
                    "2x-3x return over 7-10 years.",
    },
    {
        "label": "mid_range_investor",
        "strategy": "Mid-term hold in semi-urban growth corridors. Target plots or "
                    "2BHK apartments. 1.5x-2x return over 4-6 years.",
    },
    {
        "label": "growth_investor",
        "strategy": "Short-to-mid term in high-growth suburban areas. Target villa "
                    "plots or row houses. 1.8x-2.5x return over 3-5 years.",
    },
    {
        "label": "hni_aggressive",
        "strategy": "Rapid-growth premium corridors. Target commercial or gated "
                    "villa communities. 2x-4x return over 2-4 years.",
    },
]

FEATURE_COLUMNS = [
    "monthly_income", "max_budget", "investment_horizon",
    "risk_appetite", "target_roi_percent",
]


class InvestorSegmenter:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self._cluster_rank = None  # maps raw cluster id -> budget-sorted rank (0..3)

    @staticmethod
    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical investor fields into numeric features."""
        out = df.copy()
        out["investment_horizon"] = out["investment_horizon"].map(HORIZON_MAP)
        out["risk_appetite"] = out["risk_appetite"].map(RISK_MAP)
        out["target_roi_percent"] = out["target_roi_percent"].fillna(out["target_roi_percent"].median())
        return out[FEATURE_COLUMNS]

    def fit(self, df: pd.DataFrame):
        """
        Train the clustering model.
        df must contain columns: monthly_income, max_budget, investment_horizon
        (short/mid/long), risk_appetite (conservative/moderate/aggressive),
        target_roi_percent.
        """
        X = self._encode(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Rank clusters by average max_budget so segment labels are meaningful
        # (cluster with lowest avg budget -> "budget_buyer", highest -> "hni_aggressive")
        df = df.copy()
        df["_cluster"] = self.model.labels_
        budget_by_cluster = df.groupby("_cluster")["max_budget"].mean().sort_values()
        self._cluster_rank = {cid: rank for rank, cid in enumerate(budget_by_cluster.index)}

        score = silhouette_score(X_scaled, self.model.labels_)
        print(f"[InvestorSegmenter] trained on {len(df)} profiles. "
              f"Silhouette score: {score:.3f}")
        return self

    def predict_one(self, profile: dict) -> tuple[str, str]:
        """
        Predict segment for a single investor profile dict.
        Returns (segment_label, strategy_text).
        """
        df = pd.DataFrame([profile])
        X = self._encode(df)
        X_scaled = self.scaler.transform(X)
        raw_cluster = int(self.model.predict(X_scaled)[0])
        rank = self._cluster_rank[raw_cluster]
        profile_info = SEGMENT_PROFILES[rank]
        return profile_info["label"], profile_info["strategy"]

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "InvestorSegmenter":
        return joblib.load(path)


def generate_synthetic_training_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic but realistic investor profiles for initial model
    training before real user data is collected. Replace with real data
    once you have enough signups.
    """
    rng = np.random.default_rng(seed)
    incomes = rng.lognormal(mean=11.0, sigma=0.6, size=n)          # ~INR 30k-500k/mo
    budgets = incomes * rng.uniform(30, 60, size=n)                 # budget scales with income
    horizons = rng.choice(["short", "mid", "long"], size=n, p=[0.3, 0.45, 0.25])
    risks = rng.choice(["conservative", "moderate", "aggressive"], size=n, p=[0.35, 0.4, 0.25])
    target_roi = rng.uniform(20, 300, size=n)

    return pd.DataFrame({
        "monthly_income": incomes.round(-2),
        "max_budget": budgets.round(-3),
        "investment_horizon": horizons,
        "risk_appetite": risks,
        "target_roi_percent": target_roi.round(1),
    })


def load_training_data_from_db(db) -> pd.DataFrame:
    """
    Query all InvestorProfile records from the database and format as a DataFrame.
    Raises ValueError if there are fewer than 4 profiles.
    """
    from db.models import InvestorProfile
    profiles = db.query(InvestorProfile).all()
    if len(profiles) < 4:
        raise ValueError(f"Not enough investor profiles in the database (have {len(profiles)}, need >= 4 for clustering).")

    records = []
    for p in profiles:
        records.append({
            "monthly_income": float(p.monthly_income),
            "max_budget": float(p.max_budget),
            "investment_horizon": p.investment_horizon.value if p.investment_horizon else "mid",
            "risk_appetite": p.risk_appetite.value if p.risk_appetite else "moderate",
            "target_roi_percent": float(p.target_roi_percent or 50.0),
        })
    return pd.DataFrame(records)


# NOTE: training is intentionally done in scripts/train_segmenter.py, not here.
# Keeping this module free of a __main__ block ensures InvestorSegmenter always
# pickles with module path "ml.segmentation" (required for joblib.load() to
# resolve correctly when the FastAPI app loads the saved model).
