"""
SHAP Explainability helper for LandIQ.

Generates human-readable feature importance explanations for XGBoost predictions.

Usage:
    from ml.explainer import explain_prediction
    explanation = explain_prediction(predictor, area_dict, horizon="3yr")
    # Returns: {"top_feature": "development_index", "top_value": 0.42, "summary": "..."}
"""
# pyrefly: ignore [missing-import]
import shap
import pandas as pd
from ml.predictor import PricePredictor, FEATURE_COLUMNS

FEATURE_LABELS = {
    "zone_code":                    "Development zone classification",
    "development_index":            "Area development index score",
    "distance_to_metro_km":         "Distance to nearest metro station",
    "distance_to_highway_km":       "Distance to nearest highway",
    "current_price_sqft":           "Current price per sq.ft.",
    "price_cagr_3yr":               "3-year historical price growth (CAGR)",
    "population_growth_rate":       "Population growth rate",
    "num_upcoming_projects":        "Number of upcoming government projects",
}


def explain_prediction(
    predictor: PricePredictor,
    area_dict: dict,
    horizon: str = "3yr",
) -> dict:
    """
    Run SHAP TreeExplainer on the chosen horizon model and return:
    {
        "top_feature": str,      # feature name (human-readable)
        "top_value": float,      # SHAP value magnitude
        "contributions": list,   # [{feature, label, shap_value}] sorted by |shap|
        "summary": str           # one-line explanation
    }
    """
    model_map = {
        "1yr": predictor._model_1yr,
        "3yr": predictor._model_3yr,
        "5yr": predictor._model_5yr,
    }
    model = model_map.get(horizon, predictor._model_3yr)
    if model is None:
        return {"top_feature": "N/A", "top_value": 0.0, "contributions": [], "summary": ""}

    # Build feature row
    df = pd.DataFrame([area_dict])
    if "zone_label" not in df.columns:
        df["zone_label"] = "stable"
    from ml.predictor import _encode_zone
    df["zone_code"] = df["zone_label"].apply(_encode_zone)
    X = df[FEATURE_COLUMNS].fillna(0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (1, n_features)

    contributions = []
    for feat, val in zip(FEATURE_COLUMNS, shap_values[0]):
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "shap_value": round(float(val), 3),
        })

    # Sort by absolute SHAP value descending
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top = contributions[0]

    direction = "boosting" if top["shap_value"] > 0 else "limiting"
    summary = (
        f"{top['label']} is the key {direction} factor, "
        f"contributing {abs(top['shap_value']):.1f}% to the predicted appreciation."
    )

    return {
        "top_feature": top["feature"],
        "top_label": top["label"],
        "top_value": top["shap_value"],
        "contributions": contributions[:5],     # top 5 only
        "summary": summary,
    }
