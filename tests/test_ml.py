"""
Tests for ML models: ZoneClassifier, PricePredictor, InvestorSegmenter.

Run with:
    pytest tests/test_ml.py -v
"""
import pytest
import pandas as pd
import os

from ml.segmentation import InvestorSegmenter, generate_synthetic_training_data as gen_investors
from ml.classifier   import ZoneClassifier,    generate_synthetic_training_data as gen_zones
from ml.predictor    import PricePredictor,    generate_synthetic_training_data as gen_prices


@pytest.fixture(scope="module")
def trained_segmenter():
    df = gen_investors(n=400, seed=0)
    s = InvestorSegmenter(n_clusters=4)
    s.fit(df)
    return s


@pytest.fixture(scope="module")
def trained_classifier():
    df = gen_zones(n=400, seed=0)
    c = ZoneClassifier(n_estimators=50)
    c.fit(df)
    return c


@pytest.fixture(scope="module")
def trained_predictor():
    df = gen_prices(n=400, seed=0)
    p = PricePredictor()
    p.fit(df)
    return p


# ── Segmenter tests ──────────────────────────────────────────────────────── #

class TestInvestorSegmenter:
    VALID_PROFILE = {
        "monthly_income": 65000, "max_budget": 2800000,
        "investment_horizon": "mid", "risk_appetite": "moderate",
        "target_roi_percent": 80,
    }
    HNI_PROFILE = {
        "monthly_income": 500000, "max_budget": 25000000,
        "investment_horizon": "short", "risk_appetite": "aggressive",
        "target_roi_percent": 200,
    }

    def test_predict_returns_valid_label(self, trained_segmenter):
        label, strategy = trained_segmenter.predict_one(self.VALID_PROFILE)
        assert label in {"budget_buyer", "mid_range_investor", "growth_investor", "hni_aggressive"}
        assert len(strategy) > 20

    def test_hni_segments_as_hni(self, trained_segmenter):
        label, _ = trained_segmenter.predict_one(self.HNI_PROFILE)
        assert label == "hni_aggressive"

    def test_budget_and_hni_differ(self, trained_segmenter):
        budget_label, _ = trained_segmenter.predict_one(
            {**self.VALID_PROFILE, "monthly_income": 25000, "max_budget": 1200000}
        )
        hni_label, _ = trained_segmenter.predict_one(self.HNI_PROFILE)
        assert budget_label != hni_label

    def test_save_and_load(self, trained_segmenter, tmp_path):
        path = str(tmp_path / "seg.joblib")
        trained_segmenter.save(path)
        loaded = InvestorSegmenter.load(path)
        label1, _ = trained_segmenter.predict_one(self.VALID_PROFILE)
        label2, _ = loaded.predict_one(self.VALID_PROFILE)
        assert label1 == label2


# ── Classifier tests ─────────────────────────────────────────────────────── #

class TestZoneClassifier:
    RAPID_AREA = {
        "current_price_sqft": 7800, "price_cagr_3yr": 18.0,
        "population_growth_rate": 6.0, "distance_to_city_center_km": 16.0,
        "distance_to_metro_km": 3.0, "distance_to_highway_km": 1.0,
        "development_index": 90.0, "num_upcoming_projects": 8,
    }
    SATURATED_AREA = {
        "current_price_sqft": 18000, "price_cagr_3yr": 3.0,
        "population_growth_rate": 1.0, "distance_to_city_center_km": 4.0,
        "distance_to_metro_km": 1.0, "distance_to_highway_km": 4.0,
        "development_index": 10.0, "num_upcoming_projects": 0,
    }

    def test_predict_valid_zone(self, trained_classifier):
        zone = trained_classifier.predict_one(self.RAPID_AREA)
        assert zone in {"rapid_development", "emerging", "stable", "saturated"}

    def test_rapid_area_classified_correctly(self, trained_classifier):
        zone = trained_classifier.predict_one(self.RAPID_AREA)
        assert zone in {"rapid_development", "emerging"}   # either is acceptable

    def test_saturated_area_classified_correctly(self, trained_classifier):
        zone = trained_classifier.predict_one(self.SATURATED_AREA)
        assert zone in {"saturated", "stable"}

    def test_predict_proba_sums_to_1(self, trained_classifier):
        proba = trained_classifier.predict_proba_one(self.RAPID_AREA)
        assert abs(sum(proba.values()) - 1.0) < 0.01

    def test_feature_importances(self, trained_classifier):
        fi = trained_classifier.feature_importances()
        assert len(fi) == 8
        assert all(v >= 0 for v in fi.values())


# ── Predictor tests ──────────────────────────────────────────────────────── #

class TestPricePredictor:
    TEST_AREA = {
        "zone_label": "rapid_development", "development_index": 85.0,
        "distance_to_metro_km": 3.0, "distance_to_highway_km": 1.5,
        "current_price_sqft": 6500, "price_cagr_3yr": 16.0,
        "population_growth_rate": 6.0, "num_upcoming_projects": 7,
    }

    def test_predict_returns_all_horizons(self, trained_predictor):
        result = trained_predictor.predict_one(self.TEST_AREA)
        for key in ("1yr", "3yr", "5yr", "lower", "upper"):
            assert key in result
            assert isinstance(result[key], float)

    def test_predictions_are_non_negative(self, trained_predictor):
        result = trained_predictor.predict_one(self.TEST_AREA)
        assert result["1yr"] >= 0
        assert result["3yr"] >= 0
        assert result["5yr"] >= 0

    def test_5yr_exceeds_1yr(self, trained_predictor):
        result = trained_predictor.predict_one(self.TEST_AREA)
        assert result["5yr"] >= result["1yr"]

    def test_confidence_interval_valid(self, trained_predictor):
        result = trained_predictor.predict_one(self.TEST_AREA)
        assert result["lower"] <= result["5yr"] <= result["upper"]

    def test_save_and_load(self, trained_predictor, tmp_path):
        path = str(tmp_path / "pred.joblib")
        trained_predictor.save(path)
        loaded = PricePredictor.load(path)
        r1 = trained_predictor.predict_one(self.TEST_AREA)
        r2 = loaded.predict_one(self.TEST_AREA)
        assert abs(r1["3yr"] - r2["3yr"]) < 0.001
