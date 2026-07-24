"""
Tests for the investor profiling API + segmentation model.

Run with:
    pytest tests/ -v

Uses an isolated in-memory SQLite DB so it never touches real data.
Requires models/segmenter.joblib to exist — run `python -m scripts.train_segmenter` first.
"""
import os
import pytest
from fastapi.testclient import TestClient

TEST_DB_PATH = "./test_landiq.db"
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import api.main as main_module
from db.session import engine
from db.models import InvestorProfile

# Skip the geometry-table creation (no PostGIS in test env) — only the
# non-geometry InvestorProfile table is needed for these tests.
main_module.init_db = lambda: InvestorProfile.__table__.create(bind=engine, checkfirst=True)

from api.main import app

# create the table once, immediately, before any test runs
InvestorProfile.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def client():
    # clear table before each test so tests don't interfere with each other
    with engine.begin() as conn:
        conn.execute(InvestorProfile.__table__.delete())
    with TestClient(app) as c:
        yield c


VALID_PROFILE = {
    "name": "Test User",
    "monthly_income": 65000,
    "max_budget": 2800000,
    "investment_horizon": "mid",
    "risk_appetite": "moderate",
    "preferred_asset_type": "plot",
    "target_roi_percent": 80,
}


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "LandIQ API"


def test_create_profile_returns_segment(client):
    r = client.post("/profile", json=VALID_PROFILE)
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["segment_label"] in {
        "budget_buyer", "mid_range_investor", "growth_investor", "hni_aggressive"
    }
    assert len(body["segment_strategy"]) > 0


def test_get_profile_after_create(client):
    created = client.post("/profile", json=VALID_PROFILE).json()
    r = client.get(f"/profile/{created['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]


def test_get_nonexistent_profile_404(client):
    r = client.get("/profile/999999")
    assert r.status_code == 404


def test_invalid_income_rejected(client):
    bad = {**VALID_PROFILE, "monthly_income": -500}
    r = client.post("/profile", json=bad)
    assert r.status_code == 422


def test_invalid_horizon_rejected(client):
    bad = {**VALID_PROFILE, "investment_horizon": "forever"}
    r = client.post("/profile", json=bad)
    assert r.status_code == 422


def test_hni_profile_segments_differently_than_budget_profile(client):
    budget_profile = {**VALID_PROFILE, "monthly_income": 30000, "max_budget": 1500000}
    hni_profile = {
        **VALID_PROFILE,
        "monthly_income": 500000,
        "max_budget": 25000000,
        "risk_appetite": "aggressive",
    }
    seg_budget = client.post("/profile", json=budget_profile).json()["segment_label"]
    seg_hni = client.post("/profile", json=hni_profile).json()["segment_label"]
    assert seg_budget != seg_hni
    assert seg_hni == "hni_aggressive"
