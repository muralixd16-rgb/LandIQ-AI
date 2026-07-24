"""
Seed 35 real Hyderabad/Telangana localities into the areas table.

Run with:
    python -m scripts.seed_areas

Data is based on publicly known market data (MagicBricks/99acres averages, 2024).
Coordinates are real lat/lon from OpenStreetMap/Nominatim.
"""
import os
import sys
from sqlalchemy import text
from db.session import engine, SessionLocal
from db.models import Base, Area, ZoneLabel
from geoalchemy2.shape import from_shape
from shapely.geometry import Point

# 35 Hyderabad/Telangana localities
# (name, district, lon, lat, price_sqft, cagr_3yr%, pop_growth%, dist_center_km,
#  dist_metro_km, dist_highway_km, dev_index, zone)
SEED_AREAS = [
    # --- Core IT Corridor (high appreciation already, stable/saturated) ---
    ("Gachibowli",       "Rangareddy",  78.3376, 17.4401, 9800,  8.5, 3.2, 14.0,  2.0, 1.5,  85.0, ZoneLabel.RAPID),
    ("Madhapur",         "Hyderabad",   78.3785, 17.4485, 11200, 7.2, 2.8, 12.0,  1.5, 2.0,  80.0, ZoneLabel.RAPID),
    ("Kondapur",         "Rangareddy",  78.3607, 17.4632, 8900,  9.0, 3.5, 13.0,  2.5, 1.8,  78.0, ZoneLabel.RAPID),
    ("Hitech City",      "Hyderabad",   78.3805, 17.4484, 14500, 6.0, 2.0, 11.0,  0.5, 2.5,  70.0, ZoneLabel.SATURATED),
    ("Jubilee Hills",    "Hyderabad",   78.4066, 17.4299, 18000, 5.5, 1.5, 8.0,   3.0, 3.5,  55.0, ZoneLabel.SATURATED),

    # --- Western Corridor Growth (rapid to emerging) ---
    ("Kokapet",          "Rangareddy",  78.3242, 17.4058, 7800,  14.2, 5.2, 18.0, 4.0,  1.5,  90.0, ZoneLabel.RAPID),
    ("Narsingi",         "Rangareddy",  78.3325, 17.3947, 6200,  12.5, 4.8, 20.0, 5.5,  2.0,  82.0, ZoneLabel.RAPID),
    ("Tellapur",         "Rangareddy",  78.2907, 17.4671, 5400,  16.3, 6.1, 22.0, 7.0,  1.0,  75.0, ZoneLabel.RAPID),
    ("Mokila",           "Rangareddy",  78.2611, 17.4793, 4100,  18.5, 6.8, 26.0, 9.5,  0.8,  68.0, ZoneLabel.EMERGING),
    ("Patancheru",       "Sangareddy",  78.2657, 17.5301, 3200,  11.0, 4.5, 28.0, 11.0, 0.5,  60.0, ZoneLabel.EMERGING),

    # --- North Hyderabad (mixed growth) ---
    ("Kompally",         "Medchal",     78.4790, 17.5395, 5200,  11.5, 5.0, 16.0, 8.0,  2.5,  65.0, ZoneLabel.EMERGING),
    ("Nizampet",         "Hyderabad",   78.4025, 17.5082, 5800,  9.5,  4.0, 14.0, 6.0,  3.0,  58.0, ZoneLabel.EMERGING),
    ("Bachupally",       "Medchal",     78.4042, 17.5343, 4800,  10.2, 4.3, 17.0, 7.5,  2.8,  62.0, ZoneLabel.EMERGING),
    ("Miyapur",          "Hyderabad",   78.3675, 17.4949, 5600,  8.8,  3.8, 15.0, 4.0,  3.2,  64.0, ZoneLabel.EMERGING),
    ("Hafeezpet",        "Hyderabad",   78.3600, 17.4801, 6100,  9.0,  3.6, 14.5, 3.5,  2.8,  66.0, ZoneLabel.EMERGING),

    # --- East Hyderabad / IT Expansion ---
    ("Uppal",            "Medchal",     78.5593, 17.4052, 5000,  9.8,  4.1, 10.0, 3.0,  1.5,  62.0, ZoneLabel.EMERGING),
    ("Ghatkesar",        "Medchal",     78.6877, 17.4401, 3800,  13.5, 5.5, 22.0, 9.0,  1.0,  70.0, ZoneLabel.EMERGING),
    ("Medchal",          "Medchal",     78.5516, 17.6272, 3200,  14.0, 6.0, 28.0, 12.0, 0.8,  58.0, ZoneLabel.EMERGING),
    ("Pocharam",         "Medchal",     78.6229, 17.5139, 3500,  12.8, 5.2, 24.0, 10.5, 0.6,  65.0, ZoneLabel.EMERGING),
    ("LB Nagar",         "Rangareddy",  78.5477, 17.3483, 4600,  7.5,  3.2, 12.0, 2.5,  2.2,  50.0, ZoneLabel.STABLE),

    # --- South / Airport Corridor ---
    ("Shamshabad",       "Rangareddy",  78.4203, 17.2457, 3500,  15.0, 6.5, 30.0, 15.0, 0.5,  80.0, ZoneLabel.RAPID),
    ("Rajendranagar",    "Rangareddy",  78.3915, 17.3196, 5500,  8.0,  3.5, 18.0, 6.0,  3.0,  55.0, ZoneLabel.STABLE),
    ("Tukkuguda",        "Rangareddy",  78.5049, 17.2531, 3000,  16.5, 7.0, 32.0, 16.0, 0.4,  75.0, ZoneLabel.RAPID),
    ("Adibatla",         "Rangareddy",  78.6044, 17.2985, 2800,  18.0, 7.5, 28.0, 18.0, 0.3,  78.0, ZoneLabel.RAPID),

    # --- Outer Ring Road Nodes ---
    ("Shadnagar",        "Rangareddy",  78.1919, 17.0689, 2200,  20.5, 8.5, 48.0, 28.0, 0.5,  72.0, ZoneLabel.EMERGING),
    ("Kothur",           "Rangareddy",  78.3143, 17.0490, 1800,  19.0, 8.0, 52.0, 30.0, 0.4,  68.0, ZoneLabel.EMERGING),
    ("Yadadri",          "Yadadri",     79.0196, 17.0893, 1500,  14.0, 6.5, 65.0, 40.0, 1.5,  55.0, ZoneLabel.EMERGING),
    ("Bibinagar",        "Yadadri",     78.8920, 17.4777, 2000,  13.0, 5.8, 45.0, 22.0, 1.2,  52.0, ZoneLabel.EMERGING),
    ("Sangareddy",       "Sangareddy",  77.9999, 17.6198, 2500,  11.5, 5.0, 45.0, 25.0, 0.8,  58.0, ZoneLabel.EMERGING),

    # --- Central / Old City ---
    ("Banjara Hills",    "Hyderabad",   78.4392, 17.4130, 16000, 4.5,  1.2, 5.0,  2.5,  4.0,  48.0, ZoneLabel.SATURATED),
    ("Kukatpally",       "Hyderabad",   78.4045, 17.4944, 7200,  7.8,  3.0, 13.0, 3.5,  3.5,  60.0, ZoneLabel.STABLE),
    ("Ameerpet",         "Hyderabad",   78.4483, 17.4356, 10500, 5.0,  1.8, 6.0,  1.0,  4.5,  42.0, ZoneLabel.SATURATED),
    ("Secunderabad",     "Hyderabad",   78.4985, 17.4436, 9000,  5.5,  2.0, 4.0,  1.5,  3.0,  40.0, ZoneLabel.SATURATED),
    ("Yadagirigutta",    "Yadadri",     79.0844, 17.5783, 2200,  15.5, 6.8, 60.0, 38.0, 2.0,  62.0, ZoneLabel.EMERGING),
    ("Nalgonda",         "Nalgonda",    79.2664, 17.0584, 1600,  10.5, 4.5, 80.0, 50.0, 2.5,  42.0, ZoneLabel.STABLE),
]


def seed_areas(db) -> int:
    """Insert areas that don't already exist. Returns count inserted."""
    inserted = 0
    for row in SEED_AREAS:
        (name, district, lon, lat, price, cagr, pop_growth,
         dist_center, dist_metro, dist_highway, dev_idx, zone) = row

        existing = db.query(Area).filter(Area.name == name).first()
        if existing:
            continue

        point = from_shape(Point(lon, lat), srid=4326)
        area = Area(
            name=name,
            district=district,
            location=point,
            current_price_sqft=price,
            price_cagr_3yr=cagr,
            population_growth_rate=pop_growth,
            distance_to_city_center_km=dist_center,
            distance_to_metro_km=dist_metro,
            distance_to_highway_km=dist_highway,
            development_index=dev_idx,
            zone_label=zone,
        )
        db.add(area)
        inserted += 1

    db.commit()
    return inserted


def seed_profiles(db) -> int:
    """Insert 100 realistic investor profiles into the investor_profiles table if empty."""
    from db.models import InvestorProfile, InvestmentHorizon, RiskAppetite, AssetType
    from ml.segmentation import generate_synthetic_training_data as gen_inv
    import random

    if db.query(InvestorProfile).count() > 0:
        return 0

    df = gen_inv(n=100, seed=42)
    inserted = 0
    asset_types = [AssetType.PLOT, AssetType.HOUSE, AssetType.FARM_LAND, AssetType.VILLA, AssetType.APARTMENT]

    for _, row in df.iterrows():
        profile = InvestorProfile(
            name=f"Seed Investor {inserted + 1}",
            monthly_income=float(row["monthly_income"]),
            max_budget=float(row["max_budget"]),
            investment_horizon=InvestmentHorizon(row["investment_horizon"]),
            risk_appetite=RiskAppetite(row["risk_appetite"]),
            preferred_asset_type=random.choice(asset_types),
            target_roi_percent=float(row["target_roi_percent"]),
        )
        db.add(profile)
        inserted += 1

    db.commit()
    return inserted


if __name__ == "__main__":
    # Enable PostGIS
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        conn.commit()

    db = SessionLocal()
    try:
        count = seed_areas(db)
        print(f"[seed_areas] Inserted {count} new area records.")
        p_count = seed_profiles(db)
        print(f"[seed_areas] Inserted {p_count} new investor profile records.")
    finally:
        db.close()
