"""
PostGIS and Full-Text Search helper functions for LandIQ.

Provides:
- areas_within_radius(): Find Area rows within N km of a coordinate
- full_text_search():    Find DocumentChunk rows matching a query string
- update_development_index(): Recalculate area dev index from nearby projects
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from geoalchemy2.functions import ST_DistanceSphere, ST_MakePoint, ST_SetSRID
from db.models import Area, DevelopmentProject, DocumentChunk, ZoneLabel


# --------------------------------------------------------------------------- #
# Spatial queries                                                              #
# --------------------------------------------------------------------------- #

def areas_within_radius(
    db: Session,
    lon: float,
    lat: float,
    radius_km: float,
) -> list[Area]:
    """
    Return all Area rows whose centroid falls within `radius_km` kilometres
    of the given (lon, lat) coordinate.

    Uses PostGIS ST_DistanceSphere (meters) for efficiency without needing
    a geography type cast.
    """
    point = ST_SetSRID(ST_MakePoint(lon, lat), 4326)
    radius_m = radius_km * 1000
    return (
        db.query(Area)
        .filter(ST_DistanceSphere(Area.location, point) <= radius_m)
        .all()
    )


def heatmap_geojson(db: Session) -> dict:
    """
    Build a GeoJSON FeatureCollection of all areas for the Folium heatmap.
    """
    areas = db.query(Area).all()
    features = []
    for a in areas:
        # geoalchemy2 returns WKBElement; we ask PostGIS for coordinates
        row = db.execute(
            text("SELECT ST_X(location::geometry), ST_Y(location::geometry) "
                 "FROM areas WHERE id = :id"),
            {"id": a.id}
        ).fetchone()
        if row is None or row[0] is None:
            continue
        lon, lat = row
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "id": a.id,
                "name": a.name,
                "district": a.district,
                "zone_label": a.zone_label.value if a.zone_label else "stable",
                "development_index": a.development_index or 0.0,
                "current_price_sqft": a.current_price_sqft or 0.0,
                "price_cagr_3yr": a.price_cagr_3yr or 0.0,
                "distance_to_metro_km": a.distance_to_metro_km or 0.0,
            },
        })
    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------- #
# Full-Text Search (replaces Qdrant)                                           #
# --------------------------------------------------------------------------- #

def full_text_search(
    db: Session,
    query: str,
    limit: int = 10,
) -> list[DocumentChunk]:
    """
    Full-text search over DocumentChunk.content using Postgres plainto_tsquery.
    Falls back to ILIKE if PostGIS extension pg_trgm is unavailable.
    """
    try:
        results = (
            db.query(DocumentChunk)
            .filter(
                func.to_tsvector("english", DocumentChunk.content).op("@@")(
                    func.plainto_tsquery("english", query)
                )
            )
            .order_by(
                func.ts_rank(
                    func.to_tsvector("english", DocumentChunk.content),
                    func.plainto_tsquery("english", query),
                ).desc()
            )
            .limit(limit)
            .all()
        )
    except Exception:
        # Fallback: simple substring match
        results = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.content.ilike(f"%{query}%"))
            .limit(limit)
            .all()
        )
    return results


# --------------------------------------------------------------------------- #
# Development Index scoring                                                    #
# --------------------------------------------------------------------------- #

# Weights by project type for Development Index calculation
PROJECT_WEIGHTS = {
    "metro":       10,
    "highway":      7,
    "industrial":   8,
    "it_sez":       9,
    "park":         4,
    "education":    5,
    "hospital":     5,
    "other":        3,
}

# Impact radii by project type (km)
PROJECT_RADII = {
    "metro":       5.0,
    "highway":     3.0,
    "industrial":  8.0,
    "it_sez":      6.0,
    "park":        2.0,
    "education":   2.0,
    "hospital":    2.0,
    "other":       3.0,
}


def update_development_index(db: Session) -> None:
    """
    Recompute the development_index for every area based on all projects
    in the database. Called after a new document is ingested.

    Algorithm:
        For each area, find all projects whose impact radius covers the area.
        Score = sum(weight * budget_factor * timeline_factor) capped at 100.
    """
    from datetime import datetime
    current_year = datetime.utcnow().year

    areas = db.query(Area).all()
    projects = db.query(DevelopmentProject).all()

    for area in areas:
        # Get area coordinates
        row = db.execute(
            text("SELECT ST_X(location::geometry), ST_Y(location::geometry) "
                 "FROM areas WHERE id = :id"),
            {"id": area.id}
        ).fetchone()
        if row is None or row[0] is None:
            continue
        a_lon, a_lat = row

        score = 0.0
        project_count = 0

        for proj in projects:
            if proj.location is None:
                continue
            proj_row = db.execute(
                text("SELECT ST_X(location::geometry), ST_Y(location::geometry) "
                     "FROM development_projects WHERE id = :id"),
                {"id": proj.id}
            ).fetchone()
            if proj_row is None or proj_row[0] is None:
                continue
            p_lon, p_lat = proj_row

            # Haversine-style via PostGIS
            dist_row = db.execute(
                text("""
                    SELECT ST_DistanceSphere(
                        ST_SetSRID(ST_MakePoint(:a_lon, :a_lat), 4326),
                        ST_SetSRID(ST_MakePoint(:p_lon, :p_lat), 4326)
                    ) / 1000
                """),
                {"a_lon": a_lon, "a_lat": a_lat, "p_lon": p_lon, "p_lat": p_lat}
            ).fetchone()
            if dist_row is None:
                continue
            dist_km = dist_row[0]

            ptype = (proj.project_type or "other").lower()
            radius_km = PROJECT_RADII.get(ptype, 3.0)

            if dist_km <= radius_km:
                weight = PROJECT_WEIGHTS.get(ptype, 3)
                # Budget factor: larger budget = bigger signal (log scale)
                import math
                budget_factor = 1.0
                if proj.budget_crore and proj.budget_crore > 0:
                    budget_factor = min(2.0, 1.0 + math.log10(proj.budget_crore / 100 + 1))

                # Timeline factor: nearer completion = stronger signal
                timeline_factor = 0.8
                if proj.estimated_completion_year:
                    years_away = max(0, proj.estimated_completion_year - current_year)
                    timeline_factor = max(0.3, 1.0 - (years_away * 0.1))

                score += weight * budget_factor * timeline_factor
                project_count += 1

                # Preserve the seeded development score and add project impact
        base_index = area.development_index or 0.0

        if project_count > 0:
            area.development_index = min(100.0, base_index + score)
        else:
            area.development_index = base_index

        area.num_upcoming_projects = project_count
        

        # Update zone label based on new score
        if area.development_index >= 75:
            area.zone_label = ZoneLabel.RAPID
        elif area.development_index >= 40:
            area.zone_label = ZoneLabel.EMERGING
        elif area.development_index >= 20:
            area.zone_label = ZoneLabel.STABLE
        else:
            area.zone_label = ZoneLabel.SATURATED

    db.commit()