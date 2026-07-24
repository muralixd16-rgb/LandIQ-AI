"""
LandIQ database models.

Tables:
- Area                : Hyderabad localities with price, zone, dev index
- DevelopmentProject  : Government projects extracted from documents
- InvestorProfile     : Buyer/seller profiles with segment assignment
- PricePrediction     : XGBoost model outputs per area
- DocumentChunk       : PDF text chunks for Postgres Full-Text Search (replaces Qdrant)
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Text, Enum, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum

# geoalchemy2 is only required when connecting to a PostGIS database.
# When running against SQLite (e.g. in tests) the library may not be
# present, so we fall back to a plain Text column for geometry fields.
try:
    from geoalchemy2 import Geometry as _Geometry
    def Geometry(geom_type: str = "GEOMETRY", srid: int = 4326, **kw):  # type: ignore[misc]
        """Return a real PostGIS Geometry column type."""
        return _Geometry(geom_type, srid=srid, **kw)
except ImportError:
    def Geometry(geom_type: str = "GEOMETRY", srid: int = 4326, **kw):  # type: ignore[misc]
        """Fallback: store geometry as plain Text when geoalchemy2 is absent."""
        return Text()


class Base(DeclarativeBase):
    """SQLAlchemy 2.x declarative base."""
    pass


class ZoneLabel(str, enum.Enum):
    RAPID = "rapid_development"
    EMERGING = "emerging"
    STABLE = "stable"
    SATURATED = "saturated"


class InvestmentHorizon(str, enum.Enum):
    SHORT = "short"   # 1-2 years
    MID = "mid"        # 3-5 years
    LONG = "long"       # 5-10 years


class RiskAppetite(str, enum.Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AssetType(str, enum.Enum):
    PLOT = "plot"
    HOUSE = "house"
    FARM_LAND = "farm_land"
    VILLA = "villa"
    APARTMENT = "apartment"


class Area(Base):
    __tablename__ = "areas"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False, unique=True)
    district = Column(String(80))
    location = Column(Geometry("POINT", srid=4326))       # lat/long

    current_price_sqft = Column(Float)                     # current avg price/sqft
    price_cagr_3yr = Column(Float)                         # historical CAGR, last 3yr
    population_growth_rate = Column(Float)
    distance_to_city_center_km = Column(Float)
    distance_to_metro_km = Column(Float)
    distance_to_highway_km = Column(Float)
    num_upcoming_projects = Column(Integer, default=0)

    development_index = Column(Float, default=0.0)         # 0-100, computed score
    zone_label = Column(Enum(ZoneLabel), default=ZoneLabel.STABLE)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = relationship("PricePrediction", back_populates="area",
                               cascade="all, delete-orphan")


class DevelopmentProject(Base):
    """A government development plan extracted from an uploaded document."""
    __tablename__ = "development_projects"

    id = Column(Integer, primary_key=True)
    source_document = Column(String(255))              # original filename
    project_name = Column(String(255))
    project_type = Column(String(80))                  # metro / highway / industrial / park etc.
    location_text = Column(String(255))                # raw extracted location text
    location = Column(Geometry("POINT", srid=4326))
    budget_crore = Column(Float, nullable=True)
    estimated_completion_year = Column(Integer, nullable=True)
    impact_radius_km = Column(Float, default=5.0)

    extracted_at = Column(DateTime, default=datetime.utcnow)


class InvestorProfile(Base):
    __tablename__ = "investor_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=True)
    monthly_income = Column(Float, nullable=False)
    max_budget = Column(Float, nullable=False)
    investment_horizon = Column(Enum(InvestmentHorizon), nullable=False)
    risk_appetite = Column(Enum(RiskAppetite), nullable=False)
    preferred_asset_type = Column(Enum(AssetType), nullable=False)
    target_roi_percent = Column(Float, nullable=True)   # what they hope to earn

    segment_label = Column(String(40), nullable=True)   # assigned by KMeans
    created_at = Column(DateTime, default=datetime.utcnow)


class PricePrediction(Base):
    __tablename__ = "price_predictions"

    id = Column(Integer, primary_key=True)
    area_id = Column(Integer, ForeignKey("areas.id", ondelete="CASCADE"))

    predicted_appreciation_1yr = Column(Float)
    predicted_appreciation_3yr = Column(Float)
    predicted_appreciation_5yr = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_version = Column(String(40))
    shap_top_feature = Column(String(80), nullable=True)
    shap_top_value = Column(Float, nullable=True)

    generated_at = Column(DateTime, default=datetime.utcnow)

    area = relationship("Area", back_populates="predictions")


class DocumentChunk(Base):
    """
    Stores PDF text chunks for Postgres Full-Text Search.
    Replaces Qdrant vector store with a GIN-indexed tsvector column.

    Usage:
        db.execute(
            "UPDATE document_chunks SET tsv_content = to_tsvector('english', content)"
        )
        # search:
        db.query(DocumentChunk).filter(
            func.to_tsvector('english', DocumentChunk.content).op('@@')(
                func.plainto_tsquery('english', query)
            )
        )
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    source_document = Column(String(255), nullable=False)  # original filename
    chunk_index = Column(Integer, nullable=False)           # page / chunk order
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = ()
       