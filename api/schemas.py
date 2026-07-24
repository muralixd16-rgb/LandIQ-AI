"""Pydantic models for request validation and response shaping."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Any
from enum import Enum


class InvestmentHorizonEnum(str, Enum):
    short = "short"
    mid = "mid"
    long = "long"


class RiskAppetiteEnum(str, Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"


class AssetTypeEnum(str, Enum):
    plot = "plot"
    house = "house"
    farm_land = "farm_land"
    villa = "villa"
    apartment = "apartment"


# ── Investor Profile ─────────────────────────────────────────────────────── #

class InvestorProfileCreate(BaseModel):
    name: Optional[str] = None
    monthly_income: float = Field(..., gt=0, description="Monthly income in INR")
    max_budget: float = Field(..., gt=0, description="Max investment budget in INR")
    investment_horizon: InvestmentHorizonEnum
    risk_appetite: RiskAppetiteEnum
    preferred_asset_type: AssetTypeEnum
    target_roi_percent: Optional[float] = Field(None, ge=0, le=500)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Murali",
                "monthly_income": 65000,
                "max_budget": 2800000,
                "investment_horizon": "mid",
                "risk_appetite": "moderate",
                "preferred_asset_type": "plot",
                "target_roi_percent": 80,
            }
        }
    )


class InvestorProfileResponse(BaseModel):
    id: int
    name: Optional[str]
    segment_label: str
    segment_strategy: str
    monthly_income: float
    max_budget: float
    investment_horizon: str
    risk_appetite: str

    model_config = ConfigDict(from_attributes=True)


# ── Area Scorecard (used in buyer recommendations) ───────────────────────── #

class AreaScorecard(BaseModel):
    area_id: int
    area_name: str
    district: str
    zone_label: str
    current_price_sqft: float
    development_index: float
    predicted_roi_1yr: float
    predicted_roi_3yr: float
    predicted_roi_5yr: float
    confidence_lower: float
    confidence_upper: float
    risk_level: str
    key_driver: Optional[str] = None
    num_upcoming_projects: int = 0


# ── Buyer Recommendation ─────────────────────────────────────────────────── #

class BuyerRecommendationRequest(BaseModel):
    profile_id: int
    top_n: int = Field(5, ge=1, le=20)

    model_config = ConfigDict(json_schema_extra={"example": {"profile_id": 1, "top_n": 5}})


class BuyerRecommendationResponse(BaseModel):
    profile_id: int
    segment_label: str
    target_horizon: str
    areas: List[AreaScorecard]


# ── Seller Recommendation ─────────────────────────────────────────────────── #

class SellerRecommendationRequest(BaseModel):
    area_name: str
    property_size_sqyd: float = Field(..., gt=0)
    current_estimated_value: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "area_name": "Kompally",
                "property_size_sqyd": 200,
                "current_estimated_value": 4800000,
            }
        }
    )


class SellerRecommendationResponse(BaseModel):
    area_name: str
    zone_label: str
    current_estimated_value: float
    predicted_value_1yr: float
    predicted_value_3yr: float
    predicted_value_5yr: float
    predicted_roi_3yr: float
    recommendation: str
    reasoning: str


# ── Price Prediction ─────────────────────────────────────────────────────── #

class PricePredictRequest(BaseModel):
    area_name: str

    model_config = ConfigDict(json_schema_extra={"example": {"area_name": "Kokapet"}})


class PricePredictResponse(BaseModel):
    area_name: str
    zone_label: str
    predicted_appreciation_1yr: float
    predicted_appreciation_3yr: float
    predicted_appreciation_5yr: float
    confidence_lower: float
    confidence_upper: float
    explanation_summary: str
    top_contributors: List[Any] = []


# ── Area List Item ───────────────────────────────────────────────────────── #

class AreaListItem(BaseModel):
    id: int
    name: str
    district: str
    zone_label: str
    current_price_sqft: float
    development_index: float
    predicted_roi_3yr: Optional[float] = None
    num_upcoming_projects: int = 0


# ── Area Full Report ─────────────────────────────────────────────────────── #

class AreaDetail(BaseModel):
    id: int
    name: str
    district: str
    zone_label: str
    current_price_sqft: float
    development_index: float
    price_cagr_3yr: float
    distance_to_metro_km: float
    distance_to_highway_km: float
    num_upcoming_projects: int
    predicted_roi_1yr: float
    predicted_roi_3yr: float
    predicted_roi_5yr: float
    confidence_lower: float
    confidence_upper: float
    explanation_summary: str
    risk_level: str
    nearby_projects: List[Any] = []
    document_excerpts: List[Any] = []
