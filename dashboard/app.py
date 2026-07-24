"""
LandIQ Streamlit dashboard.

Blueprint-inspired product UI for:
  - Buyer recommendations
  - Seller hold/sell analysis
  - Area explorer map
  - Government plan upload
"""
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
from map import build_map



API_URL = os.environ.get("API_URL", "http://localhost:8000")

COLORS = {
    "bg": "#09090b",
    "bg2": "#18181b",
    "bg3": "#27272a",
    "border": "#27272a",
    "border2": "#3f3f46",
    "text": "#fafafa",
    "text2": "#a1a1aa",
    "text3": "#71717a",
    "teal": "#10b981",
    "amber": "#f59e0b",
    "purple": "#8b5cf6",
    "coral": "#f43f5e",
    "blue": "#3b82f6",
    "green": "#22c55e",
}

ZONE_COLORS = {
    "rapid_development": COLORS["teal"],
    "emerging": COLORS["amber"],
    "stable": COLORS["blue"],
    "saturated": COLORS["text2"],
}

ZONE_LABELS = {
    "rapid_development": "Rapid Dev",
    "emerging": "Emerging",
    "stable": "Stable",
    "saturated": "Saturated",
}


st.set_page_config(
    page_title="LandIQ - Smart Real Estate Advisor",
    page_icon="LIQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg: #09090b;
  --bg2: #18181b;
  --bg3: #27272a;
  --border: #27272a;
  --border2: #3f3f46;
  --text: #fafafa;
  --text2: #a1a1aa;
  --text3: #71717a;
  --teal: #10b981;
  --teal-dim: rgba(16, 185, 129, 0.08);
  --amber: #f59e0b;
  --amber-dim: rgba(245, 158, 11, 0.08);
  --purple: #8b5cf6;
  --purple-dim: rgba(139, 92, 246, 0.08);
  --coral: #f43f5e;
  --coral-dim: rgba(244, 63, 94, 0.08);
  --blue: #3b82f6;
  --blue-dim: rgba(59, 130, 246, 0.08);
  --green: #22c55e;
  --green-dim: rgba(34, 197, 94, 0.08);
}

.stApp {
  background:
    radial-gradient(circle at 80% 10%, rgba(16, 185, 129, 0.045), transparent 50%),
    radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.03), transparent 50%),
    var(--bg);
  color: var(--text);
}

html, body, [class*="css"], [class*="st-"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text);
}

.block-container {
  max-width: 1200px;
  padding-top: 40px;
  padding-bottom: 80px;
}

[data-testid="stHeader"] { background: rgba(9,9,11,0); }
[data-testid="stToolbar"] { right: 1.5rem; }

/* Hero section */
.hero {
  padding: 40px 0 30px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 30px;
}
.hero-eyebrow, .section-label, .mono {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.hero-eyebrow {
  color: var(--teal);
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 18px;
}
.hero-eyebrow:before {
  content: '';
  width: 24px;
  height: 2px;
  background: var(--teal);
  border-radius: 1px;
}
.hero h1 {
  font-family: 'Inter', sans-serif;
  font-size: clamp(38px, 5vw, 62px);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.03em;
  margin: 0;
  color: var(--text);
}
.hero h1 span {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-kicker {
  font-family: 'Inter', sans-serif;
  color: var(--text2);
  font-size: 18px;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin-top: 6px;
}
.hero-sub {
  color: var(--text3);
  font-size: 15px;
  max-width: 680px;
  margin-top: 14px;
  line-height: 1.6;
}
.hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 24px;
}
.meta-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--text2);
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 9999px;
  padding: 6px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  transition: all 0.2s ease;
}
.meta-chip:hover {
  border-color: var(--border2);
  background: var(--bg3);
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}

/* Sections */
.section {
  padding-top: 36px;
  margin-bottom: 20px;
}
.section-label {
  color: var(--teal);
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.section-label:after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}
.section h2 {
  font-family: 'Inter', sans-serif;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 8px 0;
  color: var(--text);
}
.section-intro {
  color: var(--text2);
  font-size: 14px;
  max-width: 760px;
  line-height: 1.5;
  margin-bottom: 24px;
}

/* Cards & Layout */
.card {
  background: linear-gradient(180deg, rgba(24, 24, 27, 0.85) 0%, rgba(24, 24, 27, 1) 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--border2);
  box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.3);
}
.card-accent {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
}
.tag {
  display: inline-block;
  border-radius: 6px;
  padding: 4px 10px;
  margin-bottom: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.card h3 {
  font-family: 'Inter', sans-serif;
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: var(--text);
}
.card p, .small-copy {
  color: var(--text2);
  font-size: 13px;
  line-height: 1.6;
  margin: 0;
}

/* Metrics & Metric Overrides */
.metric-card, [data-testid="stMetric"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 20px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
  transition: all 0.2s ease-in-out !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  min-height: 110px;
}
.metric-card:hover, [data-testid="stMetric"]:hover {
  border-color: var(--border2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
}
.metric-label, [data-testid="stMetricLabel"] p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  color: var(--text3) !important;
  text-transform: uppercase !important;
  margin: 0 !important;
}
.metric-value, [data-testid="stMetricValue"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  margin-top: 6px !important;
}
.metric-sub {
  color: var(--text2) !important;
  font-size: 12px !important;
  margin-top: 4px !important;
}

/* Results & Tables */
.result-card {
  background: linear-gradient(90deg, rgba(16, 185, 129, 0.03) 0%, rgba(24, 24, 27, 1) 100%);
  border: 1px solid rgba(16, 185, 129, 0.15);
  border-left: 4px solid var(--teal);
  border-radius: 12px;
  padding: 20px;
  margin-top: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.result-title {
  font-family: 'Inter', sans-serif;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-bottom: 6px;
}
.area-rank {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text3);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
}

/* Tab Overrides */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
  height: 40px;
  padding: 0 18px;
  color: var(--text3) !important;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 600;
  border-radius: 6px 6px 0 0;
  background-color: transparent !important;
  border: none !important;
  transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text2) !important;
  background-color: rgba(255, 255, 255, 0.02) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
  background-color: rgba(16, 185, 129, 0.03) !important;
}

/* Forms & Inputs Override */
div[data-testid="stForm"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 24px !important;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
  background-color: var(--bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 10px 14px !important;
  font-size: 14px !important;
  transition: all 0.2s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15) !important;
}
label, .stSlider label, .stSelectbox label, .stTextInput label, .stNumberInput label {
  color: var(--text2) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  margin-bottom: 6px !important;
}

/* Button & Download Button Overrides */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, #10b981, #059669) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 24px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  letter-spacing: -0.01em !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 4px rgba(16, 185, 129, 0.15) !important;
  width: 100% !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
  opacity: 0.95 !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translateY(0) !important;
}

/* Slider Track Overrides */
div[data-testid="stSlider"] div[role="slider"] {
  background-color: var(--teal) !important;
  border: 2px solid var(--text) !important;
}

/* Expander Overrides */
.stExpander {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
  margin-bottom: 12px !important;
  overflow: hidden !important;
}
.stExpander [data-testid="stExpanderHeader"] {
  padding: 14px 18px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  color: var(--text) !important;
  transition: background-color 0.2s ease !important;
}
.stExpander [data-testid="stExpanderHeader"]:hover {
  background-color: var(--bg3) !important;
}
.stExpander [data-testid="stExpanderDetails"] {
  padding: 18px !important;
  background-color: rgba(24, 24, 27, 0.4) !important;
}

hr {
  border-color: var(--border) !important;
}

@media (max-width: 700px) {
  .block-container { padding-left: 18px; padding-right: 18px; }
  .hero { padding-top: 20px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.warning(f"API unavailable for {path}: {exc}")
        return None


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API request failed for {path}: {exc}")
        return None


def fmt_rs(value: float | int | None) -> str:
    if value is None:
        return "Rs --"
    return f"Rs {value:,.0f}"


def fmt_zone(zone: str | None) -> str:
    if not zone:
        return "Unknown"
    return ZONE_LABELS.get(zone, zone.replace("_", " ").title())


def chip(label: str, color: str) -> str:
    return f'<div class="meta-chip"><span class="dot" style="background:{color}"></span>{label}</div>'


def section(label: str, title: str, intro: str) -> None:
    st.markdown(
        f"""
<div class="section">
  <div class="section-label">{label}</div>
  <h2>{title}</h2>
  <div class="section-intro">{intro}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", color: str = COLORS["text"]) -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value" style="color:{color}">{value}</div>
  <div class="metric-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def feature_card(tag: str, title: str, body: str, color: str) -> None:
    st.markdown(
        f"""
<div class="card">
  <div class="card-accent" style="background:{color}"></div>
  <div class="tag" style="background:color-mix(in srgb, {color} 15%, transparent);color:{color}">{tag}</div>
  <h3>{title}</h3>
  <p>{body}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-eyebrow">Project Dashboard &middot; v1.0</div>
  <h1>Land<span>IQ</span></h1>
  <div class="hero-kicker">Smart Real Estate Investment Advisor</div>
  <div class="hero-sub">
    Find undervalued Telangana areas before the market prices in metro, highway,
    zoning, and government development signals.
  </div>
  <div class="hero-meta">
    {chip("Buyer Advisor", COLORS["teal"])}
    {chip("Seller Timing", COLORS["amber"])}
    {chip("Govt Intel", COLORS["purple"])}
    {chip("ML Forecasts", COLORS["blue"])}
    {chip("Hyderabad Focus", COLORS["coral"])}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_overview(areas_data: dict[str, Any] | None) -> None:
    section(
        "00 - Overview",
        "Decision cockpit",
        "A compact operating view of the LandIQ intelligence pipeline and current coverage.",
    )

    areas = areas_data.get("areas", []) if areas_data else []
    total = areas_data.get("count", len(areas)) if areas_data else 0
    rapid = sum(1 for area in areas if area.get("zone_label") == "rapid_development")
    avg_dev = sum(area.get("development_index", 0) for area in areas) / len(areas) if areas else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Areas tracked", str(total), "Hyderabad and Telangana seed localities", COLORS["teal"])
    with c2:
        metric_card("Rapid zones", str(rapid), "High development-index areas", COLORS["amber"])
    with c3:
        metric_card("Avg dev index", f"{avg_dev:.0f}/100", "From project density and infrastructure", COLORS["purple"])
    with c4:
        metric_card("API", "Live" if areas_data else "Offline", API_URL, COLORS["green"] if areas_data else COLORS["coral"])

    st.write("")
    f1, f2, f3 = st.columns(3)
    with f1:
        feature_card(
            "System 1",
            "User Profiling",
            "Clusters income, budget, horizon, risk appetite, and target ROI into investor strategies.",
            COLORS["teal"],
        )
    with f2:
        feature_card(
            "System 2",
            "Government Intelligence",
            "Turns master plans and infrastructure PDFs into area-level development signals.",
            COLORS["purple"],
        )
    with f3:
        feature_card(
            "System 3",
            "Price Forecasting",
            "Predicts 1yr, 3yr, and 5yr appreciation using area fundamentals and ML models.",
            COLORS["amber"],
        )


def render_buyer() -> None:
    section(
        "01 - Buyer Advisor",
        "Match budget to future hotspots",
        "Submit a buyer profile and get a ranked shortlist tuned to budget, risk, asset type, and horizon.",
    )

    with st.form("buyer_form"):
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            name = st.text_input("Name", placeholder="Murali")
            monthly_income = st.number_input("Monthly income", min_value=10000, max_value=5000000, value=65000, step=5000)
            max_budget = st.number_input("Max budget", min_value=500000, max_value=100000000, value=2800000, step=100000)
        with c2:
            investment_horizon = st.selectbox("Investment horizon", ["short (1-2yr)", "mid (3-5yr)", "long (5-10yr)"])
            risk_appetite = st.selectbox("Risk appetite", ["conservative", "moderate", "aggressive"])
            preferred_asset = st.selectbox("Preferred asset", ["plot", "apartment", "house", "villa", "farm_land"])
        with c3:
            target_roi = st.slider("Target ROI", 10, 300, 80)
            top_n = st.slider("Areas to show", 3, 10, 5)
            submitted = st.form_submit_button("Find Best Areas", use_container_width=True)

    if not submitted:
        return

    horizon_map = {"short (1-2yr)": "short", "mid (3-5yr)": "mid", "long (5-10yr)": "long"}
    payload = {
        "name": name or None,
        "monthly_income": monthly_income,
        "max_budget": max_budget,
        "investment_horizon": horizon_map[investment_horizon],
        "risk_appetite": risk_appetite,
        "preferred_asset_type": preferred_asset,
        "target_roi_percent": target_roi,
    }

    with st.spinner("Building profile and ranking areas..."):
        profile = api_post("/profile", payload)
        rec = api_post("/recommend/buyer", {"profile_id": profile["id"], "top_n": top_n}) if profile else None

    if not profile:
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Investor Segment", profile["segment_label"].replace("_", " ").title())
    m2.metric("Target Horizon", rec.get("target_horizon", "-") if rec else "-")
    m3.metric("Max Budget", fmt_rs(max_budget))

    st.markdown(
        f"""
<div class="result-card">
  <div class="tag" style="background:var(--teal-dim);color:var(--teal)">Strategy</div>
  <div class="small-copy">{profile["segment_strategy"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not rec or not rec.get("areas"):
        st.info("No area recommendations returned for this profile.")
        return

    for index, area in enumerate(rec["areas"], start=1):
        zone = area.get("zone_label", "stable")
        color = ZONE_COLORS.get(zone, COLORS["text2"])
        horizon = rec["target_horizon"]
        roi = area.get(f"predicted_roi_{horizon}", 0)
        with st.expander(f"#{index} {area['area_name']} - {area['district']}", expanded=index <= 3):
            st.markdown(
                f"""
<div class="area-rank">RANK {index:02d}</div>
<div class="result-title" style="color:{color}">{area['area_name']} &middot; {fmt_zone(zone)}</div>
""",
                unsafe_allow_html=True,
            )
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Price / sqft", fmt_rs(area.get("current_price_sqft")))
            a2.metric(f"{horizon} ROI", f"+{roi:.1f}%")
            a3.metric("Dev Index", f"{area.get('development_index', 0):.0f}/100")
            a4.metric("Projects", str(area.get("num_upcoming_projects", 0)))

            r1, r2, r3 = st.columns(3)
            r1.metric("1yr ROI", f"+{area.get('predicted_roi_1yr', 0):.1f}%")
            r2.metric("3yr ROI", f"+{area.get('predicted_roi_3yr', 0):.1f}%")
            r3.metric("5yr ROI", f"+{area.get('predicted_roi_5yr', 0):.1f}%")

            if area.get("key_driver"):
                st.caption(f"Key driver: {area['key_driver']}")
            st.caption(f"Risk level: {area.get('risk_level', 'unknown')}")


def render_seller(areas_data: dict[str, Any] | None) -> None:
    section(
        "02 - Seller Advisor",
        "Hold or exit with timing logic",
        "Estimate future property value from area-level appreciation forecasts and development signals.",
    )

    areas = areas_data.get("areas", []) if areas_data else []
    area_names = [area["name"] for area in areas]

    with st.form("seller_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if area_names:
                area_name = st.selectbox("Property location", area_names)
            else:
                area_name = st.text_input("Property location", placeholder="Kompally")
        with c2:
            size_sqyd = st.number_input("Property size, sq yards", min_value=50, max_value=10000, value=200, step=10)
        with c3:
            current_value = st.number_input("Current value, optional", min_value=0, value=0, step=100000)
            submitted = st.form_submit_button("Analyze Property", use_container_width=True)

    if not submitted:
        return

    payload = {
        "area_name": area_name,
        "property_size_sqyd": size_sqyd,
        "current_estimated_value": current_value if current_value > 0 else None,
    }

    with st.spinner("Forecasting seller outcome..."):
        result = api_post("/recommend/seller", payload)

    if not result:
        return

    is_hold = "HOLD" in result["recommendation"].upper()
    color = COLORS["teal"] if is_hold else COLORS["coral"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Value", fmt_rs(result["current_estimated_value"]))
    c2.metric("Predicted 3yr", fmt_rs(result["predicted_value_3yr"]))
    c3.metric("3yr ROI", f"+{result['predicted_roi_3yr']:.1f}%")

    st.markdown(
        f"""
<div class="result-card" style="border-left-color:{color}">
  <div class="result-title" style="color:{color}">{result["recommendation"]}</div>
  <div class="small-copy">{result["reasoning"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    chart_data = pd.DataFrame(
        {
            "Year": ["Now", "1yr", "3yr", "5yr"],
            "Value": [
                result["current_estimated_value"],
                result["predicted_value_1yr"],
                result["predicted_value_3yr"],
                result["predicted_value_5yr"],
            ],
        }
    ).set_index("Year")
    st.line_chart(chart_data)


def render_explorer() -> None:
    section(
        "03 - Area Explorer",
        "Development heat map",
        "Filter Hyderabad localities by zone and development index, then inspect detailed area reports.",
    )

    filters, map_col = st.columns([1, 2.4])
    with filters:
        zone_filter = st.selectbox("Zone", ["All", "rapid_development", "emerging", "stable", "saturated"])
        min_dev = st.slider("Minimum development index", 0, 100, 0)
        report_query = st.text_input("Area report search", placeholder="Kokapet")

    with map_col:
        with st.spinner("Loading spatial signals..."):
            geojson = api_get("/heatmap")

        if not geojson:
            st.warning("Map data is not available yet. Start the API to load heatmap features.")
        else:
            features = geojson.get("features", [])
            if zone_filter != "All":
                features = [f for f in features if f["properties"].get("zone_label") == zone_filter]
            if min_dev > 0:
                features = [f for f in features if f["properties"].get("development_index", 0) >= min_dev]
            filtered_geojson = {**geojson, "features": features}
            st_folium(build_map(filtered_geojson), width="100%", height=560)

    if not report_query:
        return

    areas_data = api_get("/areas")
    if not areas_data:
        return

    match = next((area for area in areas_data["areas"] if report_query.lower() in area["name"].lower()), None)
    if not match:
        st.warning(f"No area found matching '{report_query}'.")
        return

    with st.spinner(f"Loading report for {match['name']}..."):
        report = api_get(f"/areas/{match['id']}/report", {"search_query": report_query})

    if not report:
        return

    st.markdown(f"### Report: {report['name']}")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Zone", fmt_zone(report["zone_label"]))
    r2.metric("Dev Index", f"{report['development_index']:.0f}/100")
    r3.metric("3yr ROI", f"+{report['predicted_roi_3yr']:.1f}%")
    r4.metric("Price / sqft", fmt_rs(report["current_price_sqft"]))

    if report.get("explanation_summary"):
        st.info(report["explanation_summary"])

    excerpts = report.get("document_excerpts", [])
    for excerpt in excerpts[:3]:
        with st.expander(excerpt["source"]):
            st.write(excerpt["excerpt"])


def render_upload() -> None:
    section(
        "04 - Govt Intel",
        "Upload planning documents",
        "Feed HMDA plans, budget PDFs, metro corridors, and road alignment documents into the extraction pipeline.",
    )

    uploaded = st.file_uploader("Government planning PDF", type=["pdf"])
    if uploaded:
        size_kb = len(uploaded.getvalue()) // 1024
        st.success(f"Selected {uploaded.name} ({size_kb} KB)")
        if st.button("Upload and Process", use_container_width=True):
            with st.spinner("Uploading document and starting extraction..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload-plan",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=30,
                    )
                    response.raise_for_status()
                    st.success(response.json().get("message", "Upload started."))
                    st.json(response.json())
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("Source", "HMDA Master Plan", "Zoning, growth corridors, infrastructure roadmap, and planning boundaries.", COLORS["teal"])
    with c2:
        feature_card("Source", "Metro + Highways", "Upcoming station, corridor, road widening, and access-improvement signals.", COLORS["purple"])
    with c3:
        feature_card("Source", "State Budget", "Allocated infrastructure spend and named locality development projects.", COLORS["amber"])


render_hero()
areas_payload = api_get("/areas")
render_overview(areas_payload)

buyer_tab, seller_tab, explorer_tab, upload_tab = st.tabs(
    ["Buyer Advisor", "Seller Advisor", "Area Explorer", "Upload Plan"]
)

with buyer_tab:
    render_buyer()

with seller_tab:
    render_seller(areas_payload)

with explorer_tab:
    render_explorer()

with upload_tab:
    render_upload()
