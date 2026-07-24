"""
Folium map builder for the LandIQ dashboard.

The map renders:
  - Area markers colored by development zone
  - A heatmap based on development index
  - Compact popups with price and ROI signals
"""
from typing import Any

import folium
from folium.plugins import HeatMap


ZONE_COLORS = {
    "rapid_development": "#10b981",
    "emerging": "#f59e0b",
    "stable": "#3b82f6",
    "saturated": "#71717a",
}

ZONE_MARKERS = {
    "rapid_development": "RD",
    "emerging": "EM",
    "stable": "ST",
    "saturated": "SA",
}

HYD_CENTER = [17.385, 78.486]


def build_map(geojson: dict[str, Any], height: int = 550) -> folium.Map:
    """Build a Folium map from the GeoJSON returned by /heatmap."""
    folium_map = folium.Map(
        location=HYD_CENTER,
        zoom_start=10,
        tiles="CartoDB dark_matter",
        width="100%",
        height=height,
    )

    heat_data = []
    for feature in geojson.get("features", []):
        coords = feature["geometry"]["coordinates"]
        dev_index = feature["properties"].get("development_index", 0)
        if coords and dev_index:
            heat_data.append([coords[1], coords[0], dev_index / 100.0])

    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.25,
            radius=25,
            blur=20,
            gradient={0.2: "#3b82f6", 0.5: "#f59e0b", 0.8: "#10b981", 1.0: "#ffffff"},
            name="Development Intensity",
        ).add_to(folium_map)

    for feature in geojson.get("features", []):
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]
        if not coords:
            continue

        lat, lon = coords[1], coords[0]
        zone = props.get("zone_label", "stable")
        color = ZONE_COLORS.get(zone, "#8A9298")
        marker = ZONE_MARKERS.get(zone, "AR")
        name = props.get("name", "")
        price = props.get("current_price_sqft", 0)
        dev_index = props.get("development_index", 0)
        cagr = props.get("price_cagr_3yr", 0)
        metro = props.get("distance_to_metro_km", 0)

        popup_html = f"""
        <div style="font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;width:230px;padding:10px;background-color:#18181b;color:#fafafa;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.5);border:1px solid #27272a">
          <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:2px">{marker} {name}</div>
          <div style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#a1a1aa;text-transform:uppercase;margin-bottom:8px">{zone.replace('_', ' ').title()}</div>
          <table style="width:100%;font-size:12px;border-collapse:collapse;color:#a1a1aa">
            <tr style="border-bottom:1px solid #27272a"><td style="padding:5px 0;color:#71717a">Price/sqft</td>
                <td style="text-align:right;font-weight:600;color:#fafafa;font-family:'JetBrains Mono',monospace">Rs {price:,.0f}</td></tr>
            <tr style="border-bottom:1px solid #27272a"><td style="padding:5px 0;color:#71717a">Dev Index</td>
                <td style="text-align:right;font-weight:600;color:#fafafa;font-family:'JetBrains Mono',monospace">{dev_index:.0f}/100</td></tr>
            <tr style="border-bottom:1px solid #27272a"><td style="padding:5px 0;color:#71717a">3yr CAGR</td>
                <td style="text-align:right;color:#10b981;font-weight:600;font-family:'JetBrains Mono',monospace">{cagr:.1f}%</td></tr>
            <tr><td style="padding:5px 0;color:#71717a">Metro dist</td>
                <td style="text-align:right;color:#fafafa;font-family:'JetBrains Mono',monospace">{metro:.1f} km</td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + dev_index / 15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{name} - {zone.replace('_', ' ').title()}",
        ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)
    return folium_map
