"""
radar.py  —  Real radar-proxy data via Open-Meteo Forecast API

Open-Meteo provides model-assimilated precipitation data ingested from
weather stations, radiosondes, aircraft, radar, and satellites — updated
every 1-6 hours, free, no API key, global coverage.

Variables fetched (current hour):
  - precipitation        : total precip mm  (rain + showers + snow)
  - rain                 : large-scale rain mm
  - showers              : convective rain mm  (key radar signal)
  - weather_code         : WMO code (80-82 = rain showers, 95-99 = thunderstorm)
  - wind_gusts_10m       : gusts km/h  (storm motion proxy)
  - cape                 : convective available potential energy J/kg
  - cloud_cover          : total cloud cover %
  - visibility           : meters (low = fog/precip)

Derived radar features stored as 4-d float32 vector:
  [precip_intensity, storm_signal, convective_risk, coverage_score]
  All normalized to [0, 1].
"""

import requests
from datetime import datetime
from typing import Optional


# ── WMO code severity table ───────────────────────────────────────────────────
# Maps WMO weather_code → (precip_flag, storm_flag, severity 0-1)
_WMO_TABLE = {
    0:  (0, 0, 0.00),   # clear sky
    1:  (0, 0, 0.00),   # mainly clear
    2:  (0, 0, 0.05),   # partly cloudy
    3:  (0, 0, 0.10),   # overcast
    45: (0, 0, 0.15),   # fog
    48: (0, 0, 0.15),   # rime fog
    51: (1, 0, 0.20),   # light drizzle
    53: (1, 0, 0.30),   # moderate drizzle
    55: (1, 0, 0.40),   # dense drizzle
    56: (1, 0, 0.35),   # freezing drizzle light
    57: (1, 0, 0.45),   # freezing drizzle dense
    61: (1, 0, 0.40),   # slight rain
    63: (1, 0, 0.55),   # moderate rain
    65: (1, 0, 0.75),   # heavy rain
    66: (1, 0, 0.60),   # freezing rain light
    67: (1, 0, 0.70),   # freezing rain heavy
    71: (1, 0, 0.40),   # slight snow
    73: (1, 0, 0.55),   # moderate snow
    75: (1, 0, 0.75),   # heavy snow
    77: (1, 0, 0.30),   # snow grains
    80: (1, 1, 0.60),   # slight rain showers   ← convective
    81: (1, 1, 0.75),   # moderate rain showers ← convective
    82: (1, 1, 0.90),   # violent rain showers  ← convective
    85: (1, 1, 0.65),   # slight snow showers
    86: (1, 1, 0.80),   # heavy snow showers
    95: (1, 1, 0.85),   # thunderstorm
    96: (1, 1, 0.92),   # thunderstorm + slight hail
    99: (1, 1, 1.00),   # thunderstorm + heavy hail
}


def get_radar_data(lat: Optional[float],
                   lon: Optional[float]) -> Optional[dict]:
    """
    Fetch radar-proxy data from Open-Meteo for the current hour.

    Returns a dict with keys:
        precip_intensity   float   mm/h total precipitation
        rain_mm            float   mm   large-scale rain
        showers_mm         float   mm   convective rain (radar signature)
        weather_code       int     WMO code
        wind_gusts_kmh     float   km/h gusts
        cape               float   J/kg convective energy
        cloud_cover_pct    float   %
        visibility_m       float   meters
        storm_flag         int     1 if convective/thunderstorm WMO code
        wmo_severity       float   0-1 severity from WMO table
        source             str     "open-meteo"

    Returns None on network failure (features.py will use zero vector).
    """
    if lat is None or lon is None:
        return None

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=precipitation,rain,showers,weather_code,"
        "wind_gusts_10m,cape,cloud_cover,visibility"
        "&wind_speed_unit=ms"          # gusts in m/s
        "&precipitation_unit=mm"
        "&forecast_days=1"
    )

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Radar] Open-Meteo request failed: {e}")
        return None

    cur = data.get("current", {})
    if not cur:
        print("[Radar] Empty current block from Open-Meteo")
        return None

    wmo  = int(cur.get("weather_code", 0))
    precip_flag, storm_flag, wmo_sev = _WMO_TABLE.get(wmo, (0, 0, 0.0))

    return {
        "precip_intensity":  float(cur.get("precipitation",   0.0)),
        "rain_mm":           float(cur.get("rain",            0.0)),
        "showers_mm":        float(cur.get("showers",         0.0)),
        "weather_code":      wmo,
        "wind_gusts_kmh":    float(cur.get("wind_gusts_10m",  0.0)) * 3.6,  # m/s→km/h
        "cape":              float(cur.get("cape",            0.0)),
        "cloud_cover_pct":   float(cur.get("cloud_cover",     0.0)),
        "visibility_m":      float(cur.get("visibility",  10000.0)),
        "storm_flag":        storm_flag,
        "wmo_severity":      wmo_sev,
        "source":            "open-meteo",
    }


def extract_radar_features_from_data(radar_data: Optional[dict]):
    """
    Convert the radar dict → normalized 4-d float32 numpy array.

    Dimensions:
      [0] precip_intensity  : total precip / 50 mm/h  (capped at 1.0)
      [1] storm_signal      : convective showers + CAPE proxy
      [2] wmo_severity      : WMO code severity table [0,1]
      [3] coverage_score    : cloud cover % / 100  (area coverage proxy)

    Why 50 mm/h cap? Extreme tropical rainfall rarely exceeds 50 mm/h;
    this normalizes the full range to [0,1] without clipping real events.
    """
    import numpy as np

    if radar_data is None:
        return np.zeros(4, dtype=np.float32)

    precip     = radar_data.get("precip_intensity", 0.0)
    showers    = radar_data.get("showers_mm",       0.0)
    cape       = radar_data.get("cape",             0.0)
    wmo_sev    = radar_data.get("wmo_severity",     0.0)
    cloud      = radar_data.get("cloud_cover_pct",  0.0)

    # storm_signal: showers (convective radar return) + CAPE (instability)
    # CAPE > 2500 J/kg = severe convection; normalize to [0,1]
    storm_signal = min(showers / 10.0 + cape / 2500.0, 1.0)

    return np.array([
        min(precip / 50.0,  1.0),   # precip intensity
        min(storm_signal,   1.0),   # convective / storm signal
        float(wmo_sev),             # WMO severity
        min(cloud / 100.0,  1.0),   # cloud coverage
    ], dtype=np.float32)
