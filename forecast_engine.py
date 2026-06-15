import requests
from datetime import datetime, timezone
from typing import Optional


# ── Rich 6-hour forecast snapshot ────────────────────────────────────────────

def get_forecast_6h(lat: float, lon: float,
                    owm_key: str) -> dict:
    """
    Fetch a rich 6-hour-ahead forecast snapshot from three sources:
      1. Open-Meteo forecast (free, no key) — primary
      2. OpenWeatherMap /forecast            — secondary
      3. Open-Meteo ECMWF 9km               — NWP layer

    Returns a unified dict with all forecast fields needed for both
    feature engineering and future-label generation.
    """
    result = {
        # Core fields
        "temp":              20.0,
        "humidity":          50.0,
        "wind":              0.0,
        "wind_gusts":        0.0,
        "rain":              0.0,
        "precip_prob":       0.0,
        "cape":              0.0,
        "cloud_cover":       0.0,
        "snowfall":          0.0,
        "weather_code":      0,
        # NWP layer
        "nwp_temp":          20.0,
        "nwp_precip":        0.0,
        "nwp_wind":          0.0,
        "nwp_precip_prob":   0.0,
        # Metadata
        "horizon_hours":     6,
        "valid_time":        None,
    }

    # ── Source 1: Open-Meteo main forecast ────────────────────────────────
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m,precipitation,"
            "rain,snowfall,precipitation_probability,weather_code,"
            "wind_speed_10m,wind_gusts_10m,cape,cloud_cover"
            "&wind_speed_unit=ms"
            "&forecast_hours=7"          # 0..6  → index 6 = T+6h
        )
        data   = requests.get(url, timeout=8).json()
        hourly = data.get("hourly", {})
        idx    = 6   # T+6h

        def _h(key, default):
            vals = hourly.get(key, [])
            return float(vals[idx]) if len(vals) > idx else default

        result["temp"]        = _h("temperature_2m",         20.0)
        result["humidity"]    = _h("relative_humidity_2m",   50.0)
        result["rain"]        = _h("rain",                    0.0)
        result["snowfall"]    = _h("snowfall",                0.0)
        result["precip_prob"] = _h("precipitation_probability", 0.0) / 100.0
        result["weather_code"]= int(_h("weather_code",           0))
        result["wind"]        = _h("wind_speed_10m",          0.0)
        result["wind_gusts"]  = _h("wind_gusts_10m",          0.0)
        result["cape"]        = _h("cape",                    0.0)
        result["cloud_cover"] = _h("cloud_cover",             0.0)

        times = hourly.get("time", [])
        if len(times) > idx:
            result["valid_time"] = times[idx]

    except Exception as e:
        print(f"[Forecast] Open-Meteo main failed: {e}")

    # ── Source 2: OWM /forecast (3h buckets) — fills gaps ────────────────
    try:
        url  = (f"https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={owm_key}&units=metric")
        data = requests.get(url, timeout=8).json()
        # bucket index 2 = +6h  (each bucket = 3h)
        fc   = data["list"][2] if len(data.get("list", [])) > 2 else data["list"][0]

        # Only override if Open-Meteo gave us zeros
        if result["rain"] == 0.0:
            result["rain"] = fc.get("rain", {}).get("3h", 0.0)
        if result["wind"] == 0.0:
            result["wind"] = fc["wind"]["speed"]

    except Exception as e:
        print(f"[Forecast] OWM /forecast failed: {e}")

    # ── Source 3: Open-Meteo ECMWF NWP layer ─────────────────────────────
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,precipitation,wind_speed_10m,"
            "precipitation_probability"
            "&models=ecmwf_ifs025"
            "&forecast_hours=7"
        )
        data   = requests.get(url, timeout=8).json()
        hourly = data.get("hourly", {})
        idx    = 6

        def _e(key, default):
            vals = hourly.get(key, [])
            return float(vals[idx]) if len(vals) > idx else default

        result["nwp_temp"]       = _e("temperature_2m",          20.0)
        result["nwp_precip"]     = _e("precipitation",            0.0)
        result["nwp_wind"]       = _e("wind_speed_10m",           0.0)
        result["nwp_precip_prob"]= _e("precipitation_probability",0.0) / 100.0

    except Exception as e:
        print(f"[Forecast] ECMWF NWP failed: {e}")

    return result


# ── Future-label generation ───────────────────────────────────────────────────

# WMO codes that indicate significant rain/storm events
_RAIN_CODES  = {51,53,55,61,63,65,80,81,82,95,96,99}
_SNOW_CODES  = {71,73,75,77,85,86}
_STORM_CODES = {95,96,99}

def generate_future_labels(forecast_6h: dict,
                            city: str,
                            city_thresholds: tuple) -> dict:
    """
    Generate FUTURE labels from the 6-hour forecast snapshot.

    label = 1  →  extreme event EXPECTED in the next 6 hours
    label = 0  →  no extreme event expected

    This is what makes the model a forecaster rather than a
    present-state classifier.

    Thresholds are per-city (climate-zone aware) from database.py.
    """
    heat_thresh, cold_thresh, wind_thresh = city_thresholds

    fc_temp      = forecast_6h.get("temp",        20.0)
    fc_rain      = forecast_6h.get("rain",         0.0)
    fc_snow      = forecast_6h.get("snowfall",     0.0)
    fc_wind      = forecast_6h.get("wind",         0.0)
    fc_gusts     = forecast_6h.get("wind_gusts",   0.0)
    fc_precip_p  = forecast_6h.get("precip_prob",  0.0)
    fc_cape      = forecast_6h.get("cape",         0.0)
    fc_code      = forecast_6h.get("weather_code", 0)
    fc_cloud     = forecast_6h.get("cloud_cover",  0.0)

    # NWP ensemble agreement — both sources agree = higher confidence
    nwp_rain     = forecast_6h.get("nwp_precip",     0.0)
    nwp_wind     = forecast_6h.get("nwp_wind",       0.0)
    nwp_temp     = forecast_6h.get("nwp_temp",       fc_temp)

    # ── Rain ──────────────────────────────────────────────────────────────
    # Heavy rain: >2mm in 6h, OR precip prob >60%, OR storm WMO code,
    # OR strong CAPE (thunderstorm potential), OR NWP agrees
    label_rain = int(
        fc_rain      > 2.0          or
        fc_precip_p  > 0.60         or
        fc_code      in _RAIN_CODES or
        fc_cape      > 1000         or
        (nwp_rain    > 1.0 and fc_precip_p > 0.3)
    )

    # ── Heat ─────────────────────────────────────────────────────────────
    # Extreme heat: forecast temp exceeds city threshold, NWP agrees
    label_heat = int(
        fc_temp  > heat_thresh or
        nwp_temp > heat_thresh - 2   # NWP within 2°C of threshold
    )

    # ── Wind ─────────────────────────────────────────────────────────────
    # Strong wind: sustained wind OR gusts exceed threshold
    # Gusts threshold = wind_thresh * 1.5 (gusts are always higher)
    label_wind = int(
        fc_wind  > wind_thresh      or
        fc_gusts > wind_thresh * 1.5 or
        nwp_wind > wind_thresh      or
        fc_code  in _STORM_CODES
    )

    # ── Snow ─────────────────────────────────────────────────────────────
    label_snow = int(
        fc_snow  > 0.5              or
        fc_temp  < cold_thresh      or
        fc_code  in _SNOW_CODES
    )

    # ── Haze ─────────────────────────────────────────────────────────────
    # Haze is harder to forecast from NWP — use cloud cover + temp inversion proxy
    # (high temp + low wind + high cloud = haze-prone conditions)
    label_haze = int(
        (fc_temp > 25 and fc_wind < 2.0 and fc_cloud > 40) or
        fc_code in {45, 48}    # WMO fog codes
    )

    return {
        "rain": label_rain,
        "heat": label_heat,
        "wind": label_wind,
        "snow": label_snow,
        "haze": label_haze,
        # Store confidence scores for dashboard display
        "_confidence": {
            "rain": min(fc_precip_p + fc_rain/20.0 + fc_cape/3000.0, 1.0),
            "heat": min(max(fc_temp - heat_thresh + 5, 0) / 10.0,    1.0),
            "wind": min(max(fc_wind - wind_thresh + 5, 0) / 15.0,    1.0),
            "snow": min(fc_snow / 5.0 + max(cold_thresh - fc_temp, 0) / 10.0, 1.0),
            "haze": 0.6 if label_haze else 0.1,
        }
    }


# ── Rule-based 6-hour prediction (fallback when no ML model) ─────────────────

def predict_6h_rules(forecast_6h: dict,
                     current_weather: dict,
                     city: str,
                     city_thresholds: tuple) -> dict:
    """
    Rule-based 6-hour-ahead prediction combining:
      - Future labels from forecast (70% weight)
      - Current conditions (30% weight, as leading indicator)

    Returns same format as ML predict_with_models().
    """
    future = generate_future_labels(forecast_6h, city, city_thresholds)
    conf   = future["_confidence"]

    # Current condition signals as leading indicators
    desc  = current_weather["weather"][0]["description"].lower()
    temp  = current_weather["main"]["temp"]
    wind  = current_weather.get("wind", {}).get("speed", 0)
    heat_thresh, cold_thresh, wind_thresh = city_thresholds

    # Blend: if it's already raining now AND forecast says rain → very high conf
    current_signals = {
        "rain": 0.3 if any(w in desc for w in ["rain","storm","shower"]) else 0.0,
        "heat": 0.3 if temp > heat_thresh - 3 else 0.0,
        "wind": 0.3 if wind > wind_thresh - 3  else 0.0,
        "snow": 0.3 if any(w in desc for w in ["snow","sleet"])          else 0.0,
        "haze": 0.3 if "haze" in desc or "smoke" in desc                 else 0.0,
    }

    result = {}
    for event in ["rain", "heat", "wind", "snow", "haze"]:
        base_conf = conf[event]
        cur_sig   = current_signals[event]
        final_conf = min(base_conf * 0.7 + cur_sig, 1.0)
        result[event] = {
            "detected":    bool(future[event] == 1),
            "confidence":  round(final_conf, 2),
            "horizon":     "6h",
        }
    return result
