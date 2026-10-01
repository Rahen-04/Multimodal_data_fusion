"""
main.py  —  FastAPI backend
Implements the full Step 1 → Step 9 pipeline from the project spec.

Data sources:
  • OpenWeatherMap  : current weather + 5-day forecast
  • NASA GIBS       : VIIRS cloud + MODIS thermal satellite imagery
  • Google News RSS : textual weather reports
  • Nominatim       : geocoding (lat/lon)
  • Open-Elevation  : SRTM elevation data (geospatial)
  • Nominatim rev.  : land-use category (geospatial)
  • Open-Meteo + OWM : Rich 6-hour-ahead forecast (rain,wind,CAPE,gusts,snow)
  • Open-Meteo ECMWF : NWP 9km ensemble layer
  • Open-Meteo radar : precipitation + storm signal (real, free, no key)
  • forecast_engine  : future label generation + rule-based 6h predictor
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from typing import List
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import feedparser
import os, re, json
from datetime import datetime, timedelta, timezone

from database import (init_db, save_analysis, get_history,
                      get_connection, save_labels, get_nearby,
                      get_city_thresholds)
from radar import get_radar_data
from forecast_engine import (get_forecast_6h, generate_future_labels,
                             predict_6h_rules)
from features  import (
    pixel_cloud_score, pixel_heat_score,
    extract_weather_features, extract_text_features,
    build_feature_vector,
)
from train_model import (load_models, predict_with_models,
                          load_lstm_models, predict_with_lstm)
from attention_fusion import load_attention_models, predict_attention

app = FastAPI(title="WeatherFusion API", version="2.0")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
if not WEATHER_API_KEY:
    raise ValueError("Missing WEATHER_API_KEY in .env")

init_db()

_ml_models    = load_models()
_lstm_models  = load_lstm_models()
_attn_models  = load_attention_models()

loaded = []
if _ml_models:   loaded.append(f"sklearn({list(_ml_models.keys())})")
if _lstm_models: loaded.append(f"lstm({list(_lstm_models.keys())})")
if _attn_models: loaded.append(f"attention({list(_attn_models.keys())})")
if loaded:
    print(f"[API] Models loaded: {', '.join(loaded)}")
else:
    print("[API] No trained models — using rule-based fallback")


# ── Utility ───────────────────────────────────────────────────────────────────

def save_image(pil_img, prefix):
    if pil_img is None:
        return None
    os.makedirs("data/images", exist_ok=True)
    ts = datetime.now(timezone.utc).timestamp()
    path = f"data/images/{prefix}_{ts}.png"
    pil_img.save(path)
    return path


# ── Step 1: Data collection helpers ──────────────────────────────────────────

def get_weather(city):
    """OpenWeatherMap current weather."""
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city.strip(), "appid": WEATHER_API_KEY, "units": "metric"}
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return {"error": "OWM API failed", "status": res.status_code}
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def get_forecast(lat, lon):
    """
    Rich 6-hour-ahead forecast from forecast_engine (3 fused sources).
    Returns unified dict used for both features and future-label generation.
    """
    if lat is None or lon is None:
        return {"temp": 20.0, "humidity": 50.0, "wind": 0.0, "rain": 0.0,
                "wind_gusts": 0.0, "precip_prob": 0.0, "cape": 0.0,
                "cloud_cover": 0.0, "snowfall": 0.0, "nwp_precip_prob": 0.0}
    return get_forecast_6h(lat, lon, owm_key=WEATHER_API_KEY)


def get_news(city):
    """Google News RSS → article titles + optional image URLs.
    Returns empty list on network failure (Google throttles RSS requests).
    """
    try:
        url  = (f"https://news.google.com/rss/search"
                f"?q={city}+weather&hl=en-IN&gl=IN&ceid=IN:en")
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"},
                                handlers=[])
        articles = []
        for entry in getattr(feed, "entries", [])[:5]:
            try:
                articles.append({
                    "title": entry.get("title", ""),
                    "image": (entry.media_content[0]["url"]
                              if "media_content" in entry else None),
                })
            except Exception:
                pass
        return {"articles": articles}
    except Exception as e:
        print(f"[News] {city} RSS failed (throttled/disconnected): {e}")
        return {"articles": []}


_COORDS_CACHE = {}
_ELEV_CACHE   = {}
_LAND_CACHE   = {}

def get_coordinates(city):
    """Nominatim geocoding with caching."""
    c_key = city.strip().lower()
    if c_key in _COORDS_CACHE:
        return _COORDS_CACHE[c_key]
    url     = f"https://nominatim.openstreetmap.org/search?q={city}&format=json"
    headers = {"User-Agent": "WeatherFusion-Intelligence/2.0 (contact: support@weatherfusion.local)"}
    try:
        data = requests.get(url, headers=headers, timeout=5).json()
        if data:
            coords = (float(data[0]["lat"]), float(data[0]["lon"]))
            _COORDS_CACHE[c_key] = coords
            return coords
    except Exception as e:
        print(f"[Geocoding] {e}")
    return None, None


def get_elevation(lat, lon):
    """SRTM elevation via open-elevation.com API with caching."""
    if lat is None or lon is None:
        return 0.0
    key = (round(lat, 2), round(lon, 2))
    if key in _ELEV_CACHE:
        return _ELEV_CACHE[key]
    try:
        url  = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        data = requests.get(url, timeout=5).json()
        elev = float(data["results"][0]["elevation"])
        _ELEV_CACHE[key] = elev
        return elev
    except Exception:
        return 0.0


def get_land_use(lat, lon):
    """Approximate land-use from Nominatim reverse geocoding with caching."""
    if lat is None or lon is None:
        return "unknown"
    key = (round(lat, 2), round(lon, 2))
    if key in _LAND_CACHE:
        return _LAND_CACHE[key]
    try:
        url     = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "WeatherFusion-Intelligence/2.0 (contact: support@weatherfusion.local)"}
        data    = requests.get(url, headers=headers, timeout=5).json()
        addr    = data.get("address", {})
        land_use = "unknown"
        if any(k in addr for k in ("city", "town", "suburb", "quarter")):
            land_use = "urban"
        elif "forest" in str(addr).lower() or "wood" in str(addr).lower():
            land_use = "forest"
        elif any(k in addr for k in ("farm", "village")):
            land_use = "cropland"
        elif "water" in str(addr).lower() or "lake" in str(addr).lower():
            land_use = "water"
        _LAND_CACHE[key] = land_use
        return land_use
    except Exception:
        return "unknown"


def get_nwp_data(forecast_6h: dict):
    """
    Extract NWP features from the already-fetched forecast_6h snapshot.
    NWP data (ECMWF layer) is now fetched inside get_forecast_6h().
    """
    if not forecast_6h:
        return None
    return {
        "predicted_temp":   forecast_6h.get("nwp_temp",       20.0),
        "predicted_precip": forecast_6h.get("nwp_precip",      0.0),
        "predicted_wind":   forecast_6h.get("nwp_wind",        0.0),
        "model_confidence": forecast_6h.get("nwp_precip_prob", 0.5),
    }


# ── Satellite imagery (NASA GIBS WMS) ────────────────────────────────────────

def _gibs_image(bbox, layer, fmt, style_param=""):
    now_utc   = datetime.now(timezone.utc)
    today     = now_utc.strftime('%Y-%m-%d')
    yesterday = (now_utc - timedelta(days=1)).strftime('%Y-%m-%d')
    base = (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        "SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
        f"&LAYERS={layer}{style_param}&FORMAT={fmt}"
        "&HEIGHT=512&WIDTH=512&SRS=EPSG:4326"
        f"&BBOX={bbox}"
    )
    for date in [today, yesterday]:
        try:
            res = requests.get(f"{base}&TIME={date}", timeout=10)
            if res.status_code == 200 and b"ServiceException" not in res.content:
                return Image.open(BytesIO(res.content)), f"{base}&TIME={date}"
        except Exception:
            pass
    return None, None


def get_cloud_image(lat, lon):
    bbox = f"{lon-.7},{lat-.7},{lon+.7},{lat+.7}"
    return _gibs_image(bbox, "VIIRS_SNPP_CorrectedReflectance_TrueColor",
                       "image/jpeg")


def get_thermal_image(lat, lon):
    bbox = f"{lon-.7},{lat-.7},{lon+.7},{lat+.7}"
    return _gibs_image(bbox, "MODIS_Terra_Land_Surface_Temp_Day",
                       "image/png", "&STYLES=default")


# ── Step 2: Pre-processing / scoring helpers ──────────────────────────────────

def image_analysis(lat, lon, weather_data):
    if lat is None:
        return {"cloud": 0, "heat": 0,
                "cloud_image": None, "thermal_image": None}

    cloud_img,   cloud_url   = get_cloud_image(lat, lon)
    thermal_img, thermal_url = get_thermal_image(lat, lon)

    cloud = pixel_cloud_score(cloud_img)
    heat  = pixel_heat_score(thermal_img)
    temp  = weather_data["main"]["temp"]

    if temp > 30:
        heat = heat + 0.1
    if "storm" in weather_data["weather"][0]["description"].lower():
        cloud = max(cloud, 0.5)

    # brightness sanity-check
    if cloud_img is not None:
        arr = np.array(cloud_img)
        if arr.mean() < 5:
            print(f"[Warning] Cloud image black (mean={arr.mean():.2f})")
            cloud_url = None

    cloud_path   = save_image(cloud_img, "cloud")
    thermal_path = save_image(thermal_img, "thermal")

    return {
        "cloud":         cloud,
        "heat":          heat,
        "cloud_image":   cloud_url,
        "thermal_image": thermal_url,
        "cloud_path":    cloud_path,
        "thermal_path":  thermal_path,
        "_cloud_pil":    cloud_img,
        "_thermal_pil":  thermal_img,
    }


def news_image_analysis(articles):
    rain_score, count = 0, 0
    for a in articles:
        title    = a["title"].lower()
        img_url  = a["image"]
        text_sig = any(k in title for k in ["rain", "storm"])
        image_sig = 0
        if img_url:
            try:
                img       = Image.open(BytesIO(requests.get(img_url,
                                                             timeout=5).content))
                arr       = np.array(img) / 255.0
                image_sig = np.sum(arr > 0.7) / arr.size > 0.4
            except Exception:
                pass
        if text_sig and image_sig:
            rain_score += 1
        count += 1
    return rain_score / count if count > 0 else 0


def text_analysis(articles, city, img_res):
    EVENT_KEYWORDS = {
        "rain": [r"\brain\b", r"\bflood", r"\bstorm", r"\bdownpour\b",
                 r"\bshowers\b", r"\bmonsoon", r"\bdeluge\b", r"\bwaterlogging"],
        "heat": [r"\bheatwave", r"\bextreme heat\b", r"\bhottest\b",
                 r"\bscorching\b", r"\bsunstroke\b", r"\bsweltering\b"],
        "wind": [r"\bwind", r"\bcyclone\b", r"\bstorm", r"\bgale\b",
                 r"\bhurricane\b", r"\btyphoon\b"],
        "snow": [r"\bsnow", r"\bblizzard\b", r"\bwinter\b",
                 r"\bavalanche\b", r"\bfreezing\b"],
        "haze": [r"\bsmog\b", r"\bpollution\b", r"\baqi\b",
                 r"\btoxic air\b", r"\bhaze\b"],
    }
    result = {k: 0 for k in EVENT_KEYWORDS}
    total  = len(articles)
    cloud  = img_res.get("cloud", 0)
    if total == 0:
        return result

    for a in articles:
        title = a["title"].lower()
        for event, patterns in EVENT_KEYWORDS.items():
            if any(re.search(p, title) for p in patterns):
                if event == "rain" \
                   and img_res.get("cloud_image") is not None \
                   and cloud < 0.4:
                    continue
                result[event] += 1

    for k in result:
        result[k] /= total
    return result


# ── Rule-based fusion (fallback) ──────────────────────────────────────────────

def final_decision(weather_data, text_res, img_res):
    desc       = weather_data["weather"][0]["description"].lower()
    temp       = weather_data["main"]["temp"]
    cloud      = img_res.get("cloud", 0)
    heat_img   = img_res.get("heat", 0)
    wind_speed = weather_data.get("wind", {}).get("speed", 0)

    weather_rain = any(w in desc for w in ["rain", "storm", "thunderstorm"])
    text_rain    = text_res["rain"] > 0.4
    image_rain   = cloud > 0.6
    rain_conf    = round(.6*int(weather_rain)+.3*int(text_rain)+.1*int(image_rain), 2)

    weather_heat = temp > 35
    text_heat    = text_res["heat"] > 0.3
    image_heat   = heat_img > 0.15
    heat_conf    = round(.6*int(weather_heat)+.3*int(text_heat)+.1*int(image_heat), 2)

    weather_wind = wind_speed > 10
    text_wind    = text_res["wind"] > 0.3
    wind_conf    = round(.7*int(weather_wind)+.3*int(text_wind), 2)

    weather_haze = "haze" in desc or "smoke" in desc
    weather_snow = any(w in desc for w in ["snow", "sleet", "blizzard"])
    text_snow    = text_res["snow"] > 0.3
    image_snow   = temp < 2 and cloud > 0.4
    snow_conf    = round(.6*int(weather_snow)+.3*int(text_snow)+.1*int(image_snow), 2)

    return {
        "rain": {"detected": bool(weather_rain or (text_rain and image_rain)),
                 "confidence": rain_conf,
                 "sources": {"weather": bool(weather_rain),
                             "text": bool(text_rain), "image": bool(image_rain)}},
        "heat": {"detected": bool(weather_heat or (text_heat and image_heat)),
                 "confidence": heat_conf,
                 "sources": {"weather": bool(weather_heat),
                             "text": bool(text_heat), "image": bool(image_heat)}},
        "wind": {"detected": bool(weather_wind and text_wind),
                 "confidence": wind_conf,
                 "sources": {"weather": bool(weather_wind),
                             "text": bool(text_wind)}},
        "haze": {"detected": bool(weather_haze),
                 "confidence": 0.8 if weather_haze else 0.2,
                 "sources": {"weather": bool(weather_haze)}},
        "snow": {"detected": bool(weather_snow or (text_snow and image_snow)),
                 "confidence": snow_conf,
                 "sources": {"weather": bool(weather_snow),
                             "text": bool(text_snow), "image": bool(image_snow)}},
    }


def auto_generate_labels(weather_data, text_res, img_res, forecast_data, city="", radar_data=None):
    """
    Climate-aware auto-labeling using all available data sources including
    real radar (Open-Meteo) and per-city thresholds.
    """
    desc  = weather_data["weather"][0]["description"].lower()
    temp  = weather_data["main"]["temp"]
    wind  = weather_data.get("wind", {}).get("speed", 0)
    cloud = img_res.get("cloud", 0)

    heat_thresh, cold_thresh, wind_thresh = get_city_thresholds(city)

    # radar signals for rain detection
    radar_rain = 0
    if radar_data:
        radar_rain = int(
            radar_data.get("precip_intensity", 0) > 0.5 or
            radar_data.get("showers_mm",       0) > 0.1 or
            radar_data.get("storm_flag",        0) == 1 or
            radar_data.get("wmo_severity",      0) > 0.4
        )

    return {
        "rain": int(cloud > 0.5 or forecast_data["rain"] > 0.2 or radar_rain),
        "heat": int(temp > heat_thresh or text_res["heat"] > 0.3),
        "wind": int(wind > wind_thresh or text_res["wind"] > 0.3),
        "snow": int(temp < cold_thresh or text_res["snow"] > 0.3),
        "haze": int("haze" in desc or "smoke" in desc),
    }


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message":    "WeatherFusion Multimodal API v2",
        "ml_models":  list(_ml_models.keys()),
    }


@app.get("/analyze/{city}")
def analyze(city: str):
    city = city.strip().title()
    # Step 1 – collect all modalities
    weather_data = get_weather(city)
    if "weather" not in weather_data:
        return {"error": "Could not retrieve weather data",
                "details": weather_data}

    news_data      = get_news(city)
    articles       = news_data["articles"]
    lat, lon       = get_coordinates(city)

    # Geospatial data (SRTM elevation + land-use)
    elevation  = get_elevation(lat, lon)
    land_use   = get_land_use(lat, lon)

    # Step 1b – fetch 6-hour forecast + radar (all modalities)
    forecast_data = get_forecast(lat, lon)   # rich 6h snapshot from forecast_engine
    nwp_data      = get_nwp_data(forecast_data)
    radar_data    = get_radar_data(lat, lon)

    # Step 2 – pre-process images
    img_res          = image_analysis(lat, lon, weather_data)
    news_img_score   = news_image_analysis(articles)
    text_res         = text_analysis(articles, city, img_res)
    text_res["rain"] = max(text_res["rain"], news_img_score)

    raw_titles = [a["title"] for a in articles]

    # Step 5 – build full multimodal feature vector
    features = build_feature_vector(
        weather_data,
        img_res.get("_cloud_pil"),
        img_res.get("_thermal_pil"),
        raw_titles,
        lat, lon,
        forecast_data,
        elevation  = elevation,
        land_use   = land_use,
        radar_data = radar_data,
        nwp_data   = nwp_data,
    )
    feature_vector = features.flatten().tolist()

    # Step 6 / 7 – 6-hour-ahead prediction using all model types
    city_thresh   = get_city_thresholds(city)
    future_labels = generate_future_labels(forecast_data, city, city_thresh)
    rule_analysis = predict_6h_rules(forecast_data, weather_data,
                                     city, city_thresh)

    features_reshaped = features.reshape(1, -1)

    # Sklearn (RF / GBT / SVM / late-fusion)
    ml_preds   = predict_with_models(_ml_models, features_reshaped) if _ml_models else {}
    # LSTM sequence model
    lstm_preds = predict_with_lstm(_lstm_models, features_reshaped)  if _lstm_models else {}
    # CNN+LSTM+Attention hybrid
    attn_preds = predict_attention(_attn_models, features_reshaped)  if _attn_models else {}

    any_deep = bool(ml_preds or lstm_preds or attn_preds)

    analysis = {}
    for event in ["rain", "heat", "wind", "snow", "haze"]:
        rule = rule_analysis[event]
        confs, detected = [], []

        if event in ml_preds:
            confs.append(("sklearn", 0.40, ml_preds[event]["confidence"]))
            detected.append(ml_preds[event]["detected"])
        if event in lstm_preds:
            confs.append(("lstm",    0.25, lstm_preds[event]["confidence"]))
            detected.append(lstm_preds[event]["detected"])
        if event in attn_preds:
            confs.append(("attn",    0.25, attn_preds[event]["confidence"]))
            detected.append(attn_preds[event]["detected"])

        # Rule-based always contributes
        confs.append(("rules", 0.10 if any_deep else 1.0,
                      rule["confidence"]))
        detected.append(rule["detected"])

        # Normalize weights and compute blended confidence
        total_w   = sum(w for _, w, _ in confs)
        blended   = sum(w/total_w * c for _, w, c in confs)

        # Attention weights for interpretability (from attn model)
        attn_w = (attn_preds.get(event, {}).get("attn_weights", {})
                  if attn_preds else {})

        analysis[event] = {
            "detected":     bool(blended >= 0.50),
            "confidence":   round(blended, 3),
            "horizon":      "6h",
            "model_blend":  {n: round(c, 3) for n, _, c in confs},
            "attn_weights": attn_w,
        }

    model_type = ("sklearn+lstm+attention" if (ml_preds and lstm_preds and attn_preds)
                  else "sklearn+lstm"       if (ml_preds and lstm_preds)
                  else "sklearn+attention"  if (ml_preds and attn_preds)
                  else "sklearn"            if ml_preds
                  else "rules_6h")

    # Step 3 – persist to database
    img_res_clean = {k: v for k, v in img_res.items()
                     if not k.startswith("_")}
    record_id = save_analysis(
        city, weather_data, img_res_clean, text_res, analysis,
        raw_titles, lat, lon, feature_vector,
        forecast_data = forecast_data,
        nwp_data      = nwp_data,
        radar_data    = radar_data,
        elevation     = elevation,
        land_use      = land_use,
    )

    # Auto-label for training data accumulation
    # Labels are FUTURE-aligned: based on 6h forecast, not current state
    labels = {k: v for k, v in future_labels.items() if not k.startswith("_")}
    try:
        save_labels(record_id, city, datetime.now(timezone.utc), labels)
    except Exception as e:
        print("[AutoLabel Error]", e)

    return {
        "city":           city,
        "weather":        weather_data["weather"][0]["description"],
        "temperature":    weather_data["main"]["temp"],
        "prediction_horizon": "6h",
        "forecast_summary": {
            "temp_6h":       forecast_data.get("temp"),
            "rain_6h_mm":    forecast_data.get("rain"),
            "wind_6h_ms":    forecast_data.get("wind"),
            "precip_prob":   forecast_data.get("precip_prob"),
            "cape":          forecast_data.get("cape"),
            "valid_time":    forecast_data.get("valid_time"),
        },
        "text_analysis":  text_res,
        "image_analysis": img_res_clean,
        "analysis":       analysis,
        "model":          model_type,
        "geospatial":     {"elevation": elevation, "land_use": land_use,
                           "lat": lat, "lon": lon},
    }


@app.get("/history/{city}")
def history(city: str, limit: int = 50):
    return {"city": city, "records": get_history(city, limit)}


@app.get("/nearby")
def nearby(lat: float, lon: float, radius: float = 1.0):
    """Return records from nearby locations (spatial query)."""
    return {"records": get_nearby(lat, lon, radius_deg=radius)}


@app.get("/evaluate")
def evaluate():
    path = "models/eval_results.json"
    if not os.path.exists(path):
        return {"error": "No evaluation results yet. Run: python train_model.py"}
    with open(path) as f:
        return json.load(f)


@app.get("/shap/{event}")
def shap_importance(event: str):
    """Return SHAP feature importances for a given event."""
    path = f"models/shap_{event}.json"
    if not os.path.exists(path):
        return {"error": f"No SHAP data for {event}. Run training first."}
    with open(path) as f:
        return json.load(f)


@app.post("/reload-models")
def reload_models():
    global _ml_models, _lstm_models, _attn_models
    _ml_models   = load_models()
    _lstm_models = load_lstm_models()
    _attn_models = load_attention_models()
    return {
        "sklearn":   list(_ml_models.keys()),
        "lstm":      list(_lstm_models.keys()),
        "attention": list(_attn_models.keys()),
    }


@app.get("/compare")
def compare(cities: List[str] = Query(...)):
    results = {}
    for city in cities[:5]:
        try:
            data = analyze(city)
            if "error" not in data:
                results[city] = {
                    "temperature": data["temperature"],
                    "weather":     data["weather"],
                    "geospatial":  data.get("geospatial", {}),
                    "analysis":    {
                        k: {"detected":   v["detected"],
                            "confidence": v["confidence"]}
                        for k, v in data["analysis"].items()
                    },
                }
            else:
                results[city] = {"error": data["error"]}
        except Exception as e:
            results[city] = {"error": str(e)}
    return results
