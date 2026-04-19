from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import feedparser
import os
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from database import init_db, save_analysis, get_history
from features  import (
    pixel_cloud_score, pixel_heat_score,
    extract_weather_features, extract_text_features,
    build_feature_vector,
)
from train_model import load_models, predict_with_models

app = FastAPI()

def save_image(pil_img, prefix):
    if pil_img is None:
        return None
    os.makedirs("data/images", exist_ok=True)
    path = f"data/images/{prefix}_{datetime.utcnow().timestamp()}.png"
    pil_img.save(path)
    return path

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Load ML models once at startup (silently skipped if not trained yet)
_ml_models = load_models()
if _ml_models:
    print(f"[API] ML models loaded for: {list(_ml_models.keys())}")
else:
    print("[API] No trained models found — using rule-based fallback")

# Initialise DB tables
init_db()


# ── Unchanged helpers ─────────────────────────────────────────────────────────

def get_gibs_date():
    return (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

@app.get("/")
def home():
    return {"message": "Multimodal Weather API", "ml_models": list(_ml_models.keys())}

def get_weather(city):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?q={city}&appid={WEATHER_API_KEY}&units=metric")

        res = requests.get(url, timeout=10)

        if res.status_code != 200:
            return {"error": "API failed", "status": res.status_code}

        return res.json()

    except Exception as e:
        return {"error": str(e)}
    
def get_news(city):
    url = (f"https://news.google.com/rss/search"
           f"?q={city}+weather&hl=en-IN&gl=IN&ceid=IN:en")
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:
        articles.append({
            "title": entry.title,
            "image": entry.media_content[0]["url"] if "media_content" in entry else None,
        })
    return {"articles": articles}

def get_coordinates(city):
    url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json"
    headers = {"User-Agent": "multimodal-weather-app"}
    try:
        data = requests.get(url, headers=headers).json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print("Geocoding error:", e)
    return None, None

def _gibs_image(bbox, layer, fmt, style_param=""):
    today     = datetime.utcnow().strftime('%Y-%m-%d')
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
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
    img, _ = _gibs_image(bbox,
                         "VIIRS_SNPP_CorrectedReflectance_TrueColor",
                         "image/jpeg")
    return img

def get_thermal_image(lat, lon):
    bbox = f"{lon-.7},{lat-.7},{lon+.7},{lat+.7}"
    img, _ = _gibs_image(bbox,
                         "MODIS_Terra_Land_Surface_Temp_Day",
                         "image/png",
                         "&STYLES=default")
    return img

def image_analysis(lat,lon, weather_data):
    if lat is None:
        return {"cloud": 0, "heat": 0, "cloud_image": None, "thermal_image": None}

    cloud_img   = get_cloud_image(lat, lon)
    thermal_img = get_thermal_image(lat, lon)

    cloud = pixel_cloud_score(cloud_img)
    heat  = pixel_heat_score(thermal_img)
    temp  = weather_data["main"]["temp"]

    if temp > 30:
        heat = max(heat, 0.4)
    if "storm" in weather_data["weather"][0]["description"].lower():
        cloud = max(cloud, 0.5)

    bbox      = f"{lon-.7},{lat-.7},{lon+.7},{lat+.7}"
    date_str  = get_gibs_date()
    cloud_url = (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        "SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
        "&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor"
        "&FORMAT=image/jpeg&HEIGHT=512&WIDTH=512&SRS=EPSG:4326"
        f"&TIME={date_str}&BBOX={bbox}"
    )
    thermal_url = (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        "SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
        "&LAYERS=MODIS_Terra_Land_Surface_Temp_Day&STYLES=default"
        "&FORMAT=image/png&HEIGHT=512&WIDTH=512&SRS=EPSG:4326"
        f"&TIME={date_str}&BBOX={bbox}"
    )
    cloud_path   = save_image(cloud_img, "cloud")
    thermal_path = save_image(thermal_img, "thermal")

    return {
        "cloud": cloud,
        "heat": heat,
        "cloud_image": cloud_url,
        "thermal_image": thermal_url,

        "cloud_path": cloud_path,
        "thermal_path": thermal_path,

        "_cloud_pil": cloud_img,
        "_thermal_pil": thermal_img,
    }

def news_image_analysis(articles):
    rain_score = 0
    count = 0
    for a in articles:
        title     = a["title"].lower()
        img_url   = a["image"]
        text_sig  = any(k in title for k in ["rain", "storm"])
        image_sig = 0
        if img_url:
            try:
                img = Image.open(BytesIO(requests.get(img_url, timeout=5).content))
                arr = np.array(img) / 255.0
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
                # Only suppress rain signal if image actually loaded AND shows clear sky
                if event == "rain" and img_res.get("cloud_image") is not None and cloud < 0.4:
                    continue

    for k in result:
        result[k] /= total
    return result

def get_forecast(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        data = requests.get(url, timeout=10).json()

        # take next time step
        f = data["list"][0]

        return {
            "temp": f["main"]["temp"],
            "humidity": f["main"]["humidity"],
            "wind": f["wind"]["speed"],
            "rain": f.get("rain", {}).get("3h", 0.0)
        }

    except Exception:
        return {
            "temp": 20.0,
            "humidity": 50.0,
            "wind": 0.0,
            "rain": 0.0
        }

def final_decision(weather_data, text_res, img_res):
    weather_desc = weather_data["weather"][0]["description"].lower()
    temp         = weather_data["main"]["temp"]
    cloud        = img_res.get("cloud", 0)
    heat_img     = img_res.get("heat", 0)
    wind_speed   = weather_data.get("wind", {}).get("speed", 0)

    weather_rain = any(w in weather_desc for w in ["rain","storm","thunderstorm"])
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

    weather_haze = "haze" in weather_desc or "smoke" in weather_desc
    weather_snow = any(w in weather_desc for w in ["snow","sleet","blizzard"])
    text_snow    = text_res["snow"] > 0.3
    image_snow   = temp < 2 and cloud > 0.4
    snow_conf    = round(.6*int(weather_snow)+.3*int(text_snow)+.1*int(image_snow), 2)

    return {
        "rain": {"detected": bool(weather_rain or (text_rain and image_rain)),
                 "confidence": rain_conf,
                 "reason": ("Rain confirmed by weather" if weather_rain
                            else "Rain inferred from text+cloud" if (text_rain and image_rain)
                            else "Weak signal"),
                 "sources": {"weather": bool(weather_rain), "text": bool(text_rain), "image": bool(image_rain)}},
        "heat": {"detected": bool(weather_heat or (text_heat and image_heat)),
                 "confidence": heat_conf,
                 "sources": {"weather": bool(weather_heat), "text": bool(text_heat), "image": bool(image_heat)}},
        "wind": {"detected": bool(weather_wind and text_wind),
                 "confidence": wind_conf,
                 "sources": {"weather": bool(weather_wind), "text": bool(text_wind)}},
        "haze": {"detected": bool(weather_haze),
                 "confidence": 0.8 if weather_haze else 0.2,
                 "sources": {"weather": bool(weather_haze)}},
        "snow": {"detected": bool(weather_snow or (text_snow and image_snow)),
                 "confidence": snow_conf,
                 "sources": {"weather": bool(weather_snow), "text": bool(text_snow), "image": bool(image_snow)}},
    }

def auto_generate_labels(weather_data, text_res, img_res, forecast_data):
    desc = weather_data["weather"][0]["description"].lower()
    temp = weather_data["main"]["temp"]
    wind = weather_data.get("wind", {}).get("speed", 0)
    cloud = img_res.get("cloud", 0)

    return {
        "rain": int(cloud > 0.5 or forecast_data["rain"] > 0.2),
        "heat": int(temp > 30 or text_res["heat"] > 0.3),
        "wind": int(wind > 5 or text_res["wind"] > 0.3),
        "snow": int(temp < 5 or text_res["snow"] > 0.3),
        "haze": int("haze" in desc or "smoke" in desc)
    }
    

# ── Main analysis endpoint ────────────────────────────────────────────────────

@app.get("/analyze/{city}")
def analyze(city: str):
    weather_data = get_weather(city)
    if "weather" not in weather_data:
        return {"error": "Could not retrieve weather data", "details": weather_data}

    news_data      = get_news(city)
    articles       = news_data["articles"]
    lat, lon = get_coordinates(city)
    img_res  = image_analysis(lat, lon, weather_data)
    news_img_score = news_image_analysis(articles)
    text_res       = text_analysis(articles, city, img_res)
    text_res["rain"] = max(text_res["rain"], news_img_score)
    
    raw_titles = [a["title"] for a in articles]
    
    if lat is not None and lon is not None:
        forecast_data = get_forecast(lat, lon)
    
    else:
        forecast_data = {
            "temp": 20.0,
            "humidity": 50.0,
            "wind": 0.0,
            "rain": 0.0
        }

    # ── Step 7: ALWAYS build features ────────────────────────────────────
    features = build_feature_vector(
        weather_data,
        img_res.get("_cloud_pil"),
        img_res.get("_thermal_pil"),
        raw_titles,
        lat,
        lon,
        forecast_data  
    )
    # save feature vector ALWAYS
    feature_vector = features.flatten().tolist()

    # ── Use ML model if available ────────────────────────────────────
    # Always compute rule-based fallback
    rule_analysis = final_decision(weather_data, text_res, img_res)

    if _ml_models:
        features_reshaped = features.reshape(1, -1)
        ml_analysis = predict_with_models(_ml_models, features_reshaped)

        # Merge ML + fallback
        analysis = {}
        for event in ["rain", "heat", "wind", "snow", "haze"]:
            if event in ml_analysis:
                analysis[event] = ml_analysis[event]
            else:
                analysis[event] = rule_analysis[event]
    else:
        analysis = rule_analysis
    
    # ── Step 3: persist to database ───────────────────────────────────────────
    # Strip PIL objects before saving
    img_res_clean = {k: v for k, v in img_res.items() if not k.startswith("_")}
    record_id = save_analysis(
        city,
        weather_data,
        img_res_clean,
        text_res,
        analysis,
        raw_titles,
        lat,
        lon,
        feature_vector  
    )
    labels = auto_generate_labels(weather_data, text_res, img_res, forecast_data)

    # Save labels automatically
    try:
        from database import get_connection
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO labeled_events
            (record_id, city, timestamp,
            label_rain, label_heat, label_wind, label_snow, label_haze)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            record_id,
            city,
            datetime.utcnow(),
            labels["rain"],
            labels["heat"],
            labels["wind"],
            labels["snow"],
            labels["haze"]
        ))

        conn.commit() 

        cursor.close()
        conn.close()

    except Exception as e:
        print("[AutoLabel Error]", e)

    return {
        "city":          city,
        "weather":       weather_data["weather"][0]["description"],
        "temperature":   weather_data["main"]["temp"],
        "text_analysis": text_res,
        "image_analysis": img_res_clean,
        "analysis":      analysis,
        "model":         "ml" if _ml_models else "rules",
    }


# ── New endpoints ─────────────────────────────────────────────────────────────

@app.get("/history/{city}")
def history(city: str, limit: int = 50):
    """Return past analysis records for a city."""
    return {"city": city, "records": get_history(city, limit)}


@app.get("/evaluate")
def evaluate():
    """Return saved evaluation metrics from last training run."""
    import json
    path = "models/eval_results.json"
    if not os.path.exists(path):
        return {"error": "No evaluation results yet. Run: python train_model.py"}
    with open(path) as f:
        return json.load(f)

@app.post("/reload-models")
def reload_models():
    global _ml_models
    _ml_models = load_models()
    return {"loaded": list(_ml_models.keys())}

from fastapi import Query
from typing import List

@app.get("/compare")
def compare(cities: List[str] = Query(...)):
    """
    Compare multiple cities at once.
    """
    results = {}
    for city in cities[:5]:   # cap at 5 to avoid timeout
        data = analyze(city)
        if "error" not in data:
            results[city] = {
                "temperature": data["temperature"],
                "weather":     data["weather"],
                "analysis":    {
                    k: {"detected": v["detected"], "confidence": v["confidence"]}
                    for k, v in data["analysis"].items()
                }
            }
        else:
            results[city] = {"error": data["error"]}
    return results