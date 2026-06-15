"""
features.py  —  Multimodal Feature Engineering
Modalities:
  1. Satellite imagery   → ResNet18 CNN embeddings (512-d each, cloud + thermal)
  2. Weather numerics    → OpenWeatherMap fields (6-d)
  3. Forecast numerics   → next time-step forecast (4-d)
  4. Textual data        → sentence-transformers mean-pool (384-d)
  5. Geospatial          → lat/lon + elevation + land-use (5-d)
  6. Radar               → precipitation intensity proxy (4-d)
  7. NWP model output    → GFS/ECMWF placeholder features (4-d)

Total vector size: 512 + 512 + 6 + 4 + 384 + 5 + 4 + 4 = 1431-d
"""

import numpy as np
from PIL import Image
from typing import List, Optional

# ── lazy-loaded global models ─────────────────────────────────────────────────
_resnet     = None
_transform  = None
_text_model = None


def _load_vision_model():
    global _resnet, _transform
    if _resnet is not None:
        return
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    # strip classification head → 512-d feature vector
    _resnet = torch.nn.Sequential(*list(model.children())[:-1])
    _transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    print("[Features] ResNet18 loaded (IMAGENET1K_V1)")


def _load_text_model():
    global _text_model
    if _text_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[Features] SentenceTransformer loaded (all-MiniLM-L6-v2)")


# ── 1. Image features (CNN) ───────────────────────────────────────────────────

def extract_image_features(pil_image: Optional[Image.Image]) -> np.ndarray:
    """ResNet18 → 512-d float32 vector. Falls back to zeros."""
    if pil_image is None:
        return np.zeros(512, dtype=np.float32)
    try:
        import torch
        _load_vision_model()
        img    = pil_image.convert("RGB")
        tensor = _transform(img).unsqueeze(0)       # (1, 3, 224, 224)
        with torch.no_grad():
            feat = _resnet(tensor)                  # (1, 512, 1, 1)
        return feat.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print(f"[Features] CNN fallback: {e}")
        return np.zeros(512, dtype=np.float32)


def pixel_cloud_score(pil_image: Optional[Image.Image]) -> float:
    """Heuristic: fraction of bright-white pixels → proxy cloud cover."""
    if pil_image is None:
        return 0.0
    img = pil_image.resize((224, 224))
    arr = np.array(img) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cloud_mask = np.logical_and.reduce((
        r > 0.7, g > 0.7, b > 0.7,
        np.abs(r - g) < 0.1, np.abs(r - b) < 0.1,
    ))
    return round(float(np.sum(cloud_mask) / cloud_mask.size), 4)


def pixel_heat_score(img: Optional[Image.Image]) -> float:
    """Heuristic: red-dominant pixels → proxy surface heat."""
    if img is None:
        return 0.0
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    heat_mask = (r > 0.55) & (r > g + 0.15) & (r > b + 0.2)
    return round(float(np.sum(heat_mask) / heat_mask.size), 4)


# ── 2. Weather numeric features ───────────────────────────────────────────────

def extract_weather_features(weather_data: dict) -> np.ndarray:
    """
    OpenWeatherMap current-weather JSON → 6-d vector.
    [temp, humidity, pressure, wind_speed, cloud_pct, weather_id]
    """
    main   = weather_data.get("main", {})
    wind   = weather_data.get("wind", {})
    clouds = weather_data.get("clouds", {})
    wid    = (weather_data["weather"][0]["id"]
              if weather_data.get("weather") else 800)

    return np.array([
        main.get("temp",     20.0),
        main.get("humidity", 50.0),
        main.get("pressure", 1013.0),
        wind.get("speed",    0.0),
        clouds.get("all",    0.0),
        float(wid),
    ], dtype=np.float32)


# ── 3. Forecast features (6-hour ahead) ──────────────────────────────────────

def extract_forecast_features(forecast_data: dict) -> np.ndarray:
    """
    Rich 6-hour-ahead forecast snapshot → 10-d vector.
    Uses unified forecast dict from forecast_engine.get_forecast_6h().

    [temp/50, humidity/100, wind/30, wind_gusts/50, rain/20,
     precip_prob, cape/3000, cloud_cover/100, snowfall/10, nwp_precip_prob]
    All normalized to approx [0,1].
    Backwards-compatible with older 4-field dicts.
    """
    return np.array([
        forecast_data.get("temp",            20.0) / 50.0,
        forecast_data.get("humidity",        50.0) / 100.0,
        forecast_data.get("wind",             0.0) / 30.0,
        forecast_data.get("wind_gusts",       0.0) / 50.0,
        forecast_data.get("rain",             0.0) / 20.0,
        forecast_data.get("precip_prob",      0.0),
        forecast_data.get("cape",             0.0) / 3000.0,
        forecast_data.get("cloud_cover",      0.0) / 100.0,
        forecast_data.get("snowfall",         0.0) / 10.0,
        forecast_data.get("nwp_precip_prob",  0.0),
    ], dtype=np.float32)


# ── 4. Text features (transformer embeddings) ────────────────────────────────

def extract_text_features(texts: List[str]) -> np.ndarray:
    """
    Encode article titles with sentence-transformers → mean-pooled 384-d.
    Falls back to zeros if library unavailable.
    """
    if not texts:
        return np.zeros(384, dtype=np.float32)
    try:
        _load_text_model()
        embeddings = _text_model.encode(texts, show_progress_bar=False)
        return embeddings.mean(axis=0).astype(np.float32)     # (384,)
    except Exception as e:
        print(f"[Features] Text embed fallback: {e}")
        return np.zeros(384, dtype=np.float32)


# ── 5. Geospatial features ────────────────────────────────────────────────────

def extract_geo_features(lat: Optional[float],
                         lon: Optional[float],
                         elevation: float = 0.0,
                         land_use: str = "unknown") -> np.ndarray:
    """
    Spatial context → 5-d vector.
    [lat_norm, lon_norm, elevation_norm, sin_lat, cos_lon]
    Encodes both raw position and cyclic/relative transforms.
    Elevation from SRTM (passed in from main.py).
    Land-use from Copernicus (one-hot is too large; we use a numeric code).
    """
    LAND_USE_CODES = {
        "urban": 1.0, "cropland": 2.0, "forest": 3.0,
        "grassland": 4.0, "wetland": 5.0, "water": 6.0,
        "barren": 7.0, "snow_ice": 8.0, "unknown": 0.0,
    }

    if lat is None or lon is None:
        return np.zeros(5, dtype=np.float32)

    return np.array([
        lat / 90.0,                          # [-1, 1]
        lon / 180.0,                         # [-1, 1]
        min(elevation, 8848.0) / 8848.0,     # [0, 1]  (Everest = 1.0)
        np.sin(np.radians(lat)),             # cyclic latitude
        LAND_USE_CODES.get(land_use, 0.0) / 8.0,  # [0, 1]
    ], dtype=np.float32)


# ── 6. Radar features (Open-Meteo real data) ─────────────────────────────────

from radar import extract_radar_features_from_data

def extract_radar_features(radar_data: Optional[dict]) -> np.ndarray:
    """
    Delegates to radar.py which uses real Open-Meteo precipitation data.
    4-d vector: [precip_intensity, storm_signal, wmo_severity, cloud_coverage]
    All normalized to [0, 1].
    """
    return extract_radar_features_from_data(radar_data)


# ── 7. NWP model features (GFS / ECMWF) ──────────────────────────────────────

def extract_nwp_features(nwp_data: Optional[dict]) -> np.ndarray:
    """
    Numerical Weather Prediction output → 4-d.
    [predicted_temp, predicted_precip, predicted_wind, model_confidence]

    Production: parse ECMWF GRIB2 or NOAA GFS grib2 files.
    Placeholder returns zeros when not available.
    """
    if nwp_data is None:
        return np.zeros(4, dtype=np.float32)

    return np.array([
        float(nwp_data.get("predicted_temp",    20.0)),
        float(nwp_data.get("predicted_precip",   0.0)),
        float(nwp_data.get("predicted_wind",     0.0)),
        float(nwp_data.get("model_confidence",   0.5)),
    ], dtype=np.float32)


# ── Combined feature vector ───────────────────────────────────────────────────

def build_feature_vector(
    weather_data: dict,
    cloud_img:    Optional[Image.Image],
    thermal_img:  Optional[Image.Image],
    article_titles: List[str],
    lat:          Optional[float],
    lon:          Optional[float],
    forecast_data: dict,
    elevation:    float = 0.0,
    land_use:     str   = "unknown",
    radar_data:   Optional[dict] = None,
    nwp_data:     Optional[dict] = None,
) -> np.ndarray:
    """
    Early-fusion: concatenate ALL modality feature vectors into one flat array.

    Shape breakdown:
      weather_feat  :   6
      cloud_feat    : 512   (ResNet18)
      thermal_feat  : 512   (ResNet18)
      text_feat     : 384   (MiniLM)
      forecast_feat :  10   (6h ahead: temp,humid,wind,gusts,rain,precip_prob,cape,cloud,snow,nwp)
      geo_feat      :   5
      radar_feat    :   4
      nwp_feat      :   4
                      ----
      TOTAL         : 1437
    """
    weather_feat  = extract_weather_features(weather_data)          # (6,)
    cloud_feat    = extract_image_features(cloud_img)               # (512,)
    thermal_feat  = extract_image_features(thermal_img)             # (512,)
    text_feat     = extract_text_features(article_titles)           # (384,)
    forecast_feat = extract_forecast_features(forecast_data)        # (10,)
    geo_feat      = extract_geo_features(lat, lon,
                                         elevation, land_use)       # (5,)
    radar_feat    = extract_radar_features(radar_data)              # (4,)
    nwp_feat      = extract_nwp_features(nwp_data)                  # (4,)

    return np.concatenate([
        weather_feat,
        cloud_feat,
        thermal_feat,
        text_feat,
        forecast_feat,
        geo_feat,
        radar_feat,
        nwp_feat,
    ])   # (1431,)
