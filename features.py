"""
Feature Engineering
Replacing raw pixel heuristics with:
  - ResNet18 CNN embeddings for satellite images
  - sentence-transformers for text embeddings
  - Combined numeric weather features

"""

import numpy as np
from PIL import Image

_resnet = None
_transform = None
_text_model = None


def _load_vision_model():
    global _resnet, _transform
    if _resnet is not None:
        return
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    model = models.resnet18(weights="IMAGENET1K_V1")
    # Remove classification head → 512-d feature vector
    model.eval()
    _resnet = torch.nn.Sequential(*list(model.children())[:-1])
    _transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    print("[Features] ResNet18 loaded")


def _load_text_model():
    global _text_model
    if _text_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[Features] SentenceTransformer loaded")


# ── Image features ──────────────────────────────────────────────────────────

def extract_image_features(pil_image: Image.Image) -> np.ndarray:
    """
    Returns a 512-d float32 vector from ResNet18.
    Falls back to a 512-d zero vector if torch is unavailable.
    """
    if pil_image is None:
        return np.zeros(512, dtype=np.float32)

    try:
        import torch
        _load_vision_model()
        img = pil_image.convert("RGB")
        tensor = _transform(img).unsqueeze(0)          # (1, 3, 224, 224)
        with torch.no_grad():
            feat = _resnet(tensor)                     # (1, 512, 1, 1)
        return feat.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print(f"[Features] CNN fallback: {e}")
        return np.zeros(512, dtype=np.float32)


def pixel_cloud_score(pil_image: Image.Image) -> float:
    """
    Keep the original heuristic as a cheap fallback / cross-check.
    """
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


def pixel_heat_score(img):
    if img is None:
        return 0.0
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Hot pixels: red channel dominant, green moderate, blue low
    heat_mask = (r > 0.55) & (r > g + 0.15) & (r > b + 0.2)
    return round(float(np.sum(heat_mask) / heat_mask.size), 4)

# ── Text features ────────────────────────────────────────────────────────────

def extract_text_features(texts: list[str]) -> np.ndarray:
    """
    Encode a list of article titles → mean-pooled 384-d vector.
    Falls back to zeros if sentence-transformers is unavailable.
    """
    if not texts:
        return np.zeros(384, dtype=np.float32)
    try:
        _load_text_model()
        embeddings = _text_model.encode(texts, show_progress_bar=False)
        return embeddings.mean(axis=0).astype(np.float32)   # (384,)
    except Exception as e:
        print(f"[Features] Text embed fallback: {e}")
        return np.zeros(384, dtype=np.float32)


# ── Weather numeric features ─────────────────────────────────────────────────

def extract_weather_features(weather_data: dict) -> np.ndarray:
    """
    Returns a small numeric vector from OpenWeatherMap JSON.
    Shape: (6,)  [temp, humidity, pressure, wind_speed, cloud_pct, weather_id]
    """
    main   = weather_data.get("main", {})
    wind   = weather_data.get("wind", {})
    clouds = weather_data.get("clouds", {})
    wid    = weather_data["weather"][0]["id"] if weather_data.get("weather") else 800

    vec = np.array([
        main.get("temp",     20.0),
        main.get("humidity", 50.0),
        main.get("pressure", 1013.0),
        wind.get("speed",    0.0),
        clouds.get("all",    0.0),
        float(wid),
    ], dtype=np.float32)
    return vec

def extract_forecast_features(forecast_data):
    return np.array([
        forecast_data.get("temp", 20.0),
        forecast_data.get("humidity", 50.0),
        forecast_data.get("wind", 0.0),
        forecast_data.get("rain", 0.0)
    ], dtype=np.float32)


def extract_geo_features(lat, lon):
    if lat is None or lon is None:
        return np.zeros(2, dtype=np.float32)

    return np.array([
        lat / 90.0,      # normalize
        lon / 180.0
    ], dtype=np.float32)

# ── Combined feature vector ───────────────────────────────────────────────────

def build_feature_vector(
    weather_data,
    cloud_img,
    thermal_img,
    article_titles,
    lat,
    lon,
    forecast_data
)-> np.ndarray:
    """
    Concatenates all modality features into one flat vector.
    This vector is the input to your ML model (Step 7).
    """
    weather_feat = extract_weather_features(weather_data)        # (6,)
    cloud_feat   = extract_image_features(cloud_img)             # (512,)
    thermal_feat = extract_image_features(thermal_img)           # (512,)
    text_feat    = extract_text_features(article_titles)         # (384,)
    forecast_feat = extract_forecast_features(forecast_data)
    geo_feat = extract_geo_features(lat, lon)

    return np.concatenate([
        weather_feat,
        cloud_feat,
        thermal_feat,
        text_feat,
        forecast_feat,    
        geo_feat  ])
