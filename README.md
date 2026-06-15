# 🌦️ Multimodal Weather Intelligence System

A production-grade FastAPI backend that fuses **8 data modalities** — satellite imagery, live weather, radar-proxy, NWP model output, geospatial context, news text, and 6-hour forecasts — to **detect and predict weather events** using a full deep learning pipeline with classical ML fallback.

---

## 🧠 What Makes This Different

Most weather apps read from one API. This system fuses **8 independent data streams** and learns which ones to trust per-sample using **attention-based neural fusion**. It's also a **forecaster**, not just a classifier — labels are generated from 6-hour-ahead forecast data so the model predicts what's coming, not just what's happening now.

---

## 🏗️ System Architecture

```
Request: /analyze/{city}
        │
        ├── OpenWeatherMap API        → current conditions (temp, humidity, pressure, wind)
        ├── OpenStreetMap Nominatim   → lat/lon geocoding
        ├── NASA GIBS WMS             → satellite imagery (VIIRS true color + MODIS thermal)
        ├── Open-Meteo Forecast API   → 6-hour ahead forecast (T+6h, 10 fields)
        ├── Open-Meteo ECMWF layer    → NWP ensemble model output
        ├── Open-Meteo Current        → radar-proxy (precip, CAPE, WMO codes, visibility)
        ├── Google News RSS           → news text (5 articles, sentence-transformer embeddings)
        └── SRTM / Copernicus         → elevation + land-use metadata
                │
                ▼
        Feature Engineering (features.py)
        ┌───────────────┬────────────────────────────────────────────┐
        │ Modality      │ Representation                             │
        ├───────────────┼────────────────────────────────────────────┤
        │ Weather       │ 6-d numeric (temp, humidity, pressure …)   │
        │ Cloud image   │ 512-d ResNet18 CNN embedding               │
        │ Thermal image │ 512-d ResNet18 CNN embedding               │
        │ Text          │ 384-d MiniLM sentence-transformer          │
        │ Forecast      │ 10-d (wind_gusts, CAPE, precip_prob, …)   │
        │ Geospatial    │ 5-d (lat/lon norm, elevation, land-use)   │
        │ Radar-proxy   │ 4-d (precip intensity, storm signal, …)   │
        │ NWP output    │ 4-d (ECMWF temp, precip, wind, conf)      │
        └───────────────┴────────────────────────────────────────────┘
                Total feature vector: 1437-d
                │
                ▼
        Inference (3-tier)
        ├── [Tier 1] CNN+LSTM+Attention (attention_fusion.py)
        │     CrossModalAttentionFusion → BiLSTM → Sigmoid
        │     Modality attention weights exposed for interpretability
        ├── [Tier 2] Late-fusion ensemble (train_model.py)
        │     RF + GBT + LR independently trained, soft-voted
        │     TimeSeriesSplit CV, SHAP explainability per event
        └── [Tier 3] Rule-based fallback (forecast_engine.py)
              WMO code table + weighted signal fusion (no ML needed)
                │
                ▼
        5 event predictions: Rain · Heat · Wind · Snow · Haze
        Each with: detected (bool) · confidence (float) · source breakdown
                │
                ▼
        SQLite / MySQL persistence  +  Auto-label generation
        → continuous self-improvement loop
```

---

## 🤖 ML / DL Pipeline

### Feature Engineering
- **ResNet18 CNN** (ImageNet pretrained, classification head removed) encodes satellite images to 512-d vectors
- **all-MiniLM-L6-v2** sentence-transformer mean-pools news article titles to 384-d
- **Radar-proxy** maps WMO weather codes to a severity table + computes convective storm signal from CAPE and shower intensity
- **Forecast features** are 10-dimensional covering wind gusts, CAPE, precipitation probability, snowfall, NWP ensemble agreement

### Deep Learning (attention_fusion.py)
- **ModalityAttentionFusion**: learns a scalar attention weight per modality via a small MLP, softmax-normalized — tells you which data source the model trusted
- **CrossModalAttentionFusion**: transformer-style multi-head self-attention across the 8 modality embeddings (modalities attend to each other)
- **CNNLSTMAttentionModel**: full hybrid — cross-modal attention → BiLSTM (2 layers, bidirectional, dropout 0.4) → binary prediction per event

### Classical ML (train_model.py)
- **Model selection**: TimeSeriesSplit CV across RF, GBT, Logistic Regression, SVM — best F1 wins
- **Late-fusion ensemble**: RF + GBT + LR soft-voted; GBT receives combined city-balance × class-balance sample weights
- **SHAP explainability**: TreeExplainer saves per-event mean absolute SHAP values to `models/shap_{event}.json`
- **Climate-zone-aware labels**: heat thresholds are city-specific (e.g. Delhi = 42°C, London = 30°C) so labels reflect local norms not global averages

### Forecasting (forecast_engine.py)
- Labels are generated from **6-hour-ahead forecasts** (not current conditions) making the model a genuine forecaster
- Three forecast sources fused: Open-Meteo primary, OWM 3h buckets, ECMWF NWP layer
- WMO code severity table maps 30+ codes to (precip_flag, storm_flag, severity 0-1)

### Self-improvement Loop
- Every live analysis auto-generates future labels → saved to `labeled_events` table
- `python collector.py` runs continuously across 50+ global cities, feeding the training set
- `python train_model.py` retrains on accumulated data; models hot-reload via `POST /reload-models`

---

## 🚀 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /analyze/{city}` | Full multimodal analysis + 6h forecast |
| `GET /history/{city}` | Past analysis records for a city |
| `GET /nearby?lat=&lon=` | Analysis records within 1° radius |
| `GET /compare?cities=X&cities=Y` | Side-by-side comparison (up to 5 cities) |
| `GET /evaluate` | ML model evaluation metrics from last training run |
| `POST /reload-models` | Hot-reload trained ML models without restarting |

### Example Response
```json
{
  "city": "Mumbai",
  "weather": "moderate rain",
  "temperature": 28.4,
  "analysis": {
    "rain": {
      "detected": true,
      "confidence": 0.84,
      "sources": { "weather": true, "text": true, "image": true }
    },
    "heat": { "detected": false, "confidence": 0.10 },
    "wind": { "detected": false, "confidence": 0.22 }
  },
  "model": "ml"
}
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn, Starlette |
| Deep Learning | PyTorch, torchvision (ResNet18), BiLSTM, Multi-head Attention |
| NLP | sentence-transformers (MiniLM), HuggingFace transformers |
| Classical ML | scikit-learn (RF, GBT, SVM, LR), SHAP, joblib |
| Image Processing | Pillow, NumPy |
| Database | SQLite (default) → MySQL (production), with one-command migration |
| Unstructured Store | MongoDB (optional, for raw text docs) |
| Dashboard | Streamlit |
| Data Sources | OpenWeatherMap, NASA GIBS, Open-Meteo, ECMWF, Google News RSS, OpenStreetMap |

---

## ⚙️ Setup

```bash
git clone https://github.com/Rahen-04/Multimodal_data_fusion.git
cd Multimodal_data_fusion
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Set WEATHER_API_KEY (openweathermap.org — free tier works)
# Optionally set DB_HOST, DB_USER, DB_PASSWORD for MySQL

# Start the API
uvicorn main:app --reload

# Collect data across 50+ global cities (runs continuously)
python collector.py

# Train ML + deep learning models on collected data
python train_model.py

# Launch Streamlit dashboard
streamlit run dashboard.py
```

For SQLite → MySQL migration after initial data collection:
```bash
python database.py --migrate weather_data.db
```

---

## 📁 File Structure

```
├── main.py               # FastAPI app — all endpoints, signal fusion, persistence
├── features.py           # Feature engineering: CNN, transformer, radar, geo, NWP
├── attention_fusion.py   # CrossModalAttention + CNNLSTMAttentionModel (PyTorch)
├── forecast_engine.py    # 6h forecast fetching, future label generation, rule fallback
├── train_model.py        # Training pipeline: model selection, late fusion, SHAP, LSTM
├── radar.py              # Open-Meteo radar-proxy fetching + WMO code severity table
├── database.py           # Multi-backend DB (SQLite/MySQL), schema, city climate table
├── collector.py          # Continuous multi-city data ingestion (50+ cities)
├── dashboard.py          # Streamlit visualization dashboard
└── requirements.txt
```

---

## 💡 Design Highlights

- **Climate-zone-aware labelling**: heat thresholds are calibrated per city — a 38°C day is normal in Dubai, extreme in London. Labels reflect local climate norms.
- **Three-tier inference**: attention model → ensemble ML → rule-based fallback ensures predictions are always available even without trained models.
- **Continuous self-improvement**: every API call generates future labels from forecast data, feeding the training set automatically. The model improves as more cities and weather events are observed.
- **Graceful degradation**: if satellite images return black tiles or APIs time out, the system falls back seamlessly to remaining modalities without crashing.
- **Portable DB layer**: single codebase runs on SQLite locally and MySQL in production; schema migrations are backward-compatible (ALTER TABLE IF NOT EXISTS pattern).
