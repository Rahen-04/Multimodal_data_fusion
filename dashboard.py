"""
dashboard.py  —  Streamlit frontend

Pages:
  1. Live Analysis      — current city weather + event detection
  2. History & Trends   — time-series charts + trend analysis
  3. Compare Cities     — side-by-side multi-city table
  4. Geospatial Map     — lat/lon scatter of recent records
  5. Label & Train      — manual labeling UI + model training trigger
  6. Model Insights     — eval metrics + SHAP feature importance
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv
load_dotenv()  # load .env so DB_PASSWORD etc are available
import sys
from database import get_connection, _ph
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="WeatherFusion",
    page_icon="🌩",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1117; }
    .metric-card {
        background: #1a1d2e; border-radius: 12px;
        padding: 16px 20px; border: 1px solid #2a2d3e; margin-bottom: 10px;
    }
    .detected { color: #ff6b6b; font-weight: 700; }
    .clear    { color: #51cf66; font-weight: 700; }
    .conf-bar  { height: 6px; background: #2a2d3e; border-radius: 3px; margin-top: 6px; }
    .conf-fill { height: 6px; border-radius: 3px;
                 background: linear-gradient(90deg,#4cc9f0,#7209b7); }
    .alert-critical {
        background: linear-gradient(135deg, #3d0000, #7a0000);
        border: 2px solid #ff4444; border-radius: 12px;
        padding: 18px 22px; margin-bottom: 10px;
        animation: pulse-red 1.5s infinite;
    }
    .alert-high {
        background: linear-gradient(135deg, #3d1a00, #7a3800);
        border: 2px solid #ff8800; border-radius: 12px;
        padding: 18px 22px; margin-bottom: 10px;
    }
    .alert-medium {
        background: linear-gradient(135deg, #2a2a00, #4a4a00);
        border: 2px solid #ffdd00; border-radius: 12px;
        padding: 18px 22px; margin-bottom: 10px;
    }
    .alert-title  { font-size:17px; font-weight:800; letter-spacing:.5px; margin-bottom:4px; }
    .alert-body   { font-size:13px; color:#cccccc; margin-top:4px; }
    .alert-conf   { font-size:12px; color:#aaaaaa; margin-top:6px; }
    .no-alert {
        background:#0d2b0d; border:1px solid #2a6a2a; border-radius:10px;
        padding:14px 18px; color:#51cf66; font-weight:600; font-size:15px;
    }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0px #ff4444; }
        50%  { box-shadow: 0 0 14px #ff4444; }
        100% { box-shadow: 0 0 0px #ff4444; }
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌩 WeatherFusion")
    st.caption("Multimodal Extreme Weather Predictor")
    st.divider()

    city_input = st.text_input("Enter cities (comma-separated)", value="Mumbai")
    cities     = [c.strip() for c in city_input.split(",") if c.strip()]
    if not cities:
        st.warning("Enter at least one city")
        st.stop()

    st.markdown(f"**Selected:** {', '.join(cities)}")
    run_btn = st.button("Analyze", use_container_width=True, type="primary")
    st.divider()

    page = st.radio(
        "View",
        ["Live Analysis", "History & Trends",
         "Compare Cities", "Geospatial Map",
         "Label & Train",  "Model Insights",
         "Climate Trends", "Forecast Timeline"],
    )

    st.divider()
    st.markdown("**⚙️ Alert Settings**")
    alert_threshold = st.slider(
        "Alert confidence threshold",
        min_value=0.30, max_value=0.95, value=0.60, step=0.05,
    )

    st.divider()
    st.markdown("**🔄 Real-Time Settings**")
    auto_refresh = st.checkbox("Auto-refresh (real-time)", value=False)
    refresh_interval = st.selectbox("Refresh every", [30, 60, 120, 300],
                                     index=1, format_func=lambda x: f"{x}s")
    if auto_refresh:
        import time as _time
        st.caption(f"⏳ Refreshing every {refresh_interval}s")
        _time.sleep(refresh_interval)
        st.rerun()


# ── Cached fetchers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_analysis(city):
    try:
        r    = requests.get(f"{API_BASE}/analyze/{city}", timeout=60)
        data = r.json()
        return {"error": f"{city}: {data['error']}"} if "error" in data else data
    except Exception as e:
        return {"error": f"{city}: {str(e)}"}


@st.cache_data(ttl=60)
def fetch_history(city, limit=100):
    try:
        r = requests.get(f"{API_BASE}/history/{city}?limit={limit}", timeout=10)
        return r.json().get("records", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_eval():
    try:
        return requests.get(f"{API_BASE}/evaluate", timeout=5).json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_shap(event):
    try:
        r = requests.get(f"{API_BASE}/shap/{event}", timeout=5)
        return r.json()
    except Exception:
        return {}


# ── Alert helpers ─────────────────────────────────────────────────────────────
EVENT_META = {
    "rain": {"icon": "🌧",
             "critical_msg": "Severe rainfall / flooding risk. Avoid low-lying areas.",
             "high_msg":     "Heavy rain expected. Carry rain gear.",
             "medium_msg":   "Rain likely. Keep an umbrella handy."},
    "heat": {"icon": "🔥",
             "critical_msg": "Dangerous heat conditions. Stay indoors, stay hydrated.",
             "high_msg":     "Extreme heat warning. Limit outdoor activity.",
             "medium_msg":   "High temperatures expected."},
    "wind": {"icon": "💨",
             "critical_msg": "Violent storm / cyclone risk. Seek shelter immediately.",
             "high_msg":     "Strong winds expected. Secure loose objects.",
             "medium_msg":   "Windy conditions. Be cautious while driving."},
    "snow": {"icon": "❄️",
             "critical_msg": "Blizzard / heavy snowfall. Do not travel.",
             "high_msg":     "Heavy snow expected. Roads may be slippery.",
             "medium_msg":   "Light snow possible. Drive carefully."},
    "haze": {"icon": "🌫",
             "critical_msg": "Hazardous air quality. Stay indoors.",
             "high_msg":     "Poor air quality. Wear N95 mask outdoors.",
             "medium_msg":   "Moderate haze. Sensitive groups limit exposure."},
}


def render_alerts(city, analysis, threshold):
    critical_threshold = 0.80
    medium_threshold   = max(0.30, threshold - 0.15)

    critical_events, high_events, medium_events = [], [], []

    for event, result in analysis.items():
        conf     = result.get("confidence", 0)
        detected = result.get("detected", False)
        if conf >= critical_threshold:
            critical_events.append((event, conf))
        elif conf >= threshold and detected:
            high_events.append((event, conf))
        elif conf >= medium_threshold and detected:
            medium_events.append((event, conf))

    critical_events.sort(key=lambda x: x[1], reverse=True)
    high_events.sort(    key=lambda x: x[1], reverse=True)
    medium_events.sort(  key=lambda x: x[1], reverse=True)

    st.markdown("### 🚨 6-Hour Alert Forecast")

    if not (critical_events or high_events or medium_events):
        st.markdown(
            f'<div class="no-alert">✅ &nbsp; No extreme weather expected in the next 6 hours for {city}.</div>',
            unsafe_allow_html=True,
        )
        return

    for event, conf in critical_events:
        meta = EVENT_META.get(event, {})
        st.markdown(f"""
        <div class="alert-critical">
            <div class="alert-title">🚨 {meta.get('icon','⚠️')} CRITICAL — {event.upper()} in {city}</div>
            <div class="alert-body">{meta.get('critical_msg','')}</div>
            <div class="alert-conf">Confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)

    for event, conf in high_events:
        meta = EVENT_META.get(event, {})
        st.markdown(f"""
        <div class="alert-high">
            <div class="alert-title">⚠️ {meta.get('icon','⚠️')} HIGH ALERT — {event.upper()} in {city}</div>
            <div class="alert-body">{meta.get('high_msg','')}</div>
            <div class="alert-conf">Confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)

    for event, conf in medium_events:
        meta = EVENT_META.get(event, {})
        st.markdown(f"""
        <div class="alert-medium">
            <div class="alert-title">🔔 {meta.get('icon','⚠️')} ADVISORY — {event.upper()} in {city}</div>
            <div class="alert-body">{meta.get('medium_msg','')}</div>
            <div class="alert-conf">Confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)


def safe_json(val):
    if val is None: return {}
    if isinstance(val, dict): return val
    try: return json.loads(val)
    except: return {}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "Live Analysis":
    city = cities[0]
    st.title(f"⚡ 6-Hour Forecast — {city}")

    if run_btn or "last_data" not in st.session_state:
        with st.spinner("Fetching multimodal weather data…"):
            data = fetch_analysis(city)
            st.session_state["last_data"] = data
    else:
        data = st.session_state.get("last_data", {})

    if "error" in data:
        st.error(f"API error: {data['error']}")
        st.stop()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Temp",   f"{data.get('temperature','--')} °C")
    fc = data.get("forecast_summary", {})
    col2.metric("Temp in 6h",     f"{fc.get('temp_6h','--')} °C" if fc.get('temp_6h') else "--")
    col3.metric("Rain in 6h",     f"{fc.get('rain_6h_mm', 0):.1f} mm")
    col4.metric("Precip. Prob",   f"{fc.get('precip_prob', 0)*100:.0f}%")
    col5.metric("Horizon",        data.get("prediction_horizon", "6h").upper())

    geo = data.get("geospatial", {})
    st.caption(f"📍 {city}  |  Elevation: {geo.get('elevation',0):.0f}m  |  "
               f"Land use: {geo.get('land_use','?').title()}  |  "
               f"Valid at: {fc.get('valid_time','--')}")

    st.divider()
    render_alerts(city, data.get("analysis", {}), alert_threshold)
    st.divider()

    # Event detection cards
    st.subheader("📡 6-Hour Extreme Event Forecast")
    analysis = data.get("analysis", {})
    icons    = {"rain":"🌧","heat":"🔥","wind":"💨","snow":"❄️","haze":"🌫"}
    cols     = st.columns(len(analysis))
    for col, event in zip(cols, analysis):
        ev   = analysis[event]
        det  = ev.get("detected", False)
        conf = ev.get("confidence", 0)
        with col:
            status = "🔴 EXPECTED (6h)" if det else "🟢 Not Expected"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px">{icons.get(event,'🌡')}</div>
                <div style="font-size:15px;font-weight:600;margin:4px 0">{event.upper()}</div>
                <div class="{'detected' if det else 'clear'}">{status}</div>
                <div style="font-size:12px;color:#888;margin-top:4px">
                    Confidence: {conf:.0%}</div>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:{conf*100:.0f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Attention weights — which modality the model trusted most
    if any(data.get("analysis", {}).get(ev, {}).get("attn_weights")
           for ev in ["rain","heat","wind","snow","haze"]):
        st.subheader("🧠 Attention Weights — Which Modality Mattered Most")
        st.caption("Higher = model relied on that data source more for this prediction")
        attn_cols = st.columns(5)
        for ac, event in zip(attn_cols, ["rain","heat","wind","snow","haze"]):
            aw = data["analysis"].get(event, {}).get("attn_weights", {})
            if aw:
                with ac:
                    st.caption(f"**{event.upper()}**")
                    df_aw = pd.DataFrame({"Modality": list(aw.keys()),
                                          "Weight":   list(aw.values())})
                    df_aw = df_aw.sort_values("Weight", ascending=False)
                    st.bar_chart(df_aw.set_index("Modality"), use_container_width=True)
        st.divider()

    # Model blend breakdown
    st.subheader("🔀 Model Ensemble Blend")
    blend_rows = []
    for event, ev_data in data.get("analysis", {}).items():
        blend = ev_data.get("model_blend", {})
        if blend:
            row = {"Event": event.upper()}
            row.update({k.upper(): f"{v:.0%}" for k, v in blend.items()})
            blend_rows.append(row)
    if blend_rows:
        st.dataframe(pd.DataFrame(blend_rows).set_index("Event"),
                     use_container_width=True)
        st.caption("Shows confidence score from each model type (sklearn / lstm / attention / rules)")
    st.divider()

    # Satellite images
    img_col1, img_col2 = st.columns(2)
    img_res = data.get("image_analysis", {})
    with img_col1:
        st.subheader("☁️ Cloud Cover (NASA VIIRS)")
        cloud_url = img_res.get("cloud_image")
        if cloud_url:
            st.image(cloud_url, use_container_width=True)
            st.caption(f"Cloud score: {img_res.get('cloud',0):.2f}")
        else:
            st.info("Image unavailable")
    with img_col2:
        st.subheader("🌡 Land Surface Temp (NASA MODIS)")
        thermal_url = img_res.get("thermal_image")
        if thermal_url:
            st.image(thermal_url, use_container_width=True)
            st.caption(f"Heat score: {img_res.get('heat',0):.2f}")
        else:
            st.info("Image unavailable")

    st.divider()
    st.subheader("📰 Text Analysis Scores")
    text_res = data.get("text_analysis", {})
    if text_res:
        df_text = pd.DataFrame({"Event": list(text_res.keys()),
                                 "Score": list(text_res.values())})
        st.bar_chart(df_text.set_index("Event"), use_container_width=True)

    st.divider()
    st.subheader("📈 6-Hour Forecast Confidence History (last 20 records)")
    hist = fetch_history(city, limit=20)
    if hist:
        df_hist = pd.DataFrame(hist)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist.sort_values("timestamp").tail(20)
        conf_rows = []
        for _, row in df_hist.iterrows():
            try:
                ana   = safe_json(row["analysis"])
                entry = {"timestamp": row["timestamp"]}
                for ev in ["rain","heat","wind","snow","haze"]:
                    entry[ev] = ana.get(ev, {}).get("confidence", 0)
                conf_rows.append(entry)
            except Exception:
                pass
        if conf_rows:
            st.line_chart(pd.DataFrame(conf_rows).set_index("timestamp"),
                          use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History & Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History & Trends":
    st.title("📊 History & Trends")

    all_data = []
    for c in cities:
        records = fetch_history(c)
        if records:
            df = pd.DataFrame(records)
            df["city"]      = c
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

    if not all_data:
        st.info("No historical data. Run collector first.")
        st.stop()

    df_all = pd.concat(all_data).sort_values("timestamp").reset_index(drop=True)

    df_all["analysis_parsed"]    = df_all["analysis"].apply(safe_json)
    df_all["raw_weather_parsed"] = df_all["raw_weather"].apply(safe_json)

    for event in ["rain","heat","wind","snow","haze"]:
        df_all[f"conf_{event}"] = df_all["analysis_parsed"].apply(
            lambda a: a.get(event,{}).get("confidence",0) if isinstance(a,dict) else 0)
        df_all[f"det_{event}"]  = df_all["analysis_parsed"].apply(
            lambda a: bool(a.get(event,{}).get("detected",False)) if isinstance(a,dict) else False)

    df_all["humidity"]   = df_all["raw_weather_parsed"].apply(
        lambda w: w.get("main",{}).get("humidity"))
    df_all["pressure"]   = df_all["raw_weather_parsed"].apply(
        lambda w: w.get("main",{}).get("pressure"))
    df_all["wind_speed"] = df_all["raw_weather_parsed"].apply(
        lambda w: w.get("wind",{}).get("speed"))

    # Filters
    st.markdown("### 🔍 Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        city_filter = st.multiselect("City",
            options=sorted(df_all["city"].unique()),
            default=sorted(df_all["city"].unique()))
    with f2:
        min_d = df_all["timestamp"].min().date()
        max_d = df_all["timestamp"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d))
    with f3:
        event_filter = st.multiselect("Events",
            options=["rain","heat","wind","snow","haze"],
            default=["rain","heat"])

    df_f = df_all[df_all["city"].isin(city_filter)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df_f = df_f[
            (df_f["timestamp"].dt.date >= date_range[0]) &
            (df_f["timestamp"].dt.date <= date_range[1])
        ]

    if df_f.empty:
        st.warning("No records after filtering.")
        st.stop()

    st.divider()

    # Temperature trend
    st.subheader("🌡 Temperature Over Time")
    df_temp = df_f.set_index("timestamp")[["temperature","city"]]
    if len(cities) == 1:
        st.line_chart(df_temp["temperature"], use_container_width=True)
    else:
        pivot = df_f.pivot_table(index="timestamp", columns="city",
                                  values="temperature", aggfunc="mean")
        st.line_chart(pivot, use_container_width=True)

    st.divider()

    # Confidence trends for selected events
    st.subheader("📈 Confidence Trends")
    cols_to_plot = [f"conf_{e}" for e in event_filter if f"conf_{e}" in df_f.columns]
    if cols_to_plot:
        st.line_chart(df_f.set_index("timestamp")[cols_to_plot],
                      use_container_width=True)

    st.divider()

    # Detection frequency heatmap
    st.subheader("🗓 Detection Frequency by Day")
    df_f["date"] = df_f["timestamp"].dt.date
    for event in event_filter:
        col = f"det_{event}"
        if col in df_f.columns:
            day_counts = df_f.groupby("date")[col].sum().reset_index()
            day_counts.columns = ["date", "detections"]
            st.caption(f"**{event.upper()}** detections per day")
            st.bar_chart(day_counts.set_index("date"), use_container_width=True)

    st.divider()

    # Humidity / Pressure / Wind
    st.subheader("🌬 Atmospheric Parameters")
    params = {k: k for k in ["humidity","pressure","wind_speed"]
              if k in df_f.columns and df_f[k].notna().any()}
    if params:
        p1, p2, p3 = st.columns(3)
        for col_st, param in zip([p1, p2, p3], params):
            with col_st:
                st.caption(param.replace("_"," ").title())
                st.line_chart(df_f.set_index("timestamp")[param],
                              use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Compare Cities
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare Cities":
    st.title("🌍 Compare Cities Side-by-Side")

    if len(cities) < 2:
        st.warning("Enter at least 2 cities in the sidebar.")
        st.stop()

    if run_btn or "compare_data" not in st.session_state:
        with st.spinner("Fetching data for all cities…"):
            params = "&".join(f"cities={c}" for c in cities)
            try:
                r = requests.get(f"{API_BASE}/compare?{params}", timeout=180)
                compare_data = r.json()
                st.session_state["compare_data"] = compare_data
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
    else:
        compare_data = st.session_state["compare_data"]

    st.markdown("### 🚨 Alerts Across All Cities")
    any_alert = False
    for city_name, d in compare_data.items():
        if "error" in d: continue
        city_analysis = d.get("analysis", {})
        has_alert = any(
            v.get("confidence",0) >= max(0.30, alert_threshold - 0.15)
            and v.get("detected", False)
            for v in city_analysis.values()
        )
        if has_alert:
            any_alert = True
            render_alerts(city_name, city_analysis, alert_threshold)

    if not any_alert:
        st.markdown(
            '<div class="no-alert">✅ &nbsp; No extreme weather alerts for any selected city.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    rows = []
    for city_name, d in compare_data.items():
        if "error" in d:
            rows.append({"City": city_name, "Error": d["error"]})
            continue
        geo = d.get("geospatial", {})
        row = {
            "City":      city_name,
            "Temp °C":   d.get("temperature", "--"),
            "Condition": d.get("weather", "--"),
            "Elevation": f"{geo.get('elevation',0):.0f} m",
            "Land Use":  geo.get("land_use","?"),
        }
        for event, ev in d.get("analysis", {}).items():
            row[event.capitalize()] = "🔴" if ev["detected"] else "🟢"
            row[f"{event.capitalize()} conf"] = f"{ev['confidence']:.0%}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("City"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Geospatial Map
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Geospatial Map":
    st.title("🗺 Geospatial Map — Recent Records")
    st.caption("Each point = one city record. Color = highest-confidence detected event.")

    all_records = []
    for c in cities:
        recs = fetch_history(c, limit=50)
        all_records.extend(recs)

    if not all_records:
        st.info("No records yet. Run an analysis first.")
        st.stop()

    map_rows = []
    for r in all_records:
        if r.get("lat") is None or r.get("lon") is None:
            continue
        ana = safe_json(r.get("analysis"))
        best_event = max(ana, key=lambda e: ana[e].get("confidence",0)) if ana else "none"
        best_conf  = ana.get(best_event, {}).get("confidence", 0) if ana else 0
        map_rows.append({
            "lat":        r["lat"],
            "lon":        r["lon"],
            "city":       r["city"],
            "temp":       r.get("temperature", 0),
            "event":      best_event,
            "confidence": best_conf,
        })

    if not map_rows:
        st.warning("No records with lat/lon available.")
        st.stop()

    df_map = pd.DataFrame(map_rows)

    # Streamlit native map
    st.map(df_map[["lat","lon"]], zoom=2)

    st.divider()
    st.subheader("📋 Data Table")
    st.dataframe(df_map, use_container_width=True)

    # Elevation summary for fetched cities
    st.divider()
    st.subheader("⛰ Elevation Summary")
    if "temp" in df_map.columns:
        elev_data = []
        for c in cities:
            last = fetch_history(c, limit=1)
            if last:
                r = last[0]
                elev_data.append({
                    "City":      c,
                    "Elevation": r.get("elevation", 0) or 0,
                    "Land Use":  r.get("land_use",   "?") or "?",
                    "Lat":       r.get("lat", 0),
                    "Lon":       r.get("lon", 0),
                })
        if elev_data:
            st.dataframe(pd.DataFrame(elev_data).set_index("City"),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Label & Train
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Label & Train":
    st.title("🏷️ Label Records & Trigger Training")
    st.info(
        "Label historical records, then click **Train model** to retrain. "
        "Time-based splits are used automatically to avoid data leakage."
    )

    try:
        conn   = get_connection()
        cursor = conn.cursor()
        P      = _ph()
        cursor.execute("""
            SELECT r.id, r.city, r.timestamp, r.temperature,
                   r.description, r.cloud_score, r.heat_score,
                   r.elevation, r.land_use
            FROM weather_records r
            LEFT JOIN labeled_events l ON l.record_id = r.id
            WHERE l.id IS NULL
            ORDER BY r.timestamp DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
    except Exception as e:
        st.error(f"DB error: {e}")
        rows = []

    if not rows:
        st.success("All records labeled! Ready to train.")
    else:
        st.write(f"**{len(rows)} unlabeled records**")
        label_data = []
        for row in rows:
            rid, rcity, rtime, rtemp, rdesc, rcloud, rheat, relev, rland = row
            with st.expander(
                f"#{rid} — {rcity} | {str(rtime)[:16]} | "
                f"{rtemp}°C | {rdesc} | el={relev or 0:.0f}m | {rland or '?'}"
            ):
                c1,c2,c3,c4,c5 = st.columns(5)
                rain = c1.checkbox("Rain", key=f"r_{rid}")
                heat = c2.checkbox("Heat", key=f"h_{rid}")
                wind = c3.checkbox("Wind", key=f"w_{rid}")
                snow = c4.checkbox("Snow", key=f"sn_{rid}")
                haze = c5.checkbox("Haze", key=f"hz_{rid}")
                label_data.append((
                    rid, rcity, rtime,
                    int(rain), int(heat), int(wind), int(snow), int(haze)
                ))

        if st.button("💾 Save labels", type="primary"):
            try:
                cursor.executemany(f"""
                    INSERT INTO labeled_events
                    (record_id, city, timestamp,
                     label_rain, label_heat, label_wind, label_snow, label_haze)
                    VALUES ({P},{P},{P},{P},{P},{P},{P},{P})
                """, label_data)
                conn.commit()
                st.success("Labels saved!")
            except Exception as e:
                st.error(f"Error saving labels: {e}")
        conn.close()

    st.divider()
    st.subheader("🚀 Train Model")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Train all events", type="primary"):
            st.info("⏳ Training started — this may take 2-5 minutes depending on data size.")
            output_placeholder = st.empty()
            progress_text = []

            import subprocess, threading

            result_holder = {"done": False, "stdout": "", "stderr": "", "code": None}

            def run_training():
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "train_model.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    for line in proc.stdout:
                        result_holder["stdout"] += line
                    proc.wait()
                    result_holder["code"]  = proc.returncode
                    result_holder["done"]  = True
                except Exception as e:
                    result_holder["stderr"] = str(e)
                    result_holder["code"]   = 1
                    result_holder["done"]   = True

            t = threading.Thread(target=run_training, daemon=True)
            t.start()

            import time as _time
            bar = st.progress(0, text="Training models...")
            for i in range(300):   # max 5 minutes
                _time.sleep(1)
                bar.progress(min(i/300, 0.99),
                             text=f"Training... {i}s elapsed")
                # Show last few lines of output
                lines = result_holder["stdout"].strip().split("\n")
                output_placeholder.code("\n".join(lines[-15:]))
                if result_holder["done"]:
                    break

            bar.progress(1.0, text="Done!")

            if result_holder["code"] == 0:
                st.success("✅ Training complete!")
                st.code(result_holder["stdout"][-3000:])
                try:
                    r = requests.post(f"{API_BASE}/reload-models", timeout=10)
                    loaded = r.json()
                    st.info(f"✅ Models reloaded — "
                            f"sklearn: {loaded.get('sklearn',[])} | "
                            f"lstm: {loaded.get('lstm',[])} | "
                            f"attention: {loaded.get('attention',[])}")
                except Exception as e:
                    st.warning(f"Reload failed: {e} — restart uvicorn manually")
            elif not result_holder["done"]:
                st.warning("Training is still running in the background. "
                           "Check the terminal for output.")
                st.code(result_holder["stdout"][-2000:])
            else:
                st.error("Training failed")
                st.code(result_holder["stdout"][-3000:])

    with col_b:
        st.markdown("**ℹ️ Training details:**")
        st.markdown("- Time-based 80/20 chronological split")
        st.markdown("- Models: RF, GBT, Logistic, SVM, Late-Fusion Ensemble")
        st.markdown("- LSTM optional (PyTorch)")
        st.markdown("- SHAP explainability saved per event")
        st.markdown("- Feature vector: 1437-d multimodal")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.title("🔬 Model Insights & Explainability")

    # Eval metrics
    eval_res = fetch_eval()
    if eval_res and "error" not in eval_res:
        st.subheader("📊 Evaluation Metrics (last training run)")
        rows = []
        for event, metrics in eval_res.items():
            rows.append({
                "Event":      event.capitalize(),
                "F1 Score":   f"{metrics.get('f1',0):.3f}",
                "AUC-ROC":    f"{metrics.get('auc',0) or 0:.3f}",
                "Model Type": metrics.get("model_type", "—"),
            })
        st.table(pd.DataFrame(rows).set_index("Event"))

        # F1 bar chart
        f1_vals = {e.capitalize(): v.get("f1",0)
                   for e, v in eval_res.items()}
        st.bar_chart(pd.DataFrame.from_dict({"F1": f1_vals}, orient="columns"),
                     use_container_width=True)
    else:
        st.info("No evaluation results yet. Train a model first.")

    st.divider()

    # SHAP importances
    st.subheader("🧠 SHAP Feature Importances (top 20 PCA components)")
    shap_event = st.selectbox("Select event", ["rain","heat","wind","snow","haze"])
    shap_data  = fetch_shap(shap_event)
    if shap_data and "mean_abs_shap" in shap_data:
        shap_vals = shap_data["mean_abs_shap"]
        top_n     = 20
        top_idx   = sorted(range(len(shap_vals)),
                           key=lambda i: shap_vals[i], reverse=True)[:top_n]
        df_shap   = pd.DataFrame({
            "PCA Component": [f"PC{i}" for i in top_idx],
            "Mean |SHAP|":   [shap_vals[i] for i in top_idx],
        }).set_index("PCA Component")
        st.bar_chart(df_shap, use_container_width=True)
        st.caption(
            "Higher = more important to the model. "
            "Components correspond to linear combinations of the 1431-d feature vector."
        )
    else:
        st.info("No SHAP data. Train model with shap installed (`pip install shap`).")

    st.divider()
    st.subheader("🗂 Feature Vector Breakdown")
    st.markdown("""
| Modality | Dimensions | Source |
|---|---|---|
| Weather numerics | 6 | OpenWeatherMap current |
| Cloud image (CNN) | 512 | NASA VIIRS → ResNet18 |
| Thermal image (CNN) | 512 | NASA MODIS → ResNet18 |
| Article text (NLP) | 384 | Google News → MiniLM |
| Forecast (6h ahead) | 10 | Open-Meteo: temp,wind,rain,CAPE,gusts,precip_prob,snow,cloud,NWP |
| Geospatial | 5 | Nominatim + SRTM elevation + Copernicus land-use |
| Radar | 4 | IMD/NEXRAD placeholder |
| NWP model output | 4 | GFS/ECMWF placeholder |
| **Total** | **1437** | Early fusion |
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Climate Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Climate Trends":
    st.title("🌡 Climate Trend Analysis")
    st.caption("Long-term patterns from collected data — temperature drift, "
               "event frequency shifts, seasonal baselines.")

    all_records = []
    for c in cities:
        recs = fetch_history(c, limit=500)
        for r in recs:
            r["city"] = c
        all_records.extend(recs)

    if len(all_records) < 5:
        st.info("Need more data. Let the collector run for a few days.")
        st.stop()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["week"]      = df["timestamp"].dt.to_period("W").astype(str)
    df["month"]     = df["timestamp"].dt.to_period("M").astype(str)
    df["analysis_parsed"] = df["analysis"].apply(safe_json)

    for ev in ["rain","heat","wind","snow","haze"]:
        df[f"det_{ev}"]  = df["analysis_parsed"].apply(
            lambda a: int(a.get(ev,{}).get("detected", False)) if isinstance(a,dict) else 0)
        df[f"conf_{ev}"] = df["analysis_parsed"].apply(
            lambda a: a.get(ev,{}).get("confidence", 0) if isinstance(a,dict) else 0)

    st.subheader("📈 Temperature Drift Over Time")
    if len(df["date"].unique()) > 1:
        temp_trend = df.groupby("date")["temperature"].mean().reset_index()
        temp_trend.columns = ["date", "avg_temp"]
        st.line_chart(temp_trend.set_index("date"), use_container_width=True)
        # Linear trend
        if len(temp_trend) > 3:
            x = np.arange(len(temp_trend))
            z = np.polyfit(x, temp_trend["avg_temp"].values, 1)
            slope = z[0]
            direction = "warming" if slope > 0 else "cooling"
            st.metric("Temperature trend",
                      f"{abs(slope)*7:.2f}°C / week",
                      delta=f"{'↑' if slope>0 else '↓'} {direction}")
    else:
        st.info("Need multiple days of data to show trend.")

    st.divider()
    st.subheader("📊 Extreme Event Frequency (by week)")
    event_filter = st.multiselect("Events to show",
                                   ["rain","heat","wind","snow","haze"],
                                   default=["rain","heat"])
    if df["week"].nunique() > 1:
        weekly = df.groupby("week")[[f"det_{e}" for e in event_filter]].mean()
        weekly.columns = [e.upper() for e in event_filter]
        st.bar_chart(weekly, use_container_width=True)
        st.caption("Y-axis = fraction of records in that week with event detected")
    else:
        st.info("Need data from multiple weeks to show frequency trends.")

    st.divider()
    st.subheader("🏙 City-Level Climate Baseline")
    city_baseline = df.groupby("city").agg(
        avg_temp=("temperature", "mean"),
        max_temp=("temperature", "max"),
        min_temp=("temperature", "min"),
        rain_freq=("det_rain",  "mean"),
        heat_freq=("det_heat",  "mean"),
        wind_freq=("det_wind",  "mean"),
        records=  ("temperature","count"),
    ).round(2)
    st.dataframe(city_baseline.sort_values("avg_temp", ascending=False),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Forecast Timeline
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast Timeline":
    st.title("🕐 6-Hour Forecast Timeline")
    st.caption("Visual breakdown of what each model predicts for the next 6 hours.")

    city = cities[0]
    if run_btn or "last_data" not in st.session_state:
        with st.spinner("Fetching forecast data..."):
            data = fetch_analysis(city)
            st.session_state["last_data"] = data
    else:
        data = st.session_state.get("last_data", {})

    if "error" in data:
        st.error(data["error"])
        st.stop()

    fc = data.get("forecast_summary", {})
    st.subheader(f"📍 {city} — Forecast valid at: {fc.get('valid_time','--')}")

    # Forecast summary metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Temp in 6h",      f"{fc.get('temp_6h','--')} °C")
    c2.metric("Rain expected",   f"{fc.get('rain_6h_mm',0):.1f} mm")
    c3.metric("Precip. prob",    f"{fc.get('precip_prob',0)*100:.0f}%")
    c4.metric("CAPE",            f"{fc.get('cape',0):.0f} J/kg",
              help="Convective Available Potential Energy — thunderstorm risk")

    st.divider()

    # Per-event prediction breakdown
    st.subheader("🎯 Event Predictions — All Model Types")
    analysis = data.get("analysis", {})
    model_type = data.get("model","rules_6h")
    st.caption(f"Active model: **{model_type}**")

    for event, ev_data in analysis.items():
        conf    = ev_data.get("confidence", 0)
        det     = ev_data.get("detected", False)
        blend   = ev_data.get("model_blend", {})
        attn_w  = ev_data.get("attn_weights", {})

        with st.expander(
            f"{'🔴' if det else '🟢'} {event.upper()} — "
            f"{'EXPECTED' if det else 'Not expected'} "
            f"(conf: {conf:.0%})",
            expanded=det,
        ):
            # Confidence bar
            st.progress(conf, text=f"Overall confidence: {conf:.0%}")

            b1, b2 = st.columns(2)
            with b1:
                if blend:
                    st.caption("**Model contributions:**")
                    for model_name, model_conf in blend.items():
                        st.write(f"• {model_name.upper()}: {model_conf:.0%}")
            with b2:
                if attn_w:
                    st.caption("**Top modalities (attention):**")
                    top3 = sorted(attn_w.items(), key=lambda x: x[1], reverse=True)[:3]
                    for mod, w in top3:
                        st.write(f"• {mod}: {w:.3f}")

    st.divider()
    render_alerts(city, analysis, alert_threshold)
