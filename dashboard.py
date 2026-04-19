import streamlit as st
import requests
import pandas as pd
import json
import mysql.connector
import os
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
        background: #1a1d2e;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #2a2d3e;
        margin-bottom: 10px;
    }
    .detected { color: #ff6b6b; font-weight: 700; }
    .clear    { color: #51cf66; font-weight: 700; }
    .conf-bar  { height: 6px; background: #2a2d3e; border-radius: 3px; margin-top: 6px; }
    .conf-fill { height: 6px; border-radius: 3px; background: linear-gradient(90deg,#4cc9f0,#7209b7); }

    /* ── Alert banner styles ── */
    .alert-critical {
        background: linear-gradient(135deg, #3d0000, #7a0000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
        animation: pulse-red 1.5s infinite;
    }
    .alert-high {
        background: linear-gradient(135deg, #3d1a00, #7a3800);
        border: 2px solid #ff8800;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .alert-medium {
        background: linear-gradient(135deg, #2a2a00, #4a4a00);
        border: 2px solid #ffdd00;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .alert-title {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .alert-body {
        font-size: 13px;
        color: #cccccc;
        margin-top: 4px;
    }
    .alert-conf {
        font-size: 12px;
        color: #aaaaaa;
        margin-top: 6px;
    }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0px #ff4444; }
        50%  { box-shadow: 0 0 14px #ff4444; }
        100% { box-shadow: 0 0 0px #ff4444; }
    }
    .no-alert {
        background: #0d2b0d;
        border: 1px solid #2a6a2a;
        border-radius: 10px;
        padding: 14px 18px;
        color: #51cf66;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌩 WeatherFusion")
    st.caption("Multimodal Extreme Weather Predictor")
    st.divider()

    city_input = st.text_input(
        "Enter cities (comma separated)",
        value="Mumbai"
    )
    cities = [c.strip() for c in city_input.split(",") if c.strip()]

    if not cities:
        st.warning("Enter at least one city")
        st.stop()

    st.markdown(f"**Selected Cities:** {', '.join(cities)}")

    run_btn = st.button("Analyze", use_container_width=True, type="primary")
    st.divider()

    page = st.radio("View", ["Live Analysis", "History & Trends", "Compare Cities", "Label & Train"])

    # Alert threshold slider in sidebar
    st.divider()
    st.markdown("**⚙️ Alert Settings**")
    alert_threshold = st.slider(
        "Alert confidence threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Events with confidence above this value will trigger a screen alert"
    )


# ── Cached fetchers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_analysis(city):
    try:
        r = requests.get(f"{API_BASE}/analyze/{city}", timeout=30)
        data = r.json()
        if "error" in data:
            return {"error": f"{city}: {data['error']}"}
        return data
    except Exception as e:
        return {"error": f"{city}: {str(e)}"}


@st.cache_data(ttl=60)
def fetch_history(city):
    try:
        r = requests.get(f"{API_BASE}/history/{city}?limit=100", timeout=10)
        return r.json().get("records", [])
    except Exception:
        return []


# ── Alert helper ───────────────────────────────────────────────────────────────
EVENT_META = {
    "rain": {
        "icon": "🌧",
        "critical_msg": "Severe rainfall / flooding risk. Avoid low-lying areas.",
        "high_msg":     "Heavy rain expected. Carry rain gear, expect delays.",
        "medium_msg":   "Rain likely. Keep an umbrella handy.",
    },
    "heat": {
        "icon": "🔥",
        "critical_msg": "Dangerous heat conditions. Stay indoors, stay hydrated.",
        "high_msg":     "Extreme heat warning. Limit outdoor activity.",
        "medium_msg":   "High temperatures expected. Stay cool and drink water.",
    },
    "wind": {
        "icon": "💨",
        "critical_msg": "Violent storm / cyclone risk. Seek shelter immediately.",
        "high_msg":     "Strong winds expected. Secure loose objects outdoors.",
        "medium_msg":   "Windy conditions. Be cautious while driving.",
    },
    "snow": {
        "icon": "❄️",
        "critical_msg": "Blizzard / heavy snowfall. Do not travel unless essential.",
        "high_msg":     "Heavy snow expected. Roads may be slippery.",
        "medium_msg":   "Light snow possible. Drive carefully.",
    },
    "haze": {
        "icon": "🌫",
        "critical_msg": "Hazardous air quality. Stay indoors, use air purifiers.",
        "high_msg":     "Poor air quality. Wear N95 mask outdoors.",
        "medium_msg":   "Moderate haze. Sensitive groups should limit exposure.",
    },
}

def render_alerts(city, analysis, threshold):
    """
    Renders on-screen alert banners based on prediction confidence.
    Tiers:
      critical  → confidence >= 0.80   (red pulsing banner)
      high      → confidence >= threshold and < 0.80  (orange banner)
      medium    → confidence >= threshold - 0.15 and < threshold  (yellow banner)
    Any detected event below threshold is silently ignored in alerts
    (still shown in event cards below).
    """
    critical_threshold = 0.80
    medium_threshold   = max(0.30, threshold - 0.15)

    critical_events = []
    high_events     = []
    medium_events   = []

    for event, result in analysis.items():
        conf    = result.get("confidence", 0)
        detected = result.get("detected", False)

        # Only alert if detected OR confidence is high enough even without hard detection
        if conf >= critical_threshold:
            critical_events.append((event, conf))
        elif conf >= threshold and detected:
            high_events.append((event, conf))
        elif conf >= medium_threshold and detected:
            medium_events.append((event, conf))

    # Sort by confidence desc within each tier
    critical_events.sort(key=lambda x: x[1], reverse=True)
    high_events.sort(key=lambda x: x[1], reverse=True)
    medium_events.sort(key=lambda x: x[1], reverse=True)

    has_any_alert = critical_events or high_events or medium_events

    st.markdown("### 🚨 Alert Status")

    if not has_any_alert:
        st.markdown(
            '<div class="no-alert">✅ &nbsp; No extreme weather alerts for '
            f'{city} at this time.</div>',
            unsafe_allow_html=True
        )
        return

    # ── Critical alerts ────────────────────────────────────────────────────
    for event, conf in critical_events:
        meta = EVENT_META.get(event, {})
        icon = meta.get("icon", "⚠️")
        msg  = meta.get("critical_msg", "Extreme conditions detected.")
        st.markdown(f"""
        <div class="alert-critical">
            <div class="alert-title">🚨 {icon} CRITICAL — {event.upper()} in {city}</div>
            <div class="alert-body">{msg}</div>
            <div class="alert-conf">Model confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)

    # ── High alerts ────────────────────────────────────────────────────────
    for event, conf in high_events:
        meta = EVENT_META.get(event, {})
        icon = meta.get("icon", "⚠️")
        msg  = meta.get("high_msg", "Significant weather event expected.")
        st.markdown(f"""
        <div class="alert-high">
            <div class="alert-title">⚠️ {icon} HIGH ALERT — {event.upper()} in {city}</div>
            <div class="alert-body">{msg}</div>
            <div class="alert-conf">Model confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)

    # ── Medium alerts ──────────────────────────────────────────────────────
    for event, conf in medium_events:
        meta = EVENT_META.get(event, {})
        icon = meta.get("icon", "⚠️")
        msg  = meta.get("medium_msg", "Moderate weather event possible.")
        st.markdown(f"""
        <div class="alert-medium">
            <div class="alert-title">🔔 {icon} ADVISORY — {event.upper()} in {city}</div>
            <div class="alert-body">{msg}</div>
            <div class="alert-conf">Model confidence: <strong>{conf:.0%}</strong></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "Live Analysis":
    city = cities[0]
    st.title(f"Live Analysis — {city}")

    if run_btn or "last_data" not in st.session_state:
        with st.spinner("Fetching satellite imagery and weather data..."):
            data = fetch_analysis(city)
            st.session_state["last_data"] = data
            st.session_state["last_city"] = city
    else:
        data = st.session_state.get("last_data", {})

    if "error" in data:
        st.error(f"API error: {data['error']}")
        st.stop()

    # ── Top metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{data.get('temperature', '--')} °C")
    col2.metric("Condition",   data.get("weather", "--").title())
    col3.metric("Model type",  data.get("model", "rules").upper())
    col4.metric("City",        city)

    st.divider()

    # ── ALERT BANNERS — rendered immediately after metrics ─────────────────
    analysis = data.get("analysis", {})
    render_alerts(city, analysis, alert_threshold)

    st.divider()

    # ── Event detection cards ──────────────────────────────────────────────
    st.subheader("📡 Event Detection")
    events = list(analysis.keys())
    cols   = st.columns(len(events))
    icons  = {"rain": "🌧", "heat": "🔥", "wind": "💨", "snow": "❄️", "haze": "🌫"}

    for col, event in zip(cols, events):
        ev   = analysis[event]
        det  = ev.get("detected", False)
        conf = ev.get("confidence", 0)
        with col:
            status = "🔴 DETECTED" if det else "🟢 Clear"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px">{icons.get(event, '🌡')}</div>
                <div style="font-size:15px;font-weight:600;margin:4px 0">{event.upper()}</div>
                <div class="{'detected' if det else 'clear'}">{status}</div>
                <div style="font-size:12px;color:#888;margin-top:4px">
                    Confidence: {conf:.0%}</div>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:{conf*100:.0f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Satellite images ───────────────────────────────────────────────────
    img_col1, img_col2 = st.columns(2)
    img_res = data.get("image_analysis", {})

    with img_col1:
        st.subheader("☁️ Cloud cover (VIIRS)")
        cloud_url = img_res.get("cloud_image")
        if cloud_url:
            st.image(cloud_url, use_container_width=True)
            st.caption(f"Cloud score: {img_res.get('cloud', 0):.2f}")
        else:
            st.info("Image unavailable")

    with img_col2:
        st.subheader("🌡 Land surface temperature (MODIS)")
        thermal_url = img_res.get("thermal_image")
        if thermal_url:
            st.image(thermal_url, use_container_width=True)
            st.caption(f"Heat score: {img_res.get('heat', 0):.2f}")
        else:
            st.info("Image unavailable")

    # ── Text analysis breakdown ────────────────────────────────────────────
    st.divider()
    st.subheader("📰 Text analysis scores")
    text_res = data.get("text_analysis", {})
    if text_res:
        df_text = pd.DataFrame(
            {"Event": list(text_res.keys()), "Score": list(text_res.values())}
        )
        st.bar_chart(df_text.set_index("Event"), use_container_width=True)

    # ── Confidence history chart ───────────────────────────────────────────
    st.divider()
    st.subheader("📈 Confidence History (last 20 records)")
    hist = fetch_history(city)
    if hist:
        df_hist = pd.DataFrame(hist)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist.sort_values("timestamp").tail(20)

        conf_rows = []
        for _, row in df_hist.iterrows():
            try:
                ana = (json.loads(row["analysis"])
                       if isinstance(row["analysis"], str)
                       else row["analysis"])
                entry = {"timestamp": row["timestamp"]}
                for ev in ["rain", "heat", "wind", "snow", "haze"]:
                    entry[ev] = ana.get(ev, {}).get("confidence", 0)
                conf_rows.append(entry)
            except Exception:
                pass

        if conf_rows:
            df_conf = pd.DataFrame(conf_rows).set_index("timestamp")
            st.line_chart(df_conf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History & Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History & Trends":
    st.title("📊 Multi-City Trends")

    all_data = []
    for c in cities:
        records = fetch_history(c)
        if records:
            df = pd.DataFrame(records)
            df["city"] = c
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

    if not all_data:
        st.info("No data available")
        st.stop()

    df_all = pd.concat(all_data).sort_values("timestamp")

    st.subheader("🌡 Temperature Comparison")
    pivot_temp = df_all.pivot_table(
        index="timestamp", columns="city", values="temperature", aggfunc="mean"
    )
    st.line_chart(pivot_temp, use_container_width=True)

    st.subheader("☁️ Cloud Score Comparison")
    pivot_cloud = df_all.pivot_table(
        index="timestamp", columns="city", values="cloud_score", aggfunc="mean"
    )
    st.line_chart(pivot_cloud, use_container_width=True)

    st.subheader("🔥 Heat Score Comparison")
    pivot_heat = df_all.pivot_table(
        index="timestamp", columns="city", values="heat_score", aggfunc="mean"
    )
    st.line_chart(pivot_heat, use_container_width=True)

    st.subheader("🌧 Text Rain Signal Over Time")
    pivot_text = df_all.pivot_table(
        index="timestamp", columns="city", values="text_rain", aggfunc="mean"
    )
    st.line_chart(pivot_text, use_container_width=True)

    st.divider()
    st.subheader("Raw records")
    st.dataframe(
        df_all[["timestamp", "city", "temperature",
                "cloud_score", "heat_score", "description"]].tail(20),
        use_container_width=True,
    )

    latest       = df_all.sort_values("timestamp").tail(1)
    cloud_path   = latest["cloud_img_path"].values[0]
    thermal_path = latest["thermal_img_path"].values[0]

    st.subheader("🛰 Latest Stored Images")
    col1, col2 = st.columns(2)
    with col1:
        if cloud_path and os.path.exists(str(cloud_path)):
            st.image(cloud_path, caption="Cloud Image")
        else:
            st.info("Cloud image not available on this machine")
    with col2:
        if thermal_path and os.path.exists(str(thermal_path)):
            st.image(thermal_path, caption="Thermal Image")
        else:
            st.info("Thermal image not available on this machine")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Compare Cities
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare Cities":
    st.title("🌍 Compare Cities Side-by-Side")

    if len(cities) < 2:
        st.warning("Enter at least 2 cities in the sidebar to compare.")
        st.stop()

    if run_btn or "compare_data" not in st.session_state:
        with st.spinner("Fetching data for all cities..."):
            params = "&".join(f"cities={c}" for c in cities)
            try:
                r = requests.get(f"{API_BASE}/compare?{params}", timeout=120)
                compare_data = r.json()
                st.session_state["compare_data"] = compare_data
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
    else:
        compare_data = st.session_state["compare_data"]

    # ── Alert banners for ALL cities ───────────────────────────────────────
    st.markdown("### 🚨 Alerts Across All Cities")
    any_alert = False
    for city_name, d in compare_data.items():
        if "error" in d:
            continue
        city_analysis = d.get("analysis", {})
        # Check if any event triggers an alert
        has_alert = any(
            v.get("confidence", 0) >= max(0.30, alert_threshold - 0.15)
            and v.get("detected", False)
            for v in city_analysis.values()
        )
        if has_alert:
            any_alert = True
            render_alerts(city_name, city_analysis, alert_threshold)

    if not any_alert:
        st.markdown(
            '<div class="no-alert">✅ &nbsp; No extreme weather alerts for any selected city.</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Comparison table ───────────────────────────────────────────────────
    rows = []
    for city_name, d in compare_data.items():
        if "error" in d:
            rows.append({"City": city_name, "Error": d["error"]})
            continue
        row = {
            "City":      city_name,
            "Temp °C":   d.get("temperature", "--"),
            "Condition": d.get("weather", "--"),
        }
        for event, ev in d.get("analysis", {}).items():
            row[event.capitalize()] = "🔴" if ev["detected"] else "🟢"
            row[f"{event.capitalize()} conf"] = f"{ev['confidence']:.0%}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("City"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Label & Train
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Label & Train":
    st.title("🏷️ Label Records & Trigger Training")

    st.info(
        "Label historical records so you can train the ML model. "
        "After labeling, click Train model."
    )

    # ── Connect to MySQL ───────────────────────────────────────────────────
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("DB_PASSWORD", "password"),
            database="weather_db",
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT r.id, r.city, r.timestamp, r.temperature,
                   r.description, r.cloud_score, r.heat_score
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

    # ── Labeling UI ────────────────────────────────────────────────────────
    if not rows:
        st.success("All records labeled!")
    else:
        st.write(f"**{len(rows)} unlabeled records**")
        label_data = []

        for row in rows:
            rid, rcity, rtime, rtemp, rdesc, rcloud, rheat = row
            with st.expander(
                f"#{rid} — {rcity} | {str(rtime)[:16]} | {rtemp}°C | {rdesc}"
            ):
                c1, c2, c3, c4, c5 = st.columns(5)
                rain = c1.checkbox("Rain", key=f"r_{rid}")
                heat = c2.checkbox("Heat", key=f"h_{rid}")
                wind = c3.checkbox("Wind", key=f"w_{rid}")
                snow = c4.checkbox("Snow", key=f"sn_{rid}")
                haze = c5.checkbox("Haze", key=f"hz_{rid}")
                label_data.append((
                    rid, rcity, rtime,
                    int(rain), int(heat), int(wind), int(snow), int(haze)
                ))

        if st.button("Save labels", type="primary"):
            try:
                cursor.executemany("""
                    INSERT INTO labeled_events
                    (record_id, city, timestamp,
                     label_rain, label_heat, label_wind, label_snow, label_haze)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, label_data)
                conn.commit()
                st.success("Labels saved!")
            except Exception as e:
                st.error(f"Error saving labels: {e}")

        conn.close()

    # ── Train model ────────────────────────────────────────────────────────
    st.divider()

    if st.button("🚀 Train model now"):
        with st.spinner("Training... (this may take a minute)"):
            import subprocess
            result = subprocess.run(
                ["python", "train_model.py"],
                capture_output=True, text=True,
            )

        if result.returncode == 0:
            st.success("Training complete!")
            st.code(result.stdout)
            # Reload models in the running API so new weights are used immediately
            try:
                requests.post(f"{API_BASE}/reload-models", timeout=5)
                st.info("✅ ML models reloaded in API — no restart needed")
            except Exception:
                st.warning("⚠️ Could not auto-reload API models. Restart uvicorn to apply.")
        else:
            st.error("Training failed")
            st.code(result.stderr)

    # ── Eval results ───────────────────────────────────────────────────────
    if os.path.exists("models/eval_results.json"):
        with open("models/eval_results.json") as f:
            eval_res = json.load(f)
        st.subheader("📊 Evaluation Results")
        st.json(eval_res)

    st.divider()
    st.caption("Requires at least ~30 labeled samples with positive examples per event.")