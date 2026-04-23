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
    st.title("📊 History & Trends")

    # ── Load raw records for all selected cities ───────────────────────────
    all_data = []
    for c in cities:
        records = fetch_history(c)
        if records:
            df = pd.DataFrame(records)
            df["city"] = c
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

    if not all_data:
        st.info("No historical data found for the selected cities. Run an analysis first.")
        st.stop()

    df_all = pd.concat(all_data).sort_values("timestamp").reset_index(drop=True)

    # ── Parse JSON columns into structured fields ──────────────────────────
    def safe_json(val):
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        try:
            return json.loads(val)
        except Exception:
            return {}

    df_all["analysis_parsed"]   = df_all["analysis"].apply(safe_json)
    df_all["raw_weather_parsed"] = df_all["raw_weather"].apply(safe_json)
    df_all["text_raw_parsed"]   = df_all["text_raw"].apply(safe_json)

    # Flatten analysis confidence scores into individual columns
    for event in ["rain", "heat", "wind", "snow", "haze"]:
        df_all[f"conf_{event}"] = df_all["analysis_parsed"].apply(
            lambda a: a.get(event, {}).get("confidence", 0) if isinstance(a, dict) else 0
        )
        df_all[f"det_{event}"] = df_all["analysis_parsed"].apply(
            lambda a: bool(a.get(event, {}).get("detected", False)) if isinstance(a, dict) else False
        )

    # Flatten raw_weather fields
    df_all["humidity"]   = df_all["raw_weather_parsed"].apply(lambda w: w.get("main", {}).get("humidity"))
    df_all["pressure"]   = df_all["raw_weather_parsed"].apply(lambda w: w.get("main", {}).get("pressure"))
    df_all["wind_speed"] = df_all["raw_weather_parsed"].apply(lambda w: w.get("wind", {}).get("speed"))
    df_all["weather_id"] = df_all["raw_weather_parsed"].apply(
        lambda w: w.get("weather", [{}])[0].get("id") if w.get("weather") else None
    )

    # ── Filters (sidebar-style in a horizontal bar) ────────────────────────
    st.markdown("### 🔍 Filters")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    with f1:
        city_filter = st.multiselect(
            "City", options=sorted(df_all["city"].unique()), default=sorted(df_all["city"].unique())
        )
    with f2:
        min_date = df_all["timestamp"].min().date()
        max_date = df_all["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with f3:
        event_filter = st.multiselect(
            "Event detected",
            options=["rain", "heat", "wind", "snow", "haze"],
            default=[],
            help="Show only records where these events were detected",
        )
    with f4:
        min_conf = st.slider("Min confidence (any event)", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    df_filtered = df_all[df_all["city"].isin(city_filter)].copy()

    if len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        df_filtered = df_filtered[
            (df_filtered["timestamp"] >= start_dt) &
            (df_filtered["timestamp"] <  end_dt)
        ]

    if event_filter:
        mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        for ev in event_filter:
            mask = mask | df_filtered[f"det_{ev}"]
        df_filtered = df_filtered[mask]

    if min_conf > 0:
        conf_cols = [f"conf_{e}" for e in ["rain", "heat", "wind", "snow", "haze"]]
        max_conf_per_row = df_filtered[conf_cols].max(axis=1)
        df_filtered = df_filtered[max_conf_per_row >= min_conf]

    st.caption(f"Showing **{len(df_filtered)}** records after filters (total: {len(df_all)})")
    st.divider()

    if df_filtered.empty:
        st.warning("No records match the current filters.")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Trend Charts
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("## 📈 Trend Charts")

    tab_temp, tab_scores, tab_conf, tab_signals = st.tabs([
        "🌡 Temperature", "☁️ Image Scores", "📊 Event Confidence", "📰 Text Signals"
    ])

    with tab_temp:
        pivot_temp = df_filtered.pivot_table(
            index="timestamp", columns="city", values="temperature", aggfunc="mean"
        )
        st.line_chart(pivot_temp, use_container_width=True)

        # Also show humidity & pressure if available
        if df_filtered["humidity"].notna().any():
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Humidity (%)")
                pivot_hum = df_filtered.pivot_table(
                    index="timestamp", columns="city", values="humidity", aggfunc="mean"
                )
                st.line_chart(pivot_hum, use_container_width=True)
            with c2:
                st.caption("Wind Speed (m/s)")
                pivot_wind = df_filtered.pivot_table(
                    index="timestamp", columns="city", values="wind_speed", aggfunc="mean"
                )
                st.line_chart(pivot_wind, use_container_width=True)

    with tab_scores:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Cloud Score (satellite)")
            pivot_cloud = df_filtered.pivot_table(
                index="timestamp", columns="city", values="cloud_score", aggfunc="mean"
            )
            st.line_chart(pivot_cloud, use_container_width=True)
        with c2:
            st.caption("Heat Score (thermal)")
            pivot_heat = df_filtered.pivot_table(
                index="timestamp", columns="city", values="heat_score", aggfunc="mean"
            )
            st.line_chart(pivot_heat, use_container_width=True)

    with tab_conf:
        for event, icon in [("rain","🌧"), ("heat","🔥"), ("wind","💨"), ("snow","❄️"), ("haze","🌫")]:
            if f"conf_{event}" in df_filtered.columns:
                st.caption(f"{icon} {event.upper()} confidence over time")
                pivot_ev = df_filtered.pivot_table(
                    index="timestamp", columns="city", values=f"conf_{event}", aggfunc="mean"
                )
                st.line_chart(pivot_ev, use_container_width=True)

    with tab_signals:
        text_cols = {"text_rain": "🌧 Rain", "text_heat": "🔥 Heat",
                     "text_wind": "💨 Wind", "text_snow": "❄️ Snow", "text_haze": "🌫 Haze"}
        available = [c for c in text_cols if c in df_filtered.columns]
        if available:
            for col in available:
                pivot_txt = df_filtered.pivot_table(
                    index="timestamp", columns="city", values=col, aggfunc="mean"
                )
                st.caption(text_cols[col] + " text signal")
                st.line_chart(pivot_txt, use_container_width=True)
        else:
            st.info("No text signal columns available.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Structured Records Table
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("## 🗂 Structured Records")

    # Build a clean display dataframe
    display_cols = {
        "id":          "ID",
        "city":        "City",
        "timestamp":   "Timestamp",
        "temperature": "Temp °C",
        "description": "Condition",
        "humidity":    "Humidity %",
        "wind_speed":  "Wind m/s",
        "cloud_score": "Cloud Score",
        "heat_score":  "Heat Score",
        "conf_rain":   "Rain Conf",
        "conf_heat":   "Heat Conf",
        "conf_wind":   "Wind Conf",
        "conf_snow":   "Snow Conf",
        "conf_haze":   "Haze Conf",
        "det_rain":    "Rain 🌧",
        "det_heat":    "Heat 🔥",
        "det_wind":    "Wind 💨",
        "det_snow":    "Snow ❄️",
        "det_haze":    "Haze 🌫",
    }
    avail = {k: v for k, v in display_cols.items() if k in df_filtered.columns}
    df_display = (
        df_filtered[list(avail.keys())]
        .rename(columns=avail)
        .sort_values("Timestamp", ascending=False)
        .reset_index(drop=True)
    )

    # Format confidence columns as percentages
    for col in ["Rain Conf", "Heat Conf", "Wind Conf", "Snow Conf", "Haze Conf"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.0%}")

    # Format boolean detection columns
    for col in ["Rain 🌧", "Heat 🔥", "Wind 💨", "Snow ❄️", "Haze 🌫"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: "🔴" if x else "🟢")

    st.dataframe(df_display, use_container_width=True, height=320)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — Per-Record Detail Cards (expandable)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("## 🔎 Record Detail Explorer")
    st.caption("Click any record to inspect all parsed fields — weather, image scores, event analysis, news headlines.")

    records_to_show = df_filtered.sort_values("timestamp", ascending=False).head(50)

    EVENT_ICONS = {"rain": "🌧", "heat": "🔥", "wind": "💨", "snow": "❄️", "haze": "🌫"}

    for _, row in records_to_show.iterrows():
        # Build expander label
        det_tags = " ".join(
            EVENT_ICONS[e] for e in ["rain", "heat", "wind", "snow", "haze"]
            if row.get(f"det_{e}", False)
        ) or "✅ Clear"

        label = (
            f"#{int(row['id'])}  |  {row['city']}  |  "
            f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')}  |  "
            f"{row.get('temperature', '--')}°C — {row.get('description', '').title()}  |  {det_tags}"
        )

        with st.expander(label, expanded=False):
            col_w, col_i, col_a, col_n = st.columns([1.2, 1, 1.4, 1.4])

            # ── Weather ──────────────────────────────────────────────────
            with col_w:
                st.markdown("**🌤 Weather**")
                st.markdown(f"- **Temp:** {row.get('temperature', '--')} °C")
                st.markdown(f"- **Condition:** {str(row.get('description', '--')).title()}")
                rw = row["raw_weather_parsed"]
                main = rw.get("main", {})
                wind = rw.get("wind", {})
                clouds = rw.get("clouds", {})
                st.markdown(f"- **Humidity:** {main.get('humidity', '--')} %")
                st.markdown(f"- **Pressure:** {main.get('pressure', '--')} hPa")
                st.markdown(f"- **Wind:** {wind.get('speed', '--')} m/s")
                st.markdown(f"- **Clouds:** {clouds.get('all', '--')} %")
                st.markdown(f"- **Lat/Lon:** {row.get('lat', '--')}, {row.get('lon', '--')}")

            # ── Image scores ─────────────────────────────────────────────
            with col_i:
                st.markdown("**🛰 Image Scores**")
                st.markdown(f"- **Cloud score:** `{row.get('cloud_score', 0):.3f}`")
                st.markdown(f"- **Heat score:** `{row.get('heat_score', 0):.3f}`")
                st.markdown("**📰 Text Signals**")
                for sig, label_txt in [
                    ("text_rain", "Rain"), ("text_heat", "Heat"),
                    ("text_wind", "Wind"), ("text_snow", "Snow"), ("text_haze", "Haze")
                ]:
                    val = row.get(sig, 0) or 0
                    st.markdown(f"- **{label_txt}:** `{val:.2f}`")

            # ── Event analysis ────────────────────────────────────────────
            with col_a:
                st.markdown("**📡 Event Analysis**")
                analysis = row["analysis_parsed"]
                if analysis:
                    for event, icon in EVENT_ICONS.items():
                        ev = analysis.get(event, {})
                        detected = ev.get("detected", False)
                        conf     = ev.get("confidence", 0)
                        badge    = "🔴" if detected else "🟢"
                        bar_pct  = int(conf * 100)
                        st.markdown(
                            f"{badge} **{icon} {event.upper()}** — `{conf:.0%}` conf  "
                            f"<div style='background:#2a2d3e;border-radius:3px;height:5px;margin:2px 0 6px'>"
                            f"<div style='width:{bar_pct}%;background:linear-gradient(90deg,#4cc9f0,#7209b7);height:5px;border-radius:3px'></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        sources = ev.get("sources", {})
                        if sources:
                            src_str = ", ".join(k for k, v in sources.items() if v)
                            if src_str:
                                st.caption(f"   Sources: {src_str}")
                else:
                    st.info("No analysis data")

            # ── News headlines ────────────────────────────────────────────
            with col_n:
                st.markdown("**📰 News Headlines**")
                titles = row.get("text_raw_parsed")
                if isinstance(titles, list) and titles:
                    for t in titles[:5]:
                        st.markdown(f"- {t}")
                elif isinstance(titles, dict):
                    for k, v in list(titles.items())[:5]:
                        st.markdown(f"- {v if isinstance(v, str) else k}")
                else:
                    st.caption("No headlines stored")

            # ── Satellite image URLs ──────────────────────────────────────
            cloud_path   = row.get("cloud_img_path")
            thermal_path = row.get("thermal_img_path")
            if cloud_path or thermal_path:
                st.markdown("**🛰 Stored Satellite Images**")
                ic1, ic2 = st.columns(2)
                with ic1:
                    if cloud_path and os.path.exists(str(cloud_path)):
                        st.image(cloud_path, caption="Cloud (VIIRS)", use_container_width=True)
                    else:
                        st.caption("Cloud image not on this machine")
                with ic2:
                    if thermal_path and os.path.exists(str(thermal_path)):
                        st.image(thermal_path, caption="Thermal (MODIS)", use_container_width=True)
                    else:
                        st.caption("Thermal image not on this machine")


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