import json, os, sqlite3
from datetime import datetime, timezone
import mysql.connector
from dotenv import load_dotenv
load_dotenv()

# ── optional MongoDB ──────────────────────────────────────────────────────────
try:
    from pymongo import MongoClient
    _mongo_client = MongoClient(os.getenv("MONGO_URI","mongodb://localhost:27017/"),
                                serverSelectionTimeoutMS=1000)
    _mongo_client.admin.command("ping")
    _mongo_db     = _mongo_client["weather_text"]
    _MONGO_OK     = True
    print("[DB] MongoDB connected")
except Exception:
    _MONGO_OK = False


# ── backend selection ─────────────────────────────────────────────────────────

def _use_mysql():
    """Return True if MySQL creds are configured and server is reachable."""
    if not os.getenv("DB_PASSWORD") and not os.getenv("DB_HOST"):
        return False
    try:
        c = mysql.connector.connect(
            host=os.getenv("DB_HOST","localhost"),
            user=os.getenv("DB_USER","root"),
            password=os.getenv("DB_PASSWORD",""),
            database=os.getenv("DB_NAME","weather_db"),
            connection_timeout=3,
        )
        c.close()
        return True
    except Exception:
        return False


_BACKEND = "mysql" if _use_mysql() else "sqlite"
_SQLITE_PATH = os.getenv("SQLITE_PATH", "weather_data.db")
print(f"[DB] Using backend: {_BACKEND.upper()}")


# ── connection helpers ────────────────────────────────────────────────────────

def get_connection():
    if _BACKEND == "mysql":
        return mysql.connector.connect(
            host=os.getenv("DB_HOST","localhost"),
            user=os.getenv("DB_USER","root"),
            password=os.getenv("DB_PASSWORD",""),
            database=os.getenv("DB_NAME","weather_db"),
        )
    else:
        return sqlite3.connect(_SQLITE_PATH)


def _cursor(conn, dictionary=False):
    if _BACKEND == "mysql":
        return conn.cursor(dictionary=dictionary)
    else:
        if dictionary:
            conn.row_factory = sqlite3.Row
        return conn.cursor()


def _ph():
    """Placeholder character: %s for MySQL, ? for SQLite."""
    return "%s" if _BACKEND == "mysql" else "?"


def _rows_as_dicts(cursor):
    if _BACKEND == "mysql":
        return cursor.fetchall()
    else:
        rows = cursor.fetchall()
        return [dict(r) for r in rows] if rows else []


# ── schema ────────────────────────────────────────────────────────────────────

def init_db():
    conn   = get_connection()
    cursor = conn.cursor()
    P      = _ph()

    if _BACKEND == "mysql":
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_records (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                city             VARCHAR(100),
                timestamp        DATETIME,
                temperature      FLOAT,
                description      TEXT,
                cloud_score      FLOAT,
                heat_score       FLOAT,
                text_rain        FLOAT,
                text_heat        FLOAT,
                text_wind        FLOAT,
                text_snow        FLOAT,
                text_haze        FLOAT,
                analysis         JSON,
                raw_weather      JSON,
                feature_vector   JSON,
                cloud_img_path   TEXT,
                thermal_img_path TEXT,
                text_raw         JSON,
                lat              FLOAT,
                lon              FLOAT,
                fc_temp          FLOAT,
                fc_humidity      FLOAT,
                fc_wind          FLOAT,
                fc_rain          FLOAT,
                nwp_data         JSON,
                radar_data       JSON,
                elevation        FLOAT,
                land_use         VARCHAR(50)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labeled_events (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                record_id   INT,
                city        VARCHAR(100),
                timestamp   DATETIME,
                label_rain  INT DEFAULT 0,
                label_heat  INT DEFAULT 0,
                label_wind  INT DEFAULT 0,
                label_snow  INT DEFAULT 0,
                label_haze  INT DEFAULT 0,
                FOREIGN KEY (record_id) REFERENCES weather_records(id)
            )
        """)
        for ddl in [
            "CREATE INDEX IF NOT EXISTS idx_city_time ON weather_records (city, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_lat_lon   ON weather_records (lat, lon)",
        ]:
            try: cursor.execute(ddl)
            except mysql.connector.Error: pass
    else:
        # SQLite — add new columns to existing DB without destroying data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_records (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                city             TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                temperature      REAL,
                description      TEXT,
                cloud_score      REAL,
                heat_score       REAL,
                text_rain        REAL,
                text_heat        REAL,
                text_wind        REAL,
                text_snow        REAL,
                text_haze        REAL,
                analysis         TEXT,
                raw_weather      TEXT,
                feature_vector   TEXT,
                cloud_img_path   TEXT,
                thermal_img_path TEXT,
                text_raw         TEXT,
                lat              REAL,
                lon              REAL,
                fc_temp          REAL,
                fc_humidity      REAL,
                fc_wind          REAL,
                fc_rain          REAL,
                nwp_data         TEXT,
                radar_data       TEXT,
                elevation        REAL,
                land_use         TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labeled_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id   INTEGER,
                city        TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                label_rain  INTEGER DEFAULT 0,
                label_heat  INTEGER DEFAULT 0,
                label_wind  INTEGER DEFAULT 0,
                label_snow  INTEGER DEFAULT 0,
                label_haze  INTEGER DEFAULT 0
            )
        """)
        # Safely add new columns to EXISTING SQLite DB (ALTER TABLE IF NOT EXISTS)
        new_cols = [
            ("feature_vector",   "TEXT"),
            ("cloud_img_path",   "TEXT"),
            ("thermal_img_path", "TEXT"),
            ("text_raw",         "TEXT"),
            ("lat",              "REAL"),
            ("lon",              "REAL"),
            ("fc_temp",          "REAL"),
            ("fc_humidity",      "REAL"),
            ("fc_wind",          "REAL"),
            ("fc_rain",          "REAL"),
            ("nwp_data",         "TEXT"),
            ("radar_data",       "TEXT"),
            ("elevation",        "REAL"),
            ("land_use",         "TEXT"),
        ]
        cursor.execute("PRAGMA table_info(weather_records)")
        existing = {row[1] for row in cursor.fetchall()}
        for col_name, col_type in new_cols:
            if col_name not in existing:
                try:
                    cursor.execute(
                        f"ALTER TABLE weather_records ADD COLUMN {col_name} {col_type}"
                    )
                    print(f"[DB] Added column: {col_name}")
                except Exception:
                    pass
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_city_time "
                       "ON weather_records (city, timestamp)")

    conn.commit()
    cursor.close()
    conn.close()
    os.makedirs("data/images", exist_ok=True)
    print(f"[DB] Schema ready ({_BACKEND})")


# ── write helpers ─────────────────────────────────────────────────────────────

def save_analysis(city, weather_data, img_res, text_res, analysis,
                  raw_titles, lat, lon, feature_vector=None,
                  forecast_data=None, nwp_data=None,
                  radar_data=None, elevation=None, land_use=None):

    conn   = get_connection()
    cursor = conn.cursor()
    P      = _ph()
    fd     = forecast_data or {}

    cursor.execute(f"""
        INSERT INTO weather_records (
            city, timestamp, temperature, description,
            cloud_score, heat_score,
            text_rain, text_heat, text_wind, text_snow, text_haze,
            analysis, raw_weather,
            cloud_img_path, thermal_img_path, text_raw,
            lat, lon, feature_vector,
            fc_temp, fc_humidity, fc_wind, fc_rain,
            nwp_data, radar_data, elevation, land_use
        )
        VALUES ({P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},
                {P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P},{P})
    """, (
        city, datetime.now(timezone.utc),
        weather_data["main"]["temp"],
        weather_data["weather"][0]["description"],
        img_res.get("cloud", 0),
        img_res.get("heat",  0),
        text_res.get("rain", 0),
        text_res.get("heat", 0),
        text_res.get("wind", 0),
        text_res.get("snow", 0),
        text_res.get("haze", 0),
        json.dumps(analysis),
        json.dumps(weather_data),
        img_res.get("cloud_path"),
        img_res.get("thermal_path"),
        json.dumps(raw_titles),
        lat, lon,
        json.dumps(feature_vector) if feature_vector is not None else None,
        fd.get("temp"), fd.get("humidity"), fd.get("wind"), fd.get("rain"),
        json.dumps(nwp_data)    if nwp_data    else None,
        json.dumps(radar_data)  if radar_data  else None,
        elevation, land_use,
    ))

    conn.commit()
    if _BACKEND == "mysql":
        record_id = cursor.lastrowid
    else:
        record_id = cursor.lastrowid

    cursor.close()
    conn.close()

    if _MONGO_OK and raw_titles:
        try:
            _mongo_db.text_docs.insert_one({
                "record_id": record_id, "city": city,
                "timestamp": datetime.now(timezone.utc), "titles": raw_titles,
            })
        except Exception as e:
            print(f"[DB] Mongo write error: {e}")

    return record_id


def get_history(city, limit=50):
    conn   = get_connection()
    cursor = _cursor(conn, dictionary=True)
    P      = _ph()
    cursor.execute(f"""
        SELECT * FROM weather_records
        WHERE city = {P}
        ORDER BY timestamp DESC
        LIMIT {P}
    """, (city, limit))
    rows = _rows_as_dicts(cursor)
    cursor.close()
    conn.close()
    return rows


def get_nearby(lat, lon, radius_deg=1.0, limit=20):
    conn   = get_connection()
    cursor = _cursor(conn, dictionary=True)
    P      = _ph()
    cursor.execute(f"""
        SELECT * FROM weather_records
        WHERE lat BETWEEN {P} AND {P}
          AND lon BETWEEN {P} AND {P}
        ORDER BY timestamp DESC
        LIMIT {P}
    """, (lat-radius_deg, lat+radius_deg,
          lon-radius_deg, lon+radius_deg, limit))
    rows = _rows_as_dicts(cursor)
    cursor.close()
    conn.close()
    return rows


def export_training_data(city=None):
    """
    Export labeled rows for training.
    Only rows that have a feature_vector are returned.
    """
    conn   = get_connection()
    cursor = _cursor(conn, dictionary=True)
    P      = _ph()

    base = """
        SELECT
            r.city, r.timestamp,
            r.temperature, r.cloud_score, r.heat_score, r.feature_vector,
            r.text_rain, r.text_heat, r.text_wind, r.text_snow, r.text_haze,
            r.fc_temp, r.fc_humidity, r.fc_wind, r.fc_rain,
            r.lat, r.lon, r.elevation, r.land_use,
            l.label_rain, l.label_heat, l.label_wind, l.label_snow, l.label_haze
        FROM weather_records r
        JOIN labeled_events l ON l.record_id = r.id
        WHERE r.feature_vector IS NOT NULL
    """
    if city:
        cursor.execute(base + f" AND r.city = {P} ORDER BY r.timestamp ASC",
                       (city,))
    else:
        cursor.execute(base + " ORDER BY r.timestamp ASC")

    rows = _rows_as_dicts(cursor)
    cursor.close()
    conn.close()
    return rows


def save_labels(record_id, city, timestamp, labels: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    P      = _ph()
    try:
        cursor.execute(f"""
            INSERT INTO labeled_events
            (record_id, city, timestamp,
             label_rain, label_heat, label_wind, label_snow, label_haze)
            VALUES ({P},{P},{P},{P},{P},{P},{P},{P})
        """, (
            record_id, city, timestamp,
            labels["rain"], labels["heat"],
            labels["wind"], labels["snow"], labels["haze"],
        ))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_all_labeled_cities():
    conn   = get_connection()
    cursor = conn.cursor()
    P      = _ph()
    cursor.execute("""
        SELECT DISTINCT r.city
        FROM weather_records r
        JOIN labeled_events l ON l.record_id = r.id
        ORDER BY r.city
    """)
    cities = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return cities


# ── Migration: import old SQLite records into MySQL ───────────────────────────

def migrate_sqlite_to_mysql(sqlite_path="weather_data.db"):
    """
    One-time migration: copy all records from the old SQLite DB into MySQL.
    Old records that have no feature_vector are migrated as-is;
    they will be excluded from training (feature_vector IS NOT NULL filter)
    but remain visible in the dashboard History page.

    Usage: python database.py --migrate
    """
    if _BACKEND != "mysql":
        print("[Migrate] Target backend is not MySQL — nothing to do.")
        return

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    src_cur = src.cursor()

    src_cur.execute("SELECT * FROM weather_records ORDER BY id ASC")
    old_rows = [dict(r) for r in src_cur.fetchall()]

    src_cur.execute("SELECT * FROM labeled_events ORDER BY id ASC")
    old_labels = [dict(r) for r in src_cur.fetchall()]
    src.close()

    if not old_rows:
        print("[Migrate] No rows in source SQLite DB.")
        return

    dst     = get_connection()
    dst_cur = dst.cursor()

    # build old_id → new_id map
    id_map = {}
    for r in old_rows:
        dst_cur.execute("""
            INSERT INTO weather_records
            (city, timestamp, temperature, description,
             cloud_score, heat_score,
             text_rain, text_heat, text_wind, text_snow, text_haze,
             analysis, raw_weather,
             cloud_img_path, thermal_img_path, text_raw,
             lat, lon, feature_vector,
             fc_temp, fc_humidity, fc_wind, fc_rain,
             nwp_data, radar_data, elevation, land_use)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            r["city"], r["timestamp"], r.get("temperature"),
            r.get("description"),
            r.get("cloud_score"), r.get("heat_score"),
            r.get("text_rain"), r.get("text_heat"),
            r.get("text_wind"), r.get("text_snow"), r.get("text_haze"),
            r.get("analysis"),  r.get("raw_weather"),
            r.get("cloud_img_path"), r.get("thermal_img_path"), r.get("text_raw"),
            r.get("lat"), r.get("lon"), r.get("feature_vector"),
            r.get("fc_temp"), r.get("fc_humidity"), r.get("fc_wind"), r.get("fc_rain"),
            r.get("nwp_data"), r.get("radar_data"), r.get("elevation"), r.get("land_use"),
        ))
        dst.commit()
        id_map[r["id"]] = dst_cur.lastrowid

    for lbl in old_labels:
        new_rid = id_map.get(lbl["record_id"])
        if new_rid:
            dst_cur.execute("""
                INSERT INTO labeled_events
                (record_id, city, timestamp,
                 label_rain, label_heat, label_wind, label_snow, label_haze)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                new_rid, lbl["city"], lbl["timestamp"],
                lbl.get("label_rain",0), lbl.get("label_heat",0),
                lbl.get("label_wind",0), lbl.get("label_snow",0),
                lbl.get("label_haze",0),
            ))
        dst.commit()

    dst_cur.close()
    dst.close()
    print(f"[Migrate] {len(old_rows)} records, {len(old_labels)} labels → MySQL")


# ── per-city label thresholds (fixes climate-zone bias) ──────────────────────
# These are used by main.py's auto_generate_labels() instead of fixed globals.
# Each city's "normal" temperature range is calibrated so that extreme-heat
# labels are relative to that city's climate, not a single global threshold.

CITY_CLIMATE = {
    # city            : (heat_thresh_°C, cold_thresh_°C, wind_thresh_m/s)
    "default"         : (35, 5, 10),

    # Hot / tropical
    "Mumbai"          : (38, 15, 12),
    "Delhi"           : (42, 10, 12),
    "Chennai"         : (40, 18, 12),
    "Dubai"           : (44, 15, 12),
    "Lagos"           : (38, 20, 10),
    "Cairo"           : (42, 12, 12),
    "Bangkok"         : (38, 20, 10),
    "Hyderabad"       : (40, 15, 12),
    "Bangalore"       : (36, 15, 12),
    "Kolkata"         : (40, 12, 12),

    # Temperate
    "London"          : (30, 2, 10),
    "Paris"           : (32, 2, 10),
    "Berlin"          : (32, -2, 10),
    "New York"        : (34, -2, 12),
    "Chicago"         : (33, -5, 14),
    "Toronto"         : (32, -5, 12),
    "Amsterdam"       : (28, 0, 10),
    "Madrid"          : (38, 2, 10),
    "Rome"            : (36, 2, 10),

    # Cold / subarctic
    "Moscow"          : (28, -15, 12),
    "Srinagar"        : (30, -8, 10),
    "Vancouver"       : (28, -2, 10),
    "Auckland"        : (28, 5, 14),

    # Southern Hemisphere / mixed
    "Sydney"          : (36, 8, 12),
    "Melbourne"       : (34, 5, 14),
    "Cape Town"       : (34, 8, 14),
    "Buenos Aires"    : (36, 5, 12),
    "São Paulo"       : (36, 12, 10),
    "Nairobi"         : (32, 10, 10),
}


def get_city_thresholds(city):
    """Return (heat_thresh, cold_thresh, wind_thresh) for a city."""
    if not city:
        return CITY_CLIMATE["default"]
    normalized = city.strip().title()
    return CITY_CLIMATE.get(normalized, CITY_CLIMATE.get(city, CITY_CLIMATE["default"]))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    init_db()
    if "--migrate" in sys.argv:
        src = sys.argv[sys.argv.index("--migrate") + 1] \
              if len(sys.argv) > sys.argv.index("--migrate") + 1 \
              else "weather_data.db"
        migrate_sqlite_to_mysql(src)
