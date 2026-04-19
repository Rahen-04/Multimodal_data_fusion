import mysql.connector
import json
from datetime import datetime
import os


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("DB_PASSWORD"),
        database="weather_db"
    )


def init_db():
    conn   = get_connection()
    cursor = conn.cursor()

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
            lon              FLOAT
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

    try:
        cursor.execute("""
            CREATE INDEX idx_city_time ON weather_records (city, timestamp)
        """)
    except mysql.connector.Error:
        pass  # already exists

    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Initialized MySQL database")


def save_analysis(city, weather_data, img_res, text_res, analysis,
                  raw_titles, lat, lon, feature_vector=None):
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weather_records (
            city, timestamp, temperature, description,
            cloud_score, heat_score,
            text_rain, text_heat, text_wind, text_snow, text_haze,
            analysis, raw_weather,
            cloud_img_path, thermal_img_path, text_raw,
            lat, lon, feature_vector
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        city,
        datetime.utcnow(),
        weather_data["main"]["temp"],
        weather_data["weather"][0]["description"],
        img_res.get("cloud", 0),
        img_res.get("heat",  0),
        text_res.get("rain",  0),
        text_res.get("heat",  0),
        text_res.get("wind",  0),
        text_res.get("snow",  0),
        text_res.get("haze",  0),
        json.dumps(analysis),
        json.dumps(weather_data),
        img_res.get("cloud_path"),
        img_res.get("thermal_path"),
        json.dumps(raw_titles),
        lat,
        lon,
        json.dumps(feature_vector) if feature_vector is not None else None,
    ))

    conn.commit()
    record_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return record_id


def get_history(city, limit=50):
    conn   = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM weather_records
        WHERE city = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """, (city, limit))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def export_training_data(city=None):
    """
    Export labeled rows joined with their feature vectors.
    If city is provided, only return rows for that city.
    If city is None, return ALL cities (global model).
    """
    conn   = get_connection()
    cursor = conn.cursor(dictionary=True)

    if city:
        cursor.execute("""
            SELECT
                r.city,
                r.temperature, r.cloud_score, r.heat_score, r.feature_vector,
                r.text_rain, r.text_heat, r.text_wind, r.text_snow, r.text_haze,
                l.label_rain, l.label_heat, l.label_wind, l.label_snow, l.label_haze
            FROM weather_records r
            JOIN labeled_events l ON l.record_id = r.id
            WHERE r.city = %s
            ORDER BY r.timestamp ASC
        """, (city,))
    else:
        cursor.execute("""
            SELECT
                r.city,
                r.temperature, r.cloud_score, r.heat_score, r.feature_vector,
                r.text_rain, r.text_heat, r.text_wind, r.text_snow, r.text_haze,
                l.label_rain, l.label_heat, l.label_wind, l.label_snow, l.label_haze
            FROM weather_records r
            JOIN labeled_events l ON l.record_id = r.id
            ORDER BY r.timestamp ASC
        """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_all_labeled_cities():
    """
    Return list of distinct cities that have at least one labeled record.
    Used by train_all_cities() to know which per-city models to build.
    """
    conn   = get_connection()
    cursor = conn.cursor()

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