"""
collector.py  —  Automated multi-city data ingestion 

Runs continuously, hitting /analyze/{city} every INTERVAL seconds.
Covers India, Asia, Europe, Americas, Africa, Oceania.
Logs success/failure per city.
"""

import os
import requests
import time
import logging
from datetime import datetime

# Create required folders BEFORE logging.basicConfig tries to open the log file
os.makedirs("logs",        exist_ok=True)
os.makedirs("data/images", exist_ok=True)
os.makedirs("models",      exist_ok=True)

logging.basicConfig(
    filename="logs/collector.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

API_BASE = "http://127.0.0.1:8000"

CITIES = [
    # India
    "Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Bhopal", "Patna", "Raipur", "Korba", "Chandigarh",
    "Srinagar", "Guwahati", "Thiruvananthapuram",

    # Asia
    "Tokyo", "Beijing", "Shanghai", "Seoul", "Bangkok",
    "Singapore", "Dubai", "Jakarta", "Kuala Lumpur",

    # Europe
    "London", "Paris", "Berlin", "Madrid", "Rome",
    "Amsterdam", "Moscow",

    # North America
    "New York", "Los Angeles", "Chicago", "Toronto",
    "Vancouver", "Mexico City",

    # South America
    "São Paulo", "Rio de Janeiro", "Buenos Aires",

    # Africa
    "Cairo", "Nairobi", "Cape Town", "Lagos",

    # Oceania
    "Sydney", "Melbourne", "Auckland",
]

# Collection interval in seconds (default 10 min)
INTERVAL = int(os.getenv("COLLECTOR_INTERVAL", "600"))


def collect_once():
    success, failed = 0, 0
    print(f"\n[Collector] Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("Collection cycle started")

    for city in CITIES:
        try:
            print(f"  → {city}", end="  ", flush=True)
            response = requests.get(f"{API_BASE}/analyze/{city}", timeout=90)
            data     = response.json()

            if "error" in data:
                print(f"ERROR: {data['error']}")
                logging.warning(f"{city}: {data['error']}")
                failed += 1
            else:
                print(f"OK  ({data.get('weather','?')}, "
                      f"{data.get('temperature','?')}°C, "
                      f"el={data.get('geospatial',{}).get('elevation',0):.0f}m)")
                logging.info(f"{city}: OK")
                success += 1

        except requests.exceptions.Timeout:
            print("TIMEOUT")
            logging.warning(f"{city}: timeout")
            failed += 1
        except Exception as e:
            print(f"EXCEPTION: {e}")
            logging.error(f"{city}: {e}")
            failed += 1

    print(f"[Collector] Done — {success} OK, {failed} failed")
    logging.info(f"Cycle done — {success} OK, {failed} failed")


if __name__ == "__main__":
    print("═" * 60)
    print(" WeatherFusion Continuous Data Collector")
    print(f" Cities: {len(CITIES)}  |  Interval: {INTERVAL}s")
    print("═" * 60)

    while True:
        collect_once()
        print(f"[Collector] Sleeping {INTERVAL}s …\n")
        time.sleep(INTERVAL)
