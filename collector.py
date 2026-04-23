import requests
import time
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

CITIES = [
    
    "Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Bhopal", "Patna", "Raipur", "Korba", "Chandigarh",
    "Srinagar", "Guwahati", "Thiruvananthapuram",

    #Asia
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

    #Oceania
    "Sydney", "Melbourne", "Auckland"
]

# ⏱ interval (in seconds)
INTERVAL = 600   # 10 minutes


def collect_data():
    print(f"\n[Collector] Running at {datetime.now()}")
    for city in CITIES:
        try:
            print(f"[Collector] Fetching → {city}")
            response = requests.get(f"{API_BASE}/analyze/{city}", timeout=60)
            data = response.json()
            if "error" in data:
                print(f"[Error] {city}: {data['error']}")
            else:
                print(f"[OK] {city} data stored")
        except requests.exceptions.Timeout:
            print(f"[Timeout] {city}: skipping after 60s")
        except Exception as e:
            print(f"[Exception] {city}: {str(e)}")

if __name__ == "__main__":
    print(" Continuous Data Collector Started")

    while True:
        collect_data()
        print(f"[Collector] Sleeping for {INTERVAL} seconds...\n")
        time.sleep(INTERVAL)