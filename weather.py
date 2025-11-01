import os
import requests
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()


API_BASE = "https://api.weatherapi.com/v1"
WEATHER_KEY = os.getenv("WEATHERAPI_KEY", "")

class WeatherError(Exception):
    pass

def _require_key():
    if not WEATHER_KEY:
        raise WeatherError("Missing WEATHERAPI_KEY. Add it to your .env")

def search_location(query: str):
    _require_key()
    # WeatherAPI: location search is part of forecast call; optional helper not needed.
    return {"name": query}  # passthrough

def get_14day_forecast(query: str):
    _require_key()
    url = f"{API_BASE}/forecast.json"
    params = {
        "key": WEATHER_KEY,
        "q": query,          # city or "lat,lon"
        "days": 14,          # WeatherAPI max 14
        "aqi": "no",
        "alerts": "no",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_15day_forecast(query_or_lat: str | float, lon: float | None = None):
    # route to 14-day since WeatherAPI caps at 14
    q = f"{query_or_lat},{lon}" if isinstance(query_or_lat, (int,float)) and lon is not None else str(query_or_lat)
    return get_14day_forecast(q)

def flatten_forecast(payload):
    if not payload or "forecast" not in payload:
        return []
    days = payload["forecast"].get("forecastday", [])
    rows = []
    for d in days:
        day = d.get("day", {})
        astro = d.get("astro", {})
        date_str = d.get("date")  # e.g. "2025-11-01"
        day_name = None
        if date_str:
            try:
                day_name = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
            except Exception:
                day_name = None

        rows.append({
            "Date": date_str,
            "Day": day_name,  # <-- now filled
            "Max (°C)": day.get("maxtemp_c"),
            "Min (°C)": day.get("mintemp_c"),
            "Precip (%)": day.get("daily_chance_of_rain"),
            "Sunrise": astro.get("sunrise"),
            "Sunset": astro.get("sunset"),
            "Summary": day.get("condition", {}).get("text"),
        })
    return rows

