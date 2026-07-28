"""Live weather data integration using the free Open-Meteo API (no API key required)."""

from __future__ import annotations

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_coordinates(city: str) -> tuple[float, float, str]:
    """Resolve a city name to latitude, longitude, and display label."""
    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")

    location = results[0]
    label = f"{location['name']}, {location.get('country', '')}".strip(", ")
    return location["latitude"], location["longitude"], label


def fetch_live_weather(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
) -> dict:
    """
    Fetch current weather and short-term rainfall forecast.

    Returns a dictionary with temperature, humidity, rainfall, soil moisture,
    and location metadata ready for crop prediction.
    """
    if city:
        latitude, longitude, location_name = get_coordinates(city)
    elif latitude is not None and longitude is not None:
        location_name = f"{latitude:.4f}, {longitude:.4f}"
    else:
        raise ValueError("Provide either city name or latitude/longitude.")

    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "soil_moisture_0_to_1cm",
            ],
            "hourly": ["precipitation"],
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload["current"]

    hourly_precip = payload.get("hourly", {}).get("precipitation", [])
    forecast_rainfall = sum(hourly_precip[:24]) if hourly_precip else 0.0

    soil_moisture_raw = current.get("soil_moisture_0_to_1cm")
    soil_moisture_pct = round(soil_moisture_raw * 100, 2) if soil_moisture_raw else None

    return {
        "location": location_name,
        "latitude": payload["latitude"],
        "longitude": payload["longitude"],
        "time": current["time"],
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "precipitation_mm": current.get("precipitation", 0.0),
        "forecast_rainfall_24h_mm": round(forecast_rainfall, 2),
        "rainfall": round(current.get("precipitation", 0.0) + forecast_rainfall, 2),
        "soil_moisture_pct": soil_moisture_pct,
    }
