import numpy as np
import pytest
from features import (
    extract_weather_features,
    extract_forecast_features,
    extract_geo_features,
    extract_radar_features,
    extract_nwp_features,
    build_feature_vector,
)
from radar import extract_radar_features_from_data
from forecast_engine import generate_future_labels


def test_weather_features_dimension():
    mock_weather = {
        "main": {"temp": 28.5, "humidity": 65.0, "pressure": 1012.0},
        "wind": {"speed": 4.2},
        "clouds": {"all": 20.0},
        "weather": [{"id": 800}]
    }
    feats = extract_weather_features(mock_weather)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (6,)


def test_forecast_features_dimension():
    mock_forecast = {
        "temp": 31.0, "humidity": 70.0, "wind": 5.0, "wind_gusts": 8.0,
        "rain": 1.2, "precip_prob": 0.45, "cape": 850.0, "cloud_cover": 40.0,
        "snowfall": 0.0, "nwp_precip_prob": 0.5
    }
    feats = extract_forecast_features(mock_forecast)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (10,)


def test_geo_features_dimension():
    feats = extract_geo_features(lat=19.076, lon=72.877, elevation=14.0, land_use="urban")
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (5,)


def test_radar_features_dimension():
    mock_radar = {
        "precipitation": 3.5, "rain": 2.0, "showers": 1.5,
        "weather_code": 80, "wind_gusts_10m": 35.0, "cape": 1200.0,
        "cloud_cover": 85.0, "visibility": 4000.0
    }
    feats = extract_radar_features_from_data(mock_radar)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (4,)


def test_nwp_features_dimension():
    mock_nwp = {
        "predicted_temp": 27.0, "predicted_precip": 4.0,
        "predicted_wind": 6.5, "model_confidence": 0.8
    }
    feats = extract_nwp_features(mock_nwp)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (4,)


def test_build_feature_vector_dimension():
    mock_weather = {"main": {"temp": 25.0}}
    mock_forecast = {"temp": 24.0}
    vec = build_feature_vector(
        weather_data=mock_weather,
        cloud_img=None,
        thermal_img=None,
        article_titles=[],
        lat=19.076,
        lon=72.877,
        forecast_data=mock_forecast,
    )
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1437,)


def test_future_label_generation_rain():
    mock_rain_forecast = {
        "temp": 25.0, "rain": 4.5, "snowfall": 0.0, "wind": 10.0,
        "wind_gusts": 15.0, "precip_prob": 0.85, "cape": 1200.0,
        "weather_code": 81, "cloud_cover": 90.0, "nwp_precip": 3.0,
        "nwp_wind": 10.0, "nwp_temp": 25.0
    }
    labels = generate_future_labels(mock_rain_forecast, city="Mumbai", city_thresholds=(40.0, 10.0, 15.0))
    assert labels["rain"] == 1
    assert labels["heat"] == 0
    assert labels["snow"] == 0


def test_future_label_generation_heat():
    mock_heat_forecast = {
        "temp": 44.0, "rain": 0.0, "snowfall": 0.0, "wind": 5.0,
        "wind_gusts": 8.0, "precip_prob": 0.05, "cape": 100.0,
        "weather_code": 0, "cloud_cover": 10.0, "nwp_precip": 0.0,
        "nwp_wind": 5.0, "nwp_temp": 43.5
    }
    labels = generate_future_labels(mock_heat_forecast, city="Delhi", city_thresholds=(42.0, 8.0, 15.0))
    assert labels["heat"] == 1
    assert labels["rain"] == 0
