"""
Tests du fetcher Open-Meteo (stub).
"""

import pytest

from forecaster.fetchers.openmeteo import WeatherForecast, WeatherPoint, fetch_forecast


def test_fetch_forecast_raises_not_implemented():
    """Vérifie que le stub lève NotImplementedError."""
    with pytest.raises(NotImplementedError):
        fetch_forecast("site-test-01", latitude=43.6047, longitude=1.4442, horizon_h=48)


def test_weather_point_dataclass():
    """Vérifie que WeatherPoint est instanciable."""
    from datetime import datetime, timezone
    pt = WeatherPoint(
        timestamp=datetime(2026, 4, 12, 6, 0, tzinfo=timezone.utc),
        temperature_c=15.2,
        irradiance_wm2=450.0,
        cloud_cover_pct=20.0,
    )
    assert pt.temperature_c == 15.2
    assert pt.irradiance_wm2 == 450.0


# TODO (à implémenter après fetch_forecast()) :
#
# def test_fetch_forecast_returns_correct_number_of_points(httpx_mock):
#     """Vérifie que le nombre de points correspond à horizon_h."""
#     httpx_mock.add_response(json=_mock_openmeteo_response(48))
#     result = fetch_forecast("site-01", 43.6, 1.4, horizon_h=48)
#     assert isinstance(result, WeatherForecast)
#     assert len(result.points) == 48
#     assert all(isinstance(p, WeatherPoint) for p in result.points)
