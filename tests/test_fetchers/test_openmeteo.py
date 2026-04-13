"""
Tests du fetcher Open-Meteo.

Les appels HTTP sont mockés via unittest.mock.patch — aucun appel réseau réel.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from forecaster.fetchers.openmeteo import (
    WeatherForecast,
    WeatherPoint,
    _parse_response,
    fetch_forecast,
    fetch_historical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openmeteo_response(n_points: int, variable_cloud_cover: str = "cloud_cover") -> dict:
    """Construit une réponse JSON Open-Meteo synthétique avec n_points horaires."""
    debut = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
    times = [
        (debut + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
        for h in range(n_points)
    ]
    return {
        "hourly": {
            "time": times[:n_points],
            "temperature_2m": [15.0 + i * 0.1 for i in range(n_points)],
            "shortwave_radiation": [max(0.0, 500.0 - abs(i - 12) * 50) for i in range(n_points)],
            variable_cloud_cover: [20.0 + (i % 5) * 5 for i in range(n_points)],
        }
    }


def _make_mock_response(data: dict) -> MagicMock:
    """Crée un mock httpx.Response qui retourne `data` en JSON."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Tests WeatherPoint / WeatherForecast (dataclasses)
# ---------------------------------------------------------------------------


def test_weather_point_dataclass() -> None:
    """WeatherPoint est instanciable et conserve ses valeurs."""
    pt = WeatherPoint(
        timestamp=datetime(2026, 4, 12, 6, 0, tzinfo=timezone.utc),
        temperature_c=15.2,
        irradiance_wm2=450.0,
        cloud_cover_pct=20.0,
    )
    assert pt.temperature_c == 15.2
    assert pt.irradiance_wm2 == 450.0
    assert pt.cloud_cover_pct == 20.0


def test_weather_forecast_dataclass() -> None:
    """WeatherForecast est instanciable avec une liste de points."""
    pts = [
        WeatherPoint(
            timestamp=datetime(2026, 4, 12, h, 0, tzinfo=timezone.utc),
            temperature_c=15.0,
            irradiance_wm2=0.0,
            cloud_cover_pct=50.0,
        )
        for h in range(3)
    ]
    forecast = WeatherForecast(
        site_id="site-01", latitude=43.6, longitude=1.4, points=pts
    )
    assert len(forecast.points) == 3
    assert forecast.site_id == "site-01"


# ---------------------------------------------------------------------------
# Tests _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_retourne_weather_forecast() -> None:
    """_parse_response() convertit correctement un JSON Open-Meteo en WeatherForecast."""
    data = _mock_openmeteo_response(3)
    result = _parse_response("site-test", 43.6, 1.4, data)

    assert isinstance(result, WeatherForecast)
    assert len(result.points) == 3
    assert all(isinstance(p, WeatherPoint) for p in result.points)


def test_parse_response_ajoute_timezone_utc() -> None:
    """_parse_response() ajoute tzinfo=UTC aux timestamps (Open-Meteo retourne sans TZ)."""
    data = _mock_openmeteo_response(1)
    result = _parse_response("site-test", 43.6, 1.4, data)

    assert result.points[0].timestamp.tzinfo is not None
    assert result.points[0].timestamp.tzinfo == timezone.utc


def test_parse_response_valeurs_numeriques() -> None:
    """_parse_response() conserve les valeurs numériques de temperature, irradiance, cloud."""
    data = _mock_openmeteo_response(2)
    result = _parse_response("site-test", 43.6, 1.4, data)

    pt = result.points[0]
    assert isinstance(pt.temperature_c, float)
    assert isinstance(pt.irradiance_wm2, float)
    assert isinstance(pt.cloud_cover_pct, float)


def test_parse_response_gere_valeur_none() -> None:
    """_parse_response() remplace None par 0.0 pour les valeurs manquantes."""
    data = {
        "hourly": {
            "time": ["2026-04-12T00:00"],
            "temperature_2m": [None],
            "shortwave_radiation": [None],
            "cloud_cover": [None],
        }
    }
    result = _parse_response("site-test", 43.6, 1.4, data)

    assert result.points[0].temperature_c == 0.0
    assert result.points[0].irradiance_wm2 == 0.0
    assert result.points[0].cloud_cover_pct == 0.0


# ---------------------------------------------------------------------------
# Tests fetch_forecast
# ---------------------------------------------------------------------------


def test_fetch_forecast_retourne_weather_forecast() -> None:
    """fetch_forecast() retourne un WeatherForecast avec les bons points."""
    data = _mock_openmeteo_response(48)
    mock_resp = _make_mock_response(data)

    with patch("forecaster.fetchers.openmeteo.httpx.get", return_value=mock_resp):
        result = fetch_forecast("site-01", latitude=43.6, longitude=1.4, horizon_h=48)

    assert isinstance(result, WeatherForecast)
    assert result.site_id == "site-01"
    assert len(result.points) == 48


def test_fetch_forecast_tronque_a_horizon_h() -> None:
    """fetch_forecast() tronque les points si l'API retourne plus qu'horizon_h."""
    data = _mock_openmeteo_response(72)  # API retourne 72h
    mock_resp = _make_mock_response(data)

    with patch("forecaster.fetchers.openmeteo.httpx.get", return_value=mock_resp):
        result = fetch_forecast("site-01", latitude=43.6, longitude=1.4, horizon_h=24)

    assert len(result.points) == 24


def test_fetch_forecast_retry_puis_succes() -> None:
    """fetch_forecast() réessaie après un échec et réussit à la 2ème tentative."""
    import httpx

    data = _mock_openmeteo_response(48)
    mock_ok = _make_mock_response(data)
    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock(status_code=500)
    )

    with patch("forecaster.fetchers.openmeteo.httpx.get", side_effect=[mock_fail, mock_ok]):
        with patch("forecaster.fetchers.openmeteo.time.sleep"):  # accélère le test
            result = fetch_forecast("site-01", 43.6, 1.4, horizon_h=48)

    assert isinstance(result, WeatherForecast)
    assert len(result.points) == 48


def test_fetch_forecast_leve_erreur_apres_max_retries() -> None:
    """fetch_forecast() lève RuntimeError après 3 tentatives échouées."""
    import httpx

    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=MagicMock(status_code=503)
    )

    with patch("forecaster.fetchers.openmeteo.httpx.get", return_value=mock_fail):
        with patch("forecaster.fetchers.openmeteo.time.sleep"):
            with pytest.raises(RuntimeError):
                fetch_forecast("site-01", 43.6, 1.4, horizon_h=48)


# ---------------------------------------------------------------------------
# Tests fetch_historical
# ---------------------------------------------------------------------------


def test_fetch_historical_retourne_weather_forecast() -> None:
    """fetch_historical() retourne un WeatherForecast depuis l'API archive."""
    data = _mock_openmeteo_response(24, variable_cloud_cover="cloudcover")
    # L'API archive retourne "cloudcover" (pas "cloud_cover")
    mock_resp = _make_mock_response(data)

    with patch("forecaster.fetchers.openmeteo.httpx.get", return_value=mock_resp):
        result = fetch_historical(
            "site-01",
            latitude=43.6,
            longitude=1.4,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 1),
        )

    assert isinstance(result, WeatherForecast)
    assert len(result.points) == 24
    assert all(isinstance(p, WeatherPoint) for p in result.points)


def test_fetch_historical_normalise_cloud_cover() -> None:
    """fetch_historical() normalise 'cloudcover' en 'cloud_cover' avant parsing."""
    data = {
        "hourly": {
            "time": ["2026-01-01T12:00"],
            "temperature_2m": [10.0],
            "shortwave_radiation": [300.0],
            "cloudcover": [45.0],  # nom de l'API archive
        }
    }
    mock_resp = _make_mock_response(data)

    with patch("forecaster.fetchers.openmeteo.httpx.get", return_value=mock_resp):
        result = fetch_historical("site-01", 43.6, 1.4, date(2026, 1, 1), date(2026, 1, 1))

    assert result.points[0].cloud_cover_pct == 45.0
