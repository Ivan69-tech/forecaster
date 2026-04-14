"""
Fetcher — Données météo Open-Meteo (§3.1).

API publique gratuite, pas de clé requise.
Documentation : https://open-meteo.com/en/docs

Variables récupérées :
  - temperature_2m        : température à 2m (°C)
  - shortwave_radiation   : irradiance GHI (W/m²)  — proxy pour production PV
  - cloud_cover           : nébulosité (%)

Fréquence : toutes les 6h (§3.3 — mises à jour intraday).
Horizon   : jusqu'à 48h (J et J+1).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

from forecaster.config import settings

logger = logging.getLogger(__name__)

# Variables météo demandées à Open-Meteo (API forecast)
HOURLY_VARIABLES_FORECAST = [
    "temperature_2m",
    "shortwave_radiation",
    "cloud_cover",
]

# Variables météo pour l'API archive (noms légèrement différents)
HOURLY_VARIABLES_ARCHIVE = [
    "temperature_2m",
    "shortwave_radiation",
    "cloudcover",
]

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_S = 1.0  # délai initial avant retry (secondes)


@dataclass
class WeatherPoint:
    """Données météo pour un pas horaire."""

    timestamp: datetime  # UTC
    temperature_c: float
    irradiance_wm2: float
    cloud_cover_pct: float


@dataclass
class WeatherForecast:
    """Prévisions météo pour un site, sur l'horizon demandé."""

    site_id: str
    latitude: float
    longitude: float
    points: list[WeatherPoint]  # triés par timestamp croissant


def fetch_forecast(
    site_id: str,
    latitude: float,
    longitude: float,
    horizon_h: int = 48,
) -> WeatherForecast:
    """
    Récupère les prévisions météo depuis Open-Meteo pour un site donné.

    Args:
        site_id:    Identifiant du site (pour le retour dans WeatherForecast).
        latitude:   Latitude du site (degrés décimaux).
        longitude:  Longitude du site (degrés décimaux).
        horizon_h:  Horizon de prévision en heures (défaut : 48h).

    Returns:
        WeatherForecast avec jusqu'à `horizon_h` points horaires à partir de maintenant.

    Raises:
        httpx.HTTPStatusError: Si l'API Open-Meteo retourne une erreur HTTP après les retries.
    """
    forecast_days = math.ceil(horizon_h / 24)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES_FORECAST),
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    data = _get_with_retry(settings.openmeteo_base_url + "/forecast", params, site_id)
    forecast = _parse_response(site_id, latitude, longitude, data)

    # Tronquer à horizon_h points
    if len(forecast.points) > horizon_h:
        forecast.points = forecast.points[:horizon_h]

    logger.info(
        "fetch_forecast | site=%s | points=%d | horizon_h=%d",
        site_id,
        len(forecast.points),
        horizon_h,
    )
    return forecast


def fetch_historical(
    site_id: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> WeatherForecast:
    """
    Récupère des données météo historiques depuis l'API archive Open-Meteo (ERA5).

    Utilisé par le pipeline d'entraînement pour obtenir les données météo
    correspondant à la période d'historique de production PV.

    Args:
        site_id:    Identifiant du site.
        latitude:   Latitude du site (degrés décimaux).
        longitude:  Longitude du site (degrés décimaux).
        start_date: Date de début (incluse).
        end_date:   Date de fin (incluse).

    Returns:
        WeatherForecast avec les points horaires sur la période demandée.

    Raises:
        httpx.HTTPStatusError: Si l'API retourne une erreur HTTP après les retries.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES_ARCHIVE),
        "timezone": "UTC",
    }

    data = _get_with_retry(ARCHIVE_BASE_URL, params, site_id)

    # L'API archive retourne "cloudcover" au lieu de "cloud_cover" —
    # on normalise pour utiliser le même _parse_response().
    if "hourly" in data and "cloudcover" in data["hourly"]:
        data["hourly"]["cloud_cover"] = data["hourly"].pop("cloudcover")

    forecast = _parse_response(site_id, latitude, longitude, data)

    logger.info(
        "fetch_historical | site=%s | points=%d | start=%s | end=%s",
        site_id,
        len(forecast.points),
        start_date,
        end_date,
    )
    return forecast


def _parse_response(site_id: str, lat: float, lon: float, data: dict) -> WeatherForecast:
    """
    Parse la réponse JSON Open-Meteo en WeatherForecast.

    Structure attendue de `data` :
    {
        "hourly": {
            "time": ["2026-04-12T00:00", ...],
            "temperature_2m": [12.3, ...],
            "shortwave_radiation": [0.0, ...],
            "cloud_cover": [80, ...]
        }
    }

    Les timestamps Open-Meteo sont en UTC mais sans suffixe 'Z' — on ajoute timezone.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    irradiances = hourly.get("shortwave_radiation", [])
    cloud_covers = hourly.get("cloud_cover", [])

    points = []
    for ts_str, temp, irr, cloud in zip(times, temperatures, irradiances, cloud_covers):
        # Open-Meteo retourne "2026-04-12T00:00" sans timezone — c'est UTC
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        points.append(
            WeatherPoint(
                timestamp=ts,
                temperature_c=float(temp) if temp is not None else 0.0,
                irradiance_wm2=float(irr) if irr is not None else 0.0,
                cloud_cover_pct=float(cloud) if cloud is not None else 0.0,
            )
        )

    return WeatherForecast(site_id=site_id, latitude=lat, longitude=lon, points=points)


def _get_with_retry(url: str, params: dict, site_id: str) -> dict:
    """
    Effectue un GET HTTP avec retry exponentiel (3 tentatives).

    Args:
        url:     URL complète de l'API.
        params:  Paramètres de la requête.
        site_id: Utilisé pour le logging.

    Returns:
        Données JSON de la réponse.

    Raises:
        httpx.HTTPStatusError: Si toutes les tentatives échouent.
    """
    last_exc: Exception | None = None

    for tentative in range(1, _MAX_RETRIES + 1):
        try:
            response = httpx.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            logger.warning(
                "open-meteo | site=%s | tentative=%d/%d | HTTP %d",
                site_id,
                tentative,
                _MAX_RETRIES,
                exc.response.status_code,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            logger.warning(
                "open-meteo | site=%s | tentative=%d/%d | erreur réseau : %s",
                site_id,
                tentative,
                _MAX_RETRIES,
                exc,
            )

        if tentative < _MAX_RETRIES:
            delai = _RETRY_BACKOFF_BASE_S * (2 ** (tentative - 1))
            time.sleep(delai)

    raise RuntimeError(
        f"open-meteo | site={site_id} | échec après {_MAX_RETRIES} tentatives"
    ) from last_exc
