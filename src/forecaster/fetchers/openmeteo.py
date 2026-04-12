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

from dataclasses import dataclass
from datetime import datetime

import httpx

from forecaster.config import settings

# Variables météo demandées à Open-Meteo
HOURLY_VARIABLES = [
    "temperature_2m",
    "shortwave_radiation",
    "cloud_cover",
]


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
        WeatherForecast avec `horizon_h` points horaires à partir de maintenant.

    Raises:
        httpx.HTTPStatusError: Si l'API Open-Meteo retourne une erreur HTTP.

    TODO:
        - Construire l'URL avec les bons paramètres Open-Meteo
          (forecast_days, hourly, timezone=UTC)
        - Parser la réponse JSON → liste de WeatherPoint
        - Gérer le cas où l'API retourne moins de points qu'horizon_h
    """
    raise NotImplementedError(
        "fetch_forecast() n'est pas encore implémentée. "
        "Voir TODO dans openmeteo.py."
    )


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

    TODO: implémenter le parsing.
    """
    raise NotImplementedError
