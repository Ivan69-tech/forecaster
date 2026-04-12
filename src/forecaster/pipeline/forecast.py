"""
Pipeline de prévision (§3.3).

Orchestration : fetch météo → récupération historique DB → predict → écriture DB.

Appelé par le scheduler aux heures définies :
  - 06h00 quotidien  : horizon 48h (J + J+1)
  - 12h00, 18h00, 00h00 : horizon 24h (intraday)
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from forecaster.db.models import ConsumptionForecast, PVProductionForecast, Site
from forecaster.fetchers.openmeteo import WeatherForecast, fetch_forecast
from forecaster.predictors.base import ForecastPoint
from forecaster.predictors.consumption import ConsumptionModel
from forecaster.predictors.pv_production import PVProductionModel

logger = logging.getLogger(__name__)


def run_forecast(session: Session, site_id: str, horizon_h: int = 48) -> None:
    """
    Génère et persiste les prévisions de consommation et de production PV
    pour un site donné sur l'horizon demandé.

    Args:
        session:   Session SQLAlchemy active.
        site_id:   Identifiant unique du site.
        horizon_h: Horizon de prévision en heures (48 ou 24).

    Raises:
        SiteNotFoundError: Si le site_id n'existe pas en base.
        NotImplementedError: Tant que les modèles ne sont pas implémentés.

    TODO:
        1. Charger le site depuis la DB (paramètres lat/lon, capacité, etc.)
        2. Appeler fetch_forecast() pour obtenir les données météo
        3. Lire les mesures historiques (lags J-1, J-7) depuis mesures_reelles
        4. Construire les DataFrames de features pour chaque modèle
        5. Charger les modèles actifs depuis modeles_versions (DB → chemin artefact)
        6. Appeler model.predict() → liste de ForecastPoint
        7. Persister les prévisions dans forecasts_consommation et forecasts_production_pv
        8. Logger le résultat (nombre de points écrits, MAPE courante si dispo)
    """
    logger.info("run_forecast | site=%s | horizon=%dh", site_id, horizon_h)

    site = _load_site(session, site_id)
    weather = fetch_forecast(site_id, site.latitude, site.longitude, horizon_h)

    conso_points = _predict_consumption(session, site, weather, horizon_h)
    pv_points = _predict_pv(session, site, weather, horizon_h)

    _write_conso_forecasts(session, site_id, conso_points)
    _write_pv_forecasts(session, site_id, pv_points)

    logger.info(
        "run_forecast | site=%s | écrits : %d conso + %d PV",
        site_id,
        len(conso_points),
        len(pv_points),
    )


def run_forecast_all_sites(session: Session, horizon_h: int = 48) -> None:
    """Lance run_forecast() pour tous les sites enregistrés en base."""
    sites = session.query(Site).all()
    for site in sites:
        try:
            run_forecast(session, site.site_id, horizon_h)
        except Exception:
            logger.exception("run_forecast | site=%s | échec", site.site_id)


# ---------------------------------------------------------------------------
# Helpers privés — à implémenter
# ---------------------------------------------------------------------------


def _load_site(session: Session, site_id: str) -> Site:
    site = session.query(Site).filter_by(site_id=site_id).first()
    if site is None:
        raise SiteNotFoundError(f"Site '{site_id}' introuvable en base.")
    return site


def _predict_consumption(
    session: Session, site: Site, weather: WeatherForecast, horizon_h: int
) -> list[ForecastPoint]:
    """
    TODO:
        - Lire les lags de consommation depuis mesures_reelles (J-1, J-7)
        - Construire le DataFrame de features
        - Charger ConsumptionModel depuis la version active en DB
        - Retourner les ForecastPoint
    """
    raise NotImplementedError


def _predict_pv(
    session: Session, site: Site, weather: WeatherForecast, horizon_h: int
) -> list[ForecastPoint]:
    """
    TODO:
        - Lire les lags de production PV depuis mesures_reelles (J-1, J-7)
        - Construire le DataFrame de features avec irradiance + température
        - Charger PVProductionModel depuis la version active en DB
        - Retourner les ForecastPoint
    """
    raise NotImplementedError


def _write_conso_forecasts(
    session: Session, site_id: str, points: list[ForecastPoint]
) -> None:
    """Insère ou met à jour les prévisions de consommation en base. TODO."""
    raise NotImplementedError


def _write_pv_forecasts(
    session: Session, site_id: str, points: list[ForecastPoint]
) -> None:
    """Insère ou met à jour les prévisions PV en base. TODO."""
    raise NotImplementedError


class SiteNotFoundError(Exception):
    """Levée quand le site_id demandé n'existe pas en base."""
