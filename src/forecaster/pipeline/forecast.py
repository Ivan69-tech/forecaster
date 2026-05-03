"""
Pipeline de prévision (§3.3).

Orchestration : fetch météo → récupération historique DB → predict → écriture DB.

Appelé par le scheduler aux heures définies :
  - 06h00 quotidien  : horizon 48h (J + J+1)
  - 12h00, 18h00, 00h00 : horizon 24h (intraday)
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from forecaster.db.models import Site
from forecaster.db.readers import get_active_model_version, get_mesures_recentes
from forecaster.db.writers import (
    write_consumption_forecasts,
    write_pv_forecasts,
)
from forecaster.exceptions import ForecastUnavailableError, SiteNotFoundError
from forecaster.fetchers.openmeteo import WeatherForecast, fetch_forecast
from forecaster.predictors.base import ForecastPoint
from forecaster.predictors.consumption import ConsumptionModel
from forecaster.predictors.pv_production import PVProductionModel

logger = logging.getLogger(__name__)

# Nombre de pas de 15 min par jour (utilisé pour les lags)
PAS_PAR_JOUR = 96

# Tolérance pour le lookup de lags (30 minutes)
TOLERANCE_LAG_SECONDES = 1800

__all__ = ["run_forecast", "run_forecast_all_sites", "SiteNotFoundError"]


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
        ForecastUnavailableError: Si aucun modèle actif n'est disponible.
    """
    logger.info("run_forecast | site=%s | horizon=%dh", site_id, horizon_h)

    site = _load_site(session, site_id)
    weather = fetch_forecast(site_id, site.latitude, site.longitude, horizon_h)

    conso_points = _predict_consumption(session, site, weather, horizon_h)
    pv_points = _predict_pv(session, site, weather, horizon_h)

    _write_conso_forecasts(session, site_id, conso_points, horizon_h)
    _write_pv_forecasts(session, site_id, pv_points, horizon_h)

    logger.info(
        "run_forecast | site=%s | écrits : %d conso + %d PV",
        site_id,
        len(conso_points),
        len(pv_points),
    )


def run_forecast_all_sites(session: Session, horizon_h: int = 48) -> list[str]:
    """Lance run_forecast() pour tous les sites. Retourne la liste des site_id en échec."""
    sites = session.query(Site).all()
    failed: list[str] = []
    for site in sites:
        try:
            run_forecast(session, site.site_id, horizon_h)
        except Exception:
            logger.exception("run_forecast | site=%s | échec", site.site_id)
            failed.append(site.site_id)
    return failed


# ---------------------------------------------------------------------------
# Helpers privés
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
    Charge le modèle de consommation actif, construit les features et prédit.

    Features construites :
      - Lags conso J-1 et J-7 depuis mesures_reelles
      - Température depuis la prévision météo (interpolée à 15 min)
      - Lags température J-1/J-7 : 0.0 (TODO : Open-Meteo archive)
      - Indicateurs calendaires (jours fériés)
    """
    model_version = get_active_model_version(session, "consumption", site.site_id)
    if model_version is None:
        raise ForecastUnavailableError(
            f"Aucun modèle consumption actif pour le site '{site.site_id}'."
        )

    model = ConsumptionModel(version=model_version.version)
    model.load(Path(model_version.chemin_artefact))

    now = datetime.now(tz=UTC)
    # Aligner sur le pas de 15 min inférieur pour que les timestamps soient
    # cohérents avec ceux attendus par l'optimizer (qui fait un _floor_pas).
    now_floor = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    freq = pd.Timedelta(minutes=15)
    nombre_pas = horizon_h * 4
    ts_futurs = pd.date_range(
        start=now_floor + freq,
        periods=nombre_pas,
        freq=freq,
        tz="UTC",
    )

    # Charger l'historique pour les lags (besoin de 8 jours en arrière)
    depuis_lags = now - timedelta(days=8)
    df_hist = get_mesures_recentes(session, site.site_id, depuis_lags)

    # Interpoler la météo horaire → 15 min
    df_meteo_15min = _interpoler_meteo_15min(weather)

    # Jours fériés
    annees = list({ts.year for ts in ts_futurs})
    jours_feries = set(holidays.France(years=annees).keys())

    lignes = []
    for ts in ts_futurs:
        horizon = round((ts - now_floor).total_seconds() / 3600)
        lag_1d = _lookup_conso_lag(df_hist, ts, timedelta(days=1))
        lag_7d = _lookup_conso_lag(df_hist, ts, timedelta(days=7))
        temperature = _lookup_meteo_value(df_meteo_15min, ts, "temperature_c", 15.0)

        lignes.append({
            "timestamp": ts,
            "horizon_h": horizon,
            "conso_kw_lag_1d": lag_1d,
            "conso_kw_lag_7d": lag_7d,
            "temperature_c": temperature,
            "temp_lag_1d": 0.0,
            "temp_lag_7d": 0.0,
            "is_holiday": int(ts.date() in jours_feries),
            "is_school_holiday": 0,
        })

    df_futur = pd.DataFrame(lignes)
    return model.predict(df_futur)


def _predict_pv(
    session: Session, site: Site, weather: WeatherForecast, horizon_h: int
) -> list[ForecastPoint]:
    """
    Charge le modèle PV actif, construit les features et prédit.

    Features construites :
      - Irradiance, nébulosité, température depuis la prévision météo
      - Puissance crête PV du site
    """
    model_version = get_active_model_version(session, "pv_production", site.site_id)
    if model_version is None:
        raise ForecastUnavailableError(
            f"Aucun modèle pv_production actif pour le site '{site.site_id}'."
        )

    model = PVProductionModel(version=model_version.version)
    try:
        model.load(Path(model_version.chemin_artefact))
    except FileNotFoundError:
        logger.warning(
            "_predict_pv | site=%s | artefact introuvable (%s) — réentraînement",
            site.site_id,
            model_version.chemin_artefact,
        )
        from forecaster.pipeline.training import run_training

        run_training(session, "pv_production", site.site_id)
        session.flush()
        model_version = get_active_model_version(session, "pv_production", site.site_id)
        if model_version is None:
            raise ForecastUnavailableError(
                f"Aucun modèle pv_production actif après réentraînement pour '{site.site_id}'."
            )
        model = PVProductionModel(version=model_version.version)
        model.load(Path(model_version.chemin_artefact))

    now = datetime.now(tz=UTC)
    # Aligner sur le pas de 15 min inférieur (cohérence avec l'optimizer).
    now_floor = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    freq = pd.Timedelta(minutes=15)
    nombre_pas = horizon_h * 4
    ts_futurs = pd.date_range(
        start=now_floor + freq,
        periods=nombre_pas,
        freq=freq,
        tz="UTC",
    )

    # Interpoler la météo horaire → 15 min
    df_meteo_15min = _interpoler_meteo_15min(weather)

    lignes = []
    for ts in ts_futurs:
        horizon = round((ts - now_floor).total_seconds() / 3600)
        irradiance = _lookup_meteo_value(df_meteo_15min, ts, "irradiance_wm2", 0.0)
        cloud_cover = _lookup_meteo_value(df_meteo_15min, ts, "cloud_cover_pct", 30.0)
        temperature = _lookup_meteo_value(df_meteo_15min, ts, "temperature_c", 15.0)

        lignes.append({
            "timestamp": ts,
            "horizon_h": horizon,
            "irradiance_wm2": irradiance,
            "cloud_cover_pct": cloud_cover,
            "temperature_c": temperature,
            "p_pv_peak_kw": float(site.p_pv_peak_kw),
        })

    df_futur = pd.DataFrame(lignes)
    return model.predict(df_futur)


def _write_conso_forecasts(
    session: Session, site_id: str, points: list[ForecastPoint], horizon_h: int
) -> None:
    """Délègue l'écriture des prévisions conso à db/writers.py."""
    model_version = get_active_model_version(session, "consumption", site_id)
    version_modele = model_version.version if model_version else "unknown"
    write_consumption_forecasts(session, site_id, points, horizon_h, version_modele)


def _write_pv_forecasts(
    session: Session, site_id: str, points: list[ForecastPoint], horizon_h: int
) -> None:
    """Délègue l'écriture des prévisions PV à db/writers.py."""
    model_version = get_active_model_version(session, "pv_production", site_id)
    version_modele = model_version.version if model_version else "unknown"
    write_pv_forecasts(session, site_id, points, horizon_h, version_modele)


# ---------------------------------------------------------------------------
# Utilitaires de construction de features
# ---------------------------------------------------------------------------


def _interpoler_meteo_15min(weather: WeatherForecast) -> pd.DataFrame:
    """
    Rééchantillonne la météo horaire en pas de 15 min.

    Irradiance et température : interpolation linéaire.
    Cloud cover : forward-fill (valeur discrète par heure).

    Returns:
        DataFrame indexé par timestamp avec colonnes temperature_c,
        irradiance_wm2, cloud_cover_pct.
    """
    if not weather.points:
        return pd.DataFrame(
            columns=["timestamp", "temperature_c", "irradiance_wm2", "cloud_cover_pct"]
        )

    df_meteo = pd.DataFrame(
        [
            {
                "timestamp": p.timestamp,
                "temperature_c": p.temperature_c,
                "irradiance_wm2": p.irradiance_wm2,
                "cloud_cover_pct": p.cloud_cover_pct,
            }
            for p in weather.points
        ]
    )
    df_meteo["timestamp"] = pd.to_datetime(df_meteo["timestamp"], utc=True)
    df_meteo = df_meteo.set_index("timestamp").sort_index()

    # Interpolation linéaire pour irradiance et température
    df_15min = df_meteo.resample("15min").interpolate(method="linear")
    # Forward-fill pour cloud cover
    df_15min["cloud_cover_pct"] = (
        df_meteo[["cloud_cover_pct"]].resample("15min").ffill()["cloud_cover_pct"]
    )
    df_15min = df_15min.reset_index()

    return df_15min


def _lookup_conso_lag(
    df_hist: pd.DataFrame, ts: pd.Timestamp, delta: timedelta
) -> float:
    """
    Retourne la consommation historique correspondant au lag demandé.

    Cherche la mesure la plus proche de (ts - delta) dans df_hist.
    Si aucune mesure trouvée dans la tolérance, retourne la moyenne.
    """
    if df_hist.empty:
        return 0.0

    cible = ts - delta
    diff_secs = np.abs((df_hist["timestamp"] - cible).dt.total_seconds())
    idx_min = int(diff_secs.idxmin())
    if diff_secs.loc[idx_min] <= TOLERANCE_LAG_SECONDES:
        return float(df_hist.loc[idx_min, "conso_kw"])
    return float(df_hist["conso_kw"].mean())


def _lookup_meteo_value(
    df_meteo: pd.DataFrame, ts: pd.Timestamp, colonne: str, default: float
) -> float:
    """
    Retourne la valeur météo interpolée pour un timestamp donné.

    Arrondit le timestamp au quart d'heure le plus proche et cherche
    dans le DataFrame. Retourne `default` si hors plage.
    """
    if df_meteo.empty:
        return default

    ts_norm = ts.floor("15min")
    mask = df_meteo["timestamp"] == ts_norm
    if mask.any():
        return float(df_meteo.loc[mask.idxmax(), colonne])
    return default
