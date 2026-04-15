"""
Monitoring de la qualité des prévisions (§3.4).

Calcule la MAPE en comparant les prévisions passées aux mesures réelles.
Déclenche un réentraînement immédiat si MAPE > seuil (défaut : 15%).

Exécuté toutes les heures par le scheduler.
"""

import logging
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from forecaster.config import settings
from forecaster.db.models import Site
from forecaster.db.readers import (
    get_forecasts_consommation,
    get_forecasts_production_pv,
    get_mesures_reelles_consommation,
    get_mesures_reelles_production_pv,
)

logger = logging.getLogger(__name__)

MAPE_WINDOW_HOURS = 48  # §3.4 — fenêtre d'évaluation MAPE
MIN_PAIRES_MAPE = 10  # nombre minimum de paires pour calculer une MAPE fiable
TOLERANCE_JOINTURE_SECONDES = 450  # ±7min30 pour aligner pas 15min / 5min


def compute_mape(session: Session, site_id: str, model_type: str) -> float:
    """
    Calcule la MAPE sur les `MAPE_WINDOW_HOURS` dernières heures pour un site et un modèle.

    Compare :
      - forecasts_consommation (ou forecasts_production_pv) générés dans la fenêtre
      - mesures_reelles correspondantes (même timestamp, même site_id)

    La jointure se fait sur timestamp avec une tolérance de ±7min30 pour aligner
    les pas de 15 min (prévisions) avec les pas de 5 min (mesures réelles).

    Args:
        session:    Session SQLAlchemy.
        site_id:    Identifiant du site.
        model_type: "consumption" | "pv_production"

    Returns:
        MAPE en % (float). Retourne NaN si pas assez de données (< 10 paires).
    """
    now = datetime.now(tz=UTC)
    depuis = now - timedelta(hours=MAPE_WINDOW_HOURS)

    # Charger les prévisions sur la fenêtre
    if model_type == "consumption":
        df_forecast = get_forecasts_consommation(session, site_id, depuis, now)
        df_reel = get_mesures_reelles_consommation(session, site_id, depuis)
        colonne_reel = "conso_kw"
    elif model_type == "pv_production":
        df_forecast = get_forecasts_production_pv(session, site_id, depuis, now)
        df_reel = get_mesures_reelles_production_pv(session, site_id, depuis)
        colonne_reel = "production_pv_kw"
    else:
        raise ValueError(f"model_type invalide : '{model_type}'")

    if df_forecast.empty or df_reel.empty:
        logger.info(
            "compute_mape | site=%s | model=%s | données insuffisantes "
            "(forecasts=%d, mesures=%d) — retour NaN",
            site_id,
            model_type,
            len(df_forecast),
            len(df_reel),
        )
        return float("nan")

    # Jointure sur timestamp avec tolérance ±7min30
    df_merged = _jointure_tolerante(
        df_forecast, df_reel, colonne_reel, TOLERANCE_JOINTURE_SECONDES
    )

    if len(df_merged) < MIN_PAIRES_MAPE:
        logger.info(
            "compute_mape | site=%s | model=%s | seulement %d paires "
            "(minimum=%d) — retour NaN",
            site_id,
            model_type,
            len(df_merged),
            MIN_PAIRES_MAPE,
        )
        return float("nan")

    # Calcul MAPE
    y_prev = df_merged["puissance_kw"].values
    y_reel = df_merged[colonne_reel].values
    denominateur = np.maximum(np.abs(y_reel), 1e-6)
    mape = float(np.mean(np.abs(y_prev - y_reel) / denominateur) * 100)

    return mape


def check_mape_all_sites(session: Session) -> None:
    """
    Vérifie la MAPE pour tous les sites et tous les modèles.
    Déclenche un réentraînement si MAPE > settings.mape_threshold.

    Appelé toutes les heures par le scheduler.
    """
    sites = session.query(Site).all()
    for site in sites:
        for model_type in ("consumption", "pv_production"):
            try:
                mape = compute_mape(session, site.site_id, model_type)

                if math.isnan(mape):
                    continue

                logger.info(
                    "monitoring | site=%s | model=%s | MAPE=%.2f%%",
                    site.site_id,
                    model_type,
                    mape,
                )
                if mape > settings.mape_threshold:
                    _trigger_retraining(session, site.site_id, model_type, mape)
            except Exception:
                logger.exception(
                    "monitoring | site=%s | model=%s | erreur calcul MAPE",
                    site.site_id,
                    model_type,
                )


def _trigger_retraining(
    session: Session, site_id: str, model_type: str, current_mape: float
) -> None:
    """
    Déclenche un réentraînement immédiat hors cycle.

    Pour l'instant le réentraînement est synchrone. Une évolution future
    pourra le lancer dans un thread séparé ou via un job APScheduler immédiat.
    """
    from forecaster.pipeline.training import run_training

    logger.warning(
        "monitoring | MAPE > %.1f%% (actuelle=%.2f%%) | site=%s | model=%s "
        "| réentraînement déclenché",
        settings.mape_threshold,
        current_mape,
        site_id,
        model_type,
    )

    try:
        new_mape = run_training(session, model_type)
        logger.info(
            "monitoring | réentraînement terminé | site=%s | model=%s "
            "| ancienne MAPE=%.2f%% | nouvelle MAPE=%.2f%%",
            site_id,
            model_type,
            current_mape,
            new_mape,
        )
    except Exception:
        logger.exception(
            "monitoring | échec réentraînement | site=%s | model=%s",
            site_id,
            model_type,
        )


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _jointure_tolerante(
    df_forecast: pd.DataFrame,
    df_reel: pd.DataFrame,
    colonne_reel: str,
    tolerance_secondes: int,
) -> pd.DataFrame:
    """
    Jointure entre prévisions et mesures réelles avec tolérance temporelle.

    Pour chaque prévision, cherche la mesure réelle la plus proche dans la
    tolérance. Utilise pd.merge_asof pour l'alignement temporel.

    Returns:
        DataFrame avec colonnes ['timestamp', 'puissance_kw', colonne_reel].
    """
    df_f = df_forecast.copy().sort_values("timestamp").reset_index(drop=True)
    df_r = df_reel.copy().sort_values("timestamp").reset_index(drop=True)

    df_f["timestamp"] = pd.to_datetime(df_f["timestamp"], utc=True)
    df_r["timestamp"] = pd.to_datetime(df_r["timestamp"], utc=True)

    merged = pd.merge_asof(
        df_f,
        df_r,
        on="timestamp",
        tolerance=pd.Timedelta(seconds=tolerance_secondes),
        direction="nearest",
    )

    # Supprimer les lignes sans correspondance
    merged = merged.dropna(subset=[colonne_reel])
    return merged
