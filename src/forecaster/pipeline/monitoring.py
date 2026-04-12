"""
Monitoring de la qualité des prévisions (§3.4).

Calcule la MAPE en comparant les prévisions passées aux mesures réelles.
Déclenche un réentraînement immédiat si MAPE > seuil (défaut : 15%).

Exécuté toutes les heures par le scheduler.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from forecaster.config import settings
from forecaster.db.models import Site

logger = logging.getLogger(__name__)

MAPE_WINDOW_HOURS = 48  # §3.4 — fenêtre d'évaluation MAPE


def compute_mape(session: Session, site_id: str, model_type: str) -> float:
    """
    Calcule la MAPE sur les `MAPE_WINDOW_HOURS` dernières heures pour un site et un modèle.

    Compare :
      - forecasts_consommation (ou forecasts_production_pv) générés dans la fenêtre
      - mesures_reelles correspondantes (même timestamp, même site_id)

    Args:
        session:    Session SQLAlchemy.
        site_id:    Identifiant du site.
        model_type: "consumption" | "pv_production"

    Returns:
        MAPE en % (float). Retourne NaN si pas assez de données.

    TODO:
        1. Définir la fenêtre : [now - 48h, now]
        2. Requêter la table de forecast correspondante (filtre site_id + timestamp)
        3. Requêter mesures_reelles sur la même fenêtre et le même site_id
        4. Joindre sur timestamp (tolérance ± 7min30 pour aligner les pas 15min / 5min)
        5. Calculer MAPE = mean(|prévision - réel| / max(|réel|, epsilon)) * 100
        6. Retourner la MAPE, logger le résultat
    """
    raise NotImplementedError


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
                logger.info(
                    "monitoring | site=%s | model=%s | MAPE=%.2f%%",
                    site.site_id,
                    model_type,
                    mape,
                )
                if mape > settings.mape_threshold:
                    _trigger_retraining(session, site.site_id, model_type, mape)
            except NotImplementedError:
                raise
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

    TODO: appeler pipeline.training.run_training() de façon non bloquante
         (thread séparé ou tâche APScheduler immédiate) pour ne pas bloquer
         le thread de monitoring.
    """
    logger.warning(
        "monitoring | MAPE > %.1f%% (actuelle=%.2f%%) | site=%s | model=%s | réentraînement déclenché",
        settings.mape_threshold,
        current_mape,
        site_id,
        model_type,
    )
    # TODO: lancer run_training() de façon asynchrone
    raise NotImplementedError
