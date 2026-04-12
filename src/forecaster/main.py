"""
Point d'entrée du Service de Prévision (L1 — SGE Tewa Solar).

Séquence de démarrage :
  1. Chargement de la configuration (.env)
  2. Vérification de la connexion DB
  3. Application des migrations Alembic (optionnel en production)
  4. Démarrage du scheduler APScheduler (bloquant)
"""

import logging
import sys

from forecaster.config import settings
from forecaster.db.session import engine
from forecaster.scheduler.jobs import build_scheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _check_db_connection() -> None:
    """Vérifie que la base de données est joignable au démarrage."""
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        logger.info("startup | connexion DB OK")
    except Exception as exc:
        logger.critical("startup | impossible de joindre la DB : %s", exc)
        sys.exit(1)


def main() -> None:
    logger.info("startup | Service de Prévision SGE — démarrage")
    logger.info("startup | DATABASE_URL=%s", settings.database_url.split("@")[-1])  # masque credentials
    logger.info("startup | MODELS_DIR=%s", settings.models_dir)
    logger.info("startup | MAPE_THRESHOLD=%.1f%%", settings.mape_threshold)

    _check_db_connection()

    scheduler = build_scheduler()
    logger.info("startup | scheduler configuré — %d jobs", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("startup |   · %s (next run: %s)", job.name, job.next_run_time)

    logger.info("startup | démarrage du scheduler (mode bloquant)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("shutdown | arrêt demandé")
        scheduler.shutdown()
        logger.info("shutdown | scheduler arrêté proprement")


if __name__ == "__main__":
    main()
