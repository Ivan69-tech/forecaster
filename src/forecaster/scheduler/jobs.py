"""
Définition des jobs APScheduler (§3.3).

Tableau de planification :
  ┌──────────────────────────────┬─────────────────────────────────────────────────┐
  │ Déclencheur                  │ Tâche                                           │
  ├──────────────────────────────┼─────────────────────────────────────────────────┤
  │ 06h00 quotidien              │ Prévisions conso + PV J et J+1 (48h)            │
  │ ~16h00 quotidien             │ Récupération prix spots RTE J+1                 │
  │ 12h00, 18h00, 00h00          │ Mise à jour intraday conso + PV (24h glissant)  │
  │ Dimanche 02h00 (hebdo)       │ Réentraînement modèles LightGBM                 │
  │ Toutes les heures (interval) │ Monitoring MAPE — trigger retraining si > 15%  │
  └──────────────────────────────┴─────────────────────────────────────────────────┘

Timezone : Europe/Paris (heure locale pour les horaires métier).
"""

import logging
from datetime import date, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from forecaster.db.session import SessionLocal
from forecaster.pipeline.forecast import run_forecast_all_sites
from forecaster.pipeline.monitoring import check_mape_all_sites
from forecaster.pipeline.training import run_training_all

logger = logging.getLogger(__name__)

TIMEZONE = "Europe/Paris"


def build_scheduler() -> BlockingScheduler:
    """Crée et configure le scheduler avec tous les jobs."""
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    # ── Job 1 : prévisions quotidiennes 48h (06h00) ──────────────────────────
    scheduler.add_job(
        _job_daily_forecast_48h,
        CronTrigger(hour=6, minute=0, timezone=TIMEZONE),
        id="daily_forecast_48h",
        name="Prévisions conso + PV J+J+1 (48h)",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # ── Job 2 : récupération prix spots RTE (16h00) ──────────────────────────
    scheduler.add_job(
        _job_fetch_spot_prices,
        CronTrigger(hour=16, minute=0, timezone=TIMEZONE),
        id="fetch_spot_prices",
        name="Récupération prix spots RTE J+1",
        replace_existing=True,
        misfire_grace_time=3600,  # marge d'1h — publication RTE parfois tardive
    )

    # ── Job 3 : mises à jour intraday (12h, 18h, 00h) ───────────────────────
    for hour in (12, 18, 0):
        scheduler.add_job(
            _job_intraday_forecast_24h,
            CronTrigger(hour=hour, minute=0, timezone=TIMEZONE),
            id=f"intraday_forecast_{hour:02d}h",
            name=f"Prévisions intraday 24h — {hour:02d}h00",
            replace_existing=True,
            misfire_grace_time=300,
        )

    # ── Job 4 : réentraînement hebdomadaire (dimanche 02h00) ─────────────────
    scheduler.add_job(
        _job_weekly_retraining,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=TIMEZONE),
        id="weekly_retraining",
        name="Réentraînement hebdomadaire LightGBM",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # ── Job 5 : monitoring MAPE (toutes les heures) ──────────────────────────
    scheduler.add_job(
        _job_mape_monitoring,
        IntervalTrigger(hours=1),
        id="mape_monitoring",
        name="Monitoring MAPE — trigger retraining si > seuil",
        replace_existing=True,
    )

    return scheduler


# ---------------------------------------------------------------------------
# Fonctions de job (wrappent les pipelines avec gestion de session)
# ---------------------------------------------------------------------------


def _job_daily_forecast_48h() -> None:
    logger.info("job | daily_forecast_48h | démarrage")
    with SessionLocal() as session:
        run_forecast_all_sites(session, horizon_h=48)
        session.commit()
    logger.info("job | daily_forecast_48h | terminé")


def _job_fetch_spot_prices() -> None:
    """Récupère les prix spots RTE pour J+1 et les persiste en base pour chaque site."""
    from forecaster.db.readers import get_all_sites
    from forecaster.db.writers import write_spot_prices
    from forecaster.fetchers.rte import RTEDataUnavailable, fetch_spot_prices

    tomorrow = date.today() + timedelta(days=1)
    logger.info("job | fetch_spot_prices | date=%s | démarrage", tomorrow)

    try:
        rows = fetch_spot_prices(tomorrow)
        logger.info("job | fetch_spot_prices | %d entrées récupérées", len(rows))
    except RTEDataUnavailable:
        logger.warning(
            "job | fetch_spot_prices | données non disponibles pour %s", tomorrow
        )
        return
    except Exception:
        logger.exception("job | fetch_spot_prices | erreur lors de la récupération")
        return

    with SessionLocal() as session:
        sites = get_all_sites(session)
        for site in sites:
            try:
                n = write_spot_prices(session, site.site_id, rows)
                logger.info(
                    "job | fetch_spot_prices | site=%s | %d prix écrits",
                    site.site_id,
                    n,
                )
            except Exception:
                logger.exception(
                    "job | fetch_spot_prices | site=%s | erreur écriture", site.site_id
                )
        session.commit()

    logger.info("job | fetch_spot_prices | terminé")


def _job_intraday_forecast_24h() -> None:
    logger.info("job | intraday_forecast_24h | démarrage")
    with SessionLocal() as session:
        run_forecast_all_sites(session, horizon_h=24)
        session.commit()
    logger.info("job | intraday_forecast_24h | terminé")


def _job_weekly_retraining() -> None:
    logger.info("job | weekly_retraining | démarrage")
    with SessionLocal() as session:
        results = run_training_all(session)
        session.commit()
    logger.info("job | weekly_retraining | résultats MAPE : %s", results)


def _job_mape_monitoring() -> None:
    logger.info("job | mape_monitoring | démarrage")
    with SessionLocal() as session:
        check_mape_all_sites(session)
        session.commit()
    logger.info("job | mape_monitoring | terminé")
