"""
Runner manuel des jobs définis dans forecaster.scheduler.jobs.

Permet de déclencher hors calendrier APScheduler un ou plusieurs jobs
(debug, régénération de données, tests d'intégration). Chaque job est isolé :
une exception est capturée et loggée, les jobs suivants continuent, et un
récapitulatif final indique précisément lesquels ont échoué.

Usage :
    python scripts/run_all_jobs.py --list
    python scripts/run_all_jobs.py <job_name> [<job_name> ...]
    python scripts/run_all_jobs.py all

Conçu pour être exécuté dans le conteneur forecast-service :
    docker compose -f docker/docker-compose.yml exec forecast-service \\
        python scripts/run_all_jobs.py all
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy import text

from forecaster.config import settings
from forecaster.db.session import engine
from forecaster.scheduler.jobs import (
    _job_daily_forecast_48h,
    _job_fetch_spot_prices,
    _job_intraday_forecast_24h,
    _job_mape_monitoring,
    _job_weekly_retraining,
)

JOBS = {
    "fetch_spot_prices": _job_fetch_spot_prices,
    "daily_forecast_48h": _job_daily_forecast_48h,
    "intraday_forecast_24h": _job_intraday_forecast_24h,
    "weekly_retraining": _job_weekly_retraining,
    "mape_monitoring": _job_mape_monitoring,
}

# Ordre d'exécution pour "all" : dépendances implicites
# (prix spots → modèles à jour → prévisions → monitoring).
ALL_ORDER = [
    "fetch_spot_prices",
    "weekly_retraining",
    "daily_forecast_48h",
    "intraday_forecast_24h",
    "mape_monitoring",
]

logger = logging.getLogger("run_all_jobs")


@dataclass
class JobResult:
    name: str
    ok: bool
    duration_s: float
    error: str | None


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def _check_db() -> None:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("startup | connexion DB OK")
    except Exception as exc:
        logger.critical("startup | impossible de joindre la DB : %s", exc)
        sys.exit(1)


def _run_job(name: str) -> JobResult:
    """Exécute un job et retourne son résultat. Ne propage jamais d'exception."""
    logger.info("───── run_all_jobs | %s | START ─────", name)
    t0 = perf_counter()
    try:
        JOBS[name]()
        dt = perf_counter() - t0
        logger.info("───── run_all_jobs | %s | OK (%.1fs) ─────", name, dt)
        return JobResult(name=name, ok=True, duration_s=dt, error=None)
    except Exception as exc:
        dt = perf_counter() - t0
        logger.exception("───── run_all_jobs | %s | ÉCHEC (%.1fs) ─────", name, dt)
        return JobResult(
            name=name,
            ok=False,
            duration_s=dt,
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_summary(results: list[JobResult]) -> None:
    logger.info("═════════════ RÉCAPITULATIF ═════════════")
    for r in results:
        status = "OK   " if r.ok else "ÉCHEC"
        logger.info("  [%s] %-25s %6.1fs %s", status, r.name, r.duration_s, r.error or "")
    n_ok = sum(1 for r in results if r.ok)
    n_ko = len(results) - n_ok
    logger.info(
        "═════════════ %d OK / %d ÉCHEC sur %d ═════════════",
        n_ok,
        n_ko,
        len(results),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance manuellement un ou plusieurs jobs du scheduler.",
    )
    parser.add_argument(
        "jobs",
        nargs="*",
        help="Noms des jobs à lancer (ou 'all'). Voir --list.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liste les jobs disponibles et quitte.",
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    if args.list or not args.jobs:
        print("Jobs disponibles :")
        for name in JOBS:
            print(f"  - {name}")
        print("\nSpécial : 'all' lance tous les jobs dans l'ordre logique.")
        return 0

    requested: list[str] = []
    for name in args.jobs:
        if name == "all":
            requested.extend(ALL_ORDER)
        else:
            requested.append(name)

    unknown = [n for n in requested if n not in JOBS]
    if unknown:
        logger.error("jobs inconnus : %s", unknown)
        logger.error("jobs valides : %s", list(JOBS))
        return 2

    _check_db()
    logger.info("run_all_jobs | %d job(s) à lancer : %s", len(requested), requested)

    results = [_run_job(name) for name in requested]
    _print_summary(results)

    failures = [r.name for r in results if not r.ok]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
