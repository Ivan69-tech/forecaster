"""
Requêtes d'écriture sur la base de données (§3.5).

Toutes les insertions SQL sont centralisées ici — aucune écriture dans pipeline/.
Chaque fonction documente la table écrite et le type d'opération.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from forecaster.db.models import (
    ConsumptionForecast,
    PVProductionForecast,
    SpotPriceForecast,
)
from forecaster.fetchers.rte import SpotPriceRow
from forecaster.predictors.base import ForecastPoint

logger = logging.getLogger(__name__)


def write_consumption_forecasts(
    session: Session,
    site_id: str,
    points: list[ForecastPoint],
    horizon_h: int,
    version_modele: str,
) -> int:
    """
    Écriture dans forecasts_consommation — insertion en masse.

    Supprime les prévisions existantes du même site pour le même horizon
    avant insertion (idempotence : un re-run ne duplique pas les lignes).

    Args:
        session:        Session SQLAlchemy active.
        site_id:        Identifiant du site.
        points:         Liste de ForecastPoint issus du modèle.
        horizon_h:      Horizon de la prévision (24 ou 48).
        version_modele: Version du modèle utilisé.

    Returns:
        Nombre de lignes insérées.
    """
    if not points:
        return 0

    ts_min = min(p.timestamp for p in points)
    ts_max = max(p.timestamp for p in points)

    supprimees = (
        session.query(ConsumptionForecast)
        .filter(ConsumptionForecast.site_id == site_id)
        .filter(ConsumptionForecast.timestamp >= ts_min)
        .filter(ConsumptionForecast.timestamp <= ts_max)
        .delete()
    )
    if supprimees:
        logger.info(
            "write_consumption_forecasts | site=%s | %d anciennes prévisions supprimées",
            site_id,
            supprimees,
        )

    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        ConsumptionForecast(
            site_id=site_id,
            timestamp=fp.timestamp,
            puissance_kw=max(fp.puissance_kw, 0.0),
            horizon_h=horizon_h,
            date_generation=now_utc,
            version_modele=version_modele,
        )
        for fp in points
    ]
    session.bulk_save_objects(enregistrements)

    logger.info(
        "write_consumption_forecasts | site=%s | %d lignes insérées | horizon=%dh",
        site_id,
        len(enregistrements),
        horizon_h,
    )
    return len(enregistrements)


def write_pv_forecasts(
    session: Session,
    site_id: str,
    points: list[ForecastPoint],
    horizon_h: int,
    version_modele: str,
) -> int:
    """
    Écriture dans forecasts_production_pv — insertion en masse.

    Supprime les prévisions existantes du même site sur la même plage
    avant insertion (idempotence).

    Args:
        session:        Session SQLAlchemy active.
        site_id:        Identifiant du site.
        points:         Liste de ForecastPoint issus du modèle PV.
        horizon_h:      Horizon de la prévision (24 ou 48).
        version_modele: Version du modèle utilisé.

    Returns:
        Nombre de lignes insérées.
    """
    if not points:
        return 0

    ts_min = min(p.timestamp for p in points)
    ts_max = max(p.timestamp for p in points)

    supprimees = (
        session.query(PVProductionForecast)
        .filter(PVProductionForecast.site_id == site_id)
        .filter(PVProductionForecast.timestamp >= ts_min)
        .filter(PVProductionForecast.timestamp <= ts_max)
        .delete()
    )
    if supprimees:
        logger.info(
            "write_pv_forecasts | site=%s | %d anciennes prévisions supprimées",
            site_id,
            supprimees,
        )

    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        PVProductionForecast(
            site_id=site_id,
            timestamp=fp.timestamp,
            puissance_kw=fp.puissance_kw,
            horizon_h=horizon_h,
            date_generation=now_utc,
            version_modele=version_modele,
        )
        for fp in points
    ]
    session.bulk_save_objects(enregistrements)

    logger.info(
        "write_pv_forecasts | site=%s | %d lignes insérées | horizon=%dh",
        site_id,
        len(enregistrements),
        horizon_h,
    )
    return len(enregistrements)


def write_spot_prices(
    session: Session,
    site_id: str,
    rows: list[SpotPriceRow],
) -> int:
    """
    Écriture dans forecasts_prix_spot — insertion en masse.

    Supprime les prix existants pour le même site et la même plage
    avant insertion (idempotence).

    Args:
        session: Session SQLAlchemy active.
        site_id: Identifiant du site.
        rows:    Liste de SpotPriceRow récupérés depuis l'API RTE.

    Returns:
        Nombre de lignes insérées.
    """
    if not rows:
        return 0

    ts_min = min(r.timestamp for r in rows)
    ts_max = max(r.timestamp for r in rows)

    supprimees = (
        session.query(SpotPriceForecast)
        .filter(SpotPriceForecast.site_id == site_id)
        .filter(SpotPriceForecast.timestamp >= ts_min)
        .filter(SpotPriceForecast.timestamp <= ts_max)
        .delete()
    )
    if supprimees:
        logger.info(
            "write_spot_prices | site=%s | %d anciens prix supprimés",
            site_id,
            supprimees,
        )

    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        SpotPriceForecast(
            site_id=site_id,
            timestamp=row.timestamp,
            prix_eur_mwh=row.prix_eur_mwh,
            date_generation=now_utc,
            source=row.source,
        )
        for row in rows
    ]
    session.bulk_save_objects(enregistrements)

    logger.info(
        "write_spot_prices | site=%s | %d prix insérés",
        site_id,
        len(enregistrements),
    )
    return len(enregistrements)
