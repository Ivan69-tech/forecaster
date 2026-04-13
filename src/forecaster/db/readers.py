"""
Requêtes de lecture sur la base de données (§3.5).

Toutes les requêtes SQL sont centralisées ici — aucune requête dans pipeline/.
Chaque fonction documente la table lue et le type d'opération.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from forecaster.db.models import RealMeasure


def get_mesures_reelles_consommation(
    session: Session,
    site_id: str,
    cutoff: datetime,
) -> pd.DataFrame:
    """
    Lecture de mesures_reelles — colonnes timestamp et conso_kw.

    Retourne toutes les mesures du site `site_id` dont le timestamp est
    supérieur ou égal à `cutoff`, triées par timestamp ASC.

    Args:
        session:  Session SQLAlchemy active.
        site_id:  Identifiant du site.
        cutoff:   Date à partir de laquelle charger les données.

    Returns:
        DataFrame avec colonnes ['timestamp', 'conso_kw'], trié par timestamp.
    """
    rows = (
        session.query(RealMeasure.timestamp, RealMeasure.conso_kw)
        .filter(RealMeasure.site_id == site_id)
        .filter(RealMeasure.timestamp >= cutoff)
        .order_by(RealMeasure.timestamp)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "conso_kw"])

    df = pd.DataFrame(rows, columns=["timestamp", "conso_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_mesures_reelles_production_pv(
    session: Session,
    site_id: str,
    cutoff: datetime,
) -> pd.DataFrame:
    """
    Lecture de mesures_reelles — colonnes timestamp et production_pv_kw.

    Retourne toutes les mesures du site `site_id` dont le timestamp est
    supérieur ou égal à `cutoff`, triées par timestamp ASC.

    Args:
        session:  Session SQLAlchemy active.
        site_id:  Identifiant du site.
        cutoff:   Date à partir de laquelle charger les données.

    Returns:
        DataFrame avec colonnes ['timestamp', 'production_pv_kw'], trié par timestamp.
    """
    rows = (
        session.query(RealMeasure.timestamp, RealMeasure.production_pv_kw)
        .filter(RealMeasure.site_id == site_id)
        .filter(RealMeasure.timestamp >= cutoff)
        .order_by(RealMeasure.timestamp)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "production_pv_kw"])

    df = pd.DataFrame(rows, columns=["timestamp", "production_pv_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
