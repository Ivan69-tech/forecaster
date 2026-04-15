"""
Requêtes de lecture sur la base de données (§3.5).

Toutes les requêtes SQL sont centralisées ici — aucune requête dans pipeline/.
Chaque fonction documente la table lue et le type d'opération.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from forecaster.db.models import (
    ConsumptionForecast,
    ModelVersion,
    PVProductionForecast,
    RealMeasure,
)


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


def get_active_model_version(
    session: Session,
    model_type: str,
) -> ModelVersion | None:
    """
    Lecture de modeles_versions — version active pour un type de modèle.

    Args:
        session:    Session SQLAlchemy active.
        model_type: "consumption" | "pv_production"

    Returns:
        L'objet ModelVersion actif, ou None si aucun modèle actif.
    """
    return (
        session.query(ModelVersion)
        .filter_by(type_modele=model_type, actif=True)
        .first()
    )


def get_mesures_recentes(
    session: Session,
    site_id: str,
    depuis: datetime,
) -> pd.DataFrame:
    """
    Lecture de mesures_reelles — toutes les colonnes de mesure depuis `depuis`.

    Retourne un DataFrame avec colonnes ['timestamp', 'conso_kw', 'production_pv_kw'],
    trié par timestamp ASC. Utilisé pour les lookups de lags dans le pipeline forecast.

    Args:
        session: Session SQLAlchemy active.
        site_id: Identifiant du site.
        depuis:  Date à partir de laquelle charger les données.

    Returns:
        DataFrame trié par timestamp.
    """
    rows = (
        session.query(
            RealMeasure.timestamp,
            RealMeasure.conso_kw,
            RealMeasure.production_pv_kw,
        )
        .filter(RealMeasure.site_id == site_id)
        .filter(RealMeasure.timestamp >= depuis)
        .order_by(RealMeasure.timestamp)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "conso_kw", "production_pv_kw"])

    df = pd.DataFrame(rows, columns=["timestamp", "conso_kw", "production_pv_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_forecasts_consommation(
    session: Session,
    site_id: str,
    depuis: datetime,
    jusqua: datetime,
) -> pd.DataFrame:
    """
    Lecture de forecasts_consommation sur une fenêtre temporelle.

    Args:
        session: Session SQLAlchemy active.
        site_id: Identifiant du site.
        depuis:  Début de la fenêtre (inclus).
        jusqua:  Fin de la fenêtre (inclus).

    Returns:
        DataFrame avec colonnes ['timestamp', 'puissance_kw'], trié par timestamp.
    """
    rows = (
        session.query(ConsumptionForecast.timestamp, ConsumptionForecast.puissance_kw)
        .filter(ConsumptionForecast.site_id == site_id)
        .filter(ConsumptionForecast.timestamp >= depuis)
        .filter(ConsumptionForecast.timestamp <= jusqua)
        .order_by(ConsumptionForecast.timestamp)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "puissance_kw"])

    df = pd.DataFrame(rows, columns=["timestamp", "puissance_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_forecasts_production_pv(
    session: Session,
    site_id: str,
    depuis: datetime,
    jusqua: datetime,
) -> pd.DataFrame:
    """
    Lecture de forecasts_production_pv sur une fenêtre temporelle.

    Args:
        session: Session SQLAlchemy active.
        site_id: Identifiant du site.
        depuis:  Début de la fenêtre (inclus).
        jusqua:  Fin de la fenêtre (inclus).

    Returns:
        DataFrame avec colonnes ['timestamp', 'puissance_kw'], trié par timestamp.
    """
    rows = (
        session.query(
            PVProductionForecast.timestamp, PVProductionForecast.puissance_kw
        )
        .filter(PVProductionForecast.site_id == site_id)
        .filter(PVProductionForecast.timestamp >= depuis)
        .filter(PVProductionForecast.timestamp <= jusqua)
        .order_by(PVProductionForecast.timestamp)
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "puissance_kw"])

    df = pd.DataFrame(rows, columns=["timestamp", "puissance_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
