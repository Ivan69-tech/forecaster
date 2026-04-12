"""
Script d'initialisation pour la démo locale.

Exécuté une seule fois par le service `forecast-init` au démarrage.
Idempotent : peut être relancé sans erreur si les données existent déjà.

Étapes :
  1. Création du schéma DB (Base.metadata.create_all)
  2. Insertion d'un site de démonstration
  3. Chargement de 100 jours de données historiques synthétiques (CSV)
  4. Entraînement du ConsumptionModel via run_training()
  5. Génération de la prévision 48h et insertion dans forecasts_consommation
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import text

# Chemin du CSV synthétique (copié dans l'image Docker)
CSV_PATH = Path(__file__).parent.parent / "data" / "load_history_2025.csv"

# Paramètres du site de démonstration
SITE_ID = "site-demo-01"
SITE_NOM = "Site Industriel Demo"
N_JOURS_HISTORIQUE = 100  # jours de données insérées en DB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("init_demo")


# ---------------------------------------------------------------------------
# Étape 1 — Connexion et schéma DB
# ---------------------------------------------------------------------------


def attendre_db(engine, tentatives: int = 30, delai_s: float = 2.0) -> None:
    """Attend que la base de données soit disponible."""
    for i in range(tentatives):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DB joignable ✓")
            return
        except Exception as exc:
            logger.warning("DB non disponible (%d/%d) : %s", i + 1, tentatives, exc)
            time.sleep(delai_s)
    logger.critical("DB inaccessible après %d tentatives — abandon", tentatives)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Étape 2 — Site de démonstration
# ---------------------------------------------------------------------------


def inserer_site_si_absent(session) -> None:
    """Insère le site de démonstration s'il n'existe pas encore."""
    from forecaster.db.models import Site

    existant = session.query(Site).filter_by(site_id=SITE_ID).first()
    if existant:
        logger.info("Site '%s' déjà présent — skip", SITE_ID)
        return

    site = Site(
        site_id=SITE_ID,
        nom=SITE_NOM,
        capacite_bess_kwh=500.0,
        p_max_bess_kw=250.0,
        p_pv_peak_kw=300.0,
        p_souscrite_kw=700.0,
        soc_min_pct=10.0,
        soc_max_pct=90.0,
        latitude=43.6047,   # Toulouse
        longitude=1.4442,
    )
    session.add(site)
    session.commit()
    logger.info("Site '%s' inséré", SITE_ID)


# ---------------------------------------------------------------------------
# Étape 3 — Données historiques
# ---------------------------------------------------------------------------


def charger_historique_si_absent(session) -> pd.DataFrame:
    """
    Charge N_JOURS_HISTORIQUE jours de données dans mesures_reelles.

    Les timestamps du CSV sont recalés pour se terminer à maintenant - 15 min,
    ce qui permet d'utiliser la fenêtre glissante de 90 jours pour l'entraînement.
    Retourne le DataFrame chargé (utilisé ensuite pour calculer les lags).
    """
    from forecaster.db.models import RealMeasure

    n_existant = session.query(RealMeasure).filter_by(site_id=SITE_ID).count()
    n_lignes = N_JOURS_HISTORIQUE * 96  # 96 pas de 15 min par jour

    if n_existant >= n_lignes:
        logger.info(
            "Historique déjà présent (%d lignes) — rechargement DataFrame", n_existant
        )
        rows = (
            session.query(RealMeasure.timestamp, RealMeasure.conso_kw)
            .filter_by(site_id=SITE_ID)
            .order_by(RealMeasure.timestamp)
            .all()
        )
        df = pd.DataFrame(rows, columns=["timestamp", "conso_kw"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    logger.info("Chargement de %d jours de données historiques…", N_JOURS_HISTORIQUE)

    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], nrows=n_lignes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Recalage : la dernière mesure est à maintenant - 15 min
    now = datetime.now(tz=UTC)
    fin_souhaitee = now - pd.Timedelta(minutes=15)
    decalage = fin_souhaitee - df["timestamp"].iloc[-1]
    df["timestamp"] = df["timestamp"] + decalage

    mesures = [
        RealMeasure(
            site_id=SITE_ID,
            timestamp=row["timestamp"].to_pydatetime(),
            conso_kw=float(row["conso_kw"]),
            production_pv_kw=0.0,
            soc_kwh=0.0,
            puissance_bess_kw=0.0,
            puissance_pdl_kw=0.0,
        )
        for _, row in df.iterrows()
    ]
    session.bulk_save_objects(mesures)
    session.commit()
    logger.info("%d mesures insérées dans mesures_reelles", len(mesures))
    return df


# ---------------------------------------------------------------------------
# Étape 4 — Entraînement
# ---------------------------------------------------------------------------


def entrainer_modele_si_absent(session) -> str:
    """
    Entraîne le ConsumptionModel si aucune version active n'existe.
    Retourne le chemin de l'artefact.
    """
    from forecaster.db.models import ModelVersion
    from forecaster.pipeline.training import run_training

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True)
        .first()
    )
    if version_active:
        logger.info(
            "Modèle actif déjà présent (version=%s, MAPE=%.2f%%) — skip entraînement",
            version_active.version,
            version_active.mape_validation or 0.0,
        )
        return version_active.chemin_artefact

    logger.info("Entraînement ConsumptionModel…")
    mape = run_training(session, "consumption")
    # flush() nécessaire : SessionLocal a autoflush=False, la ligne ModelVersion
    # insérée par run_training() n'est pas encore visible pour la requête suivante.
    session.flush()
    logger.info("Entraînement terminé — MAPE validation = %.2f%%", mape)

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True)
        .first()
    )
    return version_active.chemin_artefact


# ---------------------------------------------------------------------------
# Étape 5 — Prévision 48h
# ---------------------------------------------------------------------------


def generer_prevision_48h(session, df_historique: pd.DataFrame, artefact_path: str) -> None:
    """
    Génère une prévision de consommation sur les 48 prochaines heures
    et l'insère dans forecasts_consommation.

    Remplace toute prévision existante pour le site (régénération idempotente).
    """
    from forecaster.db.models import ConsumptionForecast, ModelVersion
    from forecaster.predictors.consumption import ConsumptionModel

    # Suppression des prévisions existantes pour repartir propre
    session.query(ConsumptionForecast).filter_by(site_id=SITE_ID).delete()
    session.commit()

    # Chargement du modèle
    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True)
        .first()
    )
    model = ConsumptionModel(version=version_active.version)
    model.load(Path(artefact_path))
    logger.info("Modèle chargé depuis %s", artefact_path)

    # Construction des timestamps futurs (maintenant + 15min → +48h)
    now = datetime.now(tz=UTC)
    freq = pd.Timedelta(minutes=15)
    ts_futurs = pd.date_range(
        start=now + freq,
        periods=192,  # 48h × 4 pas/h
        freq=freq,
        tz="UTC",
    )

    # Index temporel de l'historique pour les lookups de lags
    df_hist = df_historique.set_index("timestamp").sort_index()
    jours_feries_annees = set(
        holidays.France(years=[now.year, now.year + 1]).keys()
    )

    def lookup_lag(ts: pd.Timestamp, delta: pd.Timedelta) -> float:
        """Retourne la valeur historique la plus proche du timestamp demandé."""
        cible = ts - delta
        if cible in df_hist.index:
            return float(df_hist.loc[cible, "conso_kw"])
        # Recherche du point le plus proche (tolérance 30 min)
        # total_seconds() retourne un numpy array → np.abs() fiable sur toutes versions pandas
        diff_secs = np.abs((df_hist.index - cible).total_seconds())
        idx_min = int(np.argmin(diff_secs))
        if diff_secs[idx_min] <= 1800:  # 30 minutes
            return float(df_hist.iloc[idx_min]["conso_kw"])
        # Fallback : moyenne de l'historique
        return float(df_hist["conso_kw"].mean())

    # Construction du DataFrame de features pour la prédiction
    lignes = []
    for ts in ts_futurs:
        horizon_h = round((ts - now).total_seconds() / 3600)
        lag_1d = lookup_lag(ts, pd.Timedelta(days=1))
        lag_7d = lookup_lag(ts, pd.Timedelta(days=7))
        is_holiday = int(ts.date() in jours_feries_annees)

        lignes.append({
            "timestamp": ts,
            "horizon_h": horizon_h,
            "conso_kw_lag_1d": lag_1d,
            "conso_kw_lag_7d": lag_7d,
            "temperature_c": 0.0,
            "temp_lag_1d": 0.0,
            "temp_lag_7d": 0.0,
            "is_holiday": is_holiday,
            "is_school_holiday": 0,
        })

    df_futur = pd.DataFrame(lignes)

    # Prédiction
    forecast_points = model.predict(df_futur)
    logger.info("%d points de prévision générés", len(forecast_points))

    # Insertion en DB
    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        ConsumptionForecast(
            site_id=SITE_ID,
            timestamp=fp.timestamp,
            puissance_kw=max(fp.puissance_kw, 0.0),  # pas de valeur négative
            horizon_h=fp.horizon_h,
            date_generation=now_utc,
            version_modele=version_active.version,
        )
        for fp in forecast_points
    ]
    session.bulk_save_objects(enregistrements)
    session.commit()
    logger.info("%d prévisions insérées dans forecasts_consommation", len(enregistrements))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from forecaster.db.models import Base
    from forecaster.db.session import SessionLocal, engine

    logger.info("=== init_demo — démarrage ===")

    # Étape 1 : schéma DB
    attendre_db(engine)
    Base.metadata.create_all(engine)
    logger.info("Schéma DB prêt")

    with SessionLocal() as session:
        # Étape 2 : site
        inserer_site_si_absent(session)

        # Étape 3 : historique
        df_historique = charger_historique_si_absent(session)

        # Étape 4 : entraînement
        artefact_path = entrainer_modele_si_absent(session)

        # Étape 5 : prévision
        generer_prevision_48h(session, df_historique, artefact_path)

    logger.info("=== init_demo — terminé ✓ ===")


if __name__ == "__main__":
    main()
