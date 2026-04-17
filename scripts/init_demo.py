"""
Script d'initialisation pour la démo locale (multi-sites).

Exécuté une seule fois par le service `forecast-init` au démarrage.
Idempotent : peut être relancé sans erreur si les données existent déjà.

Sites créés :
  - site-demo-01 : Site Industriel Demo (Toulouse, 300 kWc PV)
  - site-demo-02 : Entrepôt Logistique Demo (Lyon, 150 kWc PV)

Étapes exécutées pour chaque site :
  1. Création du schéma DB (Base.metadata.create_all)
  2. Insertion du site
  3. Chargement de 100 jours de données historiques (CSV ou synthétique)
  3b. Ajout de la production PV synthétique sur l'historique
  4. Entraînement du ConsumptionModel (per-site)
  4b. Entraînement du PVProductionModel (per-site, données synthétiques)
  5. Génération de la prévision consommation 48h
  5b. Génération de la prévision production PV 48h (via Open-Meteo)
"""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import func, text

# Chemin du CSV synthétique (copié dans l'image Docker)
CSV_PATH = Path(__file__).parent.parent / "data" / "load_history_2025.csv"

N_JOURS_HISTORIQUE = 100  # jours de données insérées en DB

# Paramètres des sites de démonstration
SITES = [
    {
        "site_id": "site-demo-01",
        "nom": "Site Industriel Demo",
        "latitude": 43.6047,    # Toulouse
        "longitude": 1.4442,
        "p_pv_peak_kw": 300.0,
        "capacite_bess_kwh": 500.0,
        "p_max_bess_kw": 250.0,
        "p_souscrite_kw": 700.0,
        "source_conso": "csv",
    },
    {
        "site_id": "site-demo-02",
        "nom": "Entrepôt Logistique Demo",
        "latitude": 45.7640,    # Lyon
        "longitude": 4.8357,
        "p_pv_peak_kw": 150.0,
        "capacite_bess_kwh": 250.0,
        "p_max_bess_kw": 125.0,
        "p_souscrite_kw": 400.0,
        "source_conso": "synthetic",
    },
]

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


def inserer_site_si_absent(session, site: dict) -> None:
    """Insère un site de démonstration s'il n'existe pas encore."""
    from forecaster.db.models import Site

    site_id = site["site_id"]
    existant = session.query(Site).filter_by(site_id=site_id).first()
    if existant:
        logger.info("Site '%s' déjà présent — skip", site_id)
        return

    db_site = Site(
        site_id=site_id,
        nom=site["nom"],
        capacite_bess_kwh=site["capacite_bess_kwh"],
        p_max_bess_kw=site["p_max_bess_kw"],
        p_pv_peak_kw=site["p_pv_peak_kw"],
        p_souscrite_kw=site["p_souscrite_kw"],
        soc_min_pct=10.0,
        soc_max_pct=90.0,
        latitude=site["latitude"],
        longitude=site["longitude"],
    )
    session.add(db_site)
    session.commit()
    logger.info("Site '%s' inséré", site_id)


# ---------------------------------------------------------------------------
# Étape 3 — Données historiques de consommation
# ---------------------------------------------------------------------------


def charger_historique_si_absent(session, site: dict) -> pd.DataFrame:
    """
    Charge N_JOURS_HISTORIQUE jours de données dans mesures_reelles pour un site.

    - Si source_conso == "csv" : lit le fichier CSV existant.
    - Si source_conso == "synthetic" : génère un profil entrepôt synthétique.

    Les timestamps sont recalés pour se terminer à maintenant - 15 min.
    Retourne le DataFrame chargé (utilisé ensuite pour calculer les lags).
    """
    from forecaster.db.models import RealMeasure

    site_id = site["site_id"]
    n_existant = session.query(RealMeasure).filter_by(site_id=site_id).count()
    n_lignes = N_JOURS_HISTORIQUE * 96  # 96 pas de 15 min par jour

    if n_existant >= n_lignes:
        logger.info(
            "Historique site '%s' déjà présent (%d lignes) — rechargement DataFrame",
            site_id,
            n_existant,
        )
        rows = (
            session.query(RealMeasure.timestamp, RealMeasure.conso_kw)
            .filter_by(site_id=site_id)
            .order_by(RealMeasure.timestamp)
            .all()
        )
        df = pd.DataFrame(rows, columns=["timestamp", "conso_kw"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    logger.info(
        "Chargement de %d jours de données historiques pour '%s'…",
        N_JOURS_HISTORIQUE,
        site_id,
    )

    # Recalage : la dernière mesure est à maintenant - 15 min
    now = datetime.now(tz=UTC)
    fin_souhaitee = now - pd.Timedelta(minutes=15)

    if site["source_conso"] == "csv":
        df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], nrows=n_lignes)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        decalage = fin_souhaitee - df["timestamp"].iloc[-1]
        df["timestamp"] = df["timestamp"] + decalage
    else:
        # Génération synthétique d'un profil entrepôt logistique
        freq = pd.Timedelta(minutes=15)
        timestamps = pd.date_range(
            end=fin_souhaitee,
            periods=n_lignes,
            freq=freq,
            tz="UTC",
        )
        rng = np.random.default_rng(seed=43)
        consos = _generer_conso_entrepot(timestamps, rng)
        df = pd.DataFrame({"timestamp": timestamps, "conso_kw": consos})

    mesures = [
        RealMeasure(
            site_id=site_id,
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
    logger.info("%d mesures insérées dans mesures_reelles pour '%s'", len(mesures), site_id)
    return df


def _generer_conso_entrepot(
    timestamps: pd.DatetimeIndex, rng: np.random.Generator
) -> list[float]:
    """
    Génère une série de consommation synthétique pour un entrepôt logistique.

    Profil : base 120 kW, +80 kW heures ouvrées (8h-18h, lun-ven),
    −40 kW week-end. Bruit gaussien ±5 %.
    Les timestamps sont en UTC — on approxime l'heure locale avec UTC+1.
    """
    consos = []
    for ts in timestamps:
        hour_local = (ts.hour + 1) % 24  # approximation UTC → Paris (hiver)
        jour_semaine = ts.weekday()  # 0=lundi, 6=dimanche

        if jour_semaine < 5 and 8 <= hour_local < 18:
            base = 200.0  # heures ouvrées semaine
        elif jour_semaine >= 5:
            base = 80.0   # week-end
        else:
            base = 120.0  # nuit semaine

        noise = float(rng.normal(1.0, 0.05))
        consos.append(max(0.0, base * noise))
    return consos


# ---------------------------------------------------------------------------
# Étape 3b — Production PV synthétique
# ---------------------------------------------------------------------------


def ajouter_production_pv_synthetique(session, site: dict) -> pd.DataFrame:
    """
    Calcule la production PV synthétique pour l'historique existant du site et met à
    jour mesures_reelles.production_pv_kw.

    Modèle synthétique :
      - Irradiance clear-sky basée sur l'angle d'élévation solaire
      - Nébulosité aléatoire avec corrélation journalière
      - Production = p_pv_peak_kw × (irradiance/1000) × cloud_factor × bruit

    Retourne un DataFrame (timestamp, irradiance_wm2, cloud_cover_pct,
    temperature_c, production_pv_kw) utilisé pour l'entraînement du modèle PV.
    """
    from forecaster.db.models import RealMeasure

    site_id = site["site_id"]
    latitude = site["latitude"]
    p_pv_peak_kw = site["p_pv_peak_kw"]

    # Vérifier si la production PV est déjà renseignée (idempotence)
    somme = (
        session.query(func.sum(RealMeasure.production_pv_kw))
        .filter_by(site_id=site_id)
        .scalar()
    )
    if somme and somme > 0:
        logger.info(
            "Production PV site '%s' déjà présente (somme=%.1f kW·pas) — rechargement",
            site_id,
            somme,
        )
        rows = (
            session.query(
                RealMeasure.timestamp,
                RealMeasure.production_pv_kw,
            )
            .filter_by(site_id=site_id)
            .order_by(RealMeasure.timestamp)
            .all()
        )
        # On ne peut pas reconstituer irradiance/cloud_cover à partir de la DB
        # → on les recalcule avec le même seed pour la cohérence
        timestamps = [r.timestamp for r in rows]
        productions = [r.production_pv_kw for r in rows]
        rng = np.random.default_rng(seed=42)
        cloud_covers = _generer_cloud_cover_serie(len(timestamps), rng)
        df_synth = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "irradiance_wm2": [
                _irradiance_clear_sky(latitude, ts) for ts in timestamps
            ],
            "cloud_cover_pct": cloud_covers,
            "temperature_c": [_temperature_synthetique(ts) for ts in timestamps],
            "production_pv_kw": productions,
        })
        return df_synth

    logger.info("Génération de la production PV synthétique pour '%s'…", site_id)

    rows = (
        session.query(RealMeasure.id, RealMeasure.timestamp)
        .filter_by(site_id=site_id)
        .order_by(RealMeasure.timestamp)
        .all()
    )

    rng = np.random.default_rng(seed=42)  # seed fixe → reproductible
    cloud_covers = _generer_cloud_cover_serie(len(rows), rng)

    mappings = []
    meteo_lignes = []
    for i, (row_id, ts) in enumerate(rows):
        ts_utc = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        irradiance = _irradiance_clear_sky(latitude, ts_utc)
        cloud_cover = cloud_covers[i]
        temperature = _temperature_synthetique(ts_utc)
        production = _production_pv(irradiance, cloud_cover, temperature, p_pv_peak_kw, rng)

        # float() obligatoire : np.float64 n'est pas sérialisable par psycopg2
        mappings.append({"id": row_id, "production_pv_kw": float(production)})
        meteo_lignes.append({
            "timestamp": ts_utc,
            "irradiance_wm2": float(irradiance),
            "cloud_cover_pct": float(cloud_cover),
            "temperature_c": float(temperature),
            "production_pv_kw": float(production),
        })

    session.bulk_update_mappings(RealMeasure, mappings)
    session.commit()
    logger.info(
        "%d mesures mises à jour avec production PV synthétique pour '%s'",
        len(mappings),
        site_id,
    )

    df_synth = pd.DataFrame(meteo_lignes)
    df_synth["timestamp"] = pd.to_datetime(df_synth["timestamp"], utc=True)
    return df_synth


def _irradiance_clear_sky(latitude_deg: float, ts_utc: datetime) -> float:
    """
    Calcule l'irradiance GHI de ciel clair (W/m²) à partir de l'angle d'élévation
    solaire. Retourne 0.0 si le soleil est sous l'horizon.

    Modèle simplifié : GHI ≈ 1000 × sin(élévation solaire).
    """
    day_of_year = ts_utc.timetuple().tm_yday
    # Déclinaison solaire (formule de Cooper)
    declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    # Angle horaire solaire (heure UTC + correction longitude ~0 pour Toulouse)
    solar_hour = ts_utc.hour + ts_utc.minute / 60.0
    hour_angle = 15.0 * (solar_hour - 12.0)  # degrés

    lat_rad = math.radians(latitude_deg)
    decl_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)

    sin_elev = (
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad)
    )
    sin_elev = max(-1.0, min(1.0, sin_elev))  # clamp numérique
    elevation = math.degrees(math.asin(sin_elev))

    if elevation <= 0:
        return 0.0
    return min(1000.0, 1000.0 * math.sin(math.radians(elevation)))


def _generer_cloud_cover_serie(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Génère une série de nébulosité (%) avec corrélation temporelle journalière.
    Chaque journée tire un régime dominant (clair / mitigé / nuageux).
    """
    cloud_covers = np.zeros(n_points)
    n_jours = max(1, math.ceil(n_points / 96))

    for jour in range(n_jours):
        # Régime nuageux dominant : clair 40%, mitigé 35%, nuageux 25%
        regime = rng.choice([0, 1, 2], p=[0.40, 0.35, 0.25])
        if regime == 0:
            cloud_base = float(rng.uniform(0, 25))
        elif regime == 1:
            cloud_base = float(rng.uniform(20, 60))
        else:
            cloud_base = float(rng.uniform(60, 100))

        start = jour * 96
        end = min((jour + 1) * 96, n_points)
        for i in range(start, end):
            variation = float(rng.uniform(-10, 10))
            cloud_covers[i] = np.clip(cloud_base + variation, 0.0, 100.0)

    return cloud_covers


def _temperature_synthetique(ts_utc: datetime, base_temp: float = 15.0) -> float:
    """
    Température synthétique avec variation saisonnière et journalière.
    Base : 15°C (Toulouse annuel moyen), ±10°C saisonnier, ±5°C journalier.
    """
    day_of_year = ts_utc.timetuple().tm_yday
    hour = ts_utc.hour + ts_utc.minute / 60.0
    # Variation saisonnière : max en juillet (jour 172)
    seasonal = 10.0 * math.sin(math.radians(360 / 365 * (day_of_year - 172)))
    # Variation journalière : max vers 14h
    daily = 5.0 * math.sin(math.radians(360 / 24 * (hour - 6)))
    return base_temp + seasonal + daily


def _production_pv(
    irradiance_wm2: float,
    cloud_cover_pct: float,
    temperature_c: float,
    p_pv_peak_kw: float,
    rng: np.random.Generator,
) -> float:
    """
    Production PV synthétique (kW) à partir des conditions météo.

    Modèle : P = P_peak × (irr/1000) × cloud_factor × temp_factor × bruit
    """
    if irradiance_wm2 <= 0:
        return 0.0
    cloud_factor = 1.0 - 0.7 * (cloud_cover_pct / 100.0)
    temp_factor = 1.0 - 0.004 * max(0.0, temperature_c - 25.0)
    noise = float(rng.uniform(0.97, 1.03))
    production = p_pv_peak_kw * (irradiance_wm2 / 1000.0) * cloud_factor * temp_factor * noise
    return max(0.0, production)


# ---------------------------------------------------------------------------
# Étape 4 — Entraînement ConsumptionModel
# ---------------------------------------------------------------------------


def entrainer_modele_si_absent(session, site_id: str) -> str:
    """
    Entraîne le ConsumptionModel pour un site si aucune version active n'existe.
    Retourne le chemin de l'artefact.
    """
    from forecaster.db.models import ModelVersion
    from forecaster.pipeline.training import run_training

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True, site_id=site_id)
        .first()
    )
    if version_active:
        logger.info(
            "Modèle consumption site '%s' déjà présent (version=%s, MAPE=%.2f%%) — skip",
            site_id,
            version_active.version,
            version_active.mape_validation or 0.0,
        )
        return version_active.chemin_artefact

    logger.info("Entraînement ConsumptionModel pour '%s'…", site_id)
    mape = run_training(session, "consumption", site_id)
    # flush() nécessaire : SessionLocal a autoflush=False, la ligne ModelVersion
    # insérée par run_training() n'est pas encore visible pour la requête suivante.
    session.flush()
    logger.info("Entraînement terminé — MAPE validation = %.2f%%", mape)

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True, site_id=site_id)
        .first()
    )
    return version_active.chemin_artefact


# ---------------------------------------------------------------------------
# Étape 4b — Entraînement PVProductionModel
# ---------------------------------------------------------------------------


def entrainer_modele_pv_si_absent(
    session, site_id: str, df_synthetique: pd.DataFrame, p_pv_peak_kw: float
) -> str:
    """
    Entraîne le PVProductionModel pour un site sur les données synthétiques.
    Retourne le chemin de l'artefact.

    On entraîne directement sur les données synthétiques (sans passer par
    run_training() qui appelle fetch_historical()) pour garantir la cohérence
    entre les données de production et la météo utilisée pour les générer.
    """
    from datetime import UTC

    from forecaster.db.models import ModelVersion
    from forecaster.pipeline.training import (
        InsufficientDataError,
        _archive_current_version,
        _build_artifact_path,
        _register_new_version,
    )
    from forecaster.predictors.pv_production import PVProductionModel

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="pv_production", actif=True, site_id=site_id)
        .first()
    )
    if version_active:
        logger.info(
            "Modèle PV site '%s' déjà présent (version=%s, MAPE=%.2f%%) — skip",
            site_id,
            version_active.version,
            version_active.mape_validation or 0.0,
        )
        return version_active.chemin_artefact

    logger.info("Entraînement PVProductionModel sur données synthétiques pour '%s'…", site_id)

    df = df_synthetique.copy().sort_values("timestamp").reset_index(drop=True)
    df["p_pv_peak_kw"] = p_pv_peak_kw

    df = df.dropna(
        subset=["production_pv_kw", "irradiance_wm2", "cloud_cover_pct", "temperature_c"]
    ).reset_index(drop=True)

    if len(df) < 500:
        raise InsufficientDataError(
            f"Données PV synthétiques insuffisantes pour '{site_id}' : {len(df)} lignes (min 500)."
        )

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    version = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    model = PVProductionModel(version=version)
    mape = model.train(df_train, df_val)

    artifact_path = _build_artifact_path("pv_production", version, site_id)
    model.save(artifact_path)

    _archive_current_version(session, "pv_production", site_id)
    _register_new_version(session, "pv_production", site_id, version, mape, artifact_path)
    session.flush()

    logger.info("Entraînement PV terminé pour '%s' — MAPE validation = %.2f%%", site_id, mape)

    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="pv_production", actif=True, site_id=site_id)
        .first()
    )
    return version_active.chemin_artefact


# ---------------------------------------------------------------------------
# Étape 5 — Prévision consommation 48h
# ---------------------------------------------------------------------------


def generer_prevision_48h(
    session, site: dict, df_historique: pd.DataFrame, artefact_path: str
) -> None:
    """
    Génère une prévision de consommation sur les 48 prochaines heures
    et l'insère dans forecasts_consommation.

    Remplace toute prévision existante pour le site (régénération idempotente).
    """
    from forecaster.db.models import ConsumptionForecast, ModelVersion
    from forecaster.predictors.consumption import ConsumptionModel

    site_id = site["site_id"]

    # Suppression des prévisions existantes pour repartir propre
    session.query(ConsumptionForecast).filter_by(site_id=site_id).delete()
    session.commit()

    # Chargement du modèle
    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True, site_id=site_id)
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
        diff_secs = np.abs((df_hist.index - cible).total_seconds())
        idx_min = int(np.argmin(diff_secs))
        if diff_secs[idx_min] <= 1800:  # 30 minutes
            return float(df_hist.iloc[idx_min]["conso_kw"])
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
    logger.info("%d points de prévision consommation générés", len(forecast_points))

    # Insertion en DB
    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        ConsumptionForecast(
            site_id=site_id,
            timestamp=fp.timestamp,
            puissance_kw=max(fp.puissance_kw, 0.0),
            horizon_h=fp.horizon_h,
            date_generation=now_utc,
            version_modele=version_active.version,
        )
        for fp in forecast_points
    ]
    session.bulk_save_objects(enregistrements)
    session.commit()
    logger.info(
        "%d prévisions insérées dans forecasts_consommation pour '%s'",
        len(enregistrements),
        site_id,
    )


# ---------------------------------------------------------------------------
# Étape 5b — Prévision production PV 48h
# ---------------------------------------------------------------------------


def generer_prevision_pv_48h(session, site: dict, artefact_path: str) -> None:
    """
    Génère une prévision de production PV sur les 48 prochaines heures
    en utilisant les prévisions météo réelles d'Open-Meteo.

    Remplace toute prévision existante pour le site (régénération idempotente).
    """
    from forecaster.db.models import ModelVersion, PVProductionForecast
    from forecaster.fetchers.openmeteo import fetch_forecast
    from forecaster.predictors.pv_production import PVProductionModel

    site_id = site["site_id"]
    latitude = site["latitude"]
    longitude = site["longitude"]
    p_pv_peak_kw = site["p_pv_peak_kw"]

    # Suppression des prévisions existantes pour repartir propre
    session.query(PVProductionForecast).filter_by(site_id=site_id).delete()
    session.commit()

    # Chargement du modèle PV
    version_active = (
        session.query(ModelVersion)
        .filter_by(type_modele="pv_production", actif=True, site_id=site_id)
        .first()
    )
    model = PVProductionModel(version=version_active.version)
    model.load(Path(artefact_path))
    logger.info("Modèle PV site '%s' chargé depuis %s", site_id, artefact_path)

    # Récupération de la météo réelle sur 48h
    logger.info("Récupération météo Open-Meteo (48h) pour '%s'…", site_id)
    meteo = fetch_forecast(
        site_id=site_id,
        latitude=latitude,
        longitude=longitude,
        horizon_h=48,
    )

    # Construire un index horaire de la météo pour interpolation à 15 min
    df_meteo_h = pd.DataFrame(
        [
            {
                "timestamp": p.timestamp,
                "temperature_c": p.temperature_c,
                "irradiance_wm2": p.irradiance_wm2,
                "cloud_cover_pct": p.cloud_cover_pct,
            }
            for p in meteo.points
        ]
    )
    df_meteo_h["timestamp"] = pd.to_datetime(df_meteo_h["timestamp"], utc=True)
    df_meteo_h = df_meteo_h.set_index("timestamp").sort_index()

    # Rééchantillonnage horaire → 15 min
    df_meteo_15min = df_meteo_h.resample("15min").interpolate(method="linear")
    df_meteo_15min["cloud_cover_pct"] = (
        df_meteo_h[["cloud_cover_pct"]].resample("15min").ffill()["cloud_cover_pct"]
    )
    df_meteo_15min = df_meteo_15min.reset_index()

    # Construction des timestamps futurs (maintenant + 15min → +48h)
    now = datetime.now(tz=UTC)
    freq = pd.Timedelta(minutes=15)
    ts_futurs = pd.date_range(
        start=now + freq,
        periods=192,
        freq=freq,
        tz="UTC",
    )

    def lookup_meteo(ts: pd.Timestamp) -> dict:
        """Récupère la météo interpolée pour un timestamp donné."""
        ts_norm = ts.floor("15min")
        if ts_norm in df_meteo_15min["timestamp"].values:
            row = df_meteo_15min[df_meteo_15min["timestamp"] == ts_norm].iloc[0]
            return {
                "temperature_c": float(row["temperature_c"]),
                "irradiance_wm2": float(row["irradiance_wm2"]),
                "cloud_cover_pct": float(row["cloud_cover_pct"]),
            }
        # Fallback : irradiance synthétique si hors plage Open-Meteo
        return {
            "temperature_c": _temperature_synthetique(ts.to_pydatetime()),
            "irradiance_wm2": _irradiance_clear_sky(latitude, ts.to_pydatetime()),
            "cloud_cover_pct": 30.0,
        }

    # Construction du DataFrame de features
    lignes = []
    for ts in ts_futurs:
        horizon_h = round((ts - now).total_seconds() / 3600)
        meteo_ts = lookup_meteo(ts)
        lignes.append({
            "timestamp": ts,
            "horizon_h": horizon_h,
            "irradiance_wm2": meteo_ts["irradiance_wm2"],
            "cloud_cover_pct": meteo_ts["cloud_cover_pct"],
            "temperature_c": meteo_ts["temperature_c"],
            "p_pv_peak_kw": p_pv_peak_kw,
        })

    df_futur = pd.DataFrame(lignes)

    # Prédiction
    forecast_points = model.predict(df_futur)
    logger.info("%d points de prévision PV générés", len(forecast_points))

    # Insertion en DB
    now_utc = datetime.now(tz=UTC)
    enregistrements = [
        PVProductionForecast(
            site_id=site_id,
            timestamp=fp.timestamp,
            puissance_kw=fp.puissance_kw,  # déjà clippé à 0 dans predict()
            horizon_h=fp.horizon_h,
            date_generation=now_utc,
            version_modele=version_active.version,
        )
        for fp in forecast_points
    ]
    session.bulk_save_objects(enregistrements)
    session.commit()
    logger.info(
        "%d prévisions insérées dans forecasts_production_pv pour '%s'",
        len(enregistrements),
        site_id,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _appliquer_migrations() -> None:
    """Lance alembic upgrade head pour appliquer toutes les migrations en attente."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(Path(__file__).parent.parent / "alembic.ini")
    command.upgrade(alembic_cfg, "head")
    logger.info("Migrations Alembic appliquées")


def main() -> None:
    from forecaster.db.models import Base
    from forecaster.db.session import SessionLocal, engine

    logger.info("=== init_demo — démarrage (%d sites) ===", len(SITES))

    # Étape 1 : schéma DB + migrations
    attendre_db(engine)
    _appliquer_migrations()
    Base.metadata.create_all(engine)
    logger.info("Schéma DB prêt")

    with SessionLocal() as session:
        # Étapes 2 + 3 : insertion des sites et de leurs historiques
        df_hist_par_site: dict[str, pd.DataFrame] = {}
        df_pv_par_site: dict[str, pd.DataFrame] = {}

        for site in SITES:
            inserer_site_si_absent(session, site)
            df_hist_par_site[site["site_id"]] = charger_historique_si_absent(session, site)
            df_pv_par_site[site["site_id"]] = ajouter_production_pv_synthetique(session, site)

        # Étapes 4 + 5 : entraînement et prévision par site
        for site in SITES:
            site_id = site["site_id"]

            # Entraînement ConsumptionModel
            artefact_conso = entrainer_modele_si_absent(session, site_id)

            # Entraînement PVProductionModel (données synthétiques propres au site)
            artefact_pv = entrainer_modele_pv_si_absent(
                session,
                site_id,
                df_pv_par_site[site_id],
                site["p_pv_peak_kw"],
            )

            # Prévision consommation 48h
            generer_prevision_48h(session, site, df_hist_par_site[site_id], artefact_conso)

            # Prévision production PV 48h
            generer_prevision_pv_48h(session, site, artefact_pv)

    logger.info("=== init_demo — terminé ✓ ===")


if __name__ == "__main__":
    main()
