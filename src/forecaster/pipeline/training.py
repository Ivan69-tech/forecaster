"""
Pipeline de réentraînement des modèles LightGBM (§3.4).

Déclencheurs (§3.3) :
  - Dimanche 02h00 : réentraînement hebdomadaire planifié
  - Sur MAPE > 15 % : réentraînement immédiat hors cycle (via monitoring.py)

Le modèle précédent est archivé avant remplacement (actif = False).
Chaque modèle entraîné est versionné dans la table modeles_versions.
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import holidays
import pandas as pd
from sqlalchemy.orm import Session

from forecaster.config import settings
from forecaster.db.models import ModelVersion
from forecaster.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

MODEL_TYPES = ("consumption", "pv_production")
TRAINING_WINDOW_DAYS = 90  # §3.4 — 90 derniers jours


def run_training(session: Session, model_type: str) -> float:
    """
    Réentraîne le modèle `model_type` sur les 90 derniers jours de mesures réelles.

    Args:
        session:    Session SQLAlchemy active.
        model_type: "consumption" | "pv_production"

    Returns:
        MAPE de validation du nouveau modèle (%).

    Raises:
        ValueError: Si model_type est invalide.
        InsufficientDataError: Si les données d'entraînement sont insuffisantes.

    TODO:
        1. Valider model_type
        2. Lire les 90 derniers jours de mesures_reelles + forecasts météo
        3. Construire df_train (80%) et df_val (20% les plus récents)
        4. Instancier le bon modèle (ConsumptionModel ou PVProductionModel)
        5. Appeler model.train(df_train, df_val) → mape_validation
        6. Générer un chemin d'artefact versionné dans settings.models_dir
        7. Appeler model.save(path)
        8. Archiver l'ancienne version active (actif = False)
        9. Insérer la nouvelle version dans modeles_versions (actif = True)
        10. Logger le résultat
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type invalide : '{model_type}'. Attendu : {MODEL_TYPES}")

    logger.info("run_training | model=%s | démarrage", model_type)

    cutoff = datetime.now(tz=UTC) - timedelta(days=TRAINING_WINDOW_DAYS)

    df_train, df_val = _load_training_data(session, model_type, cutoff)
    model = _instantiate_model(model_type)
    mape = model.train(df_train, df_val)

    artifact_path = _build_artifact_path(model_type, model.version)
    model.save(artifact_path)

    _archive_current_version(session, model_type)
    _register_new_version(session, model_type, model.version, mape, artifact_path)

    logger.info(
        "run_training | model=%s | MAPE=%.2f%% | artefact=%s", model_type, mape, artifact_path
    )
    return mape


def run_training_all(session: Session) -> dict[str, float]:
    """Réentraîne tous les modèles. Retourne {model_type: mape}."""
    results = {}
    for model_type in MODEL_TYPES:
        try:
            results[model_type] = run_training(session, model_type)
        except Exception:
            logger.exception("run_training | model=%s | échec", model_type)
    return results


# ---------------------------------------------------------------------------
# Helpers privés — à implémenter
# ---------------------------------------------------------------------------


def _load_training_data(
    session: Session, model_type: str, cutoff: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données d'entraînement depuis mesures_reelles.

    Pour le modèle de consommation :
      1. Lit conso_kw depuis mesures_reelles depuis `cutoff`
      2. Calcule les lags J-1 et J-7 sur conso_kw
      3. Ajoute temperature_c = 0 (TODO : remplacer par Open-Meteo archive)
      4. Calcule les lags de température J-1 et J-7
      5. Ajoute les indicateurs calendaires (jours fériés, vacances)
      6. Supprime les lignes avec NaN (7 premiers jours sans lags complets)
      7. Lève InsufficientDataError si moins de 500 lignes restantes
      8. Retourne (df_train 80%, df_val 20%) — split chronologique

    Raises:
        InsufficientDataError: Si les données sont insuffisantes après nettoyage.
        NotImplementedError: Si model_type == "pv_production" (non implémenté).
    """
    if model_type == "pv_production":
        return _load_training_data_pv(session, cutoff)

    # Pour les tests, le site_id est passé via la session — on charge tous les sites disponibles.
    # En production, run_training() sera appelé par site. Pour l'instant on agrège.
    # TODO: ajouter site_id en paramètre quand run_training() sera multi-site.
    from forecaster.db.models import RealMeasure

    rows = (
        session.query(RealMeasure.timestamp, RealMeasure.conso_kw)
        .filter(RealMeasure.timestamp >= cutoff)
        .order_by(RealMeasure.timestamp)
        .all()
    )

    if not rows:
        raise InsufficientDataError("Aucune donnée dans mesures_reelles depuis le cutoff.")

    df = pd.DataFrame(rows, columns=["timestamp", "conso_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lags consommation : J-1 = 96 pas × 15 min, J-7 = 672 pas
    df["conso_kw_lag_1d"] = df["conso_kw"].shift(96)
    df["conso_kw_lag_7d"] = df["conso_kw"].shift(672)

    # Température : placeholder zéros (TODO : remplacer par données Open-Meteo archive)
    df["temperature_c"] = 0.0
    df["temp_lag_1d"] = 0.0
    df["temp_lag_7d"] = 0.0

    # Indicateurs calendaires
    annees = df["timestamp"].dt.year.unique().tolist()
    jours_feries = set(holidays.France(years=annees).keys())

    df["is_holiday"] = df["timestamp"].dt.date.isin(jours_feries).astype(int)
    df["is_school_holiday"] = 0  # TODO : intégrer le calendrier des vacances scolaires

    # Suppression des lignes sans lags (7 premiers jours)
    df = df.dropna(subset=["conso_kw_lag_1d", "conso_kw_lag_7d"]).reset_index(drop=True)

    if len(df) < 500:
        raise InsufficientDataError(
            f"Données insuffisantes après calcul des lags : {len(df)} lignes "
            f"(minimum requis : 500 ~= 5 jours)."
        )

    # Split 80/20 chronologique
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "_load_training_data | lignes_total=%d | train=%d | val=%d",
        len(df),
        len(df_train),
        len(df_val),
    )
    return df_train, df_val


def _instantiate_model(model_type: str):
    """Instancie le bon modèle avec une version horodatée."""
    from forecaster.predictors.consumption import ConsumptionModel
    from forecaster.predictors.pv_production import PVProductionModel

    version = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    if model_type == "consumption":
        return ConsumptionModel(version=version)
    return PVProductionModel(version=version)


def _build_artifact_path(model_type: str, version: str) -> Path:
    return settings.models_dir / model_type / f"{version}.joblib"


def _archive_current_version(session: Session, model_type: str) -> None:
    """Passe actif=False sur toutes les versions actives du model_type."""
    session.query(ModelVersion).filter_by(type_modele=model_type, actif=True).update(
        {"actif": False}
    )


def _register_new_version(
    session: Session,
    model_type: str,
    version: str,
    mape: float,
    artifact_path: Path,
) -> None:
    mv = ModelVersion(
        type_modele=model_type,
        version=version,
        date_entrainement=datetime.now(tz=UTC),
        mape_validation=mape,
        actif=True,
        chemin_artefact=str(artifact_path),
    )
    session.add(mv)


def _load_training_data_pv(
    session: Session, cutoff: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données d'entraînement PV depuis mesures_reelles + Open-Meteo archive.

    Pour chaque site en DB :
      1. Lit production_pv_kw depuis mesures_reelles depuis `cutoff`
      2. Récupère la météo historique (irradiance, cloud_cover, température)
         depuis l'API archive Open-Meteo pour la même période
      3. Rééchantillonne la météo horaire → 15 min (interpolation linéaire)
      4. Jointure sur timestamp
      5. Ajoute p_pv_peak_kw (constante par site)

    Tous les sites sont concaténés avant le split train/val.

    Raises:
        InsufficientDataError: Si les données sont insuffisantes après nettoyage.
    """
    from forecaster.db.models import Site
    from forecaster.db.readers import get_mesures_reelles_production_pv
    from forecaster.fetchers.openmeteo import fetch_historical

    sites = session.query(Site).all()
    if not sites:
        raise InsufficientDataError(
            "Aucun site trouvé en base — impossible d'entraîner le modèle PV."
        )

    fragments = []
    for site in sites:
        df_pv = get_mesures_reelles_production_pv(session, site.site_id, cutoff)
        if df_pv.empty:
            logger.warning(
                "_load_training_data_pv | site=%s | aucune donnée PV — site ignoré",
                site.site_id,
            )
            continue

        start_date = df_pv["timestamp"].min().date()
        end_date = df_pv["timestamp"].max().date()

        try:
            meteo = fetch_historical(
                site_id=site.site_id,
                latitude=site.latitude,
                longitude=site.longitude,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            logger.exception(
                "_load_training_data_pv | site=%s | échec fetch_historical — site ignoré",
                site.site_id,
            )
            continue

        # Construire un DataFrame hourly de météo, puis rééchantillonner en 15 min
        df_meteo = pd.DataFrame(
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
        df_meteo["timestamp"] = pd.to_datetime(df_meteo["timestamp"], utc=True)
        df_meteo = df_meteo.set_index("timestamp").sort_index()

        # Rééchantillonnage horaire → 15 min
        # Irradiance : interpolation linéaire (variation douce)
        # Cloud cover : forward-fill (valeur discrète par heure)
        df_meteo_15min = df_meteo.resample("15min").interpolate(method="linear")
        df_meteo_15min["cloud_cover_pct"] = (
            df_meteo[["cloud_cover_pct"]].resample("15min").ffill()["cloud_cover_pct"]
        )
        df_meteo_15min = df_meteo_15min.reset_index()
        df_meteo_15min = df_meteo_15min.rename(columns={"index": "timestamp"})

        # Jointure sur timestamp (inner join — on ne garde que les instants communs)
        df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], utc=True)
        df_merged = pd.merge(df_pv, df_meteo_15min, on="timestamp", how="inner")
        df_merged["p_pv_peak_kw"] = float(site.p_pv_peak_kw)

        fragments.append(df_merged)

    if not fragments:
        raise InsufficientDataError(
            "Aucun site avec suffisamment de données PV + météo pour l'entraînement."
        )

    df = pd.concat(fragments, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(
        subset=["production_pv_kw", "irradiance_wm2", "cloud_cover_pct", "temperature_c"]
    ).reset_index(drop=True)

    if len(df) < 500:
        raise InsufficientDataError(
            f"Données PV insuffisantes après jointure : {len(df)} lignes "
            f"(minimum requis : 500 ~= 5 jours)."
        )

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "_load_training_data_pv | lignes_total=%d | train=%d | val=%d",
        len(df),
        len(df_train),
        len(df_val),
    )
    return df_train, df_val


__all__ = ["run_training", "run_training_all", "InsufficientDataError"]
