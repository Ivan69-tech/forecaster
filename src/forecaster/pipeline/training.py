"""
Pipeline de réentraînement des modèles LightGBM (§3.4).

Déclencheurs (§3.3) :
  - Dimanche 02h00 : réentraînement hebdomadaire planifié
  - Sur MAPE > 15 % : réentraînement immédiat hors cycle (via monitoring.py)

Le modèle précédent est archivé avant remplacement (actif = False).
Chaque modèle entraîné est versionné dans la table modeles_versions.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from forecaster.config import settings
from forecaster.db.models import ModelVersion

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

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=TRAINING_WINDOW_DAYS)

    df_train, df_val = _load_training_data(session, model_type, cutoff)
    model = _instantiate_model(model_type)
    mape = model.train(df_train, df_val)

    artifact_path = _build_artifact_path(model_type, model.version)
    model.save(artifact_path)

    _archive_current_version(session, model_type)
    _register_new_version(session, model_type, model.version, mape, artifact_path)

    logger.info("run_training | model=%s | MAPE=%.2f%% | artefact=%s", model_type, mape, artifact_path)
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


def _load_training_data(session: Session, model_type: str, cutoff: datetime):
    """
    Charge les données depuis mesures_reelles + données météo historiques.

    TODO:
        - Requêter mesures_reelles depuis `cutoff` jusqu'à maintenant
        - Joindre avec les données météo archivées si disponibles
        - Splitter 80/20 chronologiquement → (df_train, df_val)
    """
    raise NotImplementedError


def _instantiate_model(model_type: str):
    """Instancie le bon modèle avec une version horodatée."""
    from forecaster.predictors.consumption import ConsumptionModel
    from forecaster.predictors.pv_production import PVProductionModel

    version = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
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
        date_entrainement=datetime.now(tz=timezone.utc),
        mape_validation=mape,
        actif=True,
        chemin_artefact=str(artifact_path),
    )
    session.add(mv)


class InsufficientDataError(Exception):
    """Levée quand les données d'entraînement sont insuffisantes."""
