"""
Tests pour pipeline/training.py — run_training() et _load_training_data().

La base de données SQLite en mémoire (fixture db_session de conftest.py) est
alimentée avec des données issues du CSV synthétique. Aucun appel HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from forecaster.db.models import ModelVersion, RealMeasure, Site
from forecaster.pipeline.training import (
    InsufficientDataError,
    _load_training_data,
    run_training,
)

CSV_PATH = Path(__file__).parent.parent / "fixtures" / "load_history_2025.csv"

# Nombre de lignes à insérer en DB pour les tests de pipeline
# Les lags 7j nécessitent 672 pas de "warm-up".
# 1200 lignes → 1200 - 672 = 528 lignes utiles > seuil 500 ✓
N_LIGNES_SUFFISANTES = 1200
N_LIGNES_INSUFFISANTES = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inserer_mesures(
    session: Session,
    site: Site,
    n_lignes: int,
    decaler_vers_maintenant: bool = False,
) -> None:
    """
    Insère les n_lignes premières lignes du CSV dans mesures_reelles.

    Si decaler_vers_maintenant=True, les timestamps sont recalés pour se
    terminer à maintenant - 1h (utile pour les tests qui utilisent la fenêtre
    glissante de 90 jours de run_training()).
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], nrows=n_lignes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if decaler_vers_maintenant:
        now = datetime.now(tz=UTC)
        fin_souhaitee = now - pd.Timedelta(hours=1)
        decalage = fin_souhaitee - df["timestamp"].iloc[-1]
        df["timestamp"] = df["timestamp"] + decalage

    for _, row in df.iterrows():
        session.add(
            RealMeasure(
                site_id=site.site_id,
                timestamp=row["timestamp"].to_pydatetime(),
                conso_kw=row["conso_kw"],
                production_pv_kw=0.0,
                soc_kwh=0.0,
                puissance_bess_kw=0.0,
                puissance_pdl_kw=0.0,
            )
        )
    session.flush()


@pytest.fixture
def site_test(db_session: Session) -> Site:
    """Site de test inséré dans la DB SQLite."""
    site = Site(
        site_id="site-training-test",
        nom="Site Training Test",
        capacite_bess_kwh=500.0,
        p_max_bess_kw=250.0,
        p_pv_peak_kw=300.0,
        p_souscrite_kw=700.0,
        soc_min_pct=10.0,
        soc_max_pct=90.0,
        latitude=43.6047,
        longitude=1.4442,
    )
    db_session.add(site)
    db_session.flush()
    return site


@pytest.fixture
def db_avec_donnees_suffisantes(db_session: Session, site_test: Site) -> Session:
    """DB alimentée avec assez de données pour un entraînement complet (timestamps 2025)."""
    _inserer_mesures(db_session, site_test, N_LIGNES_SUFFISANTES)
    return db_session


@pytest.fixture
def db_avec_donnees_recentes(db_session: Session, site_test: Site) -> Session:
    """DB avec données recalées dans la fenêtre glissante des 90 derniers jours."""
    _inserer_mesures(db_session, site_test, N_LIGNES_SUFFISANTES, decaler_vers_maintenant=True)
    return db_session


@pytest.fixture
def db_avec_donnees_insuffisantes(db_session: Session, site_test: Site) -> Session:
    """DB alimentée avec seulement 10 lignes (trop peu pour les lags)."""
    _inserer_mesures(db_session, site_test, N_LIGNES_INSUFFISANTES)
    return db_session


# ---------------------------------------------------------------------------
# Tests _load_training_data
# ---------------------------------------------------------------------------


def test_load_training_data_retourne_deux_dataframes(
    db_avec_donnees_suffisantes: Session,
) -> None:
    """_load_training_data() retourne un tuple (df_train, df_val) non vides."""
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    df_train, df_val = _load_training_data(db_avec_donnees_suffisantes, "consumption", cutoff)

    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_val, pd.DataFrame)
    assert len(df_train) > 0
    assert len(df_val) > 0


def test_load_training_data_split_chronologique(
    db_avec_donnees_suffisantes: Session,
) -> None:
    """Le split est chronologique : tous les timestamps de train < timestamps de val."""
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    df_train, df_val = _load_training_data(db_avec_donnees_suffisantes, "consumption", cutoff)

    ts_train_max = df_train["timestamp"].max()
    ts_val_min = df_val["timestamp"].min()
    assert ts_train_max < ts_val_min


def test_load_training_data_contient_colonnes_contexte(
    db_avec_donnees_suffisantes: Session,
) -> None:
    """Le DataFrame retourné contient toutes les colonnes attendues par build_features()."""
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    df_train, _ = _load_training_data(db_avec_donnees_suffisantes, "consumption", cutoff)

    colonnes_attendues = [
        "timestamp", "conso_kw",
        "conso_kw_lag_1d", "conso_kw_lag_7d",
        "temperature_c", "temp_lag_1d", "temp_lag_7d",
        "is_holiday", "is_school_holiday",
    ]
    for col in colonnes_attendues:
        assert col in df_train.columns, f"Colonne manquante dans df_train : {col}"


def test_load_training_data_leve_erreur_si_donnees_insuffisantes(
    db_avec_donnees_insuffisantes: Session,
) -> None:
    """_load_training_data() lève InsufficientDataError avec trop peu de données."""
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(InsufficientDataError):
        _load_training_data(db_avec_donnees_insuffisantes, "consumption", cutoff)


# ---------------------------------------------------------------------------
# Tests run_training
# ---------------------------------------------------------------------------


def test_run_training_leve_erreur_sur_type_invalide(db_session: Session) -> None:
    """run_training() lève ValueError pour un model_type inconnu."""
    with pytest.raises(ValueError, match="model_type invalide"):
        run_training(db_session, "invalid_model")


def test_run_training_consumption_enregistre_version(
    db_avec_donnees_recentes: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    run_training() pour 'consumption' :
      - retourne une MAPE (float)
      - insère 1 ligne actif=True dans modeles_versions
    """
    # Rediriger models_dir vers un dossier temporaire
    from forecaster import config as cfg
    monkeypatch.setattr(cfg.settings, "models_dir", tmp_path)

    mape = run_training(db_avec_donnees_recentes, "consumption")

    assert isinstance(mape, float)
    assert mape >= 0.0

    versions = (
        db_avec_donnees_recentes.query(ModelVersion)
        .filter_by(type_modele="consumption", actif=True)
        .all()
    )
    assert len(versions) == 1
    assert versions[0].mape_validation == pytest.approx(mape, rel=1e-3)

