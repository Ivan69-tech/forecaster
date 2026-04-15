"""
Tests pour pipeline/monitoring.py — compute_mape() et check_mape_all_sites().

La base de données SQLite en mémoire est alimentée avec des prévisions
et mesures synthétiques. Aucun appel HTTP.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from forecaster.db.models import (
    ConsumptionForecast,
    PVProductionForecast,
    RealMeasure,
    Site,
)
from forecaster.pipeline.monitoring import (
    _trigger_retraining,
    check_mape_all_sites,
    compute_mape,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def site_monitoring(db_session: Session) -> Site:
    """Site de test pour le monitoring."""
    site = Site(
        site_id="site-monitoring-test",
        nom="Site Monitoring Test",
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


def _inserer_previsions_et_mesures(
    session: Session,
    site_id: str,
    n_points: int = 20,
    erreur_pct: float = 5.0,
) -> None:
    """
    Insère des prévisions et mesures alignées sur les dernières heures.

    L'erreur entre prévision et mesure est contrôlée par erreur_pct.
    """
    now = datetime.now(tz=UTC)

    for i in range(n_points):
        ts = now - timedelta(minutes=15 * (n_points - i))
        valeur_reelle = 100.0 + i * 2.0
        valeur_prevue = valeur_reelle * (1 + erreur_pct / 100.0)

        session.add(
            ConsumptionForecast(
                site_id=site_id,
                timestamp=ts,
                puissance_kw=valeur_prevue,
                horizon_h=24,
                date_generation=now - timedelta(hours=24),
                version_modele="test_v1",
            )
        )

        session.add(
            PVProductionForecast(
                site_id=site_id,
                timestamp=ts,
                puissance_kw=valeur_prevue * 0.5,
                horizon_h=24,
                date_generation=now - timedelta(hours=24),
                version_modele="test_v1",
            )
        )

        session.add(
            RealMeasure(
                site_id=site_id,
                timestamp=ts,
                conso_kw=valeur_reelle,
                production_pv_kw=valeur_reelle * 0.5,
                soc_kwh=200.0,
                puissance_bess_kw=0.0,
                puissance_pdl_kw=valeur_reelle,
            )
        )

    session.flush()


# ---------------------------------------------------------------------------
# Tests compute_mape
# ---------------------------------------------------------------------------


def test_compute_mape_retourne_valeur_correcte(
    db_session: Session, site_monitoring: Site
) -> None:
    """compute_mape() retourne une MAPE proche de l'erreur injectée."""
    erreur_injectee = 10.0
    _inserer_previsions_et_mesures(
        db_session, site_monitoring.site_id, n_points=20, erreur_pct=erreur_injectee
    )

    mape = compute_mape(db_session, site_monitoring.site_id, "consumption")

    assert not math.isnan(mape)
    assert mape == pytest.approx(erreur_injectee, abs=2.0)


def test_compute_mape_pv_retourne_valeur_correcte(
    db_session: Session, site_monitoring: Site
) -> None:
    """compute_mape() fonctionne aussi pour pv_production."""
    _inserer_previsions_et_mesures(
        db_session, site_monitoring.site_id, n_points=20, erreur_pct=5.0
    )

    mape = compute_mape(db_session, site_monitoring.site_id, "pv_production")

    assert not math.isnan(mape)
    assert mape > 0.0


def test_compute_mape_retourne_nan_sans_donnees(
    db_session: Session, site_monitoring: Site
) -> None:
    """compute_mape() retourne NaN quand il n'y a pas de données."""
    mape = compute_mape(db_session, site_monitoring.site_id, "consumption")
    assert math.isnan(mape)


def test_compute_mape_retourne_nan_donnees_insuffisantes(
    db_session: Session, site_monitoring: Site
) -> None:
    """compute_mape() retourne NaN avec moins de 10 paires."""
    _inserer_previsions_et_mesures(
        db_session, site_monitoring.site_id, n_points=5, erreur_pct=5.0
    )
    mape = compute_mape(db_session, site_monitoring.site_id, "consumption")
    assert math.isnan(mape)


def test_compute_mape_leve_erreur_model_type_invalide(
    db_session: Session, site_monitoring: Site
) -> None:
    """compute_mape() lève ValueError pour un model_type inconnu."""
    with pytest.raises(ValueError, match="model_type invalide"):
        compute_mape(db_session, site_monitoring.site_id, "invalid_type")


# ---------------------------------------------------------------------------
# Tests _trigger_retraining
# ---------------------------------------------------------------------------


def test_trigger_retraining_appelle_run_training(
    db_session: Session, site_monitoring: Site
) -> None:
    """_trigger_retraining() appelle run_training() avec les bons arguments."""
    with patch(
        "forecaster.pipeline.training.run_training", return_value=8.0
    ) as mock_train:
        _trigger_retraining(db_session, site_monitoring.site_id, "consumption", 20.0)

    mock_train.assert_called_once_with(db_session, "consumption")


def test_trigger_retraining_ne_plante_pas_si_training_echoue(
    db_session: Session, site_monitoring: Site
) -> None:
    """_trigger_retraining() attrape les exceptions du training."""
    with patch(
        "forecaster.pipeline.training.run_training",
        side_effect=Exception("Training échoué"),
    ):
        # Ne doit pas lever d'exception
        _trigger_retraining(db_session, site_monitoring.site_id, "consumption", 20.0)


# ---------------------------------------------------------------------------
# Tests check_mape_all_sites
# ---------------------------------------------------------------------------


def test_check_mape_all_sites_ne_plante_pas(
    db_session: Session, site_monitoring: Site
) -> None:
    """check_mape_all_sites() itère sans planter même sans données."""
    check_mape_all_sites(db_session)


def test_check_mape_all_sites_declenche_retraining_si_mape_haute(
    db_session: Session, site_monitoring: Site
) -> None:
    """check_mape_all_sites() déclenche le retraining si MAPE > seuil."""
    _inserer_previsions_et_mesures(
        db_session, site_monitoring.site_id, n_points=20, erreur_pct=25.0
    )

    with patch(
        "forecaster.pipeline.monitoring._trigger_retraining"
    ) as mock_trigger:
        check_mape_all_sites(db_session)

    # Au moins un appel pour consumption (MAPE ~25% > seuil 15%)
    assert mock_trigger.called
