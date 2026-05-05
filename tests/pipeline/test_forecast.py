"""
Tests pour pipeline/forecast.py — run_forecast() et helpers.

La base de données SQLite en mémoire (fixture db_session de conftest.py) est
alimentée avec des données synthétiques. Les appels HTTP (Open-Meteo) sont mockés.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from forecaster.db.models import (
    ConsumptionForecast,
    ModelVersion,
    PVProductionForecast,
    RealMeasure,
    Site,
)
from forecaster.exceptions import ForecastUnavailableError, SiteNotFoundError
from forecaster.fetchers.openmeteo import WeatherForecast, WeatherPoint
from forecaster.pipeline.forecast import (
    _interpoler_meteo_15min,
    _lookup_conso_lag,
    _predict_consumption,
    _predict_pv,
    run_forecast,
    run_forecast_all_sites,
)

CSV_PATH = Path(__file__).parent.parent / "fixtures" / "load_history_2025.csv"

# Nombre de lignes pour les tests (8 jours de données = lags J-7 couverts)
N_LIGNES = 96 * 8  # 768 lignes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def site_forecast(db_session: Session) -> Site:
    """Site de test pour les tests de forecast."""
    site = Site(
        site_id="site-forecast-test",
        nom="Site Forecast Test",
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


def _inserer_mesures_recentes(session: Session, site: Site, n_lignes: int) -> None:
    """Insère des mesures dont les timestamps se terminent près de maintenant."""
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], nrows=n_lignes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    now = datetime.now(tz=UTC)
    fin_souhaitee = now - pd.Timedelta(minutes=15)
    decalage = fin_souhaitee - df["timestamp"].iloc[-1]
    df["timestamp"] = df["timestamp"] + decalage

    for _, row in df.iterrows():
        session.add(
            RealMeasure(
                site_id=site.site_id,
                timestamp=row["timestamp"].to_pydatetime(),
                conso_kw=row["conso_kw"],
                production_pv_kw=50.0,
                soc_kwh=0.0,
                puissance_bess_kw=0.0,
                puissance_pdl_kw=0.0,
            )
        )
    session.flush()


def _creer_modele_actif(
    session: Session, model_type: str, tmp_path: Path, site_id: str = "site-forecast-test"
) -> ModelVersion:
    """Entraîne un modèle minimal et enregistre la version active en DB."""
    if model_type == "consumption":
        from forecaster.predictors.consumption import ConsumptionModel

        model = ConsumptionModel(version="test_v1")
        # Créer des données d'entraînement minimales
        n = 600
        timestamps = pd.date_range("2025-01-08", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "conso_kw": [100.0 + i % 50 for i in range(n)],
            "conso_kw_lag_1d": [95.0] * n,
            "conso_kw_lag_7d": [98.0] * n,
            "temperature_c": [15.0] * n,
            "temp_lag_1d": [14.0] * n,
            "temp_lag_7d": [13.0] * n,
            "is_holiday": [0] * n,
            "is_school_holiday": [0] * n,
        })
    else:
        from forecaster.predictors.pv_production import PVProductionModel

        model = PVProductionModel(version="test_v1")
        n = 600
        timestamps = pd.date_range("2025-01-08", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "production_pv_kw": [50.0 + i % 30 for i in range(n)],
            "irradiance_wm2": [400.0 + i % 200 for i in range(n)],
            "cloud_cover_pct": [30.0] * n,
            "temperature_c": [20.0] * n,
            "p_pv_peak_kw": [300.0] * n,
        })

    split = int(n * 0.8)
    model.train(df.iloc[:split], df.iloc[split:])

    artifact_path = tmp_path / model_type / "test_v1.joblib"
    model.save(artifact_path)

    mv = ModelVersion(
        site_id=site_id,
        type_modele=model_type,
        version="test_v1",
        date_entrainement=datetime.now(tz=UTC),
        mape_validation=5.0,
        actif=True,
        chemin_artefact=str(artifact_path),
    )
    session.add(mv)
    session.flush()
    return mv


def _generer_weather_mock(horizon_h: int = 48) -> WeatherForecast:
    """Crée un WeatherForecast synthétique pour mocker fetch_forecast."""
    now = datetime.now(tz=UTC)
    points = [
        WeatherPoint(
            timestamp=now + timedelta(hours=h),
            temperature_c=15.0 + h * 0.1,
            irradiance_wm2=max(0.0, 500.0 - abs(h - 12) * 40),
            cloud_cover_pct=30.0,
        )
        for h in range(horizon_h + 1)
    ]
    return WeatherForecast(
        site_id="site-forecast-test",
        latitude=43.6047,
        longitude=1.4442,
        points=points,
    )


# ---------------------------------------------------------------------------
# Tests _interpoler_meteo_15min
# ---------------------------------------------------------------------------


def test_interpoler_meteo_15min_genere_pas_de_15min() -> None:
    """L'interpolation multiplie par ~4 le nombre de points horaires."""
    weather = _generer_weather_mock(horizon_h=4)
    df = _interpoler_meteo_15min(weather)

    assert len(df) >= 4 * 4  # au moins 16 points pour 4h
    assert "temperature_c" in df.columns
    assert "irradiance_wm2" in df.columns
    assert "cloud_cover_pct" in df.columns


def test_interpoler_meteo_15min_vide() -> None:
    """Avec un WeatherForecast vide, retourne un DataFrame vide."""
    weather = WeatherForecast(site_id="test", latitude=0.0, longitude=0.0, points=[])
    df = _interpoler_meteo_15min(weather)
    assert df.empty


# ---------------------------------------------------------------------------
# Tests _lookup_conso_lag
# ---------------------------------------------------------------------------


def test_lookup_conso_lag_trouve_valeur_exacte() -> None:
    """Retourne la valeur exacte quand le timestamp correspond."""
    now = datetime.now(tz=UTC)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([now - timedelta(days=1)], utc=True),
        "conso_kw": [42.0],
        "production_pv_kw": [0.0],
    })
    result = _lookup_conso_lag(df, pd.Timestamp(now), timedelta(days=1))
    assert result == pytest.approx(42.0)


def test_lookup_conso_lag_retourne_moyenne_hors_tolerance() -> None:
    """Retourne la moyenne quand aucune mesure n'est dans la tolérance."""
    now = datetime.now(tz=UTC)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([now - timedelta(days=3)], utc=True),
        "conso_kw": [100.0],
        "production_pv_kw": [0.0],
    })
    result = _lookup_conso_lag(df, pd.Timestamp(now), timedelta(days=1))
    assert result == pytest.approx(100.0)  # moyenne d'une seule valeur


def test_lookup_conso_lag_dataframe_vide() -> None:
    """Retourne 0.0 avec un DataFrame vide."""
    df = pd.DataFrame(columns=["timestamp", "conso_kw", "production_pv_kw"])
    now = pd.Timestamp(datetime.now(tz=UTC))
    assert _lookup_conso_lag(df, now, timedelta(days=1)) == 0.0


# ---------------------------------------------------------------------------
# Tests _predict_consumption
# ---------------------------------------------------------------------------


def test_predict_consumption_retourne_forecast_points(
    db_session: Session, site_forecast: Site, tmp_path: Path
) -> None:
    """_predict_consumption retourne une liste de ForecastPoint."""
    _inserer_mesures_recentes(db_session, site_forecast, N_LIGNES)
    _creer_modele_actif(db_session, "consumption", tmp_path, site_forecast.site_id)

    weather = _generer_weather_mock(horizon_h=24)
    points = _predict_consumption(db_session, site_forecast, weather, 24)

    assert len(points) == 24 * 4  # 96 pas pour 24h
    assert all(hasattr(p, "puissance_kw") for p in points)
    assert all(hasattr(p, "timestamp") for p in points)


def test_predict_consumption_leve_erreur_sans_modele_actif(
    db_session: Session, site_forecast: Site
) -> None:
    """Lève ForecastUnavailableError si aucun modèle consumption actif."""
    weather = _generer_weather_mock()
    with pytest.raises(ForecastUnavailableError):
        _predict_consumption(db_session, site_forecast, weather, 48)


def test_predict_consumption_reentrainement_si_artefact_introuvable(
    db_session: Session, site_forecast: Site, tmp_path: Path
) -> None:
    """Si l'artefact est introuvable sur disque, _predict_consumption réentraîne et prédit."""
    from forecaster.db.models import ModelVersion

    _inserer_mesures_recentes(db_session, site_forecast, N_LIGNES)

    mv_perdu = ModelVersion(
        site_id=site_forecast.site_id,
        type_modele="consumption",
        version="version_perdue",
        date_entrainement=datetime.now(tz=UTC),
        mape_validation=5.0,
        actif=True,
        chemin_artefact="/data/models/inexistant/consumption/version_perdue.joblib",
    )
    db_session.add(mv_perdu)
    db_session.flush()

    def _fake_run_training(session: Session, model_type: str, site_id: str) -> float:
        session.query(ModelVersion).filter_by(
            type_modele=model_type, actif=True, site_id=site_id
        ).update({"actif": False})
        _creer_modele_actif(session, model_type, tmp_path, site_id)
        return 5.0

    weather = _generer_weather_mock(horizon_h=24)
    with patch(
        "forecaster.pipeline.training.run_training", side_effect=_fake_run_training
    ):
        points = _predict_consumption(db_session, site_forecast, weather, 24)

    assert len(points) == 24 * 4


# ---------------------------------------------------------------------------
# Tests _predict_pv
# ---------------------------------------------------------------------------


def test_predict_pv_retourne_forecast_points(
    db_session: Session, site_forecast: Site, tmp_path: Path
) -> None:
    """_predict_pv retourne une liste de ForecastPoint."""
    _creer_modele_actif(db_session, "pv_production", tmp_path, site_forecast.site_id)

    weather = _generer_weather_mock(horizon_h=24)
    points = _predict_pv(db_session, site_forecast, weather, 24)

    assert len(points) == 24 * 4
    assert all(p.puissance_kw >= 0.0 for p in points)


def test_predict_pv_leve_erreur_sans_modele_actif(
    db_session: Session, site_forecast: Site
) -> None:
    """Lève ForecastUnavailableError si aucun modèle pv_production actif."""
    weather = _generer_weather_mock()
    with pytest.raises(ForecastUnavailableError):
        _predict_pv(db_session, site_forecast, weather, 48)


# ---------------------------------------------------------------------------
# Tests run_forecast
# ---------------------------------------------------------------------------


def test_run_forecast_ecrit_en_base(
    db_session: Session, site_forecast: Site, tmp_path: Path
) -> None:
    """run_forecast() insère des prévisions conso et PV en base."""
    _inserer_mesures_recentes(db_session, site_forecast, N_LIGNES)
    _creer_modele_actif(db_session, "consumption", tmp_path, site_forecast.site_id)
    _creer_modele_actif(db_session, "pv_production", tmp_path, site_forecast.site_id)

    weather = _generer_weather_mock(horizon_h=24)
    with patch(
        "forecaster.pipeline.forecast.fetch_forecast", return_value=weather
    ):
        run_forecast(db_session, site_forecast.site_id, horizon_h=24)

    conso_count = (
        db_session.query(ConsumptionForecast)
        .filter_by(site_id=site_forecast.site_id)
        .count()
    )
    pv_count = (
        db_session.query(PVProductionForecast)
        .filter_by(site_id=site_forecast.site_id)
        .count()
    )

    assert conso_count == 24 * 4
    assert pv_count == 24 * 4


def test_run_forecast_site_inexistant_leve_erreur(db_session: Session) -> None:
    """run_forecast() lève SiteNotFoundError pour un site inconnu."""
    with pytest.raises(SiteNotFoundError):
        run_forecast(db_session, "site-inexistant", horizon_h=48)


def test_run_forecast_all_sites_continue_sur_erreur(
    db_session: Session, site_forecast: Site
) -> None:
    """run_forecast_all_sites() continue malgré un site en erreur et retourne les échecs."""
    with patch(
        "forecaster.pipeline.forecast.fetch_forecast",
        side_effect=Exception("API indisponible"),
    ):
        failed = run_forecast_all_sites(db_session, horizon_h=24)

    assert failed == [site_forecast.site_id]
