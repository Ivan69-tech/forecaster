"""
Tests pour PVProductionModel — entraînement, prédiction, sérialisation.

Les données de test sont générées synthétiquement à partir d'un modèle solaire
simplifié — aucune connexion DB ni appel HTTP.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster.predictors.base import ForecastPoint, ModelNotLoadedError
from forecaster.predictors.pv_production import COLONNES_CONTEXTE, PVProductionModel


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _irradiance_simple(ts: datetime, lat_deg: float = 43.6) -> float:
    """Irradiance clear-sky simplifiée pour les données de test."""
    day = ts.timetuple().tm_yday
    decl = 23.45 * math.sin(math.radians(360 / 365 * (day - 81)))
    ha = 15.0 * (ts.hour + ts.minute / 60.0 - 12.0)
    lat = math.radians(lat_deg)
    sin_elev = (
        math.sin(lat) * math.sin(math.radians(decl))
        + math.cos(lat) * math.cos(math.radians(decl)) * math.cos(math.radians(ha))
    )
    return max(0.0, 1000.0 * max(0.0, sin_elev))


def _construire_df_pv(n_jours: int = 30, p_pv_peak_kw: float = 300.0) -> pd.DataFrame:
    """
    Construit un DataFrame synthétique de production PV sur n_jours.
    Pas de 15 min, avec irradiance, cloud_cover, temperature et production.
    """
    debut = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    timestamps = [debut + timedelta(minutes=15 * i) for i in range(n_jours * 96)]

    rng = np.random.default_rng(seed=123)
    lignes = []
    for ts in timestamps:
        irr = _irradiance_simple(ts)
        cloud = float(rng.uniform(0, 60))
        temp = 10.0 + 8.0 * math.sin(math.radians(360 / 24 * (ts.hour - 6)))
        cloud_factor = 1.0 - 0.7 * (cloud / 100.0)
        temp_factor = 1.0 - 0.004 * max(0.0, temp - 25.0)
        production = max(0.0, p_pv_peak_kw * (irr / 1000.0) * cloud_factor * temp_factor)

        lignes.append({
            "timestamp": ts,
            "irradiance_wm2": irr,
            "cloud_cover_pct": cloud,
            "temperature_c": temp,
            "p_pv_peak_kw": p_pv_peak_kw,
            "production_pv_kw": production,
        })

    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def df_complet() -> pd.DataFrame:
    """DataFrame de 30 jours de données PV synthétiques."""
    return _construire_df_pv(n_jours=30)


@pytest.fixture(scope="module")
def split_train_val(df_complet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80/20 chronologique."""
    split_idx = int(len(df_complet) * 0.8)
    return df_complet.iloc[:split_idx].copy(), df_complet.iloc[split_idx:].copy()


@pytest.fixture(scope="module")
def modele_entraine(split_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> PVProductionModel:
    """PVProductionModel entraîné sur les données synthétiques (1 seul entraînement)."""
    df_train, df_val = split_train_val
    model = PVProductionModel(version="test_v1")
    model.train(df_train, df_val)
    return model


# ---------------------------------------------------------------------------
# Tests build_features
# ---------------------------------------------------------------------------


def test_build_features_ajoute_colonnes_temporelles(df_complet: pd.DataFrame) -> None:
    """build_features() ajoute les 4 features temporelles cycliques."""
    model = PVProductionModel(version="test")
    features = model.build_features(df_complet.head(100))

    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert col in features.columns, f"Colonne temporelle manquante : {col}"


def test_build_features_exclut_production_et_timestamp(df_complet: pd.DataFrame) -> None:
    """build_features() n'inclut pas production_pv_kw ni timestamp dans le résultat."""
    model = PVProductionModel(version="test")
    features = model.build_features(df_complet.head(100))

    assert "production_pv_kw" not in features.columns
    assert "timestamp" not in features.columns


def test_build_features_conserve_colonnes_contexte(df_complet: pd.DataFrame) -> None:
    """build_features() conserve toutes les colonnes de contexte PV."""
    model = PVProductionModel(version="test")
    features = model.build_features(df_complet.head(100))

    for col in COLONNES_CONTEXTE:
        assert col in features.columns, f"Colonne contexte manquante : {col}"


def test_build_features_pas_de_dayofweek(df_complet: pd.DataFrame) -> None:
    """build_features() n'ajoute pas de dayofweek (la PV ne dépend pas du comportement humain)."""
    model = PVProductionModel(version="test")
    features = model.build_features(df_complet.head(100))

    assert "dayofweek_sin" not in features.columns
    assert "dayofweek_cos" not in features.columns
    assert "is_weekend" not in features.columns


def test_build_features_valeurs_sin_cos_bornees(df_complet: pd.DataFrame) -> None:
    """Les encodages sin/cos sont dans [-1, 1]."""
    model = PVProductionModel(version="test")
    features = model.build_features(df_complet.head(500))

    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert features[col].between(-1.0, 1.0).all(), f"{col} hors de [-1, 1]"


# ---------------------------------------------------------------------------
# Tests train
# ---------------------------------------------------------------------------


def test_train_retourne_mape_float(
    split_train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """train() retourne un float (MAPE en %)."""
    df_train, df_val = split_train_val
    model = PVProductionModel(version="test_mape")
    mape = model.train(df_train, df_val)

    assert isinstance(mape, float)
    assert mape >= 0.0


def test_train_mape_calculee_sur_heures_diurnes(
    split_train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """
    train() calcule la MAPE uniquement sur les heures diurnes (irradiance > 50 W/m²).
    On vérifie que le modèle entraîné sur données synthétiques a une MAPE < 15 %
    sur les heures diurnes (données cohérentes → modèle quasi-parfait).
    """
    df_train, df_val = split_train_val
    model = PVProductionModel(version="test_diurne")
    mape = model.train(df_train, df_val)

    assert mape < 15.0, f"MAPE sur heures diurnes trop élevée : {mape:.2f}% (seuil : 15%)"


def test_train_charge_modele_en_memoire(
    split_train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Après train(), _model est non None."""
    df_train, df_val = split_train_val
    model = PVProductionModel(version="test_mem")
    assert model._model is None
    model.train(df_train, df_val)
    assert model._model is not None


# ---------------------------------------------------------------------------
# Tests predict
# ---------------------------------------------------------------------------


def test_predict_retourne_liste_forecast_points(
    modele_entraine: PVProductionModel, df_complet: pd.DataFrame
) -> None:
    """predict() retourne autant de ForecastPoint que de lignes dans le DataFrame."""
    df_futur = df_complet.tail(96).copy()  # 24h de données
    df_futur["horizon_h"] = list(range(len(df_futur)))

    resultats = modele_entraine.predict(df_futur)

    assert len(resultats) == len(df_futur)
    assert all(isinstance(p, ForecastPoint) for p in resultats)


def test_predict_clippe_valeurs_negatives_a_zero(
    modele_entraine: PVProductionModel, df_complet: pd.DataFrame
) -> None:
    """predict() ne retourne jamais de puissance négative."""
    df_futur = df_complet.tail(192).copy()
    df_futur["horizon_h"] = list(range(len(df_futur)))

    resultats = modele_entraine.predict(df_futur)

    for point in resultats:
        assert point.puissance_kw >= 0.0, (
            f"Production négative détectée : {point.puissance_kw} kW à {point.timestamp}"
        )


def test_predict_production_nulle_la_nuit(modele_entraine: PVProductionModel) -> None:
    """predict() retourne 0 kW pour les heures nocturnes (irradiance = 0)."""
    # Construire un DataFrame nocturne (minuit, irradiance = 0)
    ts_nuit = datetime(2026, 6, 21, 0, 0, tzinfo=UTC)  # solstice d'été, minuit UTC
    df_nuit = pd.DataFrame([
        {
            "timestamp": ts_nuit + timedelta(minutes=15 * i),
            "horizon_h": i,
            "irradiance_wm2": 0.0,   # nuit
            "cloud_cover_pct": 0.0,
            "temperature_c": 15.0,
            "p_pv_peak_kw": 300.0,
        }
        for i in range(8)  # 2h de nuit
    ])

    resultats = modele_entraine.predict(df_nuit)

    for point in resultats:
        assert point.puissance_kw == 0.0 or point.puissance_kw < 1.0, (
            f"Production non nulle la nuit : {point.puissance_kw} kW"
        )


def test_predict_leve_erreur_sans_modele_charge() -> None:
    """predict() lève ModelNotLoadedError si le modèle n'a pas été chargé."""
    model = PVProductionModel(version="vide")
    df_vide = pd.DataFrame({"timestamp": [], "horizon_h": []})

    with pytest.raises(ModelNotLoadedError):
        model.predict(df_vide)


# ---------------------------------------------------------------------------
# Tests save / load
# ---------------------------------------------------------------------------


def test_save_et_load_roundtrip(
    modele_entraine: PVProductionModel,
    df_complet: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """save() puis load() produit un modèle qui donne les mêmes prédictions."""
    chemin = tmp_path / "pv_production" / "test_v1.joblib"
    modele_entraine.save(chemin)

    assert chemin.exists()

    model_charge = PVProductionModel(version="test_v1_rechargee")
    model_charge.load(chemin)

    df_futur = df_complet.tail(48).copy()
    df_futur["horizon_h"] = list(range(len(df_futur)))

    preds_orig = modele_entraine.predict(df_futur)
    preds_charge = model_charge.predict(df_futur)

    valeurs_orig = np.array([p.puissance_kw for p in preds_orig])
    valeurs_charge = np.array([p.puissance_kw for p in preds_charge])
    np.testing.assert_array_almost_equal(valeurs_orig, valeurs_charge)


def test_save_leve_erreur_sans_modele() -> None:
    """save() lève ModelNotLoadedError si aucun modèle en mémoire."""
    model = PVProductionModel(version="vide")
    with pytest.raises(ModelNotLoadedError):
        model.save(Path("/tmp/inexistant.joblib"))


def test_load_leve_erreur_si_fichier_absent(tmp_path: Path) -> None:
    """load() lève FileNotFoundError si le fichier n'existe pas."""
    model = PVProductionModel(version="vide")
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "inexistant.joblib")
