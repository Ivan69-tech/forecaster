"""
Tests pour ConsumptionModel — entraînement, prédiction, sérialisation.

Les données viennent de tests/fixtures/load_history_2025.csv (1 an synthétique).
Aucune connexion DB ni appel HTTP — uniquement des fichiers locaux.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pytest

from forecaster.predictors.base import ModelNotLoadedError
from forecaster.predictors.consumption import COLONNES_CONTEXTE, ConsumptionModel

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

CSV_PATH = Path(__file__).parent.parent / "fixtures" / "load_history_2025.csv"


def _charger_et_preparer_csv() -> pd.DataFrame:
    """
    Charge le CSV synthétique et calcule les colonnes de contexte attendues
    par ConsumptionModel.build_features().

    Retourne un DataFrame prêt à l'emploi avec les colonnes :
      timestamp, conso_kw, conso_kw_lag_1d, conso_kw_lag_7d,
      temperature_c, temp_lag_1d, temp_lag_7d, is_holiday, is_school_holiday
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lags consommation
    df["conso_kw_lag_1d"] = df["conso_kw"].shift(96)
    df["conso_kw_lag_7d"] = df["conso_kw"].shift(672)

    # Lags température
    df["temp_lag_1d"] = df["temperature_c"].shift(96)
    df["temp_lag_7d"] = df["temperature_c"].shift(672)

    # Indicateurs calendaires
    annees = df["timestamp"].dt.year.unique().tolist()
    jours_feries = set(holidays.France(years=annees).keys())
    df["is_holiday"] = df["timestamp"].dt.date.isin(jours_feries).astype(int)
    df["is_school_holiday"] = 0

    # Suppression des lignes sans lags
    df = df.dropna(subset=["conso_kw_lag_1d", "conso_kw_lag_7d", "temp_lag_1d", "temp_lag_7d"])
    df = df.reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def df_complet() -> pd.DataFrame:
    """DataFrame d'un an, prêt pour build_features()."""
    return _charger_et_preparer_csv()


@pytest.fixture(scope="module")
def split_train_val(df_complet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80/20 chronologique."""
    split_idx = int(len(df_complet) * 0.8)
    return df_complet.iloc[:split_idx].copy(), df_complet.iloc[split_idx:].copy()


@pytest.fixture(scope="module")
def modele_entraine(split_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> ConsumptionModel:
    """ConsumptionModel entraîné sur les données synthétiques (1 seul entraînement par module)."""
    df_train, df_val = split_train_val
    model = ConsumptionModel(version="test_v1")
    model.train(df_train, df_val)
    return model


# ---------------------------------------------------------------------------
# Tests build_features
# ---------------------------------------------------------------------------


def test_build_features_retourne_colonnes_temporelles(df_complet: pd.DataFrame) -> None:
    """build_features() ajoute les 7 features temporelles attendues."""
    model = ConsumptionModel(version="test")
    features = model.build_features(df_complet.head(100))

    colonnes_temporelles = [
        "hour_sin", "hour_cos",
        "dayofweek_sin", "dayofweek_cos",
        "month_sin", "month_cos",
        "is_weekend",
    ]
    for col in colonnes_temporelles:
        assert col in features.columns, f"Colonne temporelle manquante : {col}"


def test_build_features_exclut_conso_kw_et_timestamp(df_complet: pd.DataFrame) -> None:
    """build_features() ne doit pas inclure conso_kw ni timestamp dans le résultat."""
    model = ConsumptionModel(version="test")
    features = model.build_features(df_complet.head(100))

    assert "conso_kw" not in features.columns
    assert "timestamp" not in features.columns


def test_build_features_conserve_colonnes_contexte(df_complet: pd.DataFrame) -> None:
    """build_features() conserve toutes les colonnes de contexte (lags, temp, calendaire)."""
    model = ConsumptionModel(version="test")
    features = model.build_features(df_complet.head(100))

    for col in COLONNES_CONTEXTE:
        assert col in features.columns, f"Colonne contexte manquante : {col}"


def test_build_features_valeurs_sin_cos_bornees(df_complet: pd.DataFrame) -> None:
    """Les encodages sin/cos sont dans [-1, 1]."""
    model = ConsumptionModel(version="test")
    features = model.build_features(df_complet.head(1000))

    for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]:
        assert features[col].between(-1.0, 1.0).all(), f"{col} hors de [-1, 1]"


# ---------------------------------------------------------------------------
# Tests train
# ---------------------------------------------------------------------------


def test_train_retourne_mape_inferieure_au_seuil(
    split_train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """train() retourne une MAPE < 25 % sur les données synthétiques."""
    df_train, df_val = split_train_val
    model = ConsumptionModel(version="test_mape")
    mape = model.train(df_train, df_val)

    assert isinstance(mape, float)
    assert mape < 25.0, f"MAPE trop élevée : {mape:.2f} % (seuil : 25 %)"


def test_train_charge_modele_en_memoire(
    split_train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Après train(), _model est non None."""
    df_train, df_val = split_train_val
    model = ConsumptionModel(version="test_mem")
    assert model._model is None
    model.train(df_train, df_val)
    assert model._model is not None


# ---------------------------------------------------------------------------
# Tests predict
# ---------------------------------------------------------------------------


def test_predict_retourne_liste_forecast_points(
    modele_entraine: ConsumptionModel, df_complet: pd.DataFrame
) -> None:
    """predict() retourne autant de ForecastPoint que de lignes dans le DataFrame."""
    df_futur = df_complet.tail(96).copy()  # 24h de données
    df_futur["horizon_h"] = list(range(len(df_futur)))

    resultats = modele_entraine.predict(df_futur)

    assert len(resultats) == len(df_futur)
    # Vérification de la structure de chaque point
    from forecaster.predictors.base import ForecastPoint
    for point in resultats:
        assert isinstance(point, ForecastPoint)
        assert point.puissance_kw >= 0


def test_predict_leve_erreur_sans_modele_charge() -> None:
    """predict() lève ModelNotLoadedError si le modèle n'a pas été chargé."""
    model = ConsumptionModel(version="vide")
    df_vide = pd.DataFrame({"timestamp": [], "horizon_h": []})

    with pytest.raises(ModelNotLoadedError):
        model.predict(df_vide)


# ---------------------------------------------------------------------------
# Tests save / load
# ---------------------------------------------------------------------------


def test_save_et_load_roundtrip(
    modele_entraine: ConsumptionModel,
    df_complet: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """save() puis load() produit un modèle qui donne les mêmes prédictions."""
    chemin = tmp_path / "consumption" / "test_v1.joblib"
    modele_entraine.save(chemin)

    assert chemin.exists()

    model_charge = ConsumptionModel(version="test_v1_rechargee")
    model_charge.load(chemin)

    df_futur = df_complet.tail(48).copy()
    df_futur["horizon_h"] = list(range(len(df_futur)))

    preds_original = modele_entraine.predict(df_futur)
    preds_charge = model_charge.predict(df_futur)

    valeurs_orig = np.array([p.puissance_kw for p in preds_original])
    valeurs_charge = np.array([p.puissance_kw for p in preds_charge])
    np.testing.assert_array_almost_equal(valeurs_orig, valeurs_charge)


def test_save_leve_erreur_sans_modele() -> None:
    """save() lève ModelNotLoadedError si aucun modèle en mémoire."""
    model = ConsumptionModel(version="vide")
    with pytest.raises(ModelNotLoadedError):
        model.save(Path("/tmp/inexistant.joblib"))


def test_load_leve_erreur_si_fichier_absent(tmp_path: Path) -> None:
    """load() lève FileNotFoundError si le fichier n'existe pas."""
    model = ConsumptionModel(version="vide")
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "inexistant.joblib")
