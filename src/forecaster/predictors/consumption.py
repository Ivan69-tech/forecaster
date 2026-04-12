"""
Modèle de prévision de consommation — LightGBM (§3.2.1).

Cible   : puissance consommée par le site (kW)
Pas     : 15 min
Horizon : 48 h

Features principales (§3.2.1) :
  - Heure de la journée, jour de la semaine, mois (encodages sin/cos)
  - Indicateur week-end, jour férié / vacances scolaires
  - Température extérieure et ses lags J-1, J-7
  - Consommation historique J-1 (même heure), J-7 (même heure)

Contrat d'interface :
  L'appelant (pipeline ou test) est responsable de fournir les colonnes de
  contexte (lags, température, indicateurs calendaires) dans le DataFrame
  passé à build_features(). Cette fonction se charge uniquement d'ajouter
  les features temporelles dérivées du timestamp.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from forecaster.predictors.base import BaseForecastModel, ForecastPoint, ModelNotLoadedError

logger = logging.getLogger(__name__)

# Colonnes contextuelles attendues dans le DataFrame d'entrée de build_features()
COLONNES_CONTEXTE = [
    "conso_kw_lag_1d",
    "conso_kw_lag_7d",
    "temperature_c",
    "temp_lag_1d",
    "temp_lag_7d",
    "is_holiday",
    "is_school_holiday",
]

# Paramètres LightGBM par défaut
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
LGB_NUM_BOOST_ROUND = 500
LGB_EARLY_STOPPING_ROUNDS = 50


class ConsumptionModel(BaseForecastModel):
    """Prévision de consommation par pas de 15 min, horizon 48h."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les features temporelles à partir du timestamp et retourne
        le DataFrame de features complet prêt pour LightGBM.

        Colonnes attendues dans `df` :
          - timestamp          : datetime timezone-aware (index ou colonne)
          - conso_kw_lag_1d    : consommation J-1 même heure (kW)
          - conso_kw_lag_7d    : consommation J-7 même heure (kW)
          - temperature_c      : température prévue (°C)
          - temp_lag_1d        : température J-1 même heure (°C)
          - temp_lag_7d        : température J-7 même heure (°C)
          - is_holiday         : 0/1 — jour férié français
          - is_school_holiday  : 0/1 — vacances scolaires

        Colonnes ajoutées par cette fonction :
          - hour_sin, hour_cos            — heure cyclique
          - dayofweek_sin, dayofweek_cos  — jour de semaine cyclique
          - month_sin, month_cos          — mois cyclique
          - is_weekend                    — 0/1

        Retourne le DataFrame de features (sans les colonnes `conso_kw` et
        `timestamp` qui ne font pas partie de la matrice X).
        """
        df = df.copy()

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        else:
            ts = pd.to_datetime(df.index)

        heure = ts.dt.hour + ts.dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * math.pi * heure / 24)
        df["hour_cos"] = np.cos(2 * math.pi * heure / 24)

        jour = ts.dt.dayofweek.astype(float)
        df["dayofweek_sin"] = np.sin(2 * math.pi * jour / 7)
        df["dayofweek_cos"] = np.cos(2 * math.pi * jour / 7)

        mois = (ts.dt.month - 1).astype(float)
        df["month_sin"] = np.sin(2 * math.pi * mois / 12)
        df["month_cos"] = np.cos(2 * math.pi * mois / 12)

        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        colonnes_features = COLONNES_CONTEXTE + [
            "hour_sin", "hour_cos",
            "dayofweek_sin", "dayofweek_cos",
            "month_sin", "month_cos",
            "is_weekend",
        ]
        return df[colonnes_features]

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
        """
        Entraîne le modèle LightGBM sur les données de consommation.

        Les deux DataFrames doivent contenir les colonnes de contexte listées
        dans COLONNES_CONTEXTE ainsi qu'une colonne `conso_kw` (la cible).

        Args:
            df_train: Données d'entraînement (80 % les plus anciens).
            df_val:   Données de validation (20 % les plus récents).

        Returns:
            MAPE de validation (%).
        """
        x_train = self.build_features(df_train)
        y_train = df_train["conso_kw"].values

        x_val = self.build_features(df_val)
        y_val = df_val["conso_kw"].values

        lgb_train = lgb.Dataset(x_train, label=y_train, feature_name=list(x_train.columns))
        lgb_val = lgb.Dataset(x_val, label=y_val, reference=lgb_train)

        callbacks = [lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False)]

        self._model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            num_boost_round=LGB_NUM_BOOST_ROUND,
            valid_sets=[lgb_val],
            callbacks=callbacks,
        )

        predictions = self._model.predict(x_val)
        mape = _compute_mape(y_val, predictions)

        logger.info(
            "ConsumptionModel.train | version=%s | lignes_train=%d | lignes_val=%d | MAPE=%.2f%%",
            self.version,
            len(df_train),
            len(df_val),
            mape,
        )
        return mape

    def predict(self, df: pd.DataFrame) -> list[ForecastPoint]:
        """
        Prédit la consommation pour chaque ligne de `df`.

        Le DataFrame doit contenir les colonnes de contexte (COLONNES_CONTEXTE),
        une colonne `timestamp`, et une colonne `horizon_h` (entier, heures
        entre la génération et le pas prévu).

        Args:
            df: DataFrame de features brutes (avant build_features).

        Returns:
            Liste de ForecastPoint, un par pas de 15 min.

        Raises:
            ModelNotLoadedError: Si le modèle n'a pas été chargé via load()
                                 ou entraîné via train().
        """
        if self._model is None:
            raise ModelNotLoadedError(
                "ConsumptionModel : modèle non chargé, appeler load() ou train() d'abord."
            )

        x = self.build_features(df)
        predictions = self._model.predict(x)

        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"])
        else:
            timestamps = pd.to_datetime(df.index)
        horizons = df["horizon_h"].values if "horizon_h" in df.columns else [0] * len(df)

        return [
            ForecastPoint(
                timestamp=ts.to_pydatetime(),
                puissance_kw=float(pred),
                horizon_h=int(h),
            )
            for ts, pred, h in zip(timestamps, predictions, horizons)
        ]

    def save(self, path: Path) -> None:
        """
        Sérialise le modèle LightGBM sur disque via joblib.

        Args:
            path: Chemin complet du fichier de sortie (.joblib).

        Raises:
            ModelNotLoadedError: Si aucun modèle n'est en mémoire.
        """
        if self._model is None:
            raise ModelNotLoadedError(
                "ConsumptionModel : impossible de sauvegarder, aucun modèle en mémoire."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("ConsumptionModel.save | version=%s | path=%s", self.version, path)

    def load(self, path: Path) -> None:
        """
        Charge un modèle LightGBM depuis le disque.

        Args:
            path: Chemin vers l'artefact sérialisé (.joblib).

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artefact modèle introuvable : {path}")
        self._model = joblib.load(path)
        logger.info("ConsumptionModel.load | version=%s | path=%s", self.version, path)


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule la Mean Absolute Percentage Error (MAPE) en %.

    Utilise max(|y_true|, 1e-6) au dénominateur pour éviter la division par zéro.
    """
    denominateur = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_pred - y_true) / denominateur) * 100)
