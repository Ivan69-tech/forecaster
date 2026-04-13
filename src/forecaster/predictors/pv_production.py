"""
Modèle de prévision de production PV — LightGBM (§3.2.2).

Cible   : puissance produite par le parc PV (kW)
Pas     : 15 min
Horizon : 48 h

Features (§3.2.2) :
  - Irradiance prévue (W/m²), nébulosité prévue (%), température ambiante (°C)
  - Puissance crête installée (kW) — paramètre site
  - Heure de la journée, mois (encodages sin/cos — proxy de la position solaire)

Pas de lags de production PV : contrairement à la consommation (pilotée par les
habitudes humaines), la production PV est déterministe — elle dépend de la physique
instantanée. Si l'irradiance prévue est connue, la production d'hier n'apporte
pas d'information supplémentaire.

Contrat d'interface :
  L'appelant fournit les colonnes de contexte dans le DataFrame passé à
  build_features(). Cette fonction ajoute uniquement les features temporelles.
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
    "irradiance_wm2",
    "cloud_cover_pct",
    "temperature_c",
    "p_pv_peak_kw",
]

# Seuil d'irradiance pour les heures diurnes (W/m²)
# En dessous, la production est nulle — on exclut ces points du MAPE
SEUIL_DIURNE_WM2 = 50.0

# Paramètres LightGBM — identiques au ConsumptionModel
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


class PVProductionModel(BaseForecastModel):
    """Prévision de production PV par pas de 15 min, horizon 48h."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les features temporelles à partir du timestamp et retourne
        le DataFrame de features complet prêt pour LightGBM.

        Colonnes attendues dans `df` :
          - timestamp       : datetime timezone-aware (index ou colonne)
          - irradiance_wm2  : irradiance GHI prévue (W/m²)
          - cloud_cover_pct : nébulosité prévue (%)
          - temperature_c   : température ambiante prévue (°C)
          - p_pv_peak_kw    : puissance crête installée (kW)

        Colonnes ajoutées par cette fonction :
          - hour_sin, hour_cos    — heure cyclique (proxy élévation solaire intraday)
          - month_sin, month_cos  — mois cyclique (proxy déclinaison solaire saisonnière)

        Retourne le DataFrame de features (sans `production_pv_kw` ni `timestamp`).
        """
        df = df.copy()

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        else:
            ts = pd.to_datetime(df.index)

        heure = ts.dt.hour + ts.dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * math.pi * heure / 24)
        df["hour_cos"] = np.cos(2 * math.pi * heure / 24)

        mois = (ts.dt.month - 1).astype(float)
        df["month_sin"] = np.sin(2 * math.pi * mois / 12)
        df["month_cos"] = np.cos(2 * math.pi * mois / 12)

        colonnes_features = COLONNES_CONTEXTE + [
            "hour_sin", "hour_cos",
            "month_sin", "month_cos",
        ]
        return df[colonnes_features]

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
        """
        Entraîne le modèle LightGBM sur les données de production PV.

        Les deux DataFrames doivent contenir les colonnes de contexte listées
        dans COLONNES_CONTEXTE ainsi qu'une colonne `production_pv_kw` (la cible).

        La MAPE est calculée uniquement sur les heures diurnes (irradiance > 50 W/m²)
        pour éviter les divisions par zéro sur les valeurs nocturnes proches de 0.

        Args:
            df_train: Données d'entraînement (80 % les plus anciens).
            df_val:   Données de validation (20 % les plus récents).

        Returns:
            MAPE de validation sur les heures diurnes (%).
        """
        x_train = self.build_features(df_train)
        y_train = df_train["production_pv_kw"].values

        x_val = self.build_features(df_val)
        y_val = df_val["production_pv_kw"].values

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

        predictions_val = self._model.predict(x_val)

        # MAPE calculée uniquement sur les heures diurnes
        masque_diurne = df_val["irradiance_wm2"] > SEUIL_DIURNE_WM2
        if masque_diurne.sum() > 0:
            mape = _compute_mape(y_val[masque_diurne], predictions_val[masque_diurne])
        else:
            # Fallback si aucun point diurne dans la validation (ne devrait pas arriver)
            mape = _compute_mape(y_val, predictions_val)

        logger.info(
            "PVProductionModel.train | version=%s | lignes_train=%d | lignes_val=%d"
            " | points_diurnes_val=%d | MAPE=%.2f%%",
            self.version,
            len(df_train),
            len(df_val),
            masque_diurne.sum(),
            mape,
        )
        return mape

    def predict(self, df: pd.DataFrame) -> list[ForecastPoint]:
        """
        Prédit la production PV pour chaque ligne de `df`.

        Le DataFrame doit contenir les colonnes de contexte (COLONNES_CONTEXTE),
        une colonne `timestamp`, et une colonne `horizon_h` (entier, heures
        entre la génération et le pas prévu).

        La production est clippée à 0.0 — pas de valeur négative possible.

        Args:
            df: DataFrame de features brutes (avant build_features).

        Returns:
            Liste de ForecastPoint, un par pas de 15 min.

        Raises:
            ModelNotLoadedError: Si le modèle n'a pas été chargé via load() ou train().
        """
        if self._model is None:
            raise ModelNotLoadedError(
                "PVProductionModel : modèle non chargé, appeler load() ou train() d'abord."
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
                puissance_kw=max(0.0, float(pred)),  # pas de production négative
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
                "PVProductionModel : impossible de sauvegarder, aucun modèle en mémoire."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("PVProductionModel.save | version=%s | path=%s", self.version, path)

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
        logger.info("PVProductionModel.load | version=%s | path=%s", self.version, path)


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
