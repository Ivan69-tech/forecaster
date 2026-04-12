"""
Modèle de prévision de consommation — LightGBM (§3.2.1).

Cible   : puissance consommée par le site (kW)
Pas     : 15 min
Horizon : 48 h

Features principales (§3.2.1) :
  - Heure de la journée, jour de la semaine, mois
  - Indicateur jour férié / vacances scolaires
  - Température extérieure (valeur prévue et historique J-1, J-7)
  - Consommation historique J-1 (même heure), J-7 (même heure)
"""

import pandas as pd

from forecaster.predictors.base import BaseForecastModel, ForecastPoint, ModelNotLoadedError


class ConsumptionModel(BaseForecastModel):
    """Prévision de consommation par pas de 15 min, horizon 48h."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les features temporelles et de lag pour le modèle conso.

        Colonnes attendues dans `df` :
          - timestamp          : datetime UTC (index ou colonne)
          - temperature_c      : température prévue (°C)
          - conso_kw_lag_1d    : consommation J-1 même heure (kW)
          - conso_kw_lag_7d    : consommation J-7 même heure (kW)
          - temp_lag_1d        : température J-1 même heure (°C)
          - temp_lag_7d        : température J-7 même heure (°C)
          - is_holiday         : 0/1 — jour férié français
          - is_school_holiday  : 0/1 — vacances scolaires (optionnel)

        TODO:
          - Extraire hour, dayofweek, month depuis timestamp
          - Calculer sin/cos encodings pour les cycliques (heure, jour)
          - Retourner le DataFrame de features dans l'ordre attendu par le modèle
        """
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> list[ForecastPoint]:
        """
        Prédit la consommation pour chaque ligne de `df`.

        TODO:
          - Vérifier que _model est chargé (sinon lever ModelNotLoadedError)
          - Appeler _model.predict(features) via LightGBM
          - Construire les ForecastPoint avec le timestamp et l'horizon correspondants
        """
        if self._model is None:
            raise ModelNotLoadedError("ConsumptionModel : modèle non chargé, appeler load() d'abord.")
        raise NotImplementedError

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
        """
        Entraîne le modèle LightGBM sur les données de consommation.

        TODO:
          - Appeler build_features() sur df_train et df_val
          - Créer un lgb.Dataset avec la cible conso_kw
          - Entraîner avec lgb.train() et early_stopping_rounds
          - Calculer MAPE sur df_val et retourner le score
        """
        raise NotImplementedError
