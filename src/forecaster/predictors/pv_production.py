"""
Modèle de prévision de production PV — LightGBM (§3.2.2).

Cible   : puissance produite par le parc PV (kW)
Pas     : 15 min
Horizon : 48 h

Features principales (§3.2.2) :
  - Heure de la journée, mois (proxy déclinaison solaire)
  - Irradiance prévue (W/m²), nébulosité prévue (%)
  - Température ambiante (influence sur rendement des panneaux)
  - Production historique J-1 (même heure), J-7 (même heure)
"""

import pandas as pd

from forecaster.predictors.base import BaseForecastModel, ForecastPoint, ModelNotLoadedError


class PVProductionModel(BaseForecastModel):
    """Prévision de production PV par pas de 15 min, horizon 48h."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les features pour le modèle PV.

        Colonnes attendues dans `df` :
          - timestamp             : datetime UTC
          - irradiance_wm2        : irradiance GHI prévue (W/m²)
          - cloud_cover_pct       : nébulosité prévue (%)
          - temperature_c         : température ambiante prévue (°C)
          - pv_kw_lag_1d          : production PV J-1 même heure (kW)
          - pv_kw_lag_7d          : production PV J-7 même heure (kW)
          - p_pv_peak_kw          : puissance crête installée (kW) — paramètre site

        TODO:
          - Extraire hour, month depuis timestamp
          - Calculer sin/cos encodings pour l'heure et le mois
          - Normaliser l'irradiance par p_pv_peak_kw pour obtenir un ratio
          - Retourner le DataFrame de features dans l'ordre attendu
        """
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> list[ForecastPoint]:
        """
        Prédit la production PV pour chaque ligne de `df`.

        Note : la production ne peut être négative (pas d'injection inverse).
              Clipper les prédictions à 0 en sortie.

        TODO:
          - Vérifier que _model est chargé
          - Appeler _model.predict() via LightGBM
          - Clipper les valeurs négatives à 0.0
          - Construire les ForecastPoint
        """
        if self._model is None:
            raise ModelNotLoadedError("PVProductionModel : modèle non chargé, appeler load() d'abord.")
        raise NotImplementedError

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
        """
        Entraîne le modèle LightGBM sur les données de production PV.

        TODO:
          - Appeler build_features() sur df_train et df_val
          - Créer un lgb.Dataset avec la cible production_pv_kw
          - Attention : filtrer les pas nocturnes (irradiance = 0) pour l'évaluation MAPE
          - Entraîner avec lgb.train() et early_stopping_rounds
          - Calculer MAPE sur df_val (pas diurnes uniquement) et retourner
        """
        raise NotImplementedError
