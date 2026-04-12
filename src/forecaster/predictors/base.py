"""
Classe de base abstraite pour les modèles de prévision (§3.2).

Tous les modèles LightGBM héritent de BaseForecastModel et
implémentent les méthodes predict(), train(), load(), save().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ForecastPoint:
    """Une prévision pour un pas de 15 minutes."""

    timestamp: datetime  # début du pas (UTC)
    puissance_kw: float
    horizon_h: int       # distance depuis le moment de génération (en heures)


class BaseForecastModel(ABC):
    """
    Interface commune à tous les modèles de prévision LightGBM.

    Chaque sous-classe correspond à un type de modèle (consommation, production PV)
    et encapsule le pipeline features → prédiction.
    """

    def __init__(self, version: str) -> None:
        self.version = version
        self._model = None  # instance LightGBM chargée

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construit le DataFrame de features à partir des données brutes.

        Args:
            df: DataFrame avec au minimum une colonne `timestamp` (datetime UTC)
                et les colonnes de contexte nécessaires au modèle.

        Returns:
            DataFrame de features prêt pour predict() ou fit().
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list[ForecastPoint]:
        """
        Génère des prévisions sur l'horizon donné.

        Args:
            df: DataFrame de features (sortie de build_features()).

        Returns:
            Liste de ForecastPoint, un par pas de 15 min.

        Raises:
            ModelNotLoadedError: Si le modèle n'a pas été chargé via load().
        """

    @abstractmethod
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
        """
        Entraîne le modèle sur les données historiques.

        Args:
            df_train: DataFrame d'entraînement (features + cible).
            df_val:   DataFrame de validation pour early stopping et calcul MAPE.

        Returns:
            MAPE de validation (float, en %).

        TODO:
            - Construire les features via build_features()
            - Entraîner le modèle LightGBM avec early stopping
            - Calculer et retourner la MAPE sur df_val
        """

    def save(self, path: Path) -> None:
        """
        Sérialise le modèle sur disque (joblib).

        Args:
            path: Chemin complet du fichier de sortie (.pkl ou .joblib).

        Raises:
            ModelNotLoadedError: Si aucun modèle n'est en mémoire.

        TODO: implémenter la sérialisation joblib.
        """
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """
        Charge un modèle depuis le disque.

        Args:
            path: Chemin vers l'artefact sérialisé.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.

        TODO: implémenter le chargement joblib.
        """
        raise NotImplementedError


class ModelNotLoadedError(Exception):
    """Levée quand predict() est appelé sans avoir chargé le modèle."""
