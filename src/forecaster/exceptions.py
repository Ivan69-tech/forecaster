"""
Exceptions métier centralisées du service de prévision.

Toutes les exceptions spécifiques au domaine sont définies ici
pour éviter les imports circulaires et faciliter la lisibilité.
"""


class SiteNotFoundError(Exception):
    """Levée quand le site_id demandé n'existe pas en base."""


class InsufficientDataError(Exception):
    """Levée quand les données d'entraînement sont insuffisantes."""


class ForecastUnavailableError(Exception):
    """Levée quand aucun modèle actif n'est disponible pour générer une prévision."""


class ModelNotFoundError(Exception):
    """Levée quand l'artefact modèle est introuvable sur disque."""
