"""
Fetcher — Prix spots RTE (§3.1).

Publication quotidienne vers ~15h30 pour le lendemain (J+1).
API utilisée : RTE eCO2mix / transparency (https://data.rte-france.com).

TODO: implémenter l'authentification OAuth2 RTE et le parsing de la réponse.
"""

from dataclasses import dataclass
from datetime import date, datetime

import httpx

from forecaster.config import settings


@dataclass
class SpotPriceRow:
    """Prix spot pour un pas horaire."""

    timestamp: datetime  # début du pas (UTC)
    prix_eur_mwh: float
    source: str = "RTE"


def fetch_spot_prices(target_date: date) -> list[SpotPriceRow]:
    """
    Récupère les prix spots RTE pour la journée `target_date`.

    Retourne 24 entrées (pas horaire, minuit → 23h UTC).

    Args:
        target_date: La date pour laquelle récupérer les prix (J+1 après publication RTE).

    Returns:
        Liste de SpotPriceRow triés par timestamp croissant.

    Raises:
        httpx.HTTPStatusError: Si l'API RTE retourne une erreur HTTP.
        RTEDataUnavailable: Si les prix ne sont pas encore publiés pour cette date.

    TODO:
        - Implémenter l'authentification OAuth2 client_credentials RTE
        - Parser la réponse JSON/XML de l'API transparency RTE
        - Gérer la pagination si nécessaire
        - Mapper les prix EPEX Spot France (€/MWh) vers SpotPriceRow
    """
    raise NotImplementedError(
        "fetch_spot_prices() n'est pas encore implémentée. "
        "Voir TODO dans rte.py pour les étapes d'intégration."
    )


class RTEDataUnavailable(Exception):
    """Levée quand les prix spots RTE ne sont pas disponibles pour la date demandée."""


def _get_auth_token() -> str:
    """
    Obtient un token OAuth2 depuis l'API RTE.

    TODO:
        - POST sur https://digital.iservices.rte-france.com/token/oauth/
        - Avec les credentials encodés en base64 (client_id:client_secret)
        - Cacher le token jusqu'à expiration
    """
    raise NotImplementedError


def _build_headers() -> dict[str, str]:
    token = _get_auth_token()
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
