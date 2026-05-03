"""
Fetcher — Prix spots RTE (§3.1).

Publication quotidienne vers ~15h30 pour le lendemain (J+1).
API utilisée : RTE Wholesale Market Data v3
(https://data.rte-france.com/catalog/-/api/market-data/Wholesale-Market-Data/r/v3.0)

Authentification : OAuth2 client_credentials (POST /token/oauth/).
Endpoint prix    : GET /open_api/wholesale_market/v3/france_power_exchanges
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from forecaster.config import settings

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
_WHOLESALE_URL = (
    "https://digital.iservices.rte-france.com/open_api"
    "/wholesale_market/v3/france_power_exchanges"
)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_S = 1.0

# Cache token module-level — évite un appel OAuth2 à chaque requête
_cached_token: str | None = None
_token_expires_at: datetime | None = None  # UTC


@dataclass
class SpotPriceRow:
    """Prix spot pour un pas horaire."""

    timestamp: datetime  # début du pas (UTC)
    prix_eur_mwh: float
    source: str = "RTE"


class RTEDataUnavailable(Exception):
    """Levée quand les prix spots RTE ne sont pas disponibles pour la date demandée."""


def fetch_spot_prices(
    start_date: date, end_date: date | None = None
) -> list[SpotPriceRow]:
    """
    Récupère les prix spots RTE pour la plage [start_date, end_date] incluse.

    Si end_date est None, seule start_date est récupérée (24 entrées).
    Avec end_date = start_date + 1 jour, retourne 48 entrées (J et J+1) en un
    seul appel HTTP.

    Args:
        start_date: Première date de la plage (incluse).
        end_date:   Dernière date de la plage (incluse). Par défaut : start_date.

    Returns:
        Liste de SpotPriceRow triés par timestamp croissant.

    Raises:
        httpx.HTTPStatusError: Si l'API RTE retourne une erreur HTTP après les retries.
        RTEDataUnavailable: Si les prix ne sont pas encore publiés pour cette plage.
    """
    if end_date is None:
        end_date = start_date

    paris_tz = ZoneInfo("Europe/Paris")
    start = datetime(
        start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=paris_tz
    )
    end = datetime(
        end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=paris_tz
    )
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    data = _get_with_retry(_WHOLESALE_URL, params)
    rows = _parse_spot_prices(data, start_date, end_date)

    logger.info(
        "fetch_spot_prices | start=%s end=%s | %d entrées récupérées",
        start_date,
        end_date,
        len(rows),
    )
    return rows


def _parse_spot_prices(
    data: dict, start_date: date, end_date: date
) -> list[SpotPriceRow]:
    """
    Parse la réponse JSON de l'API Wholesale Market RTE.

    Structure attendue de `data` :
    {
        "france_power_exchanges": [
            {
                "values": [
                    {
                        "start_date": "2026-04-13T00:00:00+02:00",
                        "end_date":   "2026-04-13T01:00:00+02:00",
                        "price": 82.50
                    },
                    ...
                ]
            }
        ]
    }

    Les timestamps sont en heure Paris — on les convertit en UTC pour stockage.
    On filtre sur la date locale Paris pour gérer les transitions DST.
    """
    paris_tz = ZoneInfo("Europe/Paris")
    exchanges = data.get("france_power_exchanges", [])

    if not exchanges:
        raise RTEDataUnavailable(
            f"Réponse RTE vide — aucun échange disponible pour {start_date} / {end_date}"
        )

    rows = []
    for entry in exchanges[0].get("values", []):
        ts = datetime.fromisoformat(entry["start_date"])
        ts_utc = ts.astimezone(UTC)
        # Comparer en heure Paris pour gérer les transitions DST
        date_paris = ts.astimezone(paris_tz).date()
        if start_date <= date_paris <= end_date:
            rows.append(
                SpotPriceRow(
                    timestamp=ts_utc,
                    prix_eur_mwh=float(entry["price"]),
                )
            )

    rows.sort(key=lambda r: r.timestamp)

    if not rows:
        raise RTEDataUnavailable(
            f"Aucun prix RTE disponible pour {start_date} / {end_date}"
        )

    return rows


def _get_with_retry(url: str, params: dict) -> dict:
    """
    Effectue un GET HTTP authentifié avec retry exponentiel (3 tentatives).

    Sur 401, invalide le cache token avant le retry suivant pour forcer
    un renouvellement du token.

    Args:
        url:    URL de l'endpoint RTE.
        params: Paramètres de la requête.

    Returns:
        Données JSON de la réponse.

    Raises:
        RuntimeError: Si toutes les tentatives échouent.
    """
    global _cached_token
    last_exc: Exception | None = None

    for tentative in range(1, _MAX_RETRIES + 1):
        try:
            headers = _build_headers()
            response = httpx.get(url, headers=headers, params=params, timeout=30.0)

            if response.status_code == 401:
                # Token refusé par le serveur — invalider le cache et retry
                _cached_token = None
                last_exc = httpx.HTTPStatusError(
                    "401 Unauthorized", request=response.request, response=response
                )
                logger.warning(
                    "rte | tentative=%d/%d | 401 token refusé, rafraîchissement",
                    tentative,
                    _MAX_RETRIES,
                )
            else:
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            logger.warning(
                "rte | tentative=%d/%d | HTTP %d",
                tentative,
                _MAX_RETRIES,
                exc.response.status_code,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            logger.warning(
                "rte | tentative=%d/%d | erreur réseau : %s",
                tentative,
                _MAX_RETRIES,
                exc,
            )

        if tentative < _MAX_RETRIES:
            delai = _RETRY_BACKOFF_BASE_S * (2 ** (tentative - 1))
            time.sleep(delai)

    raise RuntimeError(
        f"rte | échec après {_MAX_RETRIES} tentatives"
    ) from last_exc


def _get_auth_token() -> str:
    """
    Obtient un token OAuth2 depuis l'API RTE (client_credentials).

    Le token est mis en cache jusqu'à 60 secondes avant son expiration.

    Returns:
        Bearer token valide.

    Raises:
        httpx.HTTPStatusError: Si l'authentification échoue (mauvais credentials, etc.).
    """
    global _cached_token, _token_expires_at

    # Retourner le token caché s'il est encore valide (avec marge de 60s)
    if _cached_token is not None and _token_expires_at is not None:
        if datetime.now(tz=UTC) < (_token_expires_at - timedelta(seconds=60)):
            return _cached_token

    credentials = f"{settings.rte_client_id}:{settings.rte_client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    response = httpx.post(
        _TOKEN_URL,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        content=b"grant_type=client_credentials",
        timeout=30.0,
    )
    response.raise_for_status()

    token_data = response.json()
    _cached_token = token_data["access_token"]
    expires_in = token_data.get("expires_in", 3600)
    _token_expires_at = datetime.now(tz=UTC) + timedelta(seconds=expires_in)

    logger.debug("rte | token obtenu, expire dans %d secondes", expires_in)
    return _cached_token


def _build_headers() -> dict[str, str]:
    token = _get_auth_token()
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
