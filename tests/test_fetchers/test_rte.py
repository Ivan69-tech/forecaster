"""
Tests du fetcher RTE.

Les appels HTTP sont mockés via unittest.mock.patch — aucun appel réseau réel.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

import forecaster.fetchers.rte as rte_module
from forecaster.fetchers.rte import (
    RTEDataUnavailable,
    SpotPriceRow,
    _get_auth_token,
    _parse_spot_prices,
    fetch_spot_prices,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_token_cache():
    """Réinitialise le cache token avant chaque test pour éviter les interactions."""
    rte_module._cached_token = None
    rte_module._token_expires_at = None
    yield
    rte_module._cached_token = None
    rte_module._token_expires_at = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_token_response(expires_in: int = 3600) -> MagicMock:
    """Mock d'une réponse OAuth2 valide."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"access_token": "test-token-abc", "expires_in": expires_in}
    mock.raise_for_status.return_value = None
    return mock


def _mock_wholesale_response(target_date: date) -> MagicMock:
    """Mock d'une réponse wholesale RTE avec 24 entrées horaires."""
    from zoneinfo import ZoneInfo

    paris_tz = ZoneInfo("Europe/Paris")
    values = []
    for h in range(24):
        start_dt = datetime(
            target_date.year, target_date.month, target_date.day, h, 0, 0, tzinfo=paris_tz
        )
        values.append({
            "start_date": start_dt.isoformat(),
            "price": 80.0 + h * 0.5,
        })

    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"france_power_exchanges": [{"values": values}]}
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Tests — SpotPriceRow
# ---------------------------------------------------------------------------


def test_spot_price_row_dataclass():
    """Vérifie que SpotPriceRow est instanciable avec les bons champs."""
    row = SpotPriceRow(
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
        prix_eur_mwh=85.5,
        source="RTE",
    )
    assert row.prix_eur_mwh == 85.5
    assert row.source == "RTE"


# ---------------------------------------------------------------------------
# Tests — fetch_spot_prices
# ---------------------------------------------------------------------------


def test_fetch_spot_prices_returns_24_entries():
    """Vérifie que 24 entrées horaires sont retournées pour une journée."""
    target_date = date(2026, 4, 13)
    token_resp = _mock_token_response()
    wholesale_resp = _mock_wholesale_response(target_date)

    with patch("httpx.post", return_value=token_resp), \
         patch("httpx.get", return_value=wholesale_resp):
        result = fetch_spot_prices(target_date)

    assert len(result) == 24
    assert all(isinstance(r, SpotPriceRow) for r in result)
    assert all(r.source == "RTE" for r in result)
    # Timestamps triés par ordre croissant
    assert result == sorted(result, key=lambda r: r.timestamp)
    # Tous les timestamps sont en UTC
    assert all(r.timestamp.tzinfo == UTC for r in result)


def test_fetch_spot_prices_raises_when_no_values():
    """Vérifie que RTEDataUnavailable est levée si la réponse ne contient pas de valeurs."""
    target_date = date(2026, 4, 13)
    token_resp = _mock_token_response()

    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"france_power_exchanges": [{"values": []}]}
    empty_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=token_resp), \
         patch("httpx.get", return_value=empty_resp):
        with pytest.raises(RTEDataUnavailable):
            fetch_spot_prices(target_date)


def test_fetch_spot_prices_raises_when_empty_response():
    """Vérifie que RTEDataUnavailable est levée si france_power_exchanges est vide."""
    target_date = date(2026, 4, 13)
    token_resp = _mock_token_response()

    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.json.return_value = {"france_power_exchanges": []}
    empty_resp.raise_for_status.return_value = None

    with patch("httpx.post", return_value=token_resp), \
         patch("httpx.get", return_value=empty_resp):
        with pytest.raises(RTEDataUnavailable):
            fetch_spot_prices(target_date)


# ---------------------------------------------------------------------------
# Tests — cache token
# ---------------------------------------------------------------------------


def test_get_auth_token_reuses_cached_token():
    """Vérifie que le token est mis en cache et qu'un seul appel HTTP est effectué."""
    token_resp = _mock_token_response(expires_in=3600)

    with patch("httpx.post", return_value=token_resp) as mock_post:
        token1 = _get_auth_token()
        token2 = _get_auth_token()

    assert token1 == token2 == "test-token-abc"
    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Tests — _parse_spot_prices
# ---------------------------------------------------------------------------


def test_parse_spot_prices_converts_to_utc():
    """Vérifie que les timestamps Paris sont correctement convertis en UTC."""
    from zoneinfo import ZoneInfo

    paris_tz = ZoneInfo("Europe/Paris")
    target_date = date(2026, 4, 13)
    # Paris UTC+2 en été — minuit Paris = 22h UTC la veille
    data = {
        "france_power_exchanges": [
            {
                "values": [
                    {
                        "start_date": datetime(2026, 4, 13, 0, 0, tzinfo=paris_tz).isoformat(),
                        "price": 75.0,
                    }
                ]
            }
        ]
    }

    rows = _parse_spot_prices(data, target_date)

    assert len(rows) == 1
    assert rows[0].timestamp == datetime(2026, 4, 12, 22, 0, tzinfo=UTC)
    assert rows[0].prix_eur_mwh == 75.0
