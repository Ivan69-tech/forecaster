"""
Tests du fetcher RTE (stub).

Les tests vérifient l'interface publique et le comportement attendu
une fois fetch_spot_prices() implémentée.
"""

from datetime import date

import pytest

from forecaster.fetchers.rte import SpotPriceRow, fetch_spot_prices


def test_fetch_spot_prices_raises_not_implemented():
    """Vérifie que le stub lève NotImplementedError en attendant l'implémentation."""
    with pytest.raises(NotImplementedError):
        fetch_spot_prices(date.today())


def test_spot_price_row_dataclass():
    """Vérifie que SpotPriceRow est instanciable avec les bons champs."""
    from datetime import timezone
    ts = date.today()
    from datetime import datetime
    row = SpotPriceRow(
        timestamp=datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc),
        prix_eur_mwh=85.5,
        source="RTE",
    )
    assert row.prix_eur_mwh == 85.5
    assert row.source == "RTE"


# TODO (à implémenter après fetch_spot_prices()) :
#
# @pytest.mark.parametrize("target_date", [date(2026, 4, 13)])
# def test_fetch_spot_prices_returns_24_entries(target_date, httpx_mock):
#     """Vérifie que 24 entrées horaires sont retournées pour une journée."""
#     httpx_mock.add_response(json=_mock_rte_response(target_date))
#     result = fetch_spot_prices(target_date)
#     assert len(result) == 24
#     assert all(isinstance(r, SpotPriceRow) for r in result)
#     assert all(r.source == "RTE" for r in result)
