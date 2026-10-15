"""Shared fixtures. All prices here are SYNTHETIC test data, not market data."""

from __future__ import annotations

from datetime import date

import pytest

from stockmaster.config import load_rules, load_strategy, load_universe
from stockmaster.domain import PortfolioState, Quote

AS_OF = date(2026, 10, 23)
TRADE_DATE = date(2026, 10, 27)


@pytest.fixture(scope="session")
def rules():
    return load_rules()


@pytest.fixture(scope="session")
def strategy():
    return load_strategy()


@pytest.fixture(scope="session")
def universe():
    return load_universe()


def synthetic_price(code: str) -> float:
    """Deterministic synthetic price in [20, 1020) per code (2330 pinned at 1500)."""
    if code == "2330":
        return 1500.0
    return float(20 + (int(code) * 37) % 1000)


@pytest.fixture(scope="session")
def quotes(universe):
    return {c: Quote.from_close(c, synthetic_price(c), AS_OF) for c in universe.codes}


@pytest.fixture()
def cash_state(rules):
    return PortfolioState.initial(rules.initial_cash)


def ranked_codes(universe) -> list[str]:
    return sorted(universe.codes)
