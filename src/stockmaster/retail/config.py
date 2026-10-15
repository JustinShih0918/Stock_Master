"""Typed loader for config/rules.retail.yaml (the simulated small retail account).

``RetailRules`` exposes the attributes the shared ledger and quote code read from competition
``Rules`` (initial_cash, lot_size, fees, weight_caps, price_limit_pct), so accounting is shared.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from stockmaster.config import Fees, WeightCaps
from stockmaster.settings import get_settings


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Fill(_Frozen):
    price: str = "vwap"
    slippage_bps: float = 0.0


class UniverseFilter(_Frozen):
    markets: list[str]
    include_etfs: bool = True
    min_price: float = 0.0
    min_adv20_ntd: float = 0.0
    min_history_days: int = 0
    exclude_name_keywords: list[str] = []


class PortfolioRules(_Frozen):
    max_positions: int
    max_weight: float
    min_order_ntd: float
    cash_buffer_pct: float
    min_hold_days: int
    max_orders_per_day: int
    candidates_for_llm: int


class RetailLLM(_Frozen):
    chain: str = ""
    temperature: float = 0.2
    max_repair_rounds: int = 1
    max_tokens: int = 4000


class RetailRules(_Frozen):
    account_id: str
    initial_cash: int
    start_date: date
    lot_size: int
    price_limit_pct: float
    fees: Fees
    fill: Fill
    universe: UniverseFilter
    portfolio: PortfolioRules
    weight_caps: WeightCaps
    benchmark: str
    llm: RetailLLM = RetailLLM()

    @property
    def strategy_version(self) -> str:
        return "retail.v1"


@lru_cache(maxsize=4)
def load_retail_rules(config_dir: Path | None = None, name: str = "rules.retail.yaml") -> RetailRules:
    path = (config_dir or get_settings().config_dir) / name
    return RetailRules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
