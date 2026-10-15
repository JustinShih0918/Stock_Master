"""Typed loaders for config/rules.*.yaml (official competition rules) and config/strategy.yaml."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from stockmaster.settings import get_settings


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Period(_Frozen):
    start: date
    end: date
    expected_trading_days: int


class Submission(_Frozen):
    window_open_prev_day: str
    window_close: str
    max_per_day: int
    min_days_for_ranking: int


class Fees(_Frozen):
    commission_rate: float
    sell_tax_rate: float
    rounding: str  # floor | round | none
    min_commission: float = 0.0
    etf_sell_tax_rate: float | None = None  # 證交稅 on ETF sells (0.1%); None = same as stocks

    def sell_tax(self, code: str) -> float:
        if self.etf_sell_tax_rate is not None and code.startswith("00"):
            return self.etf_sell_tax_rate
        return self.sell_tax_rate


class Holdings(_Frozen):
    min_count: int
    max_count: int


class WeightCaps(_Frozen):
    default: float
    overrides: dict[str, float] = {}

    def cap(self, code: str) -> float:
        return self.overrides.get(code, self.default)


class CashRule(_Frozen):
    min: float
    max_exclusive: float


class ActiveShareRule(_Frozen):
    min: float
    top_n: int
    consecutive_days_to_disqualify: int
    reference_list_date: date


class Rules(_Frozen):
    competition_id: str
    initial_cash: int
    lot_size: int
    universe_file: str
    universe_as_of: date
    period: Period
    submission: Submission
    fill_price: str
    valuation_price: str
    fees: Fees
    holdings: Holdings
    weight_caps: WeightCaps
    drift_grace_trading_days: int
    cash: CashRule
    active_share: ActiveShareRule
    price_limit_pct: float
    warnings_to_disqualify: int
    dividends: str
    prohibited: list[str]


class Buffers(_Frozen):
    add_cap_default: float
    add_cap_overrides: dict[str, float] = {}
    drift_trim_target_default: float
    drift_trim_target_overrides: dict[str, float] = {}
    drift_trim_by_grace_day: int
    stress_up: float
    stress_down: float
    stress_rest_of_portfolio: float
    stress_fill_premium: float
    cash_min_worst: float
    cash_min_halted_sell: float = 0.0
    cash_max_worst: float
    cash_target_min: float
    cash_target_max: float
    day1_invest_target: float
    holdings_target_min: int
    holdings_target_max: int
    min_position_weight: float
    active_share_target: float
    active_share_hard_floor: float
    active_share_emergency: float
    sell_proceeds_fund_buys: bool

    def add_cap(self, code: str) -> float:
        return self.add_cap_overrides.get(code, self.add_cap_default)

    def drift_trim_target(self, code: str) -> float:
        return self.drift_trim_target_overrides.get(code, self.drift_trim_target_default)


class Turnover(_Frozen):
    no_trade_band: float
    daily_budget: float
    event_day_budget: float
    freeze_last_days: int
    swap_alpha_cost_multiple: float
    rebalance_speed: float = 1.0  # fraction of the gap to target closed per day (1 = full rebalance)
    exit_rank: int = 150          # holdings are kept while their score rank is within this


class LLMKnobs(_Frozen):
    deviation_budget: float
    max_tool_calls: int
    max_check_calls: int
    max_repair_rounds: int
    temperature: float
    fallback_models: list[str] = []


class RiskLevel(_Frozen):
    core_names: int
    core_weight: float
    tsmc_weight: float
    beta_target: float


class Alpha(_Frozen):
    weights: dict[str, float] = {}


class Strategy(_Frozen):
    strategy_version: str
    alpha: Alpha = Alpha()
    buffers: Buffers
    turnover: Turnover
    llm: LLMKnobs
    risk_levels: dict[str, RiskLevel]


@dataclass(frozen=True)
class Instrument:
    code: str
    name: str
    market: str  # TWSE | TPEX


@dataclass(frozen=True)
class Universe:
    instruments: tuple[Instrument, ...]
    sha256: str

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(i.code for i in self.instruments)

    def name(self, code: str) -> str:
        for i in self.instruments:
            if i.code == code:
                return i.name
        raise KeyError(code)

    def market(self, code: str) -> str:
        for i in self.instruments:
            if i.code == code:
                return i.market
        raise KeyError(code)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_dir(config_dir: Path | None) -> Path:
    return config_dir or get_settings().config_dir


@lru_cache(maxsize=4)
def load_rules(config_dir: Path | None = None, name: str = "rules.esun2026.yaml") -> Rules:
    path = _config_dir(config_dir) / name
    return Rules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=4)
def load_strategy(config_dir: Path | None = None, name: str = "strategy.yaml") -> Strategy:
    path = _config_dir(config_dir) / name
    return Strategy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=4)
def load_universe(config_dir: Path | None = None, rules: Rules | None = None) -> Universe:
    rules = rules or load_rules(config_dir)
    path = _config_dir(config_dir) / rules.universe_file
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    instruments = tuple(Instrument(r["code"], r["name"], r["market"]) for r in rows)
    if len({i.code for i in instruments}) != len(instruments):
        raise ValueError(f"duplicate codes in {path}")
    return Universe(instruments=instruments, sha256=file_sha256(path))


def config_fingerprint(config_dir: Path | None = None) -> dict[str, str]:
    """sha256 of every config file, recorded in each run manifest."""
    d = _config_dir(config_dir)
    return {p.name: file_sha256(p) for p in sorted(d.glob("*")) if p.is_file()}
