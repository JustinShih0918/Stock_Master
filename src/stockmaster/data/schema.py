"""Database schema (SQLAlchemy Core). Mirrors migrations/versions; change both together.

Tables fall into four groups: reference data (calendar, instruments, ETFs), market data (prices,
flows, margin, valuation, revenue, corporate actions, macro), provenance (``source_fetch``: every
raw payload we stored), and runs/accounts (run, llm_call, tool_call, decision, validation, order,
fill, submission, ledger_day, position_day, warning_event).
"""

from __future__ import annotations

from sqlalchemy import (JSON, BigInteger, Boolean, Column, Date, DateTime, Float, ForeignKey, func,
                        Integer, MetaData, Numeric, String, Table, Text, UniqueConstraint)
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import Uuid

NAMING = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ix": "ix_%(table_name)s_%(column_0_name)s",
}
metadata = MetaData(naming_convention=NAMING)

JSONB = postgresql.JSONB(astext_type=Text()).with_variant(JSON(), "sqlite")
# BIGSERIAL in Postgres; SQLite (tests, local smoke runs) auto-increments only INTEGER primary keys.
ID = BigInteger().with_variant(Integer(), "sqlite")
MONEY = Numeric(20, 4)
NOW = func.now()  # now() in Postgres, CURRENT_TIMESTAMP in SQLite


def _fetch_fk() -> Column:
    return Column("source_fetch_id", BigInteger, ForeignKey("source_fetch.id"), nullable=True)


# --- reference ---------------------------------------------------------------------------------

account = Table(
    "account", metadata,
    Column("id", String(48), primary_key=True),
    Column("kind", String(24), nullable=False),
    Column("start_cash", MONEY, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("note", Text),
)

calendar_override = Table(
    "calendar_override", metadata,
    Column("date", Date, primary_key=True),
    Column("is_open", Boolean, nullable=False),
    Column("reason", Text, nullable=False),
    Column("recorded_at", DateTime(timezone=True), server_default=NOW, nullable=False),
)

etf = Table(
    "etf", metadata,
    Column("code", String(12), primary_key=True),
    Column("name", Text, nullable=False),
    Column("issuer", Text),
    Column("market", String(8)),
    Column("asset_class", String(32)),
    Column("listed_date", Date),
    Column("in_reference_list", Boolean, server_default="false", nullable=False),
)

instrument = Table(
    "instrument", metadata,
    Column("code", String(12), primary_key=True),
    Column("name", Text, nullable=False),
    Column("market", String(8), nullable=False),
    Column("industry", Text),
    Column("listed_date", Date),
)

leaderboard = Table(
    "leaderboard", metadata,
    Column("date", Date, primary_key=True),
    Column("team", Text, primary_key=True),
    Column("nav", MONEY),
    Column("rank", Integer),
    Column("recorded_at", DateTime(timezone=True), server_default=NOW, nullable=False),
)

run = Table(
    "run", metadata,
    Column("run_id", Uuid, primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("account_id", String(48)),
    Column("trade_date", Date),
    Column("slot", String(16)),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),
    Column("git_sha", String(40)),
    Column("config_sha", JSONB),
    Column("prompt_version", String(32)),
    Column("strategy_version", String(32)),
    Column("provider", String(32)),
    Column("model_id", Text),
    Column("input_manifest", JSONB),
    Column("features_sha", String(64)),
    Column("error", Text),
)

source_fetch = Table(
    "source_fetch", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("endpoint", Text, nullable=False),
    Column("params", JSONB),
    Column("for_date", Date),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("http_status", Integer),
    Column("content_sha256", String(64), nullable=False),
    Column("raw_path", Text),
    Column("row_count", Integer),
    Column("is_stale", Boolean, server_default="false", nullable=False),
    Column("is_synthetic", Boolean, server_default="false", nullable=False),
    Column("note", Text),
)

trading_calendar = Table(
    "trading_calendar", metadata,
    Column("date", Date, primary_key=True),
    Column("is_open", Boolean, nullable=False),
    Column("name", Text),
    Column("source", String(32), nullable=False),
)

universe_member = Table(
    "universe_member", metadata,
    Column("universe_id", String(48), primary_key=True),
    Column("code", String(12), primary_key=True),
    Column("list_sha256", String(64), nullable=False),
)

# --- market data -------------------------------------------------------------------------------

breadth_daily = Table(
    "breadth_daily", metadata,
    Column("market", String(8), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("payload", JSONB),
    _fetch_fk(),
)

corp_action = Table(
    "corp_action", metadata,
    Column("code", String(12), primary_key=True),
    Column("ex_date", Date, primary_key=True),
    Column("kind", String(16), primary_key=True),
    Column("cash_dividend", Float),
    Column("stock_dividend_ratio", Float),
    Column("pay_date", Date),
    _fetch_fk(),
)

etf_holding = Table(
    "etf_holding", metadata,
    Column("etf_code", String(12), primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("code", String(16), primary_key=True),
    Column("name", Text),
    Column("weight", Float, nullable=False),
    Column("shares", BigInteger),
    _fetch_fk(),
)

financials_q = Table(
    "financials_q", metadata,
    Column("code", String(12), primary_key=True),
    Column("period", String(8), primary_key=True),
    Column("revenue", BigInteger),
    Column("gross_profit", BigInteger),
    Column("operating_income", BigInteger),
    Column("pretax_income", BigInteger),
    Column("net_income", BigInteger),
    Column("eps", Float),
    Column("announced_at", DateTime(timezone=True)),
    Column("source", String(32), nullable=False),
    _fetch_fk(),
)

index_daily = Table(
    "index_daily", metadata,
    Column("index_name", Text, primary_key=True),
    Column("date", Date, primary_key=True),
    Column("close", Float),
    Column("change_pct", Float),
    _fetch_fk(),
)

inst_flow_daily = Table(
    "inst_flow_daily", metadata,
    Column("code", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("foreign_net", BigInteger),
    Column("itrust_net", BigInteger),
    Column("dealer_self_net", BigInteger),
    Column("dealer_hedge_net", BigInteger),
    Column("total_net", BigInteger),
    _fetch_fk(),
)

macro_daily = Table(
    "macro_daily", metadata,
    Column("series_id", String(64), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("value", Float),
    _fetch_fk(),
)

margin_daily = Table(
    "margin_daily", metadata,
    Column("code", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("margin_buy", BigInteger),
    Column("margin_sell", BigInteger),
    Column("margin_prev", BigInteger),
    Column("margin_balance", BigInteger),
    Column("short_sell", BigInteger),
    Column("short_buy", BigInteger),
    Column("short_prev", BigInteger),
    Column("short_balance", BigInteger),
    _fetch_fk(),
)

monthly_revenue = Table(
    "monthly_revenue", metadata,
    Column("code", String(12), primary_key=True),
    Column("ym", Date, primary_key=True),
    Column("revenue", BigInteger),
    Column("revenue_prev_month", BigInteger),
    Column("revenue_last_year", BigInteger),
    Column("mom_pct", Float),
    Column("yoy_pct", Float),
    Column("cum_revenue", BigInteger),
    Column("cum_revenue_last_year", BigInteger),
    Column("cum_yoy_pct", Float),
    Column("note", Text),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
    Column("source", String(32), nullable=False),
    _fetch_fk(),
)

price_daily = Table(
    "price_daily", metadata,
    Column("code", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("market", String(8), nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("prev_close", Float),
    Column("volume_sh", BigInteger),
    Column("value_ntd", BigInteger),
    Column("trades", BigInteger),
    Column("vwap", Float),
    Column("vwap_published", Float),
    Column("next_ref_price", Float),
    Column("next_limit_up", Float),
    Column("next_limit_down", Float),
    Column("halted", Boolean, server_default="false", nullable=False),
    _fetch_fk(),
)

valuation_daily = Table(
    "valuation_daily", metadata,
    Column("code", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("pe", Float),
    Column("pb", Float),
    Column("div_yield", Float),
    _fetch_fk(),
)

# --- runs, decisions and accounts ----------------------------------------------------------------

decision = Table(
    "decision", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("run_id", Uuid, ForeignKey("run.run_id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("stage", String(16), nullable=False),
    Column("source", String(16), nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=NOW, nullable=False),
)

ledger_day = Table(
    "ledger_day", metadata,
    Column("account_id", String(48), ForeignKey("account.id"), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("cash", MONEY, nullable=False),
    Column("holdings_value", MONEY, nullable=False),
    Column("nav", MONEY, nullable=False),
    Column("accrued_dividends", MONEY, nullable=False),
    Column("cash_pct", Float, nullable=False),
    Column("n_holdings", Integer, nullable=False),
    Column("day_status", String(16), nullable=False),
    Column("warnings", Integer, server_default="0", nullable=False),
    Column("organizer_nav", MONEY),
    Column("nav_diff", MONEY),
    Column("active_share_min", Float),
    Column("active_share_etf", String(12)),
    Column("state", JSONB, nullable=False),
)

llm_call = Table(
    "llm_call", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("run_id", Uuid, ForeignKey("run.run_id"), nullable=False),
    Column("seq", Integer, nullable=False),
    Column("provider", String(32), nullable=False),
    Column("model_id", Text, nullable=False),
    Column("request", JSONB),
    Column("response", JSONB),
    Column("reasoning", Text),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("latency_ms", Integer),
    Column("status", String(16), nullable=False),
    Column("error", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
)

position_day = Table(
    "position_day", metadata,
    Column("account_id", String(48), ForeignKey("account.id"), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("code", String(12), primary_key=True),
    Column("lots", Integer, nullable=False),
    Column("close", Float, nullable=False),
    Column("value", MONEY, nullable=False),
    Column("weight", Float, nullable=False),
    Column("overweight_days", Integer, server_default="0", nullable=False),
)

warning_event = Table(
    "warning_event", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("account_id", String(48), ForeignKey("account.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("rule_id", String(32), nullable=False),
    Column("origin", String(16), nullable=False),
    Column("detail", Text),
)

# ``order`` is an SQL keyword; the Python name carries a trailing underscore.
order_ = Table(
    "order", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("decision_id", BigInteger, ForeignKey("decision.id"), nullable=False),
    Column("account_id", String(48), ForeignKey("account.id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("code", String(12), nullable=False),
    Column("side", String(4), nullable=False),
    Column("lots", Integer, nullable=False),
    Column("ref_price", Float, nullable=False),
    Column("limit_up", Float),
    Column("limit_down", Float),
)

submission = Table(
    "submission", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("account_id", String(48), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("seq", Integer, nullable=False),
    Column("decision_id", BigInteger, ForeignKey("decision.id"), nullable=False),
    Column("artifact_paths", JSONB, nullable=False),
    Column("artifact_sha256", JSONB, nullable=False),
    Column("generated_at", DateTime(timezone=True), nullable=False),
    Column("uploaded_at", DateTime(timezone=True)),
    Column("uploaded_sha256", String(64)),
    Column("is_final", Boolean, server_default="false", nullable=False),
    UniqueConstraint("account_id", "trade_date", "seq", name="uq_submission_account_date_seq"),
)

tool_call = Table(
    "tool_call", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("run_id", Uuid, ForeignKey("run.run_id"), nullable=False),
    Column("llm_call_id", BigInteger, ForeignKey("llm_call.id")),
    Column("name", String(64), nullable=False),
    Column("args", JSONB),
    Column("result_text", Text),
    Column("result_sha", String(64)),
    Column("ms", Integer),
    Column("error", Text),
)

validation = Table(
    "validation", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("decision_id", BigInteger, ForeignKey("decision.id"), nullable=False),
    Column("rule_id", String(32), nullable=False),
    Column("severity", String(8), nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("measured", Float),
    Column("limit_value", Float),
    Column("detail", Text),
)

fill = Table(
    "fill", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("order_id", BigInteger, ForeignKey("order.id"), nullable=False),
    Column("account_id", String(48), ForeignKey("account.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("code", String(12), nullable=False),
    Column("side", String(4), nullable=False),
    Column("lots", Integer, nullable=False),
    Column("price", Numeric(14, 4), nullable=False),
    Column("gross", MONEY, nullable=False),
    Column("commission", MONEY, nullable=False),
    Column("tax", MONEY, nullable=False),
    Column("net_cash", MONEY, nullable=False),
    Column("status", String(12), nullable=False),
)
