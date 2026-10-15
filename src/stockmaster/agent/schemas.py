"""Decision schemas. ``Decision`` is what the model proposes (weights + reasons, never lots);
``FinalDecision`` is what the system validated and will submit. Reports and trade books are
rendered only from ``FinalDecision`` so they can never disagree."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CODE_PATTERN = r"^\d{4}[A-Z]?$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionTarget(_Strict):
    code: str = Field(pattern=CODE_PATTERN, description="Stock code, e.g. 2330")
    target_weight_pct: float = Field(gt=0, le=25, description="Target % of NAV after today's trades")
    conviction: int = Field(ge=1, le=5)
    rationale: str = Field(max_length=240, description="≤80 字，引用工具中的具體數據")
    signals: list[str] = Field(default_factory=list, max_length=6)


class ExitItem(_Strict):
    code: str = Field(pattern=CODE_PATTERN)
    rationale: str = Field(max_length=200)


class Decision(_Strict):
    schema_version: Literal["decision.v1"] = "decision.v1"
    trade_date: date
    regime: Literal["risk_on", "neutral", "risk_off"]
    risk_level: Literal["defensive", "normal", "aggressive"]
    target_cash_pct: float = Field(ge=0, le=24)
    market_view: str = Field(max_length=900, description="≤300 字市場觀點")
    positions: list[PositionTarget] = Field(min_length=1, max_length=30)
    exits: list[ExitItem] = Field(default_factory=list, max_length=30)
    risk_notes: str = Field(max_length=600)

    @field_validator("positions")
    @classmethod
    def _unique_positions(cls, v: list[PositionTarget]) -> list[PositionTarget]:
        codes = [p.code for p in v]
        if len(codes) != len(set(codes)):
            raise ValueError("duplicate codes in positions")
        return v

    def weights(self) -> dict[str, float]:
        return {p.code: p.target_weight_pct / 100 for p in self.positions}


DecisionSource = Literal["llm", "llm_repair", "det_repair", "fallback", "hold", "quant"]


class FinalOrder(_Strict):
    code: str
    name: str
    side: Literal["buy", "sell"]
    lots: int = Field(gt=0)
    shares: int
    ref_price: float
    limit_up: float
    limit_down: float
    est_amount: float
    rationale: str
    rationale_source: Literal["llm", "risk_control", "baseline", "quant"]


class FinalTarget(_Strict):
    code: str
    name: str
    target_weight: float
    lots_after: int
    weight_after_ref: float
    conviction: int | None = None
    rationale: str = ""


class Provenance(_Strict):
    run_id: str
    account_id: str
    provider: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    strategy_version: str
    data_as_of: date
    input_sha256: str
    git_sha: str | None = None
    llm_attempts: list[str] = Field(default_factory=list)


class CheckRow(_Strict):
    rule_id: str
    severity: str
    passed: bool
    measured: float | None = None
    limit: float | None = None
    detail: str = ""


class FinalDecision(_Strict):
    schema_version: Literal["final_decision.v1"] = "final_decision.v1"
    trade_date: date
    decision_source: DecisionSource
    regime: str
    risk_level: str
    market_view: str
    risk_notes: str
    targets: list[FinalTarget]
    exits: list[ExitItem]
    orders: list[FinalOrder]
    post_trade: dict
    checks: list[CheckRow]
    repair_notes: list[str] = Field(default_factory=list)
    stale_inputs: list[str] = Field(default_factory=list)
    provenance: Provenance

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def decision_json_schema() -> dict:
    """JSON schema of ``Decision`` for the submit_decision tool (inlined, no $defs)."""
    schema = Decision.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    return inline(schema)
