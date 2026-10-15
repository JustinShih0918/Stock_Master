"""End-to-end decision pipeline with scripted models on a SYNTHETIC panel."""

import json
from datetime import date

import pytest

from stockmaster.agent.harness import extract_json
from stockmaster.agent.llm_client import ChatResult, ScriptedModel, ToolCall
from stockmaster.alpha.composite import composite
from stockmaster.domain import PortfolioState, Quote
from stockmaster.features.compute import compute_features
from stockmaster.features.regime import market_regime
from stockmaster.pipeline.context import build_tool_context
from stockmaster.pipeline.decide import decide
from stockmaster.reporting.consistency import check_consistency
from stockmaster.reporting.render import write_artifacts

from .test_features import AS_OF, synthetic_panel

TRADE = date(2026, 10, 27)


@pytest.fixture(scope="module")
def ctx_factory(universe, rules, strategy):
    panel = synthetic_panel(sorted(universe.codes))
    feats = compute_features(panel)
    scores = composite(feats)
    quotes = {c: Quote.from_close(c, float(feats.loc[c, "close"]), AS_OF) for c in panel.codes}
    etf = {"00999A": {c: 0.07 for c in list(scores.index[:10])},
           "00998A": {c: 0.08 for c in list(scores.index[5:15])}}

    def make(state=None):
        return build_tool_context(trade_date=TRADE, as_of=AS_OF,
                                  state=state or PortfolioState.initial(rules.initial_cash),
                                  quotes=quotes, universe=universe, features=feats, scores=scores,
                                  regime=market_regime(panel), etf_holdings=etf,
                                  etf_asof={k: AS_OF for k in etf}, rules=rules, strategy=strategy)
    return make


def call(name, args, i=0):
    return ChatResult(content=None, tool_calls=[ToolCall(f"c{i}", name, json.dumps(args))],
                      reasoning=None, finish_reason="tool_calls")


def decision_from(weights, cash_pct, exits=()):
    return {"trade_date": str(TRADE), "regime": "neutral", "risk_level": "normal",
            "target_cash_pct": cash_pct, "market_view": "測試市場觀點。",
            "positions": [{"code": c, "target_weight_pct": round(w * 100, 3), "conviction": 3,
                           "rationale": f"{c} 測試理由"} for c, w in weights.items()],
            "exits": [{"code": c, "rationale": "出清"} for c in exits],
            "risk_notes": "測試風險說明。"}


def run(ctx, script, tmp_path):
    model = ScriptedModel(script)
    out = decide(ctx, [model], run_id="test-run", account_id="paper", strategy_version="t",
                 input_sha256="0" * 64)
    art = write_artifacts(out.final, tmp_path)
    md = art.report_md.read_text(encoding="utf-8")
    book = art.trade_book_csv.read_text(encoding="utf-8-sig")
    assert check_consistency(out.final, md, book) == []
    return out, model


def test_llm_accepts_baseline(ctx_factory, tmp_path):
    ctx = ctx_factory()
    base = ctx.baselines["normal"]
    out, model = run(ctx, [call("get_baseline_portfolio", {"risk_level": "normal"}),
                           call("submit_decision", decision_from(base, 13.0), 1)], tmp_path)
    f = out.final
    assert f.decision_source == "llm"
    assert all(c.passed for c in f.checks if c.severity == "hard")
    assert 20 <= f.post_trade["n_holdings"] <= 30
    assert all(o.side == "buy" for o in f.orders)
    # the tool result was fed back to the model as a tool message
    assert any(m.get("role") == "tool" for m in model.requests[-1])


def test_invalid_llm_decision_is_repaired(ctx_factory, tmp_path):
    ctx = ctx_factory()
    # Schema-valid but rule-breaking: 2330 above the 22% add-cap and only 9 holdings.
    bad = {c: 0.087 for c in [c for c in ctx.scores.index if c != "2330"][:8]}
    bad["2330"] = 0.25
    script = [call("submit_decision", decision_from(bad, 5.4), i) for i in range(3)]
    out, _ = run(ctx, script, tmp_path)
    f = out.final
    assert f.decision_source == "det_repair"
    assert all(c.passed for c in f.checks if c.severity == "hard")
    assert out.attempts[0].repair_rounds == 2


def test_dead_model_falls_back_to_baseline(ctx_factory, tmp_path):
    ctx = ctx_factory()
    out, _ = run(ctx, [], tmp_path)  # script exhausted -> provider error
    assert out.final.decision_source == "fallback"
    assert all(c.passed for c in out.final.checks if c.severity == "hard")


def test_incomplete_decision_gets_feedback(ctx_factory, tmp_path, rules):
    ctx0 = ctx_factory()
    first, _ = run(ctx0, [call("submit_decision", decision_from(ctx0.baselines["normal"], 13.0))],
                   tmp_path)
    lots = {t.code: t.lots_after for t in first.final.targets}
    held = PortfolioState(as_of=AS_OF, cash=ctx0.state.cash - sum(
        int(o.est_amount) for o in first.final.orders), lots=lots)
    ctx = ctx_factory(held)
    w = dict(ctx.current_weights())
    dropped = sorted(w)[0]
    del w[dropped]  # forgets a holding -> must be told
    full = dict(ctx.current_weights())
    script = [call("submit_decision", decision_from(w, 100 - sum(w.values()) * 100), 0),
              call("submit_decision", decision_from(full, 100 - sum(full.values()) * 100), 1)]
    out, model = run(ctx, script, tmp_path)
    assert out.final.decision_source == "llm"
    fb = [m["content"] for m in model.requests[1] if m.get("role") == "tool"]
    assert any("missing from positions/exits" in c and dropped in c for c in fb)


def test_json_protocol_model(ctx_factory, tmp_path):
    ctx = ctx_factory()
    payload = {"tool": "submit_decision", "arguments": decision_from(ctx.baselines["normal"], 13.0)}
    script = [ChatResult(content="```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
                         tool_calls=[], reasoning=None, finish_reason="stop")]
    model = ScriptedModel(script, supports_tools=False)
    out = decide(ctx, [model], run_id="t", account_id="paper", strategy_version="t", input_sha256="0")
    assert out.final.decision_source == "llm"


def test_extract_json_tolerates_prose():
    assert extract_json('好的，以下是決策：{"a": {"b": "}"}} 謝謝') == {"a": {"b": "}"}}


def test_golden_replay_reproduces_decision(ctx_factory, tmp_path):
    from stockmaster.agent.llm_client import Cassette, CassetteModel
    ctx = ctx_factory()
    script = [call("get_baseline_portfolio", {"risk_level": "normal"}),
              call("submit_decision", decision_from(ctx.baselines["normal"], 13.0), 1)]
    path = tmp_path / "cassette.jsonl"
    rec = CassetteModel(ScriptedModel(script), Cassette(path, "record"), "scripted", "scripted-v1")
    first = decide(ctx, [rec], run_id="r", account_id="paper", strategy_version="t", input_sha256="x")
    ctx2 = ctx_factory()
    rep = CassetteModel(None, Cassette(path, "replay"), "scripted", "scripted-v1")
    second = decide(ctx2, [rep], run_id="r", account_id="paper", strategy_version="t", input_sha256="x")
    assert first.final.decision_source == second.final.decision_source == "llm"
    assert first.final.sha256() == second.final.sha256()


def test_garbage_model_output_still_yields_valid_submission(ctx_factory, tmp_path):
    ctx = ctx_factory()
    junk = [ChatResult(content="我覺得市場會漲。", tool_calls=[], reasoning=None, finish_reason="stop")] * 5
    bad_args = [ChatResult(content=None, tool_calls=[ToolCall("x", "submit_decision", "{not json")],
                           reasoning=None, finish_reason="tool_calls")] * 20
    out, _ = run(ctx, junk, tmp_path)
    assert out.final.decision_source == "fallback"
    out2, _ = run(ctx_factory(), bad_args, tmp_path)
    assert out2.final.decision_source == "fallback"
    for o in (out, out2):
        assert all(c.passed for c in o.final.checks if c.severity == "hard")


def test_rationale_revenue_month_is_fact_checked(ctx_factory):
    from stockmaster.pipeline.decide import fact_check_rationale
    ctx = ctx_factory()
    code = ctx.features.index[0]
    month = ctx.features.loc[code, "rev_last_ym"].month
    wrong = (month % 12) + 1
    text, changed = fact_check_rationale(ctx, code, f"{wrong} 月營收 YoY +50%，動能強")
    assert changed and text.startswith(f"{month} 月營收")
    same, changed2 = fact_check_rationale(ctx, code, f"{month} 月營收創新高")
    assert not changed2 and same == f"{month} 月營收創新高"


def test_positions_sent_as_json_string_are_accepted(ctx_factory, tmp_path):
    ctx = ctx_factory()
    d = decision_from(ctx.baselines["normal"], 13.0)
    d["positions"] = json.dumps(d["positions"], ensure_ascii=False)
    out, _ = run(ctx, [call("submit_decision", d)], tmp_path)
    assert out.final.decision_source == "llm"
