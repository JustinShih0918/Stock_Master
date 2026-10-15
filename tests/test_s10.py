"""S10 strategy through the live decision path (SYNTHETIC panel)."""

import pytest

from stockmaster.pipeline.decide import decide_quant
from stockmaster.pipeline.run import day_inputs
from stockmaster.reporting.consistency import check_consistency
from stockmaster.reporting.render import write_artifacts
from stockmaster.strategies import s10

from .test_pipeline import ctx_factory  # noqa: F401  (fixture re-export)


@pytest.mark.parametrize("name", list(s10.VARIANTS))
def test_day1_build_is_valid_for_every_variant(ctx_factory, name, tmp_path):  # noqa: F811
    ctx = ctx_factory()
    env = s10.Env(rules=ctx.rules, strategy=ctx.strategy, universe=ctx.universe.codes,
                  etf_holdings=ctx.etf_holdings)
    w, cash, action = s10.decide(day_inputs(ctx, 1, None), s10.VARIANTS[name], env)
    assert action == "build"
    assert cash == pytest.approx(1 - ctx.strategy.buffers.day1_invest_target)
    out = decide_quant(ctx, w, cash, strategy_label=name, action=action, run_id="t", account_id="paper",
                       strategy_version="t", input_sha256="0")
    f = out.final
    assert f.decision_source == "quant"
    assert all(c.passed for c in f.checks if c.severity == "hard"), [c for c in f.checks if not c.passed]
    pt = f.post_trade["weights"]
    assert 20 <= len(pt) <= 30
    if s10.VARIANTS[name].tsmc_w == 0:
        assert "2330" not in pt
    else:
        # nominal 10%; scaled with the core to fit the 87% day-1 investment (~8.1%), ~9.1% afterwards
        assert 0.075 <= pt["2330"] <= 0.10
    core = sorted(pt.values(), reverse=True)[:10]
    assert max(v for k, v in pt.items() if k != "2330") <= 0.0875 + 1e-9
    assert sum(core) > 0.6  # concentrated: most capital in the core
    art = write_artifacts(f, tmp_path)
    assert check_consistency(f, art.report_md.read_text(encoding="utf-8"),
                             art.trade_book_csv.read_text(encoding="utf-8-sig")) == []


def test_weekly_schedule_matches_lab():
    p = s10.VARIANTS["s10a"]
    assert [d for d in range(1, 25) if s10.is_rebalance_day(d, p)] == [1, 2, 7, 12, 17, 22]
