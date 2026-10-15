"""Command-line entry point: ``sm <command>``."""

from __future__ import annotations

import logging
import subprocess
from datetime import date, datetime

import typer

from stockmaster.data.ingest import DAILY_PARTS
from stockmaster.settings import PROJECT_ROOT
from stockmaster.timeutil import today_taipei

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _date(s: str | None) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else today_taipei()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Apply database migrations."""
    subprocess.run(["alembic", "upgrade", "head"], cwd=PROJECT_ROOT, check=True)


@app.command("sync-ref")
def sync_ref() -> None:
    """Load the universe and TWSE holiday schedule."""
    from stockmaster.data import ingest
    from stockmaster.data.http import Fetcher
    from stockmaster.data.store import get_engine
    typer.echo(ingest.sync_reference(get_engine(), Fetcher()))


@app.command()
def ingest(on: str = typer.Option(None, "--date", help="YYYY-MM-DD, default today"),
           parts: str = typer.Option(",".join(DAILY_PARTS), help="comma-separated parts")) -> None:
    """Ingest one trading day of market data."""
    from stockmaster.data import ingest as ing
    from stockmaster.data.http import Fetcher
    from stockmaster.data.store import get_engine
    typer.echo(ing.ingest_day(get_engine(), Fetcher(), _date(on), parts.split(",")))


@app.command()
def backfill(start: str = typer.Option(..., help="YYYY-MM-DD"),
             end: str = typer.Option(None, help="YYYY-MM-DD, default yesterday"),
             parts: str = typer.Option("twse_quotes,tpex_quotes"),
             newest_first: bool = typer.Option(True),
             only_missing: bool = typer.Option(False, help="fetch only parts with no stored rows")) -> None:
    """Backfill trading days between start and end (inclusive). Slow by design (TWSE pacing)."""
    from stockmaster.data import ingest as ing
    from stockmaster.data.http import Fetcher
    from stockmaster.data.reference import load_calendar
    from stockmaster.data.store import get_engine
    engine, fetcher = get_engine(), Fetcher()
    cal = load_calendar(engine)
    days = cal.trading_days(_date(start), _date(end) if end else cal.prev_open(today_taipei()))
    for d in (reversed(days) if newest_first else days):
        todo = ing.missing_parts(engine, d, parts.split(",")) if only_missing else parts.split(",")
        if todo:
            rep = ing.ingest_day(engine, fetcher, d, todo)
            typer.echo(rep)
            ing.mark_closed_if_no_data(engine, rep)


@app.command()
def revenue(ym: str = typer.Option(..., help="YYYY-MM")) -> None:
    """Ingest monthly revenue (both markets) for one month."""
    from stockmaster.data import ingest as ing
    from stockmaster.data.http import Fetcher
    from stockmaster.data.store import get_engine
    y, m = map(int, ym.split("-"))
    typer.echo(ing.ingest_monthly_revenue(get_engine(), Fetcher(), y, m))


@app.command()
def decide(account: str = typer.Option("paper", help="account id (live | paper | shadow_baseline)"),
           trade_date: str = typer.Option(None, help="YYYY-MM-DD; default next trading day"),
           slot: str = typer.Option("manual", help="A | B | watchdog | manual"),
           no_llm: bool = typer.Option(False, help="deterministic baseline only"),
           chain: str = typer.Option(None, help="provider:model chain; default per account"),
           strategy: str = typer.Option(None, help="deterministic strategy (s10a|s10b|s10a_t10|s10b_t10); default per account")) -> None:
    """Run the daily decision pipeline and write the report and trade book."""
    from stockmaster.agent.llm_client import build_chain
    from stockmaster.data.store import get_engine
    from stockmaster.ops.jobs import ACCOUNT_CHAINS, ACCOUNT_STRATEGIES, ACCOUNTS
    from stockmaster.pipeline.run import run_decision
    strat = strategy or ACCOUNT_STRATEGIES.get(account)
    spec = chain or ACCOUNT_CHAINS.get(account)
    models = build_chain(spec)[0] if (spec and not no_llm and not strat) else None
    out = run_decision(get_engine(), account_id=account, slot=slot, use_llm=not no_llm, models=models,
                       trade_date=_date(trade_date) if trade_date else None, strategy_name=strat,
                       start_date=ACCOUNTS[account][1] if account in ACCOUNTS else None)
    f = out.final
    typer.echo(f"{f.trade_date} account={account} source={f.decision_source} orders={len(f.orders)} "
               f"holdings={f.post_trade.get('n_holdings')} final={out.is_final}")
    typer.echo(f"attempts: {f.provenance.llm_attempts}")
    typer.echo(f"report: {out.artifacts.report_md}")
    bad = [c for c in f.checks if not c.passed]
    for c in bad:
        typer.echo(f"  [{c.severity}] {c.rule_id}: {c.detail}")


@app.command()
def settle(account: str = typer.Option("paper"), on: str = typer.Option(None, "--date")) -> None:
    """Apply the day's final orders at VWAP and value the account at the close."""
    from stockmaster.config import load_rules
    from stockmaster.data.store import get_engine
    from stockmaster.pipeline.accounts import settle as do_settle
    r = do_settle(get_engine(), account, _date(on), load_rules())
    v = r.valuation
    typer.echo(f"{v.date} {account}: NAV={v.nav:,.0f} cash={v.cash_pct:.2%} holdings={v.n_holdings} "
               f"fills={len(r.fills)}")


@app.command()
def reconcile(nav: float = typer.Option(..., help="organizer NAV for the day"),
              account: str = typer.Option("live"), on: str = typer.Option(None, "--date")) -> None:
    """Reconcile our ledger with the organizer's published NAV (detects voided days)."""
    from decimal import Decimal
    from stockmaster.config import load_rules
    from stockmaster.data.store import get_engine
    from stockmaster.pipeline.accounts import reconcile as do_reconcile
    typer.echo(do_reconcile(get_engine(), account, _date(on), Decimal(str(nav)), load_rules()))


@app.command()
def bakeoff(models: str = typer.Option(..., help="comma-separated provider:model, e.g. nvidia:z-ai/glm-5.3"),
            dates: str = typer.Option(..., help="comma-separated as-of dates YYYY-MM-DD"),
            repeats: int = typer.Option(3)) -> None:
    """Compare LLMs on identical frozen inputs (reliability first)."""
    from stockmaster.data.store import get_engine
    from stockmaster.pipeline.bakeoff import run_bakeoff
    from stockmaster.settings import get_settings
    trials, scores = run_bakeoff(get_engine(), models.split(","), [_date(d) for d in dates.split(",")],
                                 repeats)
    out = get_settings().artifacts_dir / "bakeoff" / now_stamp()
    out.mkdir(parents=True, exist_ok=True)
    trials.to_json(out / "trials.json", orient="records", force_ascii=False, indent=1, default_handler=str)
    scores.to_csv(out / "scores.csv", index=False)
    typer.echo(scores.to_string(index=False))
    typer.echo(f"saved to {out}")


@app.command()
def compsim(windows: int = typer.Option(200), start: str = typer.Option("2025-09-01"),
            seed: int = typer.Option(2026)) -> None:
    """Competition simulator over random 24-day windows of stored history."""
    from stockmaster.backtest.compsim import run_compsim
    from stockmaster.backtest.engine import FeatureCache, baseline_policy, equal_weight_policy
    from stockmaster.config import load_rules, load_strategy, load_universe
    from stockmaster.data.store import get_engine
    from stockmaster.features.panel import load_panel
    from stockmaster.settings import get_settings
    rules, strategy, universe = load_rules(), load_strategy(), load_universe()
    end = today_taipei()
    panel = load_panel(get_engine(), sorted(universe.codes), end,
                       lookback_days=(end - _date(start)).days)
    cache = FeatureCache(panel)
    from stockmaster.pipeline.run import load_etf_holdings
    etf, _, _ = load_etf_holdings(get_engine(), end)  # current snapshot as a proxy for history
    policies = {"adaptive": baseline_policy(strategy, rules, universe),
                **{lvl: baseline_policy(strategy, rules, universe, lvl) for lvl in strategy.risk_levels},
                "equal_weight_top24": equal_weight_policy(strategy, rules)}
    summary, rows, _ = run_compsim(cache, policies, n_windows=windows, universe=universe, rules=rules,
                                   strategy=strategy, seed=seed, etf_holdings=etf,
                                   progress=lambda i, n: typer.echo(f"window {i}/{n}") if i % 10 == 0 else None)
    out = get_settings().artifacts_dir / "compsim" / now_stamp()
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    __import__("pandas").DataFrame(rows).to_csv(out / "windows.csv", index=False)
    typer.echo(summary.to_string())
    typer.echo(f"saved to {out}")


@app.command("strategy-doc")
def strategy_doc(model: str = typer.Option(None, help="provider:model; default first in LLM_CHAIN"),
                 compsim_dir: str = typer.Option(None, help="artifacts/compsim/<stamp> to cite"),
                 outline: str = typer.Option(None, help="path to the organizer's template outline (text)")) -> None:
    """Have the AI Agent write the ETF投資策略說明文件 from verified facts."""
    import pandas as pd
    from pathlib import Path
    from stockmaster.agent.llm_client import OpenAICompatModel, parse_chain
    from stockmaster.reporting.strategy_doc import default_facts, generate
    from stockmaster.settings import get_settings
    provider, model_id = model.split(":", 1) if model else parse_chain()[0]
    summary = None
    if compsim_dir:
        summary = pd.read_csv(Path(compsim_dir) / "summary.csv").round(4).to_dict(orient="records")
    facts = default_facts(compsim_summary=summary,
                          template_outline=Path(outline).read_text(encoding="utf-8") if outline else None)
    doc = generate(OpenAICompatModel(provider, model_id), facts,
                   get_settings().artifacts_dir / "strategy_doc")
    typer.echo(doc.markdown[:2000])


@app.command()
def signals(start: str = typer.Option("2025-09-01", help="history start"),
            every: int = typer.Option(3, help="evaluate every Nth trading day")) -> None:
    """Rank-IC study of every feature vs forward returns (VWAP entry, close exit)."""
    from stockmaster.backtest.engine import FeatureCache
    from stockmaster.backtest.signals import ic_table
    from stockmaster.config import load_universe
    from stockmaster.data.store import get_engine
    from stockmaster.features.panel import load_panel
    from stockmaster.settings import get_settings
    universe = load_universe()
    end = today_taipei()
    panel = load_panel(get_engine(), sorted(universe.codes), end, lookback_days=(end - _date(start)).days)
    cache = FeatureCache(panel)
    days = [d for d in panel.close.index if panel.close.loc[d].notna().sum() > 50]
    dates = days[130::every]
    table = ic_table(cache, dates)
    out = get_settings().artifacts_dir / "signals" / now_stamp()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ic.csv")
    typer.echo(f"{len(dates)} dates {dates[0] if dates else None}..{dates[-1] if dates else None}")
    typer.echo(table.round(3).to_string())
    typer.echo(f"saved to {out}")


@app.command()
def submission(account: str = typer.Option("live"), trade_date: str = typer.Option(None)) -> None:
    """Show the system-selected final submission (files to upload and their sha256)."""
    from sqlalchemy import select
    from stockmaster.data import schema
    from stockmaster.data.store import get_engine
    s, d, r = schema.submission, schema.decision, schema.run
    td = _date(trade_date) if trade_date else None
    with get_engine().connect() as conn:
        q = (select(s.c.trade_date, s.c.seq, s.c.artifact_paths, s.c.artifact_sha256, s.c.uploaded_at,
                    d.c.source, d.c.sha256).join(d, d.c.id == s.c.decision_id)
             .join(r, r.c.run_id == d.c.run_id).where(r.c.account_id == account)
             .where(s.c.is_final.is_(True)).order_by(s.c.trade_date.desc()).limit(1))
        if td:
            q = q.where(s.c.trade_date == td)
        row = conn.execute(q).first()
    if row is None:
        typer.echo("no final submission")
        raise typer.Exit(1)
    typer.echo(f"trade_date={row.trade_date} seq={row.seq} source={row.source} decision_sha={row.sha256}")
    for k, path in row.artifact_paths.items():
        typer.echo(f"  {k}: {path}\n    sha256={row.artifact_sha256[k]}")
    typer.echo(f"uploaded_at={row.uploaded_at}")


@app.command("mark-uploaded")
def mark_uploaded(file: str = typer.Argument(..., help="the exact file that was uploaded"),
                  account: str = typer.Option("live"), trade_date: str = typer.Option(...)) -> None:
    """Record a (clerical) upload after verifying the file is byte-identical to the generated one."""
    import hashlib
    from pathlib import Path
    from sqlalchemy import select, update
    from stockmaster.data import schema
    from stockmaster.data.store import get_engine
    from stockmaster.timeutil import now_taipei
    sha = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    s, d, r = schema.submission, schema.decision, schema.run
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(s.c.id, s.c.artifact_sha256).join(d, d.c.id == s.c.decision_id)
                           .join(r, r.c.run_id == d.c.run_id).where(r.c.account_id == account)
                           .where(s.c.trade_date == _date(trade_date)).where(s.c.is_final.is_(True))).first()
        if row is None:
            typer.echo("no final submission for that date")
            raise typer.Exit(1)
        if sha not in row.artifact_sha256.values():
            typer.echo(f"MISMATCH: {sha} is not an artifact of the final submission — do not upload it")
            raise typer.Exit(2)
        conn.execute(update(s).where(s.c.id == row.id).values(uploaded_at=now_taipei(), uploaded_sha256=sha))
    typer.echo(f"recorded upload {sha}")


@app.command("ranker-eval")
def ranker_eval(start: str = typer.Option("2025-09-01"), horizon: int = typer.Option(10)) -> None:
    """Walk-forward LightGBM ranker vs the composite score (gate for adoption)."""
    import json
    from stockmaster.alpha.ranker import walk_forward
    from stockmaster.backtest.engine import FeatureCache
    from stockmaster.config import load_universe
    from stockmaster.data.store import get_engine
    from stockmaster.features.panel import load_panel
    universe = load_universe()
    end = today_taipei()
    panel = load_panel(get_engine(), sorted(universe.codes), end, lookback_days=(end - _date(start)).days)
    cache = FeatureCache(panel)
    days = [d for d in panel.close.index if panel.close.loc[d].notna().sum() > 50][130:]
    ev = walk_forward(cache, days, horizon=horizon, min_train_dates=40, retrain_every=10)
    typer.echo(json.dumps(ev.summary(), indent=2))


@app.command()
def perf(account: str = typer.Option("paper")) -> None:
    """Daily NAV of an account vs TAIEX since its first day."""
    import pandas as pd
    from sqlalchemy import select
    from stockmaster.data import schema
    from stockmaster.data.store import get_engine
    from stockmaster.features.panel import TAIEX
    eng = get_engine()
    with eng.connect() as conn:
        nav = pd.read_sql(select(schema.ledger_day.c.date, schema.ledger_day.c.nav, schema.ledger_day.c.cash_pct,
                                 schema.ledger_day.c.n_holdings, schema.ledger_day.c.day_status,
                                 schema.ledger_day.c.warnings)
                          .where(schema.ledger_day.c.account_id == account).order_by(schema.ledger_day.c.date), conn)
        idx = pd.read_sql(select(schema.index_daily.c.date, schema.index_daily.c.close)
                          .where(schema.index_daily.c.index_name == TAIEX), conn).set_index("date")["close"]
    if nav.empty:
        typer.echo("no settled days yet")
        return
    nav["nav"] = nav["nav"].astype(float)
    start = nav["date"].iloc[0]
    base_idx = idx[idx.index < start].iloc[-1] if (idx.index < start).any() else None
    nav["return"] = nav["nav"] / 1e9 - 1
    nav["taiex_return"] = nav["date"].map(idx) / base_idx - 1 if base_idx else None
    typer.echo(nav.to_string(index=False))


@app.command("rl-train")
def rl_train(split: str = typer.Option("2026-07-01", help="train on windows before, evaluate after"),
             timesteps: int = typer.Option(20000), n_envs: int = typer.Option(8),
             n_train: int = typer.Option(60), n_eval: int = typer.Option(30),
             start: str = typer.Option("2025-09-01")) -> None:
    """Train the PPO meta-allocator and run its out-of-sample gate vs the adaptive baseline."""
    import json
    from stockmaster.backtest.engine import FeatureCache
    from stockmaster.config import load_rules, load_strategy, load_universe
    from stockmaster.data.store import get_engine
    from stockmaster.features.panel import load_panel
    from stockmaster.pipeline.run import load_etf_holdings
    from stockmaster.rl.train import evaluate, save_gate, split_windows, train
    from stockmaster.settings import get_settings
    rules, strategy, universe = load_rules(), load_strategy(), load_universe()
    end = today_taipei()
    panel = load_panel(get_engine(), sorted(universe.codes), end, lookback_days=(end - _date(start)).days)
    cache = FeatureCache(panel)
    etf, _, _ = load_etf_holdings(get_engine(), end)
    tr, ev = split_windows(cache, _date(split), n_train, n_eval)
    typer.echo(f"train windows {len(tr)} ({tr[0][1][0]}..{tr[-1][1][0]}), eval windows {len(ev)} "
               f"({ev[0][1][0] if ev else None}..{ev[-1][1][0] if ev else None})")
    out = get_settings().artifacts_dir / "rl" / now_stamp()
    model = train(cache, tr, universe=universe, rules=rules, strategy=strategy, timesteps=timesteps,
                  n_envs=n_envs, out_dir=out)
    report = evaluate(model, cache, ev, universe=universe, rules=rules, strategy=strategy)
    save_gate(report, out)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    typer.echo(f"saved to {out}")


def now_stamp() -> str:
    from stockmaster.timeutil import now_taipei
    return now_taipei().strftime("%Y%m%d_%H%M%S")


@app.command()
def collect() -> None:
    """Run the unattended scheduler (data collection; decision jobs once enabled)."""
    from stockmaster.ops.scheduler import build_scheduler
    sched = build_scheduler()
    for job in sched.get_jobs():
        typer.echo(f"scheduled {job.name}: {job.trigger}")
    sched.start()


if __name__ == "__main__":
    app()
