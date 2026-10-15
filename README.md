# Stock Master

Stock Master is a Taiwan stock portfolio system and AI-agent fund manager for **AI CUP 2026 玉山人工智慧公開挑戰賽 — Agent 基金經理人**. All trading is simulated. An LLM agent makes the daily decision. Deterministic code converts it to whole-lot orders and validates every competition rule. The same code produces the daily 決策報告 (decision report) and 交易書 (trade book).

Competition facts the code encodes (`config/rules.esun2026.yaml`):
- **Capital and dates:** NT$1B, from 10/27 to 11/27 (24 trading days; 10/26 is a market holiday).
- **Universe:** 150 stocks. Holdings must stay between 20 and 30 names.
- **Weight caps:** 2330 at most 25% of NAV, every other stock at most 10%.
- **Cash:** must stay between 0% and 25% of NAV.
- **Fills and valuation:** orders fill at the day's VWAP, and NAV is measured at the close.
- **Active Share:** must stay at least 20% against every active ETF.
- **Penalties:** 3 warnings disqualify the team.

## Architecture

```
data/        TWSE/TPEx/MOPS/active-ETF/overnight adapters → Postgres (+ raw payload archive, provenance)
features/    momentum, 投信/外資 flows, margin, monthly revenue, valuation, regime
alpha/       cross-sectional composite score
portfolio/   quant baseline (core + satellites, risk levels), whole-lot order builder, deterministic repair
risk/        validator (all competition rules + internal buffers, worst-case price-limit stress), Active Share
accounting/  fees, ledger (VWAP fills, close valuation, voided days, dividend accrual)
agent/       OpenAI-compatible LLM client (NVIDIA / Gemini, failover, cassettes), tools, harness, prompts (zh-TW)
pipeline/    decide (LLM → validate → LLM repair → deterministic repair → fallback), accounts, bake-off
reporting/   決策報告 + 交易書 from ONE FinalDecision, consistency audit, strategy document
backtest/    exact-rules simulator, competition simulator (random 24-day windows), signal IC study
rl/          gymnasium env + PPO meta-allocator over factor sleeves (gated: used only if it beats baseline OOS)
ops/         scheduler (ingest, ETF snapshots, Run A / Run B / watchdog, settlement), alerts
```

The model proposes target weights and reasons; it never places orders. Decision sources, best first:
1. `llm` — the model's decision was accepted as submitted.
2. `llm_repair` — the model fixed violations after feedback.
3. `det_repair` — deterministic projection of the model's intent onto the rules.
4. `fallback` — quant baseline used because no model was available.

Every run records:
- input hashes;
- model and prompt version;
- every LLM call and tool call;
- proposals and validation outcomes;
- orders and submission artifact hashes.

## Setup

```sh
cp .env.example .env              # add NVIDIA_API_KEY (and optionally GEMINI_*, LLM_CHAIN, ALERT_WEBHOOK_URL)
sudo scripts/fix-permissions.sh   # once per machine: workspace rw for all users (containers run with umask 0000)
docker compose up -d --build
docker compose exec dev bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for the RL extra
.venv/bin/pip install -e '.[dev,ml,rl]'
.venv/bin/sm db-upgrade && .venv/bin/sm sync-ref
.venv/bin/sm backfill --start 2025-09-01 --parts twse_quotes,tpex_quotes,twse_t86,tpex_t86,twse_margin,tpex_margin,twse_valuation,tpex_valuation
docker compose --profile ops up -d scheduler                  # unattended daily jobs
```

## Daily operation (Asia/Taipei)

| Time | Job |
|---|---|
| 15:00 / 17:30 / 21:40 | ingest the day's market data (each dataset publishes at a different time) |
| 17:45 | settle paper accounts at the day's VWAP and value them at the close |
| 18:30 / 21:10 / 07:00 | snapshot holdings of the 22 reference active ETFs |
| 19:00 | reference data, ex-dividend calendar, monthly revenue |
| 21:30 / 06:45 | overnight US data (SOX, NDX, TSM ADR, USD/TWD) |
| 21:50 (eve of a trading day) | **Run A**: decision for the next trading day; the submission window opens at 19:30 |
| 07:15 | **Run B**: retry if Run A could not use the LLM, or after a large overnight US move |
| 08:30 | **watchdog**: guarantees a valid final submission before the 08:55 deadline |

Artifacts are written to `artifacts/<trade_date>/<account>/<slot>_<run>/`:
- `decision_report_*.md` / `.html` (決策報告);
- `trade_book_*.csv` / `.json` (交易書);
- `final_decision_*.json`.

Upload the files of the **final** submission. The `submission` table marks it with `is_final`; the system, not a person, picks it. After the organizer publishes NAV at 19:00, run `sm reconcile --nav <value>` to detect voided days.

## Useful commands

```sh
sm decide --account paper            # run a decision now (default: next trading day)
sm settle --account paper --date YYYY-MM-DD
sm compsim --windows 200             # rules stress test + policy comparison over history
sm bakeoff --models nvidia:nvidia/nemotron-3-ultra-550b-a55b,nvidia:z-ai/glm-5.3 --dates 2026-10-07,2026-10-08
sm strategy-doc --compsim-dir artifacts/compsim/<stamp>   # agent-written ETF投資策略說明文件
.venv/bin/pytest
```

## Rules for working on this repo
See `CLAUDE.md`. Trading stays simulated. Secrets live only in `.env`. Times are Asia/Taipei. Synthetic or stale data is always flagged.

After the 10/23 code freeze, only logged bug fixes are allowed. Do not tune parameters in reaction to the market: decisions must remain the agent's.
