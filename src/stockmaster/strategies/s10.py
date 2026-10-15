"""S10 concentrated structure (top-6 research, 2026-10-03).

Structure: 10 core names at <= 8.5% of NAV, satellites at the minimum position size to reach 22
holdings, 3% cash after day 1 (day 1 invests 87%), hysteresis (a held core name stays while its rank
< 18, a held satellite while < 60), no re-targeting in the last ``freeze_last`` competition days, and a
pre-trade guard that runs the live repair -> order builder -> validator path and steps back (0.6 / 0.4 /
0.25 of the way) or holds if the day's orders would be rejected.

Variants (pre-registered in research/preregistration_20261003.json):
    s10b      ranking = 投信 20-day net buying / 20-day ADV, re-target daily       (lab: s10_itrust20_daily)
    s10a      ranking = engine composite (alpha v2, locked), re-target every 5 days (lab: s10_composite_weekly)
    s10b_t10  s10b with 2330 at a nominal 10% (user decision 2026-10-03: keep part of 2330; untested)
    s10a_t10  s10a with 2330 at a nominal 10%
              2330 is scaled together with the core names to fit the invested total, so its effective
              weight is ~8.1% on day 1 (87% invested) and ~9.1% afterwards (97% invested).

Ported from the research lab (scratchpad/top6/concentration/structure.py + lab helpers) with explicit
inputs instead of lab globals. tests/test_s10.py replays the lab version and requires identical targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from stockmaster.config import Rules, Strategy
from stockmaster.domain import PortfolioState, Quote

TSMC = "2330"
GUARD_STEPS = (0.6, 0.4, 0.25)


@dataclass(frozen=True)
class S10Params:
    ranking: str  # 'itrust20' | 'composite'
    n_core: int = 10
    core_w: float = 0.0875
    core_max: float | None = 0.085
    tsmc_w: float = 0.0
    n_total: int = 22
    mode: str = "daily"  # 'daily' | 'weekly'
    rebal_every: int = 5
    speed: float = 1.0
    core_exit: int = 18
    sat_exit: int = 60
    cash: float = 0.03
    freeze_last: int = 5
    guard: bool = True


_B = S10Params(ranking="itrust20", mode="daily", speed=1.0)
_A = S10Params(ranking="composite", mode="weekly", speed=0.35)
VARIANTS: dict[str, S10Params] = {
    "s10b": _B,
    "s10a": _A,
    "s10b_t10": replace(_B, tsmc_w=0.10),
    "s10a_t10": replace(_A, tsmc_w=0.10),
}


@dataclass(frozen=True)
class Env:
    """Everything the structure needs besides the day's inputs."""

    rules: Rules
    strategy: Strategy
    universe: frozenset[str]
    etf_holdings: Mapping[str, Mapping[str, float]]


# --- weights helpers (lab: nav, current_weights, cash_target, _fit, build_weights, partial) ---------

def nav(state: PortfolioState, quotes: Mapping[str, Quote], lot: int) -> float:
    return float(state.cash) + sum(n * lot * quotes[c].ref_price for c, n in sorted(state.lots.items())
                                   if n > 0 and c in quotes)


def current_weights(state: PortfolioState, quotes: Mapping[str, Quote], lot: int) -> dict[str, float]:
    v = nav(state, quotes, lot)
    return {c: n * lot * quotes[c].ref_price / v for c, n in sorted(state.lots.items())
            if n > 0 and c in quotes}


def cash_target(state: PortfolioState, env: Env, normal: float) -> float:
    if not state.held_codes():
        return 1.0 - env.strategy.buffers.day1_invest_target
    return float(normal)


def _fit(w: dict, total: float, lo: dict, hi: dict, movable) -> dict:
    """Scale ``movable`` names proportionally so sum(w) == total, keeping lo <= w <= hi."""
    w = dict(w)
    order = [c for c in w if c in set(movable)]  # deterministic: insertion order of w
    for _ in range(100):
        gap = total - sum(w.values())
        if abs(gap) < 1e-10:
            break
        free = [c for c in order if (gap > 0 and w[c] < hi[c] - 1e-12) or (gap < 0 and w[c] > lo[c] + 1e-12)]
        base = sum(w[c] for c in free)
        if not free or base <= 0:
            break
        f = 1 + gap / base
        for c in free:
            w[c] = min(max(w[c] * f, lo[c]), hi[c])
    return w


def build_weights(ranked: list[str], x, env: Env, n_core: int, core_w: float, tsmc_w: float,
                  n_total: int) -> dict[str, float]:
    """Core + minimum-size satellites, fitted to invest = 1 - cash (lab: build_weights with sat_w=None).
    Halted holdings keep their current weight and count toward n_total."""
    rules, buf = env.rules, env.strategy.buffers
    lot = rules.lot_size
    invest = 1.0 - cash_target(x.state, env, normal=0.03)
    v = nav(x.state, x.quotes, lot)
    held = x.state.held_codes()
    cur = current_weights(x.state, x.quotes, lot)

    def one_lot(code):
        return lot * x.quotes[code].ref_price / v

    def ok(code):
        return (code in env.universe and code in x.quotes and not x.quotes[code].halted
                and (code in held or one_lot(code) <= buf.add_cap(code)))

    lo_n = max(rules.holdings.min_count, buf.holdings_target_min)
    n_total = min(max(int(n_total), lo_n), rules.holdings.max_count)
    frozen = {k: cur[k] for k in sorted(held) if k in x.quotes and x.quotes[k].halted and k in cur}
    use_tsmc = tsmc_w > 0 and ok(TSMC)
    seen = set(frozen) | {TSMC}
    names = []
    for k in list(ranked) + list(x.scores.index):
        if k not in seen and ok(k):
            seen.add(k)
            names.append(k)
    slots = n_total - len(frozen) - (1 if use_tsmc else 0)
    if len(names) < slots:
        raise ValueError(f"S10: only {len(names)} tradable names for {slots} slots")
    nc = min(n_core, slots)
    core, sats = names[:nc], names[nc:slots]
    keys = core + sats + ([TSMC] if use_tsmc else [])
    wmin = {k: max(buf.min_position_weight, min(one_lot(k), buf.add_cap(k))) for k in keys}
    cap = {k: buf.add_cap(k) for k in keys}
    w: dict[str, float] = {}
    if use_tsmc:
        w[TSMC] = min(float(tsmc_w), cap[TSMC])
    for k in core:
        w[k] = min(max(float(core_w), wmin[k]), cap[k])
    for k in sats:
        w[k] = min(wmin[k], cap[k])
    target = invest - sum(frozen.values())
    big = list(core) + ([TSMC] if use_tsmc else [])
    lo = {k: wmin[k] for k in w}
    w = _fit(w, target, lo, cap, big)  # satellites stay at the minimum
    if abs(target - sum(w.values())) > 1e-9:
        w = _fit(w, target, lo, cap, list(w))
    w.update(frozen)
    return w


def partial(target: dict[str, float], x, env: Env, speed: float) -> dict[str, float]:
    """Move ``speed`` of the way from current weights to ``target``; exits are complete; new names
    enter at max(speed * target, minimum position); rescaled to sum(target), clipped to add-caps."""
    buf = env.strategy.buffers
    cur = current_weights(x.state, x.quotes, env.rules.lot_size)
    if not cur:
        return dict(target)
    out = {}
    for c, t in target.items():
        if t <= 0:
            continue
        now = cur.get(c, 0.0)
        out[c] = now + speed * (t - now) if now > 0 else max(speed * t, buf.min_position_weight)
    goal, total = sum(target.values()), sum(out.values())
    if total <= 0:
        return dict(target)
    out = {c: val * goal / total for c, val in out.items()}
    caps = {c: buf.add_cap(c) for c in out}
    if any(val > caps[c] + 1e-12 and cur.get(c, 0.0) <= caps[c] for c, val in out.items()):
        lo = {c: min(val, buf.min_position_weight) for c, val in out.items()}
        hi = {c: (caps[c] if cur.get(c, 0.0) <= caps[c] else max(out[c], caps[c])) for c in out}
        out = {c: min(val, hi[c]) for c, val in out.items()}
        out = _fit(out, goal, lo, hi, list(out))
    return out


# --- ranking, selection, guard --------------------------------------------------------------------

def rank_codes(x, ranking: str, env: Env) -> list[str]:
    """Universe codes (2330 excluded) best first, as of the close of x.as_of."""
    if ranking == "composite":
        r = [c for c in x.scores.index if c != TSMC]
    elif ranking == "itrust20":
        s = x.features["itrust_20d"].astype(float).sort_index()
        r = [c for c in s.dropna().sort_values(ascending=False, kind="mergesort").index if c != TSMC]
        rs = set(r)
        r += [c for c in x.scores.index if c not in rs and c != TSMC]
    else:
        raise ValueError(f"unknown S10 ranking {ranking!r}")
    return [c for c in r if c in env.universe]


def select(x, ranked: list[str], p: S10Params, env: Env) -> tuple[list[str], list[str]]:
    cur = current_weights(x.state, x.quotes, env.rules.lot_size)
    pos = {c: i for i, c in enumerate(ranked)}
    n_other = p.n_total - (1 if p.tsmc_w > 0 else 0)
    big = [c for c in ranked if cur.get(c, 0.0) >= 0.025 and pos[c] < p.core_exit]
    core = big[:p.n_core]
    for c in ranked:
        if len(core) >= p.n_core:
            break
        if c not in core:
            core.append(c)
    core = sorted(core, key=lambda c: pos[c])
    n_sat = n_other - len(core)
    keep = [c for c in ranked if c in cur and c not in core and pos[c] < p.sat_exit]
    sats = keep[:n_sat]
    for c in ranked:
        if len(sats) >= n_sat:
            break
        if c not in core and c not in sats:
            sats.append(c)
    return core, sorted(sats, key=lambda c: pos[c])


def target_weights(x, p: S10Params, env: Env) -> dict[str, float]:
    ranked = rank_codes(x, p.ranking, env)
    core, sats = select(x, ranked, p, env)
    w = build_weights(core + sats, x, env, n_core=len(core), core_w=p.core_w, tsmc_w=p.tsmc_w,
                      n_total=p.n_total)
    if p.core_max is not None:  # margin under the 8.75% add-cap; the excess goes to smaller names
        over = {k: val - p.core_max for k, val in w.items() if k != TSMC and val > p.core_max}
        if over:
            for k in over:
                w[k] = p.core_max
            small = [k for k in w if k != TSMC and k not in over]
            if small:
                add = sum(over.values()) / len(small)
                for k in small:
                    w[k] = min(w[k] + add, p.core_max)
    return w


def passes(x, w: dict[str, float], cash: float, env: Env) -> bool:
    """Pre-trade check with the live path: repair -> whole-lot orders -> validator."""
    from stockmaster.portfolio.order_builder import build_orders
    from stockmaster.portfolio.repair import repair_targets
    from stockmaster.risk.validator import ValidationContext, validate_orders
    rep = repair_targets(w, state=x.state, quotes=x.quotes, universe=env.universe,
                         ranking=list(x.scores.index), etf_holdings=env.etf_holdings,
                         rules=env.rules, strategy=env.strategy, cash_target=cash)
    b = build_orders(x.state, rep.weights, x.quotes, env.rules, env.strategy)
    r = validate_orders(b.orders, ValidationContext(
        trade_date=x.trade_date, state=x.state, quotes=x.quotes, universe=env.universe,
        rules=env.rules, strategy=env.strategy, etf_holdings=env.etf_holdings))
    return r.passed


def _hold(x, env: Env) -> tuple[dict[str, float], float]:
    cur = current_weights(x.state, x.quotes, env.rules.lot_size)
    return cur, float(x.state.cash) / nav(x.state, x.quotes, env.rules.lot_size)


def guarded(x, w, cash, env: Env, guard: bool) -> tuple[dict[str, float], float, str]:
    if not guard or not x.state.held_codes() or passes(x, w, cash, env):
        return w, cash, "target"
    for s in GUARD_STEPS:
        step = partial(w, x, env, speed=s)
        if passes(x, step, cash, env):
            return step, cash, f"guard_step_{s}"
    cur, c = _hold(x, env)
    return cur, c, "guard_hold"


def is_rebalance_day(day_index: int, p: S10Params) -> bool:
    """Weekly schedule from the lab: day 1 (build), day 2 (top-up), then every ``rebal_every`` days."""
    return day_index in (1, 2) or (day_index - 2) % p.rebal_every == 0


def decide(x, p: S10Params, env: Env) -> tuple[dict[str, float], float, str]:
    """Target weights, cash target and the action taken for one trading day.

    ``x`` has the engine's DayInputs fields (trade_date, as_of, state, features, scores, quotes,
    day_index, days_total). ``days_total`` may be None outside the competition (no freeze)."""
    if not x.state.held_codes():
        return target_weights(x, p, env), cash_target(x.state, env, p.cash), "build"
    c_t = cash_target(x.state, env, p.cash)
    if p.freeze_last and x.days_total and x.day_index > x.days_total - p.freeze_last:
        w, c = _hold(x, env)
        return w, c, "freeze"
    if p.mode == "daily":
        return guarded(x, partial(target_weights(x, p, env), x, env, speed=p.speed), c_t, env, p.guard)
    if p.mode == "weekly":
        if is_rebalance_day(x.day_index, p):
            return guarded(x, target_weights(x, p, env), c_t, env, p.guard)
        w, c = _hold(x, env)
        return w, c, "hold"
    raise ValueError(f"unknown S10 mode {p.mode!r}")


def engine_policy(name: str, env: Env):
    """Backtest-engine policy for a variant: DayInputs -> (weights, cash)."""
    p = VARIANTS[name]

    def policy(x):
        w, c, _ = decide(x, p, env)
        return w, c
    return policy
