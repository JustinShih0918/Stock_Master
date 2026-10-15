"""Active Share between our top-N holdings and each reference active ETF's top-N (rule 六(十三)).

Active Share = 1/2 * sum |w_i(portfolio) - w_i(benchmark)|, computed on the two top-10 lists.
The rulebook does not say whether weights are renormalised within each top-10, so both variants are
computed and the minimum is enforced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def top_n(weights: Mapping[str, float], n: int) -> dict[str, float]:
    ranked = sorted(((c, w) for c, w in weights.items() if w > 0), key=lambda x: (-x[1], x[0]))
    return dict(ranked[:n])


def active_share_raw(p: Mapping[str, float], b: Mapping[str, float], n: int = 10) -> float:
    pt, bt = top_n(p, n), top_n(b, n)
    names = sorted(set(pt) | set(bt))  # fixed order: float sums must not depend on hash seed
    return 0.5 * sum(abs(pt.get(c, 0.0) - bt.get(c, 0.0)) for c in names)


def active_share_normalized(p: Mapping[str, float], b: Mapping[str, float], n: int = 10) -> float:
    pt, bt = top_n(p, n), top_n(b, n)
    sp, sb = sum(pt.values()), sum(bt.values())
    if sp <= 0 or sb <= 0:
        return 1.0
    names = sorted(set(pt) | set(bt))
    return 0.5 * sum(abs(pt.get(c, 0.0) / sp - bt.get(c, 0.0) / sb) for c in names)


@dataclass(frozen=True)
class ActiveShareResult:
    etf: str
    raw: float
    normalized: float
    overlap: tuple[str, ...]

    @property
    def value(self) -> float:
        return min(self.raw, self.normalized)


def active_share_report(portfolio: Mapping[str, float],
                        etf_holdings: Mapping[str, Mapping[str, float]],
                        n: int = 10) -> list[ActiveShareResult]:
    """One result per ETF, lowest (most binding) first."""
    out = []
    pt = top_n(portfolio, n)
    for etf, holdings in etf_holdings.items():
        bt = top_n(holdings, n)
        out.append(ActiveShareResult(
            etf=etf,
            raw=active_share_raw(portfolio, holdings, n),
            normalized=active_share_normalized(portfolio, holdings, n),
            overlap=tuple(sorted(set(pt) & set(bt), key=lambda c: (-pt[c], c))),
        ))
    return sorted(out, key=lambda r: (r.value, r.etf))
