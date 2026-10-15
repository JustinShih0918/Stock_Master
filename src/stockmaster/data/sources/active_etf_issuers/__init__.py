"""Issuer adapters for active-ETF holdings. ``registry()`` maps each ETF code to its official
adapter and to the aggregator fallback."""

from __future__ import annotations

import importlib

from stockmaster.data.sources.active_etf_issuers.base import Adapter

MODULES = ("alliancebernstein", "allianz", "capital", "cathay", "ctbc", "first_sitc", "fubon", "fuh_hwa",
           "jpmorgan", "kgi", "mega", "nomura", "sinopac", "taishin", "uni_president", "yuanta",
           "aggregator_etfinfo")


def registry() -> tuple[dict[str, Adapter], dict[str, Adapter]]:
    official: dict[str, Adapter] = {}
    fallback: dict[str, Adapter] = {}
    for name in MODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        for a in mod.ADAPTERS:
            for code in a.codes:
                (official if a.official else fallback).setdefault(code, a)
    return official, fallback
