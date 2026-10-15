"""安聯投信 (Allianz GI) PCF JSON, same vendor format as Nomura; FundID is an internal id."""

from stockmaster.data.sources.active_etf_issuers.base import Adapter, http_fetch, no_url
from stockmaster.data.sources.active_etf_issuers.nomura import parse_entries

FUND_IDS = {"00984A": "E0001"}
CODES = ("00984A", "00993A")


def parse(raw: bytes, code: str):
    return parse_entries(raw, code, FUND_IDS.get(code.upper()))


ADAPTERS = [Adapter("allianz", CODES, http_fetch(no_url, parse))]
