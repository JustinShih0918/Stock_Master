"""Operator alerts: always logged and appended to data/alerts.log; also POSTed to
ALERT_WEBHOOK_URL (e.g. a Slack incoming webhook) when configured."""

from __future__ import annotations

import json
import logging

import httpx

from stockmaster.settings import get_settings
from stockmaster.timeutil import now_taipei

log = logging.getLogger("stockmaster.alerts")


def alert(message: str, level: str = "warning") -> None:
    getattr(log, level if level in ("info", "warning", "error") else "warning")(message)
    settings = get_settings()
    path = settings.data_dir / "alerts.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_taipei().isoformat(), "level": level, "message": message},
                           ensure_ascii=False) + "\n")
    url = settings.alert_webhook_url
    if url is not None and url.get_secret_value():
        try:
            httpx.post(url.get_secret_value(), json={"text": f"[{level}] {message}"}, timeout=10)
        except httpx.HTTPError as e:
            log.warning("alert webhook failed: %s", e)
