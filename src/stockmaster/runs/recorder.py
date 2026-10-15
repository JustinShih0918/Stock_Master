"""Persist runs for reproducibility: run manifest, every LLM/tool call, proposals and the final
decision, rule outcomes, orders, and submission artifacts (with sha256)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime

from sqlalchemy import Engine, func, select, update

from stockmaster.agent.harness import AgentResult
from stockmaster.agent.schemas import FinalDecision
from stockmaster.data import schema
from stockmaster.reporting.render import Artifacts
from stockmaster.settings import PROJECT_ROOT
from stockmaster.timeutil import now_taipei

_SECRET_KEYS = {"authorization", "api_key", "apikey", "x-api-key"}


def new_run_id() -> uuid.UUID:
    return uuid.uuid4()


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("+dirty" if dirty else ""))[:40] if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("***" if k.lower() in _SECRET_KEYS else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def start_run(engine: Engine, run_id: uuid.UUID, *, kind: str, account_id: str | None,
              trade_date, slot: str | None, config_sha: dict, strategy_version: str,
              input_manifest: dict, features_sha: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(schema.run.insert().values(
            run_id=run_id, kind=kind, account_id=account_id, trade_date=trade_date, slot=slot,
            started_at=now_taipei(), status="running", git_sha=git_sha(), config_sha=config_sha,
            strategy_version=strategy_version, input_manifest=input_manifest,
            features_sha=features_sha))


def finish_run(engine: Engine, run_id: uuid.UUID, *, status: str, provider: str | None = None,
               model_id: str | None = None, prompt_version: str | None = None,
               error: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(update(schema.run).where(schema.run.c.run_id == run_id).values(
            finished_at=now_taipei(), status=status, provider=provider, model_id=model_id,
            prompt_version=prompt_version, error=error))


def record_attempts(engine: Engine, run_id: uuid.UUID, attempts: list[AgentResult],
                    trade_date) -> None:
    with engine.begin() as conn:
        seq = 0
        for a in attempts:
            last_llm = None
            for entry in a.log:
                if entry["type"] == "llm":
                    seq += 1
                    last_llm = conn.execute(schema.llm_call.insert().values(
                        run_id=run_id, seq=seq, provider=a.provider, model_id=a.model_id,
                        request=None, response=redact({k: entry.get(k) for k in
                                                       ("content", "tool_calls", "finish_reason")}),
                        reasoning=entry.get("reasoning"), prompt_tokens=entry.get("prompt_tokens"),
                        completion_tokens=entry.get("completion_tokens"),
                        latency_ms=entry.get("latency_ms"), status=entry.get("status", "ok"),
                        error=entry.get("error"),
                        started_at=datetime.fromisoformat(entry["started_at"]) if entry.get("started_at")
                        else now_taipei()).returning(schema.llm_call.c.id)).scalar_one()
                elif entry["type"] == "tool":
                    conn.execute(schema.tool_call.insert().values(
                        run_id=run_id, llm_call_id=last_llm, name=entry["name"], args=entry["args"],
                        result_text=entry["result"],
                        result_sha=hashlib.sha256(entry["result"].encode()).hexdigest(),
                        ms=entry.get("ms")))
                elif entry["type"] == "submit" and entry.get("decision"):
                    conn.execute(schema.decision.insert().values(
                        run_id=run_id, trade_date=trade_date, stage="proposal", source="llm",
                        payload={"decision": entry["decision"], "accepted": entry["accepted"],
                                 "feedback": entry.get("detail"), "model": a.model_id},
                        sha256=sha256_json(entry["decision"])))


def record_final(engine: Engine, run_id: uuid.UUID, final: FinalDecision, art: Artifacts,
                 account_id: str) -> int:
    """Store the final decision, its checks and orders, and register a submission (next seq)."""
    with engine.begin() as conn:
        did = conn.execute(schema.decision.insert().values(
            run_id=run_id, trade_date=final.trade_date, stage="final",
            source=final.decision_source, payload=final.model_dump(mode="json"),
            sha256=final.sha256()).returning(schema.decision.c.id)).scalar_one()
        for c in final.checks:
            conn.execute(schema.validation.insert().values(
                decision_id=did, rule_id=c.rule_id, severity=c.severity, passed=c.passed,
                measured=c.measured, limit_value=c.limit, detail=c.detail))
        for o in final.orders:
            conn.execute(schema.order_.insert().values(
                decision_id=did, account_id=account_id, trade_date=final.trade_date, code=o.code,
                side=o.side, lots=o.lots, ref_price=o.ref_price, limit_up=o.limit_up,
                limit_down=o.limit_down))
        seq = (conn.execute(select(func.max(schema.submission.c.seq))
                            .where(schema.submission.c.account_id == account_id)
                            .where(schema.submission.c.trade_date == final.trade_date)).scalar() or 0) + 1
        paths = {k: str(getattr(art, k)) for k in ("report_md", "report_html", "trade_book_csv",
                                                   "trade_book_json", "decision_json")}
        conn.execute(schema.submission.insert().values(
            account_id=account_id, trade_date=final.trade_date, seq=seq, decision_id=did,
            artifact_paths=paths,
            artifact_sha256=art.sha256, generated_at=now_taipei(), is_final=False))
    return did


def mark_final(engine: Engine, decision_id: int) -> None:
    """Exactly one final submission per (account, trade date): the system decides, not a human."""
    s = schema.submission
    with engine.begin() as conn:
        row = conn.execute(select(s.c.trade_date, s.c.account_id)
                           .where(s.c.decision_id == decision_id)).one()
        conn.execute(update(s).where(s.c.trade_date == row.trade_date)
                     .where(s.c.account_id == row.account_id).values(is_final=False))
        conn.execute(update(schema.submission).where(schema.submission.c.decision_id == decision_id)
                     .values(is_final=True))
