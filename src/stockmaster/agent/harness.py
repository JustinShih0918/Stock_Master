"""Agent loop: prompt -> tool calls -> submit_decision -> system evaluation -> bounded repair.

The harness never trusts model output: a submitted ``Decision`` is schema-validated, checked for
completeness against current holdings, then passed to ``evaluate`` (the deterministic order
builder + validator). Rejections are fed back to the model at most ``max_repair_rounds`` times.
Every model call and tool call is appended to ``AgentResult.log`` for the run record.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from importlib import resources

from pydantic import ValidationError

from stockmaster.agent.llm_client import ChatModel, ChatResult, ToolsUnsupported
from stockmaster.agent.schemas import Decision
from stockmaster.agent.tools import (Tool, ToolContext, get_market_context, get_portfolio_state,
                                     openai_tool_specs, registry, run_tool)
from stockmaster.timeutil import now_taipei

PROMPT_VERSION = "v1"
WEEKDAYS = "一二三四五六日"
MAX_NUDGES = 2

Evaluate = Callable[[Decision], tuple[bool, str]]


@dataclass
class AgentResult:
    status: str  # accepted | rejected | failed
    decision: Decision | None
    provider: str
    model_id: str
    rounds: int = 0
    repair_rounds: int = 0
    log: list[dict] = field(default_factory=list)
    error: str | None = None
    last_feedback: str | None = None


def load_prompt(name: str, version: str = PROMPT_VERSION) -> str:
    return (resources.files("stockmaster.agent.prompts") / version / f"{name}.md").read_text(encoding="utf-8")


def prompt_sha(version: str = PROMPT_VERSION) -> str:
    h = hashlib.sha256()
    for n in ("system", "user", "json_protocol"):
        h.update(load_prompt(n, version).encode())
    return h.hexdigest()[:16]


def extract_json(text: str) -> dict | None:
    """First balanced JSON object in ``text`` (tolerates code fences and surrounding prose)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _parse_args(raw: str) -> dict:
    try:
        v = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        v = extract_json(raw) or {}
    return v if isinstance(v, dict) else {}


def check_completeness(d: Decision, ctx: ToolContext) -> list[str]:
    problems = []
    if d.trade_date != ctx.trade_date:
        problems.append(f"trade_date must be {ctx.trade_date}, got {d.trade_date}")
    pos = {p.code for p in d.positions}
    ex = {e.code for e in d.exits}
    both = pos & ex
    if both:
        problems.append(f"codes in both positions and exits: {sorted(both)}")
    missing = sorted(ctx.state.held_codes() - pos - ex)
    if missing:
        problems.append(f"current holdings missing from positions/exits: {missing}")
    outside = sorted((pos | ex) - ctx.universe.codes)
    if outside:
        problems.append(f"codes outside the 150-stock universe: {outside}")
    not_held_exits = sorted(ex - ctx.state.held_codes())
    if not_held_exits:
        problems.append(f"exits for stocks not held: {not_held_exits}")
    total = sum(p.target_weight_pct for p in d.positions) + d.target_cash_pct
    if abs(total - 100) > 10:  # within ±10% the pipeline rescales proportionally (noted in the report)
        problems.append(f"weights + cash = {total:.1f}%, must be ~100%")
    return problems


def build_messages(ctx: ToolContext, tools: dict[str, Tool], native_tools: bool,
                   version: str = PROMPT_VERSION, extra_notes: str = "") -> list[dict]:
    system = load_prompt("system", version)
    if not native_tools:
        spec = "\n".join(f"- {t.name}: {t.description} params={json.dumps(t.parameters, ensure_ascii=False)}"
                         for t in tools.values())
        system += load_prompt("json_protocol", version).replace("{tool_list}", spec)
    comp = (f"今天是比賽第 {ctx.day_index} 個交易日，含今天尚餘 {ctx.days_remaining} 個交易日。"
            if ctx.day_index else "（非比賽期間：模擬交易）")
    user = load_prompt("user", version).format(
        trade_date=ctx.trade_date, weekday=WEEKDAYS[ctx.trade_date.weekday()], as_of=ctx.as_of,
        competition_line=comp, portfolio_state=get_portfolio_state(ctx),
        market_context=get_market_context(ctx), extra_notes=extra_notes + ("\n" if extra_notes else ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_agent(ctx: ToolContext, model: ChatModel, evaluate: Evaluate, *, max_tokens: int = 6000,
              extra_notes: str = "") -> AgentResult:
    tools = registry()
    knobs = ctx.strategy.llm
    native = model.supports_tools
    try:
        return _loop(ctx, model, evaluate, tools, native, knobs, max_tokens, extra_notes)
    except ToolsUnsupported:
        model.supports_tools = False  # type: ignore[misc]
        return _loop(ctx, model, evaluate, tools, False, knobs, max_tokens, extra_notes)


def _loop(ctx, model, evaluate, tools, native, knobs, max_tokens, extra_notes) -> AgentResult:
    res = AgentResult(status="failed", decision=None, provider=model.provider, model_id=model.model_id)
    messages = build_messages(ctx, tools, native, extra_notes=extra_notes)
    specs = openai_tool_specs(tools) if native else None
    tool_calls_used, nudges = 0, 0
    max_rounds = knobs.max_tool_calls + knobs.max_repair_rounds + MAX_NUDGES + 4

    for _ in range(max_rounds):
        res.rounds += 1
        t0 = now_taipei()
        try:
            out: ChatResult = model.complete(messages, specs, temperature=knobs.temperature,
                                             max_tokens=max_tokens)
        except ToolsUnsupported:
            raise
        except Exception as e:  # noqa: BLE001 - provider errors end this attempt
            res.log.append({"type": "llm", "started_at": t0.isoformat(), "status": "error",
                            "error": f"{type(e).__name__}: {e}"[:500], "seq": res.rounds})
            res.error = f"{type(e).__name__}: {e}"[:500]
            return res
        res.log.append({"type": "llm", "seq": res.rounds, "started_at": t0.isoformat(), "status": "ok",
                        "latency_ms": out.latency_ms, "prompt_tokens": out.prompt_tokens,
                        "completion_tokens": out.completion_tokens, "finish_reason": out.finish_reason,
                        "content": out.content, "reasoning": out.reasoning,
                        "tool_calls": [tc.__dict__ for tc in out.tool_calls]})

        calls: list[tuple[str, str, dict]] = []  # (call_id, name, args)
        if native and out.tool_calls:
            messages.append({"role": "assistant", "content": out.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.name, "arguments": tc.arguments}}
                                            for tc in out.tool_calls]})
            calls = [(tc.id, tc.name, _parse_args(tc.arguments)) for tc in out.tool_calls]
        else:
            obj = extract_json(out.content or "")
            messages.append({"role": "assistant", "content": out.content or ""})
            if obj and "tool" in obj:
                calls = [("json", str(obj["tool"]), obj.get("arguments") or {})]
            elif obj and "positions" in obj:
                calls = [("json", "submit_decision", obj)]
            else:
                nudges += 1
                if nudges > MAX_NUDGES:
                    res.error = "model did not call submit_decision"
                    return res
                messages.append({"role": "user", "content": "請使用工具繼續，完成後呼叫 submit_decision 提交決策。"})
                continue

        for call_id, name, args in calls:
            if name == "submit_decision":
                reply, done = _handle_submit(ctx, args, evaluate, res, knobs.max_repair_rounds)
            elif name in tools:
                tool_calls_used += 1
                if tool_calls_used > knobs.max_tool_calls:
                    reply, done = "error: tool budget exhausted; call submit_decision now.", False
                else:
                    t = time.monotonic()
                    reply = run_tool(ctx, tools[name], args)
                    res.log.append({"type": "tool", "name": name, "args": args, "result": reply,
                                    "ms": int((time.monotonic() - t) * 1000)})
                    done = False
            else:
                reply, done = f"error: unknown tool {name!r}", False
            if done:
                return res
            if native and call_id != "json":
                messages.append({"role": "tool", "tool_call_id": call_id, "content": reply})
            else:
                messages.append({"role": "user", "content": f"TOOL_RESULT {name}:\n{reply}"})
            if res.status == "rejected":
                return res
    res.error = res.error or "round limit reached without an accepted decision"
    return res


def _handle_submit(ctx: ToolContext, args: dict, evaluate: Evaluate, res: AgentResult,
                   max_repairs: int) -> tuple[str, bool]:
    """Returns (reply to the model, finished?)."""
    if "decision" in args and isinstance(args["decision"], dict) and "positions" not in args:
        args = args["decision"]
    # Some models send nested arrays as JSON-encoded strings; decode them instead of bouncing.
    for key in ("positions", "exits"):
        if isinstance(args.get(key), str):
            try:
                args = {**args, key: json.loads(args[key])}
            except json.JSONDecodeError:
                pass
    try:
        d = Decision.model_validate(args)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:10])
        res.log.append({"type": "submit", "accepted": False, "stage": "schema", "detail": msg})
        return f"error: decision does not match the schema: {msg}", False
    problems = check_completeness(d, ctx)
    if problems:
        res.log.append({"type": "submit", "accepted": False, "stage": "completeness", "detail": problems})
        return "error: " + "; ".join(problems), False
    ok, feedback = evaluate(d)
    res.decision = d
    res.last_feedback = feedback
    res.log.append({"type": "submit", "accepted": ok, "stage": "rules", "detail": feedback,
                    "decision": d.model_dump(mode="json")})
    if ok:
        res.status = "accepted"
        return "accepted", True
    if res.repair_rounds >= max_repairs:
        res.status = "rejected"
        return "rejected", True
    res.repair_rounds += 1
    return (f"系統驗證未通過（第 {res.repair_rounds}/{max_repairs} 次修正機會）：{feedback}\n"
            "請修正後再次呼叫 submit_decision。"), False


def trade_date_line(d: date) -> str:
    return f"{d} ({WEEKDAYS[d.weekday()]})"
