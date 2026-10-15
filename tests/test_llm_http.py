"""The real OpenAI-compatible client against a local mock server (no network, no real key):
verifies request shape, tool-call round trips and response parsing end to end."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import SecretStr

from stockmaster.agent.llm_client import OpenAICompatModel
from stockmaster.pipeline.decide import decide
from stockmaster.settings import Settings

from .test_pipeline import ctx_factory, decision_from  # noqa: F401  (fixture re-export)


class MockLLM(BaseHTTPRequestHandler):
    decision: dict = {}
    requests: list = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockLLM.requests.append({"auth": self.headers.get("Authorization"), "body": body})
        msgs = body["messages"]
        if msgs[-1]["role"] == "user":
            call = {"id": "call_1", "type": "function",
                    "function": {"name": "get_market_context", "arguments": "{}"}}
        else:
            call = {"id": "call_2", "type": "function",
                    "function": {"name": "submit_decision", "arguments": json.dumps(MockLLM.decision)}}
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None, "tool_calls": [call],
                                         "reasoning_content": "thinking..."}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), MockLLM)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_compat_model_end_to_end(ctx_factory, mock_server):  # noqa: F811
    ctx = ctx_factory()
    MockLLM.decision = decision_from(ctx.baselines["normal"], 13.0)
    MockLLM.requests = []
    settings = Settings(nvidia_api_key=SecretStr("test-key"), nvidia_base_url=mock_server)
    model = OpenAICompatModel("nvidia", "mock/model", settings)
    out = decide(ctx, [model], run_id="http", account_id="paper", strategy_version="t", input_sha256="x")
    assert out.final.decision_source == "llm"
    assert out.final.provenance.model_id == "mock/model"
    first, second = MockLLM.requests[0]["body"], MockLLM.requests[1]["body"]
    assert MockLLM.requests[0]["auth"] == "Bearer test-key"
    assert {t["function"]["name"] for t in first["tools"]} >= {"submit_decision", "check_portfolio"}
    # tool result was returned with the matching tool_call_id
    tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
    log = out.attempts[0].log
    assert any(e.get("reasoning") == "thinking..." for e in log if e["type"] == "llm")
