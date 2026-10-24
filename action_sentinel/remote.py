"""A Sentinel running as a separate service, reached over its HTTP API.

    from action_sentinel.remote import RemoteSentinel
    sentinel = RemoteSentinel("http://sentinel.internal:8000", api_key=os.environ["SENTINEL_API_KEY"])
    guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.4.0", source_system="infra-agent")

`Guard` accepts one in place of an in-process `Sentinel`. The API key's principal must have the agent role
and the same id as the guard's `agent_id`, because agents may only submit their own actions. Standard
library only (urllib); pass `transport` to route requests elsewhere (the tests use FastAPI's TestClient).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from action_sentinel.contracts import ActionProposal, DecisionQuestion, ExecutionResult, GovernanceDecision

# (method, path, JSON body or None, headers) -> (HTTP status, response body)
Transport = Callable[[str, str, bytes | None, dict[str, str]], tuple[int, bytes]]


class RemoteError(Exception):
    """The Sentinel API answered with an error, or could not be reached (status 0)."""

    def __init__(self, status: int, code: str, detail: str):
        super().__init__("Sentinel API %s %s: %s" % (status, code, detail))
        self.status, self.code, self.detail = status, code, detail


@dataclass
class RemoteEvaluation:
    decision: GovernanceDecision
    review_id: str | None
    authorization_token: str | None
    duplicate: bool = False


class RemoteSentinel:
    def __init__(self, base_url: str, api_key: str, *, timeout_s: float = 300.0,
                 transport: Transport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s  # a cold VegaML provider can take minutes on CPU; warm up the server first
        self._transport = transport or self._urllib

    # ---------------------------------------------------------------- calls
    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def evaluate(self, action: ActionProposal, *, question: DecisionQuestion | None = None,
                 providers: list[str] | None = None) -> RemoteEvaluation:
        out = self._call("POST", "/decisions/evaluate", _request(action, question, providers))
        return RemoteEvaluation(_decision(out["decision"]), out.get("review_id"), out.get("authorization_token"),
                                bool(out.get("duplicate")))

    def assess(self, action: ActionProposal, *, question: DecisionQuestion | None = None,
               providers: list[str] | None = None) -> GovernanceDecision:
        """What Sentinel would decide, storing nothing but audit events (shadow mode)."""
        return _decision(self._call("POST", "/decisions/assess", _request(action, question, providers))["decision"])

    def begin_execution(self, token: str, action: ActionProposal, adapter: str = "remote") -> ExecutionResult:
        out = self._call("POST", "/executions/begin",
                         {"token": token, "action": action.model_dump(mode="json"), "adapter": adapter})
        return ExecutionResult.model_validate_json(json.dumps(out))

    def finish_execution(self, execution_id: str, ok: bool, output: dict[str, Any] | None = None) -> ExecutionResult:
        out = self._call("POST", "/executions/%s/finish" % execution_id, {"ok": ok, "output": output or {}})
        return ExecutionResult.model_validate_json(json.dumps(out))

    def record_shadow(self, record: dict[str, Any], action_hash: str) -> None:
        self._call("POST", "/shadow-records", {"record": record, "action_hash": action_hash})

    # ---------------------------------------------------------------- transport
    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, raw = self._transport(method, path, data, headers)
        try:
            out = json.loads(raw or b"{}")
        except ValueError:
            out = {"detail": raw[:500].decode("utf-8", "replace")}
        if status >= 400:
            detail = out.get("detail") if isinstance(out, dict) else out
            code = out.get("error", "HTTP_%d" % status) if isinstance(out, dict) else "HTTP_%d" % status
            raise RemoteError(status, code, json.dumps(detail)[:1000] if not isinstance(detail, str) else detail)
        return out

    def _urllib(self, method: str, path: str, data: bytes | None, headers: dict[str, str]) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 (the operator's URL)
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RemoteError(0, "UNREACHABLE", "%s: %s" % (self.base_url, exc)) from exc


def _request(action: ActionProposal, question: DecisionQuestion | None, providers: list[str] | None) -> dict:
    body: dict[str, Any] = {"action": action.model_dump(mode="json")}
    if question is not None:
        body["question"] = question.model_dump(mode="json")
    if providers is not None:
        body["providers"] = providers
    return body


def _decision(d: dict[str, Any]) -> GovernanceDecision:
    return GovernanceDecision.model_validate_json(json.dumps(d))


__all__ = ["RemoteSentinel", "RemoteEvaluation", "RemoteError"]
