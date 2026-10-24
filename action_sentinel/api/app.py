"""Thin FastAPI layer over the Sentinel service.

Authentication: `X-API-Key` header, compared by SHA-256 hash against configured principals.
Authorization: role checks per endpoint; reviewer identity is taken from the authenticated principal,
never from the request body; agents may only submit actions under their own agent_id.
"""


import hmac
import json
import warnings
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from action_sentinel import __version__
from action_sentinel.config import Principal, Settings, hash_key
from action_sentinel.contracts import ActionProposal, DecisionQuestion, EvaluationRunConfig, GovernanceDecision, new_id
from action_sentinel.evaluation import ConfidenceLab, DatasetError, LabError, render_comparison
from action_sentinel.execution import ExecutionStateError
from action_sentinel.governance import ReviewError
from action_sentinel.providers import ProviderConfigError
from action_sentinel.service import DuplicateEventConflict, Sentinel


class EvaluateRequest(BaseModel):
    action: ActionProposal
    providers: list[str] | None = None
    strategy: Literal["single", "sequential", "parallel"] | None = None
    question: DecisionQuestion | None = None


class ReasonBody(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class OverrideBody(BaseModel):
    reason: str = Field(max_length=2000)
    changes: dict[str, Any]


class ExecuteBody(BaseModel):
    token: str = Field(max_length=200)
    action: ActionProposal


class BeginBody(ExecuteBody):
    adapter: str = Field(default="remote", max_length=64)


class FinishBody(BaseModel):
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)


class ShadowBody(BaseModel):
    """What a shadow-mode guard recorded about one call (it ran the tool regardless)."""

    record: dict[str, Any]
    action_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RevokeBody(BaseModel):
    reason: str = Field(min_length=5, max_length=2000)


def decision_view(d: GovernanceDecision) -> dict[str, Any]:
    return d.model_dump(mode="json")


def create_app(settings: Settings, service: Sentinel | None = None, vega_service=None) -> FastAPI:
    if service is None:
        with warnings.catch_warnings():
            if settings.environment != "prod":
                warnings.simplefilter("ignore", UserWarning)
            service = Sentinel(settings)
        if settings.warm_up:
            for name, detail in service.warm_up().items():
                print("warm-up %s: %s" % (name, detail), flush=True)
    svc = service
    lab = ConfidenceLab(settings, registry=None)
    app = FastAPI(title="Sentinel", version=__version__,
                  description="Decision governance for AI agent actions. Local development build.")
    app.state.sentinel = svc

    # ------------------------------------------------------------------ auth
    def principal(x_api_key: Annotated[str | None, Header()] = None) -> Principal:
        if not x_api_key:
            raise HTTPException(401, "missing X-API-Key")
        digest = hash_key(x_api_key)
        for p in settings.principals:
            if hmac.compare_digest(p.key_sha256, digest):
                return p
        raise HTTPException(401, "invalid API key")

    def require(*roles: str):
        def dep(p: Annotated[Principal, Depends(principal)]) -> Principal:
            if "admin" in p.roles or any(r in p.roles for r in roles):
                return p
            raise HTTPException(403, "requires one of roles %s" % list(roles))
        return dep

    Auth = Annotated[Principal, Depends(principal)]
    Agent = Annotated[Principal, Depends(require("agent"))]
    Reviewer = Annotated[Principal, Depends(require("reviewer"))]
    Auditor = Annotated[Principal, Depends(require("auditor"))]
    Evaluator = Annotated[Principal, Depends(require("evaluator"))]
    Admin = Annotated[Principal, Depends(require("admin"))]
    Limit = Annotated[int, Query(ge=1, le=500)]
    Offset = Annotated[int, Query(ge=0)]

    # ------------------------------------------------------------------ errors
    @app.exception_handler(ReviewError)
    async def _review_error(request: Request, exc: ReviewError):
        return JSONResponse(status_code=exc.http_status, content={"error": exc.code, "detail": str(exc)})

    @app.exception_handler(DuplicateEventConflict)
    async def _conflict(request: Request, exc: DuplicateEventConflict):
        return JSONResponse(status_code=409, content={"error": "EVENT_CONFLICT", "detail": str(exc)})

    for exc_type in (LabError, DatasetError, ProviderConfigError):
        @app.exception_handler(exc_type)
        async def _bad_request(request: Request, exc: Exception):
            return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.exception_handler(ExecutionStateError)
    async def _execution_state(request: Request, exc: ExecutionStateError):
        return JSONResponse(status_code=404 if exc.code == "NOT_FOUND" else 409,
                            content={"error": exc.code, "detail": str(exc)})

    def own(action: ActionProposal, p: Principal, what: str) -> None:
        if "admin" not in p.roles and action.agent_id != p.principal_id:
            raise HTTPException(403, "agents may only %s their own actions" % what)

    def reviewer_id(p: Principal) -> str:
        if not p.reviewer_id:
            raise HTTPException(403, "principal is not linked to a reviewer identity")
        return p.reviewer_id

    # ------------------------------------------------------------------ endpoints
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "policy_version": svc.policy_version,
                "environment": settings.environment}

    @app.get("/providers")
    def providers(p: Auth) -> dict[str, Any]:
        return {"active": settings.active_providers, "strategy": settings.strategy,
                "providers": [h.model_dump() for h in svc.provider_health()]}

    @app.post("/decisions/evaluate")
    def evaluate(body: EvaluateRequest, p: Agent) -> dict[str, Any]:
        own(body.action, p, "submit")
        out = svc.evaluate(body.action, question=body.question, providers=body.providers, strategy=body.strategy,
                           actor=p.principal_id)
        return {"decision": decision_view(out.decision), "duplicate": out.duplicate,
                "review_id": out.review.review_id if out.review else None,
                "authorization_token": out.authorization.token if out.authorization else None}

    @app.post("/decisions/assess")
    def assess(body: EvaluateRequest, p: Agent) -> dict[str, Any]:
        """What Sentinel would decide, for shadow mode: audited, but no decision, review or grant is stored."""
        own(body.action, p, "assess")
        corr = new_id("shadow")
        d = svc.assess(body.action, correlation_id=corr, question=body.question, providers=body.providers,
                       strategy=body.strategy)
        return {"decision": decision_view(d), "correlation_id": corr}

    @app.post("/shadow-records")
    def shadow_record(body: ShadowBody, p: Agent) -> dict[str, Any]:
        if len(json.dumps(body.record, default=str)) > 20_000:
            raise HTTPException(413, "shadow record larger than 20 kB")
        corr = str(body.record.get("correlation_id") or new_id("shadow"))[:100]
        codes = body.record.get("reason_codes")
        ev = svc.audit.emit("shadow.decision", corr, "guard", actor=p.principal_id,
                            reason_codes=[str(c) for c in codes] if isinstance(codes, list) else [],
                            data={**body.record, "action_hash": body.action_hash})
        return {"event_id": ev.event_id}

    @app.get("/decisions")
    def list_decisions(p: Auditor, limit: Limit = 50, offset: Offset = 0,
                       status: str | None = None) -> dict[str, Any]:
        items = svc.store.list_decisions(limit=limit, offset=offset, status=status)
        return {"items": [decision_view(d) for d in items], "limit": limit, "offset": offset}

    @app.get("/decisions/{decision_id}")
    def get_decision(decision_id: str, p: Auth) -> dict[str, Any]:
        d = svc.store.get_decision(decision_id)
        if d is None:
            raise HTTPException(404, "no such decision")
        if not ({"auditor", "admin", "reviewer"} & set(p.roles)) and d.action.agent_id != p.principal_id:
            raise HTTPException(403, "not permitted to read this decision")
        return decision_view(d)

    @app.get("/decisions/{decision_id}/events")
    def decision_events(decision_id: str, p: Auditor, limit: Limit = 500, offset: Offset = 0) -> dict[str, Any]:
        d = svc.store.get_decision(decision_id)
        if d is None:
            raise HTTPException(404, "no such decision")
        events = svc.store.list_events(correlation_id=d.correlation_id, limit=limit, offset=offset)
        return {"items": [e.model_dump(mode="json") for e in events], "limit": limit, "offset": offset}

    @app.get("/reviews/pending")
    def pending(p: Reviewer, limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
        items = svc.store.list_reviews(state="PENDING", limit=limit, offset=offset)
        return {"items": [r.model_dump(mode="json") for r in items], "limit": limit, "offset": offset}

    @app.get("/reviews/{review_id}")
    def get_review(review_id: str, p: Reviewer) -> dict[str, Any]:
        r = svc.store.get_review(review_id)
        if r is None:
            raise HTTPException(404, "no such review")
        return r.model_dump(mode="json")

    @app.post("/reviews/{review_id}/approve")
    def approve(review_id: str, body: ReasonBody, p: Reviewer) -> dict[str, Any]:
        out = svc.approve_review(review_id, reviewer_id(p), body.reason)
        return {"review": out.review.model_dump(mode="json"), "authorization_token": out.authorization.token}

    @app.post("/reviews/{review_id}/reject")
    def reject(review_id: str, body: ReasonBody, p: Reviewer) -> dict[str, Any]:
        return {"review": svc.reject_review(review_id, reviewer_id(p), body.reason or "").model_dump(mode="json")}

    @app.post("/reviews/{review_id}/override")
    def override(review_id: str, body: OverrideBody, p: Reviewer) -> dict[str, Any]:
        out = svc.override_review(review_id, reviewer_id(p), body.reason, body.changes)
        return {"review": out.review.model_dump(mode="json"), "decision": decision_view(out.decision),
                "authorization_token": out.authorization.token}

    @app.post("/executions")
    def execute(body: ExecuteBody, p: Agent) -> dict[str, Any]:
        own(body.action, p, "execute")
        r = svc.execute(body.token, body.action, actor=p.principal_id)
        return r.model_dump(mode="json")

    @app.post("/executions/begin")
    def begin_execution(body: BeginBody, p: Agent) -> dict[str, Any]:
        """For an agent that runs the action itself: every gateway check; STARTED means it may run it now."""
        own(body.action, p, "execute")
        return svc.begin_execution(body.token, body.action, actor=p.principal_id,
                                   adapter=body.adapter).model_dump(mode="json")

    @app.post("/executions/{execution_id}/finish")
    def finish_execution(execution_id: str, body: FinishBody, p: Agent) -> dict[str, Any]:
        e = svc.store.get_execution(execution_id)
        if e is None:
            raise HTTPException(404, "no such execution")
        if "admin" not in p.roles and e.started_by != p.principal_id:
            raise HTTPException(403, "only the agent that started an execution may finish it")
        return svc.finish_execution(execution_id, body.ok, body.output, actor=p.principal_id).model_dump(mode="json")

    @app.post("/authorizations/{grant_id}/revoke")
    def revoke(grant_id: str, body: RevokeBody, p: Admin) -> dict[str, Any]:
        from action_sentinel.execution import AuthorizationError
        try:
            return svc.revoke_authorization(grant_id, body.reason, actor=p.principal_id).model_dump(mode="json")
        except AuthorizationError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/evaluations/run")
    def run_evaluation(cfg: EvaluationRunConfig, p: Evaluator) -> dict[str, Any]:
        report = lab.run(cfg)
        svc.store.save_evaluation(report)
        return {"run_id": report.run_id, "n_examples": report.n_examples,
                "governance_metrics": report.governance_metrics, "notes": report.notes}

    @app.get("/evaluations")
    def list_evaluations(p: Auditor, limit: Limit = 50) -> dict[str, Any]:
        return {"items": svc.store.list_evaluations(limit)}

    @app.get("/evaluations/{run_id}")
    def get_evaluation(run_id: str, p: Auditor) -> dict[str, Any]:
        r = svc.store.get_evaluation(run_id)
        if r is None:
            raise HTTPException(404, "no such evaluation run")
        return r.model_dump(mode="json")

    @app.get("/evaluations/{run_id}/report", response_class=PlainTextResponse)
    def evaluation_report(run_id: str, p: Auditor) -> str:
        r = svc.store.get_evaluation(run_id)
        if r is None:
            raise HTTPException(404, "no such evaluation run")
        return render_comparison([r], "Confidence Lab run %s" % run_id)

    # ------------------------------------------------------------------ Vega decision service (optional)
    vsvc = vega_service
    if vsvc is None and settings.vegalab.enabled:
        from action_sentinel.vegalab.service import VegaDecisionService
        from action_sentinel.vegalab.vega_core import LiveVegaRunner
        vl = settings.vegalab
        vsvc = VegaDecisionService(LiveVegaRunner(vl.model, device=vl.device), model_id="vegaml-%s" % vl.model,
                                   default_adapter_mode=vl.default_adapter_mode, lazy_load=vl.load == "lazy",
                                   timeout_s=vl.timeout_s)
        if vl.load == "eager":
            vsvc.load()
    if vsvc is not None:
        from action_sentinel.vegalab.api import create_vega_router
        app.include_router(create_vega_router(vsvc, principal, require("agent", "evaluator")))
        app.state.vega = vsvc

    return app


def app_from_env() -> FastAPI:
    """Entry point for `uvicorn action_sentinel.api.app:app_from_env --factory`."""
    from action_sentinel.config import load_settings
    return create_app(load_settings())
