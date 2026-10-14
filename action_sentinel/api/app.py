"""Thin FastAPI layer over the Sentinel service.

Authentication: `X-API-Key` header, compared by SHA-256 hash against configured principals.
Authorization: role checks per endpoint; reviewer identity is taken from the authenticated principal,
never from the request body; agents may only submit actions under their own agent_id.
"""


import hmac
import warnings
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from action_sentinel import __version__
from action_sentinel.config import Principal, Settings, hash_key
from action_sentinel.contracts import ActionProposal, EvaluationRunConfig, GovernanceDecision
from action_sentinel.evaluation import ConfidenceLab, DatasetError, LabError, render_comparison
from action_sentinel.governance import ReviewError
from action_sentinel.providers import ProviderConfigError
from action_sentinel.service import DuplicateEventConflict, Sentinel


class EvaluateRequest(BaseModel):
    action: ActionProposal
    providers: list[str] | None = None
    strategy: Literal["single", "sequential", "parallel"] | None = None


class ReasonBody(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class OverrideBody(BaseModel):
    reason: str = Field(max_length=2000)
    changes: dict[str, Any]


class ExecuteBody(BaseModel):
    token: str = Field(max_length=200)
    action: ActionProposal


class RevokeBody(BaseModel):
    reason: str = Field(min_length=5, max_length=2000)


def decision_view(d: GovernanceDecision) -> dict[str, Any]:
    return d.model_dump(mode="json")


def create_app(settings: Settings, service: Sentinel | None = None) -> FastAPI:
    if service is None:
        with warnings.catch_warnings():
            if settings.environment != "prod":
                warnings.simplefilter("ignore", UserWarning)
            service = Sentinel(settings)
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
        if "admin" not in p.roles and body.action.agent_id != p.principal_id:
            raise HTTPException(403, "agents may only submit their own actions")
        out = svc.evaluate(body.action, providers=body.providers, strategy=body.strategy, actor=p.principal_id)
        return {"decision": decision_view(out.decision), "duplicate": out.duplicate,
                "review_id": out.review.review_id if out.review else None,
                "authorization_token": out.authorization.token if out.authorization else None}

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
        if "admin" not in p.roles and body.action.agent_id != p.principal_id:
            raise HTTPException(403, "agents may only execute their own actions")
        r = svc.execute(body.token, body.action, actor=p.principal_id)
        return r.model_dump(mode="json")

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

    return app


def app_from_env() -> FastAPI:
    """Entry point for `uvicorn action_sentinel.api.app:app_from_env --factory`."""
    from action_sentinel.config import load_settings
    return create_app(load_settings())
