"""FastAPI router for the Vega decision service: /vega/health, /vega/ready, /vega/model, /vega/decide."""

import time
import uuid
from typing import Annotated, Any, Callable, Literal

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .contracts import DecisionRequest, VegaInferenceError
from .service import VegaDecisionService

STATUS = {"INVALID_INPUT": 422, "INPUT_TOO_LONG": 413, "MODEL_NOT_READY": 503, "MODEL_LOAD_FAILED": 503,
          "TIMEOUT": 504, "INFERENCE_FAILED": 500, "MALFORMED_NATIVE_OUTPUT": 502}


class DecideBody(BaseModel):
    request: dict[str, Any]   # validated by VegaCore so bad input gets a typed INVALID_INPUT error
    adapter_mode: Literal["off", "auto", "on"] | None = None


def create_vega_router(service: VegaDecisionService, read_auth: Callable, decide_auth: Callable) -> APIRouter:
    r = APIRouter(prefix="/vega", tags=["vega"])
    Reader = Annotated[Any, Depends(read_auth)]
    Decider = Annotated[Any, Depends(decide_auth)]

    @r.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_state": service.state}

    @r.get("/ready")
    def ready() -> JSONResponse:
        body = service.ready()
        return JSONResponse(status_code=200 if body["ready"] else 503, content=body)

    @r.get("/model")
    def model(p: Reader) -> dict[str, Any]:
        return service.info()

    @r.post("/decide")
    def decide(body: DecideBody, p: Decider,
               x_request_id: Annotated[str | None, Header()] = None) -> JSONResponse:
        req = dict(body.request)
        req.setdefault("request_id", (x_request_id or "vreq_%s" % uuid.uuid4().hex)[:128])
        t0 = time.perf_counter()
        try:
            res = service.decide(req, body.adapter_mode)
        except VegaInferenceError as exc:
            return JSONResponse(status_code=STATUS.get(exc.error.code, 500),
                                headers={"X-Request-ID": str(req["request_id"])},
                                content={"error": exc.error.model_dump(mode="json")})
        return JSONResponse(content={"result": res.model_dump(mode="json"),
                                     "server_ms": round(1000 * (time.perf_counter() - t0), 2)},
                            headers={"X-Request-ID": res.request_id})

    return r


__all__ = ["create_vega_router", "DecideBody", "DecisionRequest"]
