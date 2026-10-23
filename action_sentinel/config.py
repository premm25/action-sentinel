"""Settings: one YAML file plus environment overrides. Secrets come only from the environment."""

from __future__ import annotations

import hashlib
import os
import secrets
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from action_sentinel.contracts import ApprovalLevel, Reviewer
from action_sentinel.contracts.common import StrictModel
from action_sentinel.governance.policy import GovernancePolicy

_SOURCE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "sentinel.yaml"


def default_config_path() -> Path:
    """$SENTINEL_CONFIG, else ./config/sentinel.yaml (run from the project root), else the source checkout's."""
    if os.environ.get("SENTINEL_CONFIG"):
        return Path(os.environ["SENTINEL_CONFIG"])
    cwd = Path.cwd() / "config" / "sentinel.yaml"
    return cwd if cwd.is_file() else _SOURCE_CONFIG


class Principal(StrictModel):
    """An API caller. Keys are stored only as SHA-256 hashes."""

    principal_id: str
    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roles: list[Literal["agent", "reviewer", "auditor", "admin", "evaluator"]]
    reviewer_id: str | None = None


class VegaLabSettings(StrictModel):
    """The VegaML decision service. Off by default: enabling it loads a ~3.5 GB model (0.8B on CPU)."""

    enabled: bool = False
    model: str = "0.8b"
    device: str | None = "cpu"
    load: Literal["lazy", "eager"] = "lazy"
    default_adapter_mode: Literal["off", "auto", "on"] = "auto"
    timeout_s: float = Field(default=300.0, gt=0)


class Settings(StrictModel):
    environment: Literal["dev", "test", "prod"] = "dev"
    db_path: str = "var/sentinel.db"
    rule_packs: list[str] = Field(default_factory=lambda: ["examples/rule_packs/core.yaml"])
    providers: list[dict[str, Any]] = Field(default_factory=list)
    active_providers: list[str] = Field(default_factory=list)
    strategy: Literal["single", "sequential", "parallel"] = "parallel"
    provider_timeout_s: float = Field(default=30.0, gt=0)
    governance: GovernancePolicy = Field(default_factory=GovernancePolicy)
    accepted_policy_versions: list[str] = Field(default_factory=list)  # extra versions still honoured
    calibration_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    reviewers: list[Reviewer] = Field(default_factory=list)
    principals: list[Principal] = Field(default_factory=list)
    report_dir: str = "reports"
    vegalab: VegaLabSettings = Field(default_factory=VegaLabSettings)
    # Load model providers (and fit or load VegaML heads) when the API starts, not on the first request.
    warm_up: bool = False
    base_dir: str = "."

    @model_validator(mode="after")
    def _unique_reviewers(self) -> "Settings":
        ids = [r.reviewer_id for r in self.reviewers]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate reviewer ids")
        return self

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() or p == ":memory:" else Path(self.base_dir) / q

    def reviewer(self, reviewer_id: str) -> Reviewer | None:
        return next((r for r in self.reviewers if r.reviewer_id == reviewer_id), None)


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    p = Path(path) if path else default_config_path()
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.is_file() else {}
    raw.setdefault("base_dir", str(p.parent.parent if p.parent.name == "config" else p.parent))
    if os.environ.get("SENTINEL_DB_PATH"):
        raw["db_path"] = os.environ["SENTINEL_DB_PATH"]
    if os.environ.get("SENTINEL_ENV"):
        raw["environment"] = os.environ["SENTINEL_ENV"]
    raw.update(overrides)
    return Settings.model_validate(raw)


def signing_key(settings: Settings) -> bytes:
    """HMAC key for authorization tokens. Required in prod; an ephemeral random key is used otherwise."""
    env = os.environ.get("SENTINEL_SIGNING_KEY")
    if env:
        if len(env) < 32:
            raise ValueError("SENTINEL_SIGNING_KEY must be at least 32 characters")
        return env.encode("utf-8")
    if settings.environment == "prod":
        raise RuntimeError("SENTINEL_SIGNING_KEY must be set when environment is prod")
    warnings.warn("SENTINEL_SIGNING_KEY not set: using an ephemeral key; tokens die with this process",
                  stacklevel=2)
    return secrets.token_bytes(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


__all__ = ["Settings", "Principal", "load_settings", "signing_key", "hash_key", "ApprovalLevel"]
