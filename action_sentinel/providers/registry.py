"""Provider registry: build providers from configuration, look them up, report health."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from action_sentinel.contracts import ProviderHealth
from action_sentinel.contracts.common import StrictModel

from .base import DecisionProvider
from .mock import MockDecisionProvider
from .ollama import OllamaDecisionProvider
from .rule_based import RuleBasedDecisionProvider
from .vegaml_provider import VegaMLDecisionProvider


class ProviderSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    type: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class RuleBasedConfig(StrictModel):
    classifier_path: str


class MockConfig(StrictModel):
    script: dict[str, dict[str, Any]] = Field(default_factory=dict)
    default: dict[str, Any] = Field(default_factory=lambda: {"abstain": True})


class OllamaConfig(StrictModel):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    timeout_s: float = Field(default=120.0, gt=0, le=3600)
    max_retries: int = Field(default=1, ge=0, le=5)


class VegaMLConfig(StrictModel):
    checkpoint: str = "0.8b"
    device: Literal["cpu", "cuda", "mps"] | None = "cpu"
    adapter_mode: Literal["off", "auto", "on"] = "auto"
    framing: Literal["rich", "compact"] = "rich"
    readout: Literal["engine", "ttt"] = "engine"
    ttt_fit_dataset: str | None = None
    ttt_fit_split: Literal["dev", "calibration"] = "calibration"   # never "test"


class ProviderConfigError(ValueError):
    pass


Factory = Callable[[ProviderSpec, Path], DecisionProvider]


def _rule_based(spec: ProviderSpec, base: Path) -> DecisionProvider:
    cfg = RuleBasedConfig.model_validate(spec.config)
    path = Path(cfg.classifier_path)
    return RuleBasedDecisionProvider.from_file(spec.name, path if path.is_absolute() else base / path)


def _mock(spec: ProviderSpec, base: Path) -> DecisionProvider:
    cfg = MockConfig.model_validate(spec.config)
    return MockDecisionProvider(spec.name, script=cfg.script, default=cfg.default)


def _ollama(spec: ProviderSpec, base: Path) -> DecisionProvider:
    cfg = OllamaConfig.model_validate(spec.config)
    return OllamaDecisionProvider(spec.name, model=cfg.model, base_url=cfg.base_url, timeout_s=cfg.timeout_s,
                                  max_retries=cfg.max_retries)


def _vegaml(spec: ProviderSpec, base: Path) -> DecisionProvider:
    cfg = VegaMLConfig.model_validate(spec.config)
    ds = cfg.ttt_fit_dataset
    if ds and not Path(ds).is_absolute():
        ds = str(base / ds)
    return VegaMLDecisionProvider(spec.name, checkpoint=cfg.checkpoint, device=cfg.device,
                                  adapter_mode=cfg.adapter_mode, framing=cfg.framing, readout=cfg.readout,
                                  ttt_fit_dataset=ds, ttt_fit_split=cfg.ttt_fit_split)


FACTORIES: dict[str, Factory] = {"rule_based": _rule_based, "mock": _mock, "ollama": _ollama, "vegaml": _vegaml}


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, DecisionProvider] = {}
        self._factories: dict[str, Factory] = dict(FACTORIES)

    def register_type(self, type_name: str, factory: Factory) -> None:
        """Add a new provider type without touching the governance engine."""
        self._factories[type_name] = factory

    def register(self, provider: DecisionProvider, replace: bool = False) -> None:
        if provider.name in self._providers and not replace:
            raise ProviderConfigError("provider %r is already registered" % provider.name)
        self._providers[provider.name] = provider

    def build(self, specs: list[dict[str, Any] | ProviderSpec], base_dir: Path) -> list[str]:
        """Build and register enabled providers. Returns the names registered."""
        names = []
        for raw in specs:
            try:
                spec = raw if isinstance(raw, ProviderSpec) else ProviderSpec.model_validate(raw)
            except ValidationError as exc:
                raise ProviderConfigError("invalid provider spec: %s" % exc) from exc
            if spec.type not in self._factories:
                raise ProviderConfigError("unknown provider type %r for %r (known: %s)"
                                          % (spec.type, spec.name, ", ".join(sorted(self._factories))))
            if not spec.enabled:
                continue
            try:
                provider = self._factories[spec.type](spec, base_dir)
            except (ValidationError, OSError, ValueError) as exc:
                raise ProviderConfigError("cannot build provider %r: %s" % (spec.name, exc)) from exc
            self.register(provider)
            names.append(spec.name)
        return names

    def get(self, name: str) -> DecisionProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise ProviderConfigError("no provider named %r" % name) from None

    def names(self) -> list[str]:
        return list(self._providers)

    def health(self) -> list[ProviderHealth]:
        out = []
        for p in self._providers.values():
            try:
                out.append(p.health())
            except Exception as exc:  # health must never take the service down
                out.append(ProviderHealth(name=p.name, provider_type=p.provider_type, version=p.version,
                                          kind=p.kind, available=False, detail="health check failed: %s" % exc))
        return out
