"""Systems evaluated by the lab. Each maps one dataset example to a prediction dict.

* VegaSystem          the real VegaML engine (adapter mode "off" or "auto"), through VegaCore.
* BackboneOptionScorer the SAME frozen backbone VegaML reads (Qwen/Qwen3.5-0.8B for the 0.8b checkpoint) used
                       as a plain language model. The prompt token ids come from vegaml's own build_prompt with
                       the checkpoint's settings (max_len, opt_max_tokens, query_first_max_state), so the backbone
                       reads exactly the tokens VegaML's perception reads. Each option is then scored by the mean
                       per-token log-probability of " <option text>" as the continuation after "Outcome:".
                       softmax over those scores gives probabilities (temperature 1, no calibration). No decision
                       engine is involved.
* MajorityBaseline     predicts the most frequent gold label of the task's calibration+dev examples
                       (training-split data only); probabilities are the Laplace-smoothed label prior.

A prediction dict has: predicted_label, probabilities, confidence, and where the system provides them
abstain, prediction_set, unbound, settle_steps, adapter, native. Missing ones are absent (None).
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from typing import Any

from .contracts import AdapterMode, DecisionQuestion, DecisionRequest
from .vega_core import VegaCore


class VegaSystem:
    def __init__(self, core: VegaCore, adapter_mode: AdapterMode):
        self.core = core
        self.adapter_mode = adapter_mode
        self.name = "vega-%s" % adapter_mode

    def info(self) -> dict[str, Any]:
        return {"system": self.name, "kind": "vegaml", "adapter_mode": self.adapter_mode, "live": self.core.live,
                "runner": self.core.runner.info()}

    def predict(self, example: dict[str, Any]) -> dict[str, Any]:
        q = DecisionQuestion.model_validate(example["question"])
        res = self.core.decide(DecisionRequest(request_id=example["example_id"][:128], state=example["state"],
                                               questions=[q]), adapter_mode=self.adapter_mode)
        d = res.decisions[q.question_id]
        u = d.uncertainty
        return {"predicted_label": d.selected_label, "probabilities": u.probabilities, "confidence": u.confidence,
                "abstain": u.abstain, "prediction_set": u.prediction_set, "unbound": u.unbound,
                "settle_steps": u.settle_steps, "adapter": u.adapter, "native": res.native["answers"][q.question_id],
                "live": res.live_inference}


class MajorityBaseline:
    name = "majority"

    def __init__(self, fit_examples: list[dict[str, Any]]):
        self.prior: dict[str, dict[str, float]] = {}
        self.majority: dict[str, str] = {}
        by_task: dict[str, list[dict]] = {}
        for ex in fit_examples:
            by_task.setdefault(ex["task"], []).append(ex)
        for task, exs in by_task.items():
            labels = [o["label"] for o in exs[0]["question"]["options"]]
            c = Counter(e["gold_label"] for e in exs)
            n = len(exs)
            self.prior[task] = {lab: (c[lab] + 1) / (n + len(labels)) for lab in labels}
            self.majority[task] = max(labels, key=lambda lab: (c[lab], -labels.index(lab)))

    def info(self) -> dict[str, Any]:
        return {"system": self.name, "kind": "deterministic", "majority": self.majority}

    def predict(self, example: dict[str, Any]) -> dict[str, Any]:
        task = example["task"]
        labels = [o["label"] for o in example["question"]["options"]]
        if task not in self.prior:
            raise KeyError("no fit data for task %s" % task)
        prior = self.prior[task]
        if set(prior) != set(labels):  # option-count variants: renormalize over the options offered
            sub = {lab: prior.get(lab, 0.0) for lab in labels}
            z = sum(sub.values()) or 1.0
            probs = {k: v / z for k, v in sub.items()}
        else:
            probs = dict(prior)
        pred = max(labels, key=lambda lab: (probs[lab], -labels.index(lab)))
        return {"predicted_label": pred, "probabilities": probs, "confidence": probs[pred]}


class BackboneOptionScorer:
    name = "backbone-optscore"

    def __init__(self, checkpoint: str = "0.8b", device: str = "cpu", max_prompt_tokens: int = 8192):
        self.checkpoint = checkpoint
        self.backbone: str | None = None
        self.cfg: dict[str, Any] = {}
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens
        self.model: Any = None
        self.tok: Any = None
        self.cache_mode = "unknown"

    def load(self) -> None:
        import json
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vegaml.hub import fetch
        t0 = time.perf_counter()
        with open(os.path.join(fetch(self.checkpoint), "vega_config.json")) as f:
            self.cfg = json.load(f)
        self.backbone = self.cfg["backbone_name"]
        self.tok = AutoTokenizer.from_pretrained(self.backbone)
        self.model = AutoModelForCausalLM.from_pretrained(self.backbone, dtype=torch.float32,
                                                          attn_implementation="sdpa").to(self.device).eval()
        self.load_seconds = round(time.perf_counter() - t0, 2)

    def info(self) -> dict[str, Any]:
        import torch
        import transformers
        return {"system": self.name, "kind": "frozen_backbone_lm", "checkpoint": self.checkpoint,
                "backbone": self.backbone, "device": self.device, "dtype": "float32",
                "prompt": "vegaml._common.build_prompt (template 0, checkpoint settings)",
                "scoring": "mean token log-prob of ' <option text>' after the prompt; softmax, temperature 1",
                "torch": torch.__version__, "transformers": transformers.__version__, "cache_mode": self.cache_mode,
                "load_seconds": getattr(self, "load_seconds", None)}

    def prompt_ids_and_options(self, example: dict[str, Any]) -> tuple[list[int], list[str], list[str]]:
        """Exactly the prompt VegaML builds (same function, same settings), plus the option texts."""
        from vegaml._common import build_prompt, option_texts, serialize_state
        q = DecisionQuestion.model_validate(example["question"])
        internal = {"t": q.type, "ins": q.instructions, "crit": q.to_vega().get("criteria")}
        ids, _ = build_prompt(self.tok, serialize_state(example["state"]), internal, self.cfg["max_len"], template=0,
                              opt_max_tokens=self.cfg["opt_max_tokens"],
                              query_first_max_state=self.cfg.get("query_first_max_state", 0))
        return ids, option_texts(internal), q.labels

    def predict(self, example: dict[str, Any]) -> dict[str, Any]:
        import torch
        prompt_ids, texts, labels = self.prompt_ids_and_options(example)
        ids = torch.tensor([prompt_ids], device=self.device)
        if ids.shape[1] > self.max_prompt_tokens:
            raise ValueError("prompt of %d tokens exceeds max_prompt_tokens" % ids.shape[1])
        conts = [self.tok(" " + t, add_special_tokens=False)["input_ids"] for t in texts]
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=True)
            last = out.logits[0, -1].float()
            cache = out.past_key_values
            sums = self._batched_logprobs(ids, last, cache, conts)
        scores = [lp / len(c) for lp, c in zip(sums, conts, strict=True)]
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        z = sum(ex)
        probs = {lab: e / z for lab, e in zip(labels, ex, strict=True)}
        pred = max(labels, key=lambda lab: probs[lab])
        return {"predicted_label": pred, "probabilities": probs, "confidence": probs[pred],
                "native": {"mean_token_logprob": dict(zip(labels, scores, strict=True))}}

    def _batched_logprobs(self, ids, last_logits, cache, conts: list[list[int]], batch: int = 16) -> list[float]:
        """Summed log-prob of each continuation. Options are scored in batches on a repeated copy of the
        prompt cache; right padding sits after every real token, so it cannot influence scored positions.
        Falls back to one option at a time if the cache cannot be copied or repeated."""
        import torch
        if self.cache_mode != "full_sequence":
            try:
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
                res: list[float] = []
                for i in range(0, len(conts), batch):
                    chunk = conts[i:i + batch]
                    L = max(len(c) for c in chunk)
                    x = torch.tensor([c + [pad] * (L - len(c)) for c in chunk], device=self.device)
                    c = _repeat_cache(copy.deepcopy(cache), len(chunk))
                    o = self.model(input_ids=x, past_key_values=c, use_cache=True).logits.float()
                    for b, toks in enumerate(chunk):
                        logits = torch.cat([last_logits[None], o[b, :len(toks) - 1]], 0)
                        t = torch.tensor(toks, device=self.device)[:, None]
                        res.append(float(torch.log_softmax(logits, -1).gather(1, t).sum()))
                self.cache_mode = "kv_batched"
                return res
            except Exception:
                self.cache_mode = "full_sequence"
        out = []
        for toks in conts:
            out.append(self._continuation_logprob(ids, last_logits, cache, torch.tensor([toks], device=self.device)))
        return out

    def _continuation_logprob(self, ids, last_logits, cache, cont) -> float:
        import torch
        if self.cache_mode != "full_sequence":
            try:
                o = self.model(input_ids=cont, past_key_values=copy.deepcopy(cache), use_cache=True)
                logits = torch.cat([last_logits[None], o.logits[0, :-1].float()], 0)
                self.cache_mode = "kv_copy"
                return float(torch.log_softmax(logits, -1).gather(1, cont[0][:, None]).sum())
            except Exception:
                self.cache_mode = "full_sequence"   # cache cannot be copied for this architecture
        full = torch.cat([ids, cont], 1)
        logits = self.model(input_ids=full).logits[0, ids.shape[1] - 1:-1].float()
        return float(torch.log_softmax(logits, -1).gather(1, cont[0][:, None]).sum())


def _repeat_cache(cache: Any, n: int) -> Any:
    """Repeat every cached state n times along the batch dimension.

    transformers' DynamicCache.batch_repeat_interleave fails on Qwen3.5's LinearAttentionLayer (no such
    method in transformers 5.19), so tensors are repeated directly: attention keys/values and the
    linear-attention conv/recurrent states (held in dicts). Verified equal to one-at-a-time scoring in
    tests/integration/test_vega_live.py.
    """
    def rep(v: Any) -> Any:
        import torch
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
            return v.repeat_interleave(n, dim=0)
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        return v
    for layer in cache.layers:
        for name, value in list(vars(layer).items()):
            setattr(layer, name, rep(value))
    return cache
