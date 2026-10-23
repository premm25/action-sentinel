"""Saved VegaML few-shot heads: fit once with `action-sentinel vega fit`, load at startup.

A head reads VegaML's features of one exact prompt: the framing of the action, the question's wording and
its options, on one checkpoint. A head fitted for one of those and served with another would be scoring
features it never saw, so the file records all of them and `check_head` refuses any mismatch.

The file is a NumPy `.npz`: the head's arrays plus one `meta` entry of JSON. Nothing in it is executed
(it is loaded with allow_pickle=False).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "action-sentinel-vegaml-head/1"
ARRAYS = ("mu", "sd", "c", "P", "s", "W", "b")


class HeadFileError(ValueError):
    pass


def question_fingerprint(question_id: str, instructions: str, categories: dict[str, str]) -> str:
    payload = json.dumps({"id": question_id, "instructions": instructions, "categories": categories},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def head_arrays(head: Any) -> dict[str, Any]:
    """The fitted state of a vegaml TTTHead (or anything with the same attributes)."""
    import numpy as np
    if not getattr(head, "ready", False):
        raise HeadFileError("the head is not fitted")
    return {k: np.asarray(getattr(head, k)) for k in ARRAYS}


def build_head(labels: list[str], arrays: dict[str, Any], k: int = 128, l2: float = 1e-2) -> Any:
    """A vegaml TTTHead restored from saved arrays, ready to predict."""
    from vegaml.ttt import TTTHead
    head = TTTHead(labels, k=k, l2=l2)
    for name in ARRAYS:
        setattr(head, name, arrays[name])
    head.ready = True
    return head


def save_head(path: str | Path, arrays: dict[str, Any], meta: dict[str, Any]) -> Path:
    import numpy as np
    missing = [k for k in ARRAYS if k not in arrays]
    if missing:
        raise HeadFileError("head arrays missing: %s" % missing)
    for key in ("question_id", "question_sha256", "labels", "framing", "checkpoint"):
        if key not in meta:
            raise HeadFileError("head metadata needs %r" % key)
    meta = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(), **meta}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **{k: arrays[k] for k in ARRAYS})
    tmp.replace(path)
    return path


def load_head(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    path = Path(path)
    if not path.is_file():
        raise HeadFileError("no head file at %s; create one with `action-sentinel vega fit`" % path)
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in ARRAYS}
    except (KeyError, ValueError, OSError) as exc:
        raise HeadFileError("%s is not a saved head: %s" % (path, exc)) from exc
    if meta.get("format") != FORMAT:
        raise HeadFileError("%s has format %r, expected %r" % (path, meta.get("format"), FORMAT))
    return arrays, meta


def check_head(meta: dict[str, Any], *, framing: str, checkpoint: str, engine_sha256: str | None) -> None:
    """Refuse a head that was fitted for a different prompt framing, checkpoint or engine."""
    problems = []
    if meta.get("framing") != framing:
        problems.append("framing %r (this provider uses %r)" % (meta.get("framing"), framing))
    if meta.get("checkpoint") != checkpoint:
        problems.append("checkpoint %r (this provider uses %r)" % (meta.get("checkpoint"), checkpoint))
    saved_sha = meta.get("engine_sha256")
    if saved_sha and engine_sha256 and saved_sha != engine_sha256:
        problems.append("engine %s… (loaded engine is %s…)" % (saved_sha[:12], engine_sha256[:12]))
    if problems:
        raise HeadFileError("head was fitted for " + "; ".join(problems) + ". Refit it with `action-sentinel vega fit`.")
