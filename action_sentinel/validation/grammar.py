"""The restricted condition grammar used by rule packs and the rule-based provider.

A condition is plain data: never code. The forms are:

    {"all": [cond, ...]}          every sub-condition holds (an empty list is rejected)
    {"any": [cond, ...]}          at least one holds
    {"not": cond}                 negation
    {"field": PATH, "op": OP, "value": V}

PATH is a dotted path of [A-Za-z0-9_] segments into the evaluation context (no dunder segments,
at most 8 segments). Lists are indexed with numeric segments. OP is one of OPERATORS. Nothing in a
rule pack is ever passed to eval, exec, a shell or an import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Union

OPERATORS = frozenset({
    "eq", "ne", "in", "not_in", "contains", "not_contains", "starts_with", "ends_with", "glob",
    "matches", "gt", "gte", "lt", "lte", "exists", "missing",
})
_NO_VALUE_OPS = frozenset({"exists", "missing"})
_SEGMENT = re.compile(r"^[A-Za-z0-9_]{1,64}$")
MAX_DEPTH = 8
MAX_CHILDREN = 50
MAX_PATTERN_LEN = 200
MAX_MATCH_INPUT = 10_000
_MISSING = object()
MISSING = _MISSING  # sentinel returned by resolve() for an absent path


class GrammarError(ValueError):
    """Raised when a condition does not conform to the grammar."""


@dataclass(frozen=True)
class FieldCond:
    path: tuple[str, ...]
    op: str
    value: Any
    pattern: re.Pattern | None = None


@dataclass(frozen=True)
class AllCond:
    children: tuple["Condition", ...]


@dataclass(frozen=True)
class AnyCond:
    children: tuple["Condition", ...]


@dataclass(frozen=True)
class NotCond:
    child: "Condition"


Condition = Union[FieldCond, AllCond, AnyCond, NotCond]


def parse_condition(raw: Any, depth: int = 0) -> Condition:
    if depth > MAX_DEPTH:
        raise GrammarError("condition nested deeper than %d levels" % MAX_DEPTH)
    if not isinstance(raw, dict):
        raise GrammarError("a condition must be a mapping, got %s" % type(raw).__name__)
    keys = set(raw)
    if keys in ({"all"}, {"any"}):
        kind = keys.pop()
        items = raw[kind]
        if not isinstance(items, list) or not items:
            raise GrammarError("'%s' needs a non-empty list" % kind)
        if len(items) > MAX_CHILDREN:
            raise GrammarError("'%s' has more than %d children" % (kind, MAX_CHILDREN))
        children = tuple(parse_condition(c, depth + 1) for c in items)
        return AllCond(children) if kind == "all" else AnyCond(children)
    if keys == {"not"}:
        return NotCond(parse_condition(raw["not"], depth + 1))
    if "field" in keys and "op" in keys and keys <= {"field", "op", "value"}:
        return _parse_field(raw)
    raise GrammarError("unrecognised condition keys %s" % sorted(keys))


def _parse_field(raw: dict) -> FieldCond:
    path_s, op = raw["field"], raw["op"]
    if not isinstance(path_s, str) or not path_s:
        raise GrammarError("'field' must be a non-empty string")
    path = tuple(path_s.split("."))
    if len(path) > 8 or any(not _SEGMENT.match(s) or s.startswith("__") for s in path):
        raise GrammarError("invalid field path %r" % path_s)
    if op not in OPERATORS:
        raise GrammarError("unknown operator %r (allowed: %s)" % (op, ", ".join(sorted(OPERATORS))))
    has_value = "value" in raw
    if op in _NO_VALUE_OPS:
        if has_value:
            raise GrammarError("operator %r takes no value" % op)
        return FieldCond(path, op, None)
    if not has_value:
        raise GrammarError("operator %r needs a value" % op)
    value = raw["value"]
    if isinstance(value, (dict,)):
        raise GrammarError("condition values must be scalars or lists of scalars")
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        raise GrammarError("condition value lists must contain scalars")
    if op in ("in", "not_in") and not isinstance(value, list):
        raise GrammarError("operator %r needs a list value" % op)
    if op in ("gt", "gte", "lt", "lte") and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise GrammarError("operator %r needs a numeric value" % op)
    if op in ("starts_with", "ends_with", "glob", "matches") and not isinstance(value, str):
        raise GrammarError("operator %r needs a string value" % op)
    pattern = None
    if op == "matches":
        if len(value) > MAX_PATTERN_LEN:
            raise GrammarError("regular expression longer than %d characters" % MAX_PATTERN_LEN)
        try:
            pattern = re.compile(value)
        except re.error as exc:
            raise GrammarError("invalid regular expression %r: %s" % (value, exc)) from exc
    return FieldCond(path, op, value, pattern)


def resolve(ctx: Any, path: tuple[str, ...]) -> Any:
    cur = ctx
    for seg in path:
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit():
            idx = int(seg)
            if idx >= len(cur):
                return _MISSING
            cur = cur[idx]
        else:
            return _MISSING
    return cur


def evaluate(cond: Condition, ctx: dict) -> bool:
    if isinstance(cond, AllCond):
        return all(evaluate(c, ctx) for c in cond.children)
    if isinstance(cond, AnyCond):
        return any(evaluate(c, ctx) for c in cond.children)
    if isinstance(cond, NotCond):
        return not evaluate(cond.child, ctx)
    return _eval_field(cond, resolve(ctx, cond.path))


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _eval_field(c: FieldCond, actual: Any) -> bool:
    op, value = c.op, c.value
    if op == "exists":
        return actual is not _MISSING and actual is not None
    if op == "missing":
        return actual is _MISSING or actual is None
    if actual is _MISSING:
        # A missing field never satisfies a positive comparison; negative operators hold vacuously.
        return op in ("ne", "not_in", "not_contains")
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "in":
        return actual in value
    if op == "not_in":
        return actual not in value
    if op in ("contains", "not_contains"):
        if isinstance(actual, (list, str)):
            found = value in actual if isinstance(actual, list) or isinstance(value, str) else False
        else:
            found = False
        return found if op == "contains" else not found
    if op in ("gt", "gte", "lt", "lte"):
        a = _num(actual)
        if a is None:
            return False
        return {"gt": a > value, "gte": a >= value, "lt": a < value, "lte": a <= value}[op]
    if not isinstance(actual, str):
        return False
    if op == "starts_with":
        return actual.startswith(value)
    if op == "ends_with":
        return actual.endswith(value)
    if op == "glob":
        return fnmatchcase(actual, value)
    if op == "matches":
        assert c.pattern is not None
        return c.pattern.search(actual[:MAX_MATCH_INPUT]) is not None
    raise GrammarError("unhandled operator %r" % op)  # unreachable: parse rejects unknown ops


def canonical(cond: Condition) -> Any:
    """A hashable normal form, used to find rules with identical conditions."""
    if isinstance(cond, AllCond):
        return ("all", tuple(sorted((canonical(c) for c in cond.children), key=repr)))
    if isinstance(cond, AnyCond):
        return ("any", tuple(sorted((canonical(c) for c in cond.children), key=repr)))
    if isinstance(cond, NotCond):
        return ("not", canonical(cond.child))
    v = tuple(cond.value) if isinstance(cond.value, list) else cond.value
    return ("field", cond.path, cond.op, v)
