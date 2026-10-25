"""Action Sentinel in one minute: an agent's tools behind @guard.tool.

    python examples/guard_demo.py

Three tools, three outcomes under the shipped rule pack: a safe call runs, a dangerous one is blocked
before it runs, and a risky one waits for a human. Offline, rules only, in-memory database.
"""

from __future__ import annotations

import time
import warnings

from action_sentinel.config import load_settings
from action_sentinel.guard import ActionBlocked, ActionHeld, Guard
from action_sentinel.service import Sentinel

warnings.simplefilter("ignore")  # the dev signing key warning; production sets SENTINEL_SIGNING_KEY

sentinel = Sentinel(load_settings(db_path=":memory:"))
guard = Guard(sentinel, agent_id="ops-agent", agent_version="1.0", source_system="infra-agent", mode="enforce")


@guard.tool(target=lambda a: "staging/%s" % a["service"])
def read_logs(service: str, lines: int = 50) -> str:
    return "%d log lines from %s" % (lines, service)


@guard.tool(target=lambda a: "prod/%s" % a["database"],
            evidence=lambda a: [{"ref_id": "ticket-881", "kind": "ticket", "stance": "supports"}])
def delete_database(database: str) -> str:
    return "database %s dropped" % database


@guard.tool(target=lambda a: "staging/%s" % a["deployment"],
            evidence=lambda a: [{"ref_id": "alert-17", "kind": "alert", "stance": "supports"}])
def scale_deployment(deployment: str, replicas: int) -> str:
    return "%s scaled to %d replicas" % (deployment, replicas)


def say(text: str = "", pause: float = 0.0) -> None:
    print(text, flush=True)
    time.sleep(pause)


def main(pause: float = 0.6) -> None:
    say("$ agent calls read_logs('checkout-api')", pause)
    say("  APPROVE   ran: %s" % read_logs("checkout-api"), pause)
    say()

    say("$ agent calls delete_database('orders')", pause)
    try:
        delete_database("orders")
    except ActionBlocked as e:
        say("  BLOCK     never ran: %s" % ", ".join(e.reason_codes), pause)
    say()

    say("$ agent calls scale_deployment('web', 8)", pause)
    try:
        scale_deployment("web", 8)
    except ActionHeld as held:
        say("  REVIEW    held for a human: %s" % held.review_id[:16], pause)
        approved = sentinel.approve_review(held.review_id, "ada", "traffic spike confirmed")
        say("  reviewer ada approves: 'traffic spike confirmed'", pause)
        say("  RESUMED   ran: %s" % guard.resume(held.review_id, approved.authorization.token), pause)
    say()

    execs = sentinel.store.list_executions()
    events = sentinel.store.list_events(limit=1000)
    say("audit: %d executions, %d audit events; every decision has reason codes" % (len(execs), len(events)))


if __name__ == "__main__":
    import sys
    main(0.0 if "--fast" in sys.argv else 0.6)
