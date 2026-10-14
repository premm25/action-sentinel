# Demo guide

```bash
action-sentinel demo                      # temporary database, removed afterwards
action-sentinel demo --db var/demo.db     # keep the records, then: SENTINEL_DB_PATH=var/demo.db action-sentinel dashboard
```

The demo runs offline in a few seconds. It uses the deterministic `rules` provider and two
**scripted test doubles** (`mock-a`, `mock-b`), whose outputs are fixed in `action_sentinel/demo.py`. Each
step checks its expected result, and the process exits with code 1 if any check fails. The last
line reads `Result: ALL CHECKS PASSED` when everything held.

| step | what happens | what to look for |
|---|---|---|
| 1 | Three proposals are evaluated by rules + mock-a | A staging restart is `APPROVE`; a production restart and a 900 refund go to `HUMAN_REVIEW` |
| 2 | Three providers disagree (mock-b says `unsafe`) | `HUMAN_REVIEW` with `DISAGREEMENT:CATEGORY`, not a majority vote |
| 3 | A production database delete; mock-a says `safe` at 0.98 | `BLOCK` with `PROD_DATABASE_DELETE`; policy beats confidence |
| 4 | mock-a abstains on a cache clear | `ABSTAIN`, no authorization |
| 5 | The operator tries the high-risk refund, then approves the prod restart and rejects the disputed case | `INSUFFICIENT_APPROVAL_LEVEL`, then `APPROVED` and `REJECTED` |
| 6 | Senior reviewer overrides the refund: first to 25,000, then to 300 | The 25,000 override fails re-validation (`REFUND_HARD_LIMIT`); 300 is accepted as a new derived decision |
| 7 | Misuse of authorizations | The original 900 payload with the 300 token, a tampered payload, a revoked grant and a forged token are all rejected |
| 8 | Real execution through the mock executor | The approved restart and the overridden refund run once; a repeat gives `DUPLICATE`; the executor saw exactly 2 actions |
| 9–10 | Confidence Lab on the 44-example test split | Naive mock-a vs full Sentinel: false approvals and hard violations blocked |
| 11 | The audit trail for the refund workflow | Ordered events from `proposal.received` through `review.override_rejected`, `review.overridden` and `execution.executed` |

To see the same flows over HTTP, start `action-sentinel serve` and use the requests in `tests/api/test_api.py`
as examples. The interactive docs are at `/docs`.
