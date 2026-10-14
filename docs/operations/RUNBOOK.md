# Runbook

All commands assume the repository root with the virtualenv active (`.venv/Scripts/activate` on
Windows). `python -m sentinel` can replace `sentinel` anywhere.

## Start

```bash
export SENTINEL_SIGNING_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"  # shared by all processes
sentinel serve --port 8000          # API at http://127.0.0.1:8000 (docs at /docs)
sentinel dashboard --port 8501      # dashboard at http://127.0.0.1:8501
```

Both use `db_path` from `config/sentinel.yaml` (default `var/sentinel.db`) unless `SENTINEL_DB_PATH`
or `--db` overrides it. Use the same database and the same signing key in every process that
should see the same decisions and tokens.

## Health checks

```bash
curl -s http://127.0.0.1:8000/health
sentinel providers
sentinel validate-rules
```

## Routine operations

| task | command |
|---|---|
| Pending reviews | `sentinel reviews list` or `GET /reviews/pending` |
| Approve / reject / override | `sentinel reviews approve <id> --reviewer <rid> --reason "..."` |
| Expire overdue reviews | Happens on access. Batch: `python -c "from sentinel.config import load_settings; from sentinel.service import Sentinel; print(Sentinel(load_settings()).expire_due_reviews())"` |
| Revoke an authorization | `POST /authorizations/{grant_id}/revoke` (admin) |
| Export the audit log | `sentinel audit-export --out var/audit.jsonl [--correlation <id>]` |
| Change policy | Edit or add a rule pack, bump its `version`, run `sentinel validate-rules`, restart. Grants issued under the old version stop executing unless that version is listed in `accepted_policy_versions` |
| Evaluate | `sentinel lab suite` (offline) / `sentinel lab suite --model ollama --second ""` |

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ollama ... UNAVAILABLE ollama unreachable` | Ollama not running | Start the Ollama app or `ollama serve`. Sentinel keeps working with the other providers |
| `model llama3.1:8b not pulled` | Model missing | `ollama pull llama3.1:8b`, or change `providers[].config.model` |
| LLM decisions time out | CPU inference is slow (12 s median, more than 3 min on cold load) | Raise `provider_timeout_s` and the provider's `timeout_s`, or warm the model once |
| `vegaml is not installed` | Optional dependency | `pip install vegaml` (large). Otherwise ignore it |
| Execution returns `INVALID_TOKEN` across processes | Ephemeral signing keys | Set the same `SENTINEL_SIGNING_KEY` for every process |
| `POLICY_VERSION_STALE` | Policy changed after approval | Re-evaluate the action, or temporarily list the old version in `accepted_policy_versions` |
| `RulePackError` at startup | Invalid or conflicting rule pack | `sentinel validate-rules` lists every issue with its location |
| HTTP 409 on evaluate | `event_id` reused with a different action | Use a new `event_id` per proposal |
| `database is locked` | Long write in another process | Retry. SQLite waits up to 10 s; avoid running many writers |
| Dashboard shows old data | Different database path | Check the caption under the title; pass the same `SENTINEL_DB_PATH` |

## Shutdown

- API and dashboard: press Ctrl+C in their terminals, or stop the processes. No in-flight state
  is lost: decisions, reviews, grants and audit events are committed as they happen.
- In-flight provider calls are abandoned. Their decisions were not persisted and can be resubmitted
  with the same `event_id`.
- Optionally stop Ollama (`ollama stop llama3.1:8b`) to free memory.
- Backup: copy `var/sentinel.db` together with any `-wal`/`-shm` files while no process is writing,
  or use `sqlite3 var/sentinel.db ".backup var/backup.db"`.
