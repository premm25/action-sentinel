# MCP tool calls: Action Sentinel behind the sluice proxy

[sluice](https://github.com/premm25/sluice) is an MCP proxy. It sits between an MCP client (Claude Code,
Claude Desktop, Cursor, VS Code) and each MCP server, and sees every tool call, so agents are protected
without changing their code. It can ask Action Sentinel about each call.

```
agent (Claude Code, Cursor, ...) ──> sluice ──> MCP server (files, GitHub, database, ...)
                                       │
                                       └──> Action Sentinel ──> a person, for calls that need approval
```

The two check different things:

| | sluice | Action Sentinel |
|---|---|---|
| looks at | where data comes from and where it goes | what the action does |
| catches | private data sent out (even encoded), poisoned tool descriptions, tools that change after approval, leaked secrets in results | forbidden actions, dangerous arguments, risky actions that need a person (rules, optional VegaML) |

## Two modes

| | shadow | enforce |
|---|---|---|
| Sentinel's verdict | recorded next to sluice's decision | applied |
| BLOCK | the call still runs | the call never reaches the server |
| HUMAN_REVIEW | the call still runs | the call waits for a person; it runs once when the agent retries after approval |
| Sentinel down | nothing changes | calls are blocked (`on_error = "block"`, the default) or let through (`"allow"`) |
| the agent waits for Sentinel | never (a background thread) | yes, one or two requests per call (milliseconds with the rules profile) |

Start in shadow mode, look at `sluice sentinel report` for a few days, then switch to enforce. Sluice checks
first either way: a call sluice blocks is never sent for a decision (in enforce mode Sentinel's view of it is
still recorded, in the background).

## Set up

### 1. Run Sentinel with the MCP profile

```bash
pip install "action-sentinel[api]"
export SENTINEL_SIGNING_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")   # keep it
action-sentinel --config config/sentinel-mcp.yaml --db ~/.sentinel/mcp.db serve --port 8000
```

It must be running whenever your agent is, if you use enforce mode. The profile's keys are the public
development keys (`dev-mcp-key` for the proxy, `dev-operator-key` for the reviewer Olivia). That's fine on
your own machine, but replace them before anyone else can reach the port.

### 2. Tell sluice about it

Add to sluice's policy (`~/.sluice/policy.toml`; `sluice init` creates it):

```toml
[sentinel]
url = "http://127.0.0.1:8000"
api_key_env = "SENTINEL_API_KEY"   # the key comes from this environment variable, never from this file
agent_id = "mcp-proxy"
mode = "shadow"                    # later: "enforce"
# on_error = "block"               # enforce: what to do when Sentinel can't be reached (block | allow)
```

Set `SENTINEL_API_KEY=dev-mcp-key` where your MCP client starts its servers: in the client's `env` block for
each server (below) or in your user environment. Then check: `sluice sentinel check`.

### 3. Put sluice in front of your MCP servers

Each server's command becomes `sluice run --name <name> -- <the original command>`.

**Claude Code** (`claude mcp add`, or `~/.claude.json`):

```bash
claude mcp add files --env SENTINEL_API_KEY=dev-mcp-key -- sluice run --name files -- npx -y @modelcontextprotocol/server-filesystem C:/work
```

**Claude Desktop, Cursor, VS Code, Windsurf** (their `mcpServers` JSON):

```json
{
  "mcpServers": {
    "files": {
      "command": "sluice",
      "args": ["run", "--name", "files", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/work"],
      "env": {"SENTINEL_API_KEY": "dev-mcp-key"}
    }
  }
}
```

`sluice wrap <config.json> --in-place` does this rewrite for every server in a client's config (keeping a
backup); `sluice unwrap` undoes it. Restart the client afterwards.
For remote (HTTP) MCP servers use `sluice serve --upstream <url>` and point the client at sluice instead.

## Using it, in enforce mode

- **Safe calls** (reads, lists, searches) run as before.
- **Blocked calls** come back to the agent as an error that says why, for example
  `Action Sentinel decided BLOCK (RULE_MCP_DESTRUCTIVE_SQL)`. The agent is told not to retry.
- **Held calls** come back as: *needs a person's approval (Action Sentinel review rev_…). It has NOT run.*
  The agent tells you. You look at it and decide:

  ```bash
  action-sentinel --config config/sentinel-mcp.yaml --db ~/.sentinel/mcp.db reviews list
  action-sentinel --config config/sentinel-mcp.yaml --db ~/.sentinel/mcp.db reviews approve rev_… --reviewer olivia --reason "expected"
  action-sentinel --config config/sentinel-mcp.yaml --db ~/.sentinel/mcp.db reviews reject  rev_… --reviewer olivia --reason "not this file"
  ```

  or use the dashboard's Human Review tab (`action-sentinel --config config/sentinel-mcp.yaml dashboard`,
  with the reviewer's key). Then tell the agent "approved, go ahead": it calls the tool again with the same
  arguments, and the call runs, once. After a rejection the retry is stopped with the reviewer's verdict.

What counts as "the same call": the same sluice session, server, tool and arguments. Sluice keeps held calls
in memory, so if the client restarts in between, the retry needs a fresh approval. Reviews expire after 24
hours and approvals after 15 minutes (`review_ttl_seconds`, `authorization_ttl_seconds` in the profile).

## Collecting data (for VegaML)

Every call Sentinel sees is logged in sluice's audit log with both verdicts and the arguments, secrets
redacted (secret-looking values, and values under names like `api_key`, `password`, `token`).

```bash
sluice sentinel report                         # how often they agree; in enforce mode, what ran or was stopped
sluice sentinel export --out calls.jsonl       # one row per call, with "label": null
```

Label each row `safe`, `needs_review` or `unsafe` (what *should* have happened, not what did), then:

```bash
action-sentinel vega dataset --from calls.jsonl --out mcp_dataset.jsonl
action-sentinel vega fit --dataset mcp_dataset.jsonl --split calibration --out mcp_head.npz --evaluate test
```

Splits come from a hash of each call's id (60% calibration, 20% dev, 20% test) and stay put as the file grows.
Fit on calibration and measure on test, never the reverse. Add the head to the profile (`vegaml-head`
provider, as in `config/sentinel-vegaml.yaml`) only if it beats the rules on the test split. A few hundred
labelled calls, with every label represented, is a sensible minimum before trusting the number.

## How calls reach Sentinel

Each `tools/call` becomes one Sentinel action:

| field | value |
|---|---|
| `action_type` | `mcp_tool_call` (MCP tool names are chosen by each server, so there is one generic type) |
| `action_payload` | `{server, tool, arguments}`; arguments over 48 KB or 8 levels deep are replaced by a marked preview |
| `target_resource` | `mcp/<server>/<tool>` |
| `source_system`, `agent_id` | `mcp-proxy` (the API key's principal) |
| `context` | sluice's own decision and reason codes |

Shadow mode uses `POST /decisions/assess`, which stores no decision. Enforce mode uses
`POST /decisions/evaluate`, then `POST /executions/begin` with the one-time token before forwarding the call,
and `POST /executions/{id}/finish` when the server answers. A held call's retry resubmits the identical
proposal; once a reviewer has approved it, `evaluate` returns that approval's token (`review_state` says
where the review stands), and the gateway still runs it at most once.

The arguments are sent as they are, because Sentinel's rules need to see them. Point sluice only at a
Sentinel service you run.

## The MCP profile

`config/sentinel-mcp.yaml` uses:
- `examples/rule_packs/mcp.yaml`: admits `mcp_tool_call` and blocks a few clearly dangerous calls (SQL that
  drops or truncates tables, `rm -rf`, credentials in arguments). Write rules for your own servers on
  `action_payload.tool`, `action_payload.arguments.<name>` or the target.
- `examples/rule_packs/mcp_classifier.yaml`: judges calls by tool name, cautiously. Read-only names
  (`get_`, `list_`, `read_`, `search_`, ...) are safe; names that change, send or delete need a person,
  and so does any tool it doesn't recognise. Expect many holds at first; add rules for the tools you trust.

`action-sentinel init` copies the profile and rule packs into a folder so you can edit them.

## Verified end to end

**Shadow (2026-10-26).** A real Sentinel, two `sluice run` proxies in front of sluice's demo notes and web
servers, five calls. Each side caught one problem the other missed:

| call | sluice | Sentinel |
|---|---|---|
| `notes/list_notes`, `notes/read_note`, `web/fetch` an ordinary URL | ran | APPROVE |
| `web/fetch` with an `api_key` argument | ran (warning) | **BLOCK** `RULE_MCP_SECRET_IN_ARGUMENTS` |
| `web/fetch` carrying the private note to attacker.example | **blocked** (exfiltration) | APPROVE |

**Enforce (2026-10-27).** A real Sentinel, `sluice run` in enforce mode in front of a small MCP server (the
`mcp` SDK) with real file and SQLite tools, and the real `action-sentinel reviews` CLI as the person:

| call | result |
|---|---|
| `list_files` | ran |
| `write_file plan.txt` | held; the retry before approval was still held; approved with `reviews approve`; the next retry ran and the file appeared; calling it once more needed a new review |
| `run_query DROP TABLE users` | stopped: `RULE_MCP_DESTRUCTIVE_SQL` |
| `write_file` with an `api_key` argument | stopped: `RULE_MCP_SECRET_IN_ARGUMENTS` |
| `delete_file plan.txt` | held; rejected with `reviews reject`; the retry was stopped and the file is still there |
| any call with Sentinel stopped | stopped: "Action Sentinel could not be asked" |

Sentinel recorded both runs as EXECUTED, `sluice sentinel report` showed 2 ran, 1 waiting, 3 stopped, and
`sluice sentinel export` → `action-sentinel vega dataset` produced a valid dataset. The tests are
`tests/api/test_mcp_profile.py` and `tests/unit/test_from_calls.py` here, and `tests/test_sentinel.py` in sluice.
