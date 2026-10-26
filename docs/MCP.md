# MCP tool calls: Action Sentinel behind the sluice proxy

[sluice](https://github.com/premm25/sluice) is an MCP proxy. It sits between an MCP client (Claude
Desktop, Cursor, VS Code) and each MCP server, and sees every tool call, so agents are protected
without changing their code. It can ask Action Sentinel what it would decide about each call.

The two check different things:

| | sluice | Action Sentinel |
|---|---|---|
| looks at | where data comes from and where it goes | what the action does |
| catches | private data sent out (even encoded), poisoned tool descriptions, tools that change after approval, leaked secrets in results | forbidden actions, dangerous arguments, risky actions that need a person (rules, optional VegaML) |

## Status: shadow mode

So far sluice only **records** Sentinel's verdict next to its own decision. Sentinel never changes what
sluice does. Use this to see how often the two agree on your real traffic before anything is enforced.
Enforcement (Sentinel blocking calls and holding them for human review) is the next step.

## Set up

1. Start Sentinel with the MCP profile:

   ```bash
   pip install "action-sentinel[api]"
   action-sentinel --config config/sentinel-mcp.yaml serve --port 8000
   ```

2. Add a `[sentinel]` section to sluice's policy (`~/.sluice/policy.toml`), and give sluice the API key
   through an environment variable (it is never read from the policy file):

   ```toml
   [sentinel]
   url = "http://127.0.0.1:8000"
   api_key_env = "SENTINEL_API_KEY"   # dev profile key: dev-mcp-key
   agent_id = "mcp-proxy"
   ```

3. Check the connection, use your agent as usual, then compare:

   ```bash
   sluice sentinel check
   sluice sentinel report
   ```

## How calls reach Sentinel

Each `tools/call` becomes one Sentinel action:

| field | value |
|---|---|
| `action_type` | `mcp_tool_call` (MCP tool names are chosen by each server, so there is one generic type) |
| `action_payload` | `{server, tool, arguments}`; arguments over 48 KB or 8 levels deep are replaced by a marked preview |
| `target_resource` | `mcp/<server>/<tool>` |
| `source_system`, `agent_id` | `mcp-proxy` (the API key's principal) |
| `context` | sluice's own decision and reason codes |

sluice sends these to `POST /decisions/assess`, which stores no decision, from a background thread. The
proxy never waits for Sentinel; if Sentinel falls behind, calls are left out of the comparison and
counted.

The arguments are sent as they are, because Sentinel's rules need to see them. Point sluice only at a
Sentinel service you run.

## The MCP profile

`config/sentinel-mcp.yaml` uses:
- `examples/rule_packs/mcp.yaml`: admits `mcp_tool_call` and blocks a few clearly dangerous calls (SQL that
  drops or truncates tables, `rm -rf`, credentials in arguments). Write rules for your own servers on
  `action_payload.tool`, `action_payload.arguments.<name>` or the target.
- `examples/rule_packs/mcp_classifier.yaml`: judges calls by tool name, cautiously. Read-only names
  (`get_`, `list_`, `read_`, `search_`, ...) are safe; names that change, send or delete need a person,
  and so does any tool it doesn't recognise.

To add VegaML's judgement, fit its head on your own labelled MCP calls (`action-sentinel vega fit`) and
add the `vegaml-head` provider as in `config/sentinel-vegaml.yaml`.

## Verified end to end (2026-10-26)

A real Sentinel (`config/sentinel-mcp.yaml`), two real `sluice run` proxies in front of sluice's demo
notes and web servers, and an MCP client making five calls:

| call | sluice | Sentinel |
|---|---|---|
| `notes/list_notes` | ran | APPROVE |
| `notes/read_note` | ran | APPROVE |
| `web/fetch` an ordinary URL | ran | APPROVE |
| `web/fetch` with an `api_key` argument | ran (warning) | **BLOCK** `RULE_MCP_SECRET_IN_ARGUMENTS` |
| `web/fetch` carrying the private note to attacker.example | **blocked** (exfiltration) | APPROVE |

Each caught one problem the other missed. The tests are `tests/api/test_mcp_profile.py` here and
`tests/test_sentinel.py` in sluice.
