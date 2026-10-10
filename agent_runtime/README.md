# Pi business runtime

This directory embeds the real official `@earendil-works/pi-agent-core` and
`@earendil-works/pi-ai` packages at **1.1.0**, pinned in `package-lock.json`.
It does not import the coding-agent CLI or discover shell tools, extensions,
MCP servers, local instruction files, sessions, or provider credentials.

Pi orchestrates one fixed document through four explicitly registered tools:
`detect_document`, `search_evidence`, `read_evidence`, and `recheck`. Python owns
the documents, evidence allowlists, deterministic verification, result files,
and shared budget ledger. A model sentence never becomes a verification verdict.

## Build and verification

Node.js 22.19+ is required. From this directory:

```powershell
npm.cmd ci --ignore-scripts --no-audit --no-fund
npm.cmd run build
npm.cmd test
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_python_bridge.py -v
```

The host launches `node agent_runtime/dist/cli.js` with stdin/stdout pipes and
keeps stdin open until the final response. Do not pass credentials on the command
line. `dist/` and `node_modules/` are generated and ignored; build after checkout.

Tests run the actual Pi Agent, official OpenAI SSE adapter, a localhost HTTP
fixture, and the JSONL child process. No paid model, remote corpus, or OpenViking
service is needed. These tests validate orchestration and failure boundaries,
not model quality or retrieval accuracy.

The Python test runs the real `run_pi` bridge and `EvidenceWorkflow` against the
actual Node SDK process. It verifies four tool calls, checks a committed SQLite
reservation exists before every HTTP request, settles five model calls, validates
source-file hashes, and proves an empty shared budget prevents any HTTP call.
Its retrieval/detection/recheck responses are explicit fixtures, not claims about
OpenViking retrieval quality or financial detection accuracy.

## Protocol version 1

One UTF-8 JSON object per line, one run per process. Stdout contains protocol
objects only. The first input must be `start`:

```json
{
  "type": "start",
  "protocolVersion": 1,
  "requestId": "run-1",
  "task": {
    "documentId": "host-owned-document-id",
    "question": "Check the document and request missing evidence.",
    "context": {"scene": "research report"}
  },
  "model": {
    "baseUrl": "http://127.0.0.1:8080/v1",
    "apiKey": "",
    "modelId": "configured-model",
    "contextWindow": 32768,
    "maxOutputTokens": 2048,
    "thinking": "off"
  },
  "limits": {
    "maxRounds": 6,
    "maxToolCalls": 8,
    "timeoutMs": 90000,
    "toolTimeoutMs": 15000,
    "maxInputTokens": 24000,
    "maxTotalTokens": 100000,
    "maxCostCny": 0.2
  },
  "pricing": {"inputCnyPerMtok": 1, "outputCnyPerMtok": 4}
}
```

These prices are protocol examples, **not current provider prices**. Python must
validate its existing pricing source, validity date, authorization, and ledger
before approving each model call. Price and limit fields stay outside the prompt.
Remote model endpoints require HTTPS and a nonempty explicit API key; localhost
may use HTTP. Credentials in URLs, query strings, and fragments are rejected.
Provider redirects are blocked and SDK retries are disabled.

Thinking accepts `off/minimal/low/medium/high/xhigh/max` for generic compatible
models. DeepSeek endpoints or model IDs explicitly use its protocol and accept
`off/low/high/max`: `thinking.type` is disabled/enabled and `reasoning_effort` is
preserved exactly, including `max`. `model_authorized` trace events record the
effective Pi and provider settings. Unsupported DeepSeek levels are rejected,
not silently downgraded. Python maps its enabled-without-effort setting to low.

Each model round emits, **before any HTTP request**:

```json
{"type":"model_request","id":"model-1","round":1,"estimatedInputTokens":6000,"maxOutputTokens":2048,"maximumCostCny":0.014192,"estimateMethod":"provider_payload_utf8_bytes_plus_512"}
```

The estimate is a conservative upper bound based on the actual serialized
provider payload, not the model's own token count. Python must atomically reserve
the maximum in the shared `BudgetLedger`, then reply:

```json
{"type":"model_permission","id":"model-1","allowed":true,"reservationId":"ledger-call-id"}
```

`allowed:false`, timeout, cancellation, or disconnection prevents the HTTP
request. There is no bypass for local endpoints. Each authorized round emits:

```json
{"type":"model_usage","id":"model-1","reservationId":"ledger-call-id","inputTokens":120,"outputTokens":40,"status":"ok","usageKnown":true,"costCny":0.00028}
```

`status` is `ok`, `error`, or `aborted`. Reported prompt tokens include cache
tokens and are conservatively priced at the configured input rate. If actual
usage is missing, `usageKnown:false` and `costCny` equals the full reservation;
the runtime stops. Python must retain the reserve for unknown usage and for a
process that exits without settlement, even if the network request may not have
reached the provider. Over-reservation usage is exposed and stops the run; it
must never be clamped to hide possible spend. Python processes usage before the
next permission. No settlement acknowledgment is required.

Tool dispatch:

```json
{"type":"tool_call","id":"provider-call-id","name":"search_evidence","args":{"query":"2025 revenue","limit":3}}
{"type":"tool_result","id":"provider-call-id","result":{"hits":[{"evidence_id":"ev-1"}]}}
```

The first line is runtime → Python; the second is Python → runtime. On failure,
Python sends `{"type":"tool_result","id":"...","error":"short-code"}`.

| Tool | Arguments | Host result |
| --- | --- | --- |
| `detect_document` | `{}` | `{status, summary, pending_checks, evidence_requests, result_ref}` |
| `search_evidence` | `{query: string, limit?: 1..10}` | Small search result with opaque evidence IDs |
| `read_evidence` | `{evidence_id: string}` | Original text, provenance, and locator for that authorized ID |
| `recheck` | `{evidence_ids: string[]}` (1..10) | `{status, summary, pending_checks, evidence_requests, result_ref}` |

Tool results must be objects below 64 KiB. `detect_document` and `recheck`
require a string `status` and `result_ref`. Python retains full results; the
model gets compact summaries. Node enforces detect-first and read-before-recheck;
Python must independently enforce document scope, permitted evidence IDs,
provenance, source quality, company/period/scope matching, and allowed result paths.
After a tool error, Pi may choose another allowed action within the limits.

The runtime emits compact `event` records and ends with exactly one `final`:

```json
{"type":"final","protocolVersion":1,"requestId":"run-1","status":"needs_review","stopReason":"agent_finished","businessResult":{"status":"needs_review","result_ref":"host-result-1"},"lastMessage":"Missing source evidence.","trace":[],"usage":{"modelCalls":1,"toolCalls":1,"inputTokens":120,"outputTokens":40,"accountedTokens":160,"costCny":0.00028,"usageKnown":true}}
```

`final.status` describes runtime completion: `completed`, `needs_review`,
`stopped`, or `failed`. `businessResult` is only the latest successful host
`detect_document`/`recheck` payload. A verified host status of `confirmed_error`,
`no_issue`, `completed`, or `complete` can produce `completed` only with empty
pending checks and evidence requests. `lastMessage` is untrusted model prose;
never use it to override the host verdict. No detection yields `needs_review`.
Stopping or failure preserves the latest host result without claiming full completion.

The host can send `{"type":"cancel"}` at any time. Wall-clock timeout aborts
both provider HTTP and pending host operations. Limits cover rounds, attempted
tool calls, input/output/context tokens, conservative cumulative tokens, and CNY.
Raw provider errors and API keys are not logged; the CLI redacts the supplied key
from every outbound JSON line. The host must keep the initial configuration out
of logs as well.

## Official references

- [Pi Agent core](https://github.com/earendil-works/pi/tree/main/packages/agent)
- [Pi AI provider adapters](https://github.com/earendil-works/pi/tree/main/packages/ai)
- [OpenViking Pi extension](https://github.com/volcengine/OpenViking/tree/main/examples/pi-coding-agent-extension)

This runtime uses the business evidence tools supplied by Python. It does not
enable the generic OpenViking memory extension, which has a broader tool surface
and cross-session behavior than this fixed-document verification workflow.
