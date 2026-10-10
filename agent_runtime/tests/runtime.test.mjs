import assert from "node:assert/strict";
import { test } from "node:test";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { validateStart } from "../dist/config.js";
import { createFixture } from "./fixtures/openai-sse.mjs";

const cwd = fileURLToPath(new URL("..", import.meta.url));
function config(url, changes = {}) {
  const value = { type: "start", protocolVersion: 1, requestId: "fixture-run",
    task: { documentId: "fixed-doc", question: "核对收入，缺材料时补证" },
    model: { baseUrl: url, apiKey: "test-secret-never-print", modelId: "fixture-model", contextWindow: 20000, maxOutputTokens: 1000, thinking: "off" },
    limits: { maxRounds: 6, maxToolCalls: 8, timeoutMs: 5000, toolTimeoutMs: 700, maxInputTokens: 16000, maxTotalTokens: 100000, maxCostCny: 1 },
    pricing: { inputCnyPerMtok: 1, outputCnyPerMtok: 4 },
  };
  return { ...value, ...changes, model: { ...value.model, ...changes.model }, limits: { ...value.limits, ...changes.limits } };
}
const business = (status = "needs_review") => ({ status, summary: { checked: 1 }, pending_checks: status === "needs_review" ? ["revenue"] : [], evidence_requests: [], result_ref: "host-result-1" });
async function processRun(start, onTool = async () => business(), options = {}) {
  const child = spawn(process.execPath, ["dist/cli.js"], { cwd, stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
  const messages = [], tools = [];
  let stdout = "", stderr = "", final, handlerError;
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.stdin.on("error", () => {});
  const reader = createInterface({ input: child.stdout, crlfDelay: Infinity });
  reader.on("line", async (line) => {
    try {
      const message = JSON.parse(line);
      messages.push(message);
      if (message.type === "model_request") {
        options.onPermission?.(message);
        if (options.ignorePermission) return;
        child.stdin.write(JSON.stringify({ type: "model_permission", id: message.id, allowed: options.allow !== false, reservationId: "host-reservation-" + message.round }) + "\n");
      } else if (message.type === "tool_call") {
        tools.push(message);
        try { child.stdin.write(JSON.stringify({ type: "tool_result", id: message.id, result: await onTool(message) }) + "\n"); }
        catch { child.stdin.write(JSON.stringify({ type: "tool_result", id: message.id, error: "business_tool_failed" }) + "\n"); }
      } else if (message.type === "final") final = message;
    } catch (error) { handlerError = error; child.kill(); }
  });
  const killed = setTimeout(() => child.kill(), 10000);
  child.stdin.write(JSON.stringify(start) + "\n");
  const exitCode = await new Promise((resolve, reject) => { child.on("exit", resolve); child.on("error", reject); });
  clearTimeout(killed);
  assert.ifError(handlerError);
  assert.ok(final, `runtime must produce final JSON; exit=${exitCode}; stderr=${stderr}`);
  assert.ok(!stdout.includes(start.model.apiKey), "stdout must not contain credentials");
  assert.ok(!stderr.includes(start.model.apiKey), "stderr must not contain credentials");
  return { final, tools, messages, exitCode, stdout, stderr };
}

test("actual Pi SDK and HTTP SSE orchestrate detect/search/read/recheck with host budget permission", async () => {
  const fixture = await createFixture([
    { tool: "detect_document" }, { tool: "search_evidence", args: { query: "2025 revenue", limit: 3 } },
    { tool: "read_evidence", args: { evidence_id: "ev-1" } }, { tool: "recheck", args: { evidence_ids: ["ev-1"] } }, { text: "依据已核查。" },
  ]);
  try {
    let authorized = 0;
    const result = await processRun(config(fixture.url), async ({ name }) => {
      if (name === "detect_document") return business();
      if (name === "search_evidence") return { hits: [{ evidence_id: "ev-1" }] };
      if (name === "read_evidence") return { evidence_id: "ev-1", text: "收入120万元", page: 2, block_id: "b-1" };
      return business("confirmed_error");
    }, { onPermission: () => { assert.equal(fixture.requests.length, authorized); authorized++; } });
    assert.deepEqual(result.tools.map((item) => item.name), ["detect_document", "search_evidence", "read_evidence", "recheck"]);
    assert.equal(fixture.requests.length, 5);
    assert.equal(result.final.status, "completed");
    assert.equal(result.final.businessResult.status, "confirmed_error");
    assert.equal(result.final.usage.modelCalls, 5);
    assert.equal(result.messages.filter((m) => m.type === "model_usage").length, 5);
    assert.equal(result.final.usage.inputTokens, 600);
    assert.equal(result.final.usage.outputTokens, 200);
    assert.equal(result.final.usage.usageKnown, true);
    assert.deepEqual(fixture.requests[0].tools.map((tool) => tool.function.name), ["detect_document", "search_evidence", "read_evidence", "recheck"]);
    assert.equal(fixture.requests[0].max_tokens, 1000);
    assert.equal(fixture.requests[0].stream, true);
  } finally { await fixture.close(); }
});

test("denied model permission sends no HTTP request", async () => {
  const fixture = await createFixture([{ text: "not reached" }]);
  try {
    const result = await processRun(config(fixture.url), undefined, { allow: false });
    assert.equal(fixture.requests.length, 0);
    assert.equal(result.final.stopReason, "model_permission_denied");
    assert.equal(result.final.status, "stopped");
    assert.equal(result.messages.filter((m) => m.type === "model_usage").length, 0);
  } finally { await fixture.close(); }
});

test("unreported usage keeps full reservation and stops before tools or another call", async () => {
  const fixture = await createFixture([{ tool: "detect_document", usage: false }]);
  try {
    const result = await processRun(config(fixture.url));
    assert.equal(result.tools.length, 0);
    assert.equal(result.final.stopReason, "model_usage_unknown");
    const request = result.messages.find((m) => m.type === "model_request");
    const usage = result.messages.find((m) => m.type === "model_usage");
    assert.equal(usage.usageKnown, false);
    assert.equal(usage.costCny, request.maximumCostCny);
  } finally { await fixture.close(); }
});

test("model prose cannot manufacture a confirmed business result", async () => {
  const fixture = await createFixture([{ text: '{"status":"confirmed_error","result_ref":"invented"}' }]);
  try {
    const result = await processRun(config(fixture.url));
    assert.equal(result.final.status, "needs_review");
    assert.equal(result.final.businessResult, null);
    assert.equal(result.final.stopReason, "detection_not_performed");
  } finally { await fixture.close(); }
});

test("unregistered shell tool and out-of-order recheck never reach the host", async () => {
  const fixture = await createFixture([
    { tool: "bash", args: { command: "echo unsafe" } },
    { tool: "recheck", args: { evidence_ids: ["invented"] } }, { text: "无法核实" },
  ]);
  try {
    const result = await processRun(config(fixture.url));
    assert.equal(result.tools.length, 0);
    assert.equal(result.final.status, "needs_review");
    assert.equal(result.final.businessResult, null);
  } finally { await fixture.close(); }
});

test("recheck requires evidence read after detection", async () => {
  const fixture = await createFixture([{ tool: "detect_document" }, { tool: "recheck", args: { evidence_ids: ["unread"] } }, { text: "缺原文" }]);
  try {
    const result = await processRun(config(fixture.url));
    assert.deepEqual(result.tools.map((t) => t.name), ["detect_document"]);
    assert.equal(result.final.businessResult.status, "needs_review");
  } finally { await fixture.close(); }
});

test("tool failure remains observable and cannot become a business verdict", async () => {
  const fixture = await createFixture([{ tool: "detect_document" }, { text: "confirmed_error" }]);
  try {
    const result = await processRun(config(fixture.url), async () => { throw new Error("failure"); });
    assert.equal(result.final.status, "needs_review");
    assert.equal(result.final.businessResult, null);
    assert.ok(result.final.trace.some((item) => item.event === "tool_finished" && item.isError));
  } finally { await fixture.close(); }
});

test("tool and round ceilings stop real SDK automatic continuation", async () => {
  const fixture = await createFixture([{ tools: [{ name: "detect_document" }, { name: "search_evidence", args: { query: "revenue" } }] }]);
  try {
    const result = await processRun(config(fixture.url, { limits: { maxToolCalls: 1 } }));
    assert.equal(result.final.stopReason, "max_tool_calls");
    assert.equal(fixture.requests.length, 1);
    assert.equal(result.tools.length, 1);
  } finally { await fixture.close(); }
  const second = await createFixture([{ tool: "detect_document" }]);
  try {
    const result = await processRun(config(second.url, { limits: { maxRounds: 1 } }));
    assert.equal(result.final.stopReason, "max_rounds");
    assert.equal(second.requests.length, 1);
  } finally { await second.close(); }
});

test("deadline aborts HTTP and reports unsettled usage conservatively", async () => {
  const fixture = await createFixture([{ hang: true }]);
  try {
    const result = await processRun(config(fixture.url, { limits: { timeoutMs: 300 } }));
    assert.equal(result.final.status, "stopped");
    assert.equal(result.final.stopReason, "timeout");
    assert.equal(fixture.requests.length, 1);
    assert.equal(result.messages.find((m) => m.type === "model_usage").usageKnown, false);
  } finally { await fixture.close(); }
});

test("HTTP failure has no SDK retry and cannot silently succeed", async () => {
  const fixture = await createFixture([{ error: 500 }]);
  try {
    const result = await processRun(config(fixture.url));
    assert.equal(fixture.requests.length, 1);
    assert.equal(result.final.status, "failed");
    assert.equal(result.final.stopReason, "model_error");
    assert.equal(result.messages.find((m) => m.type === "model_usage").usageKnown, false);
  } finally { await fixture.close(); }
});

test("input token and cost bounds block HTTP before requesting host budget", async () => {
  const fixture = await createFixture([{ text: "not reached" }]);
  try {
    const tokens = await processRun(config(fixture.url, { limits: { maxInputTokens: 128 } }));
    assert.equal(tokens.final.stopReason, "max_input_tokens");
    const cost = await processRun(config(fixture.url, { limits: { maxCostCny: 0.000001 } }));
    assert.equal(cost.final.stopReason, "max_cost_cny");
    assert.equal(fixture.requests.length, 0);
    assert.equal(cost.messages.filter((m) => m.type === "model_request").length, 0);
  } finally { await fixture.close(); }
});

test("configuration rejects insecure remote URLs, URL credentials, missing prices and invalid limits", () => {
  assert.throws(() => validateStart(config("http://example.com/v1")), /invalid_base_url/);
  assert.throws(() => validateStart(config("https://name:password@example.com/v1")), /invalid_base_url/);
  assert.throws(() => validateStart(config("https://example.com/v1", { pricing: { inputCnyPerMtok: 0, outputCnyPerMtok: 0 } })), /price/);
  assert.throws(() => validateStart(config("http://127.0.0.1:1/v1", { limits: { maxRounds: 0 } })), /max_rounds/);
});

test("DeepSeek low/max/off are sent explicitly without Pi effort clamping", async () => {
  for (const thinking of ["low", "max", "off"]) {
    const fixture = await createFixture([{ text: "需先执行核查" }]);
    try {
      const result = await processRun(config(fixture.url, { model: { modelId: "deepseek-flash", thinking } }));
      assert.equal(fixture.requests[0].thinking.type, thinking === "off" ? "disabled" : "enabled");
      assert.equal(fixture.requests[0].reasoning_effort, thinking === "off" ? undefined : thinking);
      const trace = result.final.trace.find((row) => row.event === "model_authorized");
      assert.equal(trace.effectiveThinking, thinking);
      assert.equal(trace.providerReasoningEffort, thinking === "off" ? null : thinking);
    } finally { await fixture.close(); }
  }
});

test("host permission timeout fails visibly and cannot reach the provider", async () => {
  const fixture = await createFixture([{ text: "not reached" }]);
  try {
    const result = await processRun(config(fixture.url, { limits: { toolTimeoutMs: 100 } }), undefined, { ignorePermission: true });
    assert.equal(result.final.status, "failed");
    assert.equal(result.final.stopReason, "model_permission_timeout_or_disconnect");
    assert.equal(fixture.requests.length, 0);
  } finally { await fixture.close(); }
});

test("outbound JSON redacts credentials even when JSON escaping changes the secret", async () => {
  const apiKey = 'fixture-"quoted\\secret';
  const fixture = await createFixture([{ text: `Untrusted provider echoed ${apiKey}` }]);
  try {
    const result = await processRun(config(fixture.url, { model: { apiKey } }));
    assert.equal(result.final.lastMessage, "Untrusted provider echoed [REDACTED]");
    assert.ok(!result.stdout.includes(JSON.stringify(apiKey).slice(1, -1)));
  } finally { await fixture.close(); }
});
