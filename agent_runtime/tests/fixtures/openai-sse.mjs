import http from "node:http";

/** Local deterministic OpenAI SSE fixture; no SDK or Agent mocking. */
export async function createFixture(script) {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    if (req.url !== "/v1/chat/completions" || req.method !== "POST") { res.writeHead(404).end(); return; }
    let input = "";
    for await (const chunk of req) input += chunk;
    requests.push(JSON.parse(input));
    const index = requests.length - 1;
    const step = script[index] ?? { error: 500 };
    if (step.hang) return;
    if (step.error) {
      res.writeHead(step.error, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "fixture_failure", type: "server_error" } }));
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
    const base = { id: `fixture-${index}`, object: "chat.completion.chunk", created: 1700000000, model: "fixture-model" };
    const push = (choice, usage) => res.write(`data: ${JSON.stringify({ ...base, choices: choice === null ? [] : [choice], ...(usage ? { usage } : {}) })}\n\n`);
    push({ index: 0, delta: { role: "assistant" }, finish_reason: null });
    const calls = step.tools ?? (step.tool ? [{ name: step.tool, args: step.args ?? {} }] : []);
    if (calls.length) {
      push({ index: 0, delta: { tool_calls: calls.map((tool, n) => ({ index: n, id: `call-${index}-${n}`, type: "function", function: { name: tool.name, arguments: JSON.stringify(tool.args ?? {}) } })) }, finish_reason: null });
    } else {
      push({ index: 0, delta: { content: step.text ?? "done" }, finish_reason: null });
    }
    push({ index: 0, delta: {}, finish_reason: step.finishReason ?? (calls.length ? "tool_calls" : "stop") });
    if (step.usage !== false) push(null, step.usage ?? { prompt_tokens: 120, completion_tokens: 40, total_tokens: 160 });
    res.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${server.address().port}/v1`, requests,
    close: async () => { server.closeAllConnections(); await new Promise((resolve) => server.close(resolve)); } };
}
