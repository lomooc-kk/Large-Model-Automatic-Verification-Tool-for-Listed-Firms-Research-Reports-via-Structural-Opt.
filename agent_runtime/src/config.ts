import type { JsonObject } from "./protocol.js";

export interface StartConfig {
  type: "start";
  protocolVersion: 1;
  requestId: string;
  task: { documentId: string; question: string; context?: unknown };
  model: { baseUrl: string; apiKey: string; modelId: string; contextWindow: number; maxOutputTokens: number; thinking?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" };
  limits: { maxRounds: number; maxToolCalls: number; timeoutMs: number; toolTimeoutMs: number; maxInputTokens: number; maxTotalTokens: number; maxCostCny: number };
  pricing: { inputCnyPerMtok: number; outputCnyPerMtok: number };
}

function object(value: unknown, name: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`invalid_${name}`);
  return value as JsonObject;
}
function text(value: unknown, name: string, max: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > max) throw new Error(`invalid_${name}`);
  return value;
}
function numeric(value: unknown, name: string, min: number, max: number, integer = true): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) throw new Error(`invalid_${name}`);
  return value;
}

export function validateStart(value: unknown): StartConfig {
  const root = object(value, "start");
  if (root.type !== "start" || root.protocolVersion !== 1) throw new Error("invalid_protocol_version");
  const task = object(root.task, "task"), model = object(root.model, "model");
  const limits = object(root.limits, "limits"), pricing = object(root.pricing, "pricing");
  const baseUrl = text(model.baseUrl, "base_url", 2048);
  let url: URL;
  try { url = new URL(baseUrl); } catch { throw new Error("invalid_base_url"); }
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (url.username || url.password || url.search || url.hash || (url.protocol !== "https:" && !(url.protocol === "http:" && local))) throw new Error("invalid_base_url");
  const apiKey = typeof model.apiKey === "string" ? model.apiKey : "";
  if ((!local && !apiKey.trim()) || apiKey.length > 16384) throw new Error("invalid_api_key");
  const thinking = model.thinking ?? "off";
  if (!["off", "minimal", "low", "medium", "high", "xhigh", "max"].includes(String(thinking))) throw new Error("invalid_thinking");
  const config: StartConfig = {
    type: "start", protocolVersion: 1,
    requestId: text(root.requestId, "request_id", 128),
    task: { documentId: text(task.documentId, "document_id", 256), question: text(task.question, "question", 32000), ...(task.context === undefined ? {} : { context: task.context }) },
    model: { baseUrl, apiKey, modelId: text(model.modelId, "model_id", 256), contextWindow: numeric(model.contextWindow, "context_window", 2048, 2_000_000), maxOutputTokens: numeric(model.maxOutputTokens, "max_output_tokens", 1, 131072), thinking: thinking as StartConfig["model"]["thinking"] },
    limits: {
      maxRounds: numeric(limits.maxRounds, "max_rounds", 1, 32),
      maxToolCalls: numeric(limits.maxToolCalls, "max_tool_calls", 1, 128),
      timeoutMs: numeric(limits.timeoutMs, "timeout_ms", 100, 600000),
      toolTimeoutMs: numeric(limits.toolTimeoutMs, "tool_timeout_ms", 50, 300000),
      maxInputTokens: numeric(limits.maxInputTokens, "max_input_tokens", 128, 2_000_000),
      maxTotalTokens: numeric(limits.maxTotalTokens, "max_total_tokens", 128, 4_000_000),
      maxCostCny: numeric(limits.maxCostCny, "max_cost_cny", 0, 1000, false),
    },
    pricing: { inputCnyPerMtok: numeric(pricing.inputCnyPerMtok, "input_price", 0, 10000, false), outputCnyPerMtok: numeric(pricing.outputCnyPerMtok, "output_price", 0, 10000, false) },
  };
  if (config.limits.maxInputTokens + config.model.maxOutputTokens > config.model.contextWindow) throw new Error("invalid_context_budget");
  if (isDeepSeek(config) && !["off", "low", "high", "max"].includes(String(thinking))) throw new Error("invalid_deepseek_thinking");
  if (Buffer.byteLength(JSON.stringify(config.task), "utf8") > 262144) throw new Error("task_too_large");
  if (!local && (!config.limits.maxCostCny || !(config.pricing.inputCnyPerMtok + config.pricing.outputCnyPerMtok))) throw new Error("remote_price_and_budget_required");
  return config;
}

export function isDeepSeek(config: StartConfig): boolean {
  return new URL(config.model.baseUrl).hostname === "api.deepseek.com" || /^deepseek[-/]/i.test(config.model.modelId);
}
