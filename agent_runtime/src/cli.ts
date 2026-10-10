import { JsonlChannel } from "./protocol.js";
import { runSession } from "./runtime.js";

const channel = new JsonlChannel(process.stdin, process.stdout);
let requestId = "unknown";
let startTimer: ReturnType<typeof setTimeout>;
const startDeadline = new Promise<never>((_resolve, reject) => { startTimer = setTimeout(() => reject(new Error("start_timeout")), 10000); });
try {
  const input = await Promise.race([channel.start(), startDeadline]);
  clearTimeout(startTimer!);
  if (typeof input.requestId === "string") requestId = input.requestId.slice(0, 128);
  const model = input.model as Record<string, unknown> | undefined;
  if (typeof model?.apiKey === "string") channel.setSecret(model.apiKey);
  const result = await runSession(input, channel, channel.controller.signal);
  channel.send(result);
  process.exitCode = result.status === "failed" ? 2 : 0;
} catch (error) {
  clearTimeout(startTimer!);
  channel.send({ type: "final", protocolVersion: 1, requestId, status: "failed", stopReason: error instanceof Error && error.message === "start_timeout" ? "start_timeout" : "invalid_start_or_protocol", businessResult: null, lastMessage: "", trace: [], usage: null });
  process.exitCode = 2;
} finally {
  channel.close();
  process.stdin.destroy();
}
