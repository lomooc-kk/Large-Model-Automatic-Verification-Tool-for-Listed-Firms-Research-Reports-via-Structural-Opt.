import { createInterface } from "node:readline";
import type { Readable, Writable } from "node:stream";

export type JsonObject = Record<string, unknown>;
export interface HostChannel {
  send(message: JsonObject): void;
  request(message: JsonObject, responseType: string, timeoutMs: number, signal?: AbortSignal): Promise<JsonObject>;
}

/** One isolated run per process. No logs, transcripts, or credentials on stdout. */
export class JsonlChannel implements HostChannel {
  private pending = new Map<string, { type: string; resolve: (v: JsonObject) => void; reject: (e: Error) => void }>();
  private first?: JsonObject;
  private firstResolve?: (v: JsonObject) => void;
  private firstReject?: (e: Error) => void;
  private receivedStart = false;
  private closed = false;
  private secret = "";
  readonly controller = new AbortController();
  private reader;

  constructor(input: Readable, private output: Writable) {
    this.reader = createInterface({ input, crlfDelay: Infinity, terminal: false });
    this.reader.on("line", (line) => {
      if (this.closed) return;
      try {
        if (Buffer.byteLength(line, "utf8") > 1_048_576) throw new Error("protocol_message_too_large");
        const message: unknown = JSON.parse(line);
        if (!message || typeof message !== "object" || Array.isArray(message)) throw new Error("protocol_invalid_message");
        const item = message as JsonObject;
        if (!this.receivedStart) {
          if (item.type !== "start") throw new Error("protocol_start_required");
          this.receivedStart = true;
          if (this.firstResolve) this.firstResolve(item); else this.first = item;
          return;
        }
        if (item.type === "cancel") { this.fail("host_cancelled"); return; }
        if (typeof item.id !== "string") throw new Error("protocol_response_id_required");
        const pending = this.pending.get(item.id);
        if (!pending || pending.type !== item.type) throw new Error("protocol_unexpected_response");
        pending.resolve(item);
      } catch (error) {
        this.fail(error instanceof Error && error.message.startsWith("protocol_") ? error.message : "protocol_invalid_json");
      }
    });
    this.reader.on("close", () => { if (!this.closed) this.fail("host_disconnected"); });
    input.on("error", () => this.fail("host_input_error"));
  }

  setSecret(secret: string): void { this.secret = secret; }
  start(): Promise<JsonObject> {
    if (this.first) return Promise.resolve(this.first);
    if (this.closed) return Promise.reject(new Error("host_disconnected"));
    return new Promise((resolve, reject) => { this.firstResolve = resolve; this.firstReject = reject; });
  }
  send(message: JsonObject): void {
    const line = JSON.stringify(message, (_key, value) => typeof value === "string" && this.secret ? value.replaceAll(this.secret, "[REDACTED]") : value);
    this.output.write(line + "\n");
  }
  request(message: JsonObject, responseType: string, timeoutMs: number, signal?: AbortSignal): Promise<JsonObject> {
    if (this.closed || signal?.aborted) return Promise.reject(new Error("host_disconnected_or_aborted"));
    const id = String(message.id);
    return new Promise((resolve, reject) => {
      const done = (error?: Error, value?: JsonObject) => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        this.pending.delete(id);
        if (error) reject(error); else resolve(value!);
      };
      const onAbort = () => done(new Error("operation_aborted"));
      const timer = setTimeout(() => done(new Error("host_response_timeout")), timeoutMs);
      this.pending.set(id, { type: responseType, resolve: (v) => done(undefined, v), reject: (e) => done(e) });
      signal?.addEventListener("abort", onAbort, { once: true });
      this.send(message);
    });
  }
  private fail(code: string): void {
    if (this.closed) return;
    this.closed = true;
    this.controller.abort(code);
    this.firstReject?.(new Error(code));
    for (const entry of [...this.pending.values()]) entry.reject(new Error(code));
    this.pending.clear();
  }
  close(): void {
    this.closed = true;
    this.reader.close();
  }
}
