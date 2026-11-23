import { NextResponse } from "next/server";
import { evidenceStatus, loadPolicyIndex, searchPolicyIndex, type RetrievedEvidence } from "@/lib/policy-search";
import { generateGroundedAnswerStream, GeminiError, hasValidCitationReferences } from "@/lib/gemini";

export const runtime = "nodejs";
// 60 seconds is within Hobby's configured ceiling even when Fluid Compute is off.
export const maxDuration = 60;

const MAX_BODY_BYTES = 16 * 1024;
const MAX_MESSAGE_LENGTH = 1_000;
const RATE_LIMIT = 8;
const RATE_WINDOW_MS = 60_000;
const requestWindows = new Map<string, number[]>();

type ChatBody = {
  message?: unknown;
  department?: unknown;
  history?: unknown;
};

function response(body: object, status = 200) {
  return NextResponse.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

async function readBoundedJson(request: Request): Promise<ChatBody> {
  const declaredLength = Number(request.headers.get("content-length") || "0");
  if (declaredLength > MAX_BODY_BYTES) throw new Error("request_too_large");
  if (!request.body) throw new Error("invalid_json");
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) {
        await reader.cancel();
        throw new Error("request_too_large");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes)) as ChatBody;
  } catch {
    throw new Error("invalid_json");
  }
}

function withinRateLimit(request: Request): boolean {
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  const key = forwarded || request.headers.get("x-real-ip") || "local";
  const now = Date.now();
  const active = (requestWindows.get(key) || []).filter((timestamp) => now - timestamp < RATE_WINDOW_MS);
  if (active.length >= RATE_LIMIT) {
    requestWindows.set(key, active);
    return false;
  }
  active.push(now);
  requestWindows.set(key, active);
  if (requestWindows.size > 4_000) {
    for (const [entry, timestamps] of requestWindows) {
      if (!timestamps.some((timestamp) => now - timestamp < RATE_WINDOW_MS)) requestWindows.delete(entry);
    }
  }
  return true;
}

function detailOnlyAnswer(status: string, requestedDetail: string | null): string | null {
  if (status !== "related_policy_found_detail_missing" || !requestedDetail) return null;
  const label = requestedDetail.replaceAll("_", " ");
  return `I found a related policy passage, but it does not state the requested ${label}. Please check the cited source or contact the relevant campus office.`;
}

function generationWarning(error: unknown): string {
  if (!(error instanceof GeminiError)) return "Gemini generation failed; retrieved passages are shown below.";
  if (error.category === "gemini_not_configured") return "Gemini is not configured; retrieved passages are shown below.";
  if (error.category === "gemini_timeout") return "Gemini timed out; retrieved passages are shown below.";
  if (error.category === "gemini_request_failed" && error.httpStatus === 404) return "Gemini returned HTTP 404 model_or_endpoint_not_found; retrieved passages are shown below.";
  if (error.category === "gemini_request_failed" && error.httpStatus === 429) return "Gemini rate limit reached; retrieved passages are shown below.";
  if (error.category === "gemini_request_failed" && (error.httpStatus === 401 || error.httpStatus === 403)) return "Gemini authorization failed; retrieved passages are shown below.";
  if (error.category === "gemini_invalid_citation") return "Gemini returned a citation outside the supplied evidence; retrieved passages are shown below.";
  return "Gemini generation failed; retrieved passages are shown below.";
}

function retrievalOnlyAnswer(results: RetrievedEvidence[]): string {
  return results.length
    ? "I found relevant policy passages. Please review the linked citations for the supported details."
    : "I couldn't find a reliable matching policy passage. Please rephrase or contact the relevant campus office.";
}

function sseEvent(encoder: TextEncoder, event: string, data: object): Uint8Array {
  return encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

export async function POST(request: Request) {
  if (!withinRateLimit(request)) return response({ detail: "Too many requests. Please wait a minute and try again." }, 429);

  let body: ChatBody;
  try {
    body = await readBoundedJson(request);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "invalid_json";
    return response({ detail: reason === "request_too_large" ? "Request body is too large." : "Request body must be valid JSON." }, reason === "request_too_large" ? 413 : 400);
  }

  if (typeof body.message !== "string") return response({ detail: "message must be a string" }, 422);
  const message = body.message.trim();
  if (!message || message.length > MAX_MESSAGE_LENGTH) return response({ detail: "message must contain 1 to 1000 characters" }, 422);
  if (body.department !== undefined && body.department !== null &&
      (typeof body.department !== "string" || body.department.length > 40)) {
    return response({ detail: "department must be a string of at most 40 characters" }, 422);
  }

  const history = Array.isArray(body.history)
    ? body.history.slice(-6).flatMap((turn) => {
      if (!turn || typeof turn !== "object") return [];
      const item = turn as { role?: unknown; content?: unknown };
      if ((item.role !== "user" && item.role !== "assistant") || typeof item.content !== "string") return [];
      return [{ role: item.role, content: item.content.slice(0, 800) }] as Array<{ role: "user" | "assistant"; content: string }>;
    })
    : [];

  const priorQuestion = [...history].reverse().find((turn) => turn.role === "user")?.content || "";
  const query = priorQuestion && /\b(what about|how about|and the|when is it|those|that policy|it)\b/i.test(message)
    ? `${priorQuestion} ${message}`
    : message;

  let index;
  try {
    index = await loadPolicyIndex();
  } catch {
    return response({ detail: "The configured policy search index is invalid." }, 500);
  }
  if (!index) return response({
    status: "policy_index_unavailable",
    answer: "Policy retrieval is unavailable until the approved private policy index is provisioned.",
    departments: [], citations: [], evidence: [], evidence_status: "no_reliable_evidence",
    retrieval_method: "none", answer_mode: "retrieval",
  }, 503);

  const searched = searchPolicyIndex(index, query, typeof body.department === "string" ? body.department : null);
  const results = searched.results;
  const departments = [...new Set(results.map((item) => item.citation.department))];
  const verified = evidenceStatus(message, results);
  if (!results.length) {
    return response({
      message, status: "no_policy_match", answer: retrievalOnlyAnswer(results), departments,
      citations: [], evidence: [], evidence_status: "no_reliable_evidence", requested_detail: null,
      retrieval_method: "none", answer_mode: "retrieval",
    });
  }

  const citations = results.slice(0, 8).map((item) => item.citation);
  const evidence = results.slice(0, 8);
  const detailFallback = detailOnlyAnswer(verified.status, verified.requested_detail);
  const commonReply = {
    message, status: "evidence_found", departments,
    citations, evidence, evidence_status: verified.status,
    requested_detail: verified.requested_detail,
    retrieval_method: "keyword" as const,
  };

  // Keep the evidence-missing guard ahead of Gemini. Never ask the model to
  // invent a requested clock time, amount, date, deadline, or duration.
  if (detailFallback) return response({
    ...commonReply, answer: detailFallback, answer_mode: "retrieval", generation_warning: null,
  });
  if (!process.env.GEMINI_API_KEY) return response({
    ...commonReply, answer: retrievalOnlyAnswer(results), answer_mode: "retrieval",
    generation_warning: "Gemini is not configured; retrieved passages are shown below.",
  });

  const encoder = new TextEncoder();
  const routeAbort = new AbortController();
  let cancelled = false;
  const onRequestAbort = () => { cancelled = true; routeAbort.abort(); };
  request.signal.addEventListener("abort", onRequestAbort, { once: true });
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const emit = (event: string, data: object) => {
        if (!cancelled) controller.enqueue(sseEvent(encoder, event, data));
      };
      emit("start", { ...commonReply, answer: "", answer_mode: "gemini", generation_warning: null });
      void (async () => {
        let answer = "";
        let pendingCitation = "";
        const emitVerifiedText = (text: string) => {
          pendingCitation += text;
          while (pendingCitation) {
            const open = pendingCitation.indexOf("[");
            if (open < 0) {
              emit("delta", { text: pendingCitation });
              pendingCitation = "";
              return;
            }
            if (open > 0) {
              emit("delta", { text: pendingCitation.slice(0, open) });
              pendingCitation = pendingCitation.slice(open);
            }
            const close = pendingCitation.indexOf("]");
            if (close < 0) return;
            const marker = pendingCitation.slice(0, close + 1);
            if (!/^\[\d+\]$/.test(marker) || !hasValidCitationReferences(marker, Math.min(5, results.length))) {
              throw new GeminiError("gemini_invalid_citation");
            }
            emit("delta", { text: marker });
            pendingCitation = pendingCitation.slice(close + 1);
          }
        };
        try {
          for await (const token of generateGroundedAnswerStream(message, history, results, { signal: routeAbort.signal })) {
            if (cancelled) return;
            answer += token;
            emitVerifiedText(token);
          }
          if (pendingCitation || !hasValidCitationReferences(answer, Math.min(5, results.length))) {
            throw new GeminiError("gemini_invalid_citation");
          }
          emit("done", { answer_mode: "gemini" });
        } catch (error) {
          if (cancelled) return;
          emit("fallback", {
            ...commonReply,
            answer: retrievalOnlyAnswer(results),
            answer_mode: "retrieval",
            generation_warning: generationWarning(error),
          });
          emit("done", { answer_mode: "retrieval" });
        } finally {
          request.signal.removeEventListener("abort", onRequestAbort);
          if (!cancelled) controller.close();
        }
      })();
    },
    cancel() {
      cancelled = true;
      routeAbort.abort();
      request.signal.removeEventListener("abort", onRequestAbort);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
