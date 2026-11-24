import type { RetrievedEvidence } from "./policy-search";

const MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";
const DEFAULT_TIMEOUT_MS = 45_000;
const MAX_TIMEOUT_MS = 55_000;

export class GeminiError extends Error {
  readonly category: string;
  readonly httpStatus: number | null;

  constructor(category: string, httpStatus: number | null = null) {
    super(category);
    this.name = "GeminiError";
    this.category = category;
    this.httpStatus = httpStatus;
  }
}

export function getGeminiTimeoutMs(value = process.env.YITI_GEMINI_TIMEOUT_MS): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 1_000) return DEFAULT_TIMEOUT_MS;
  return Math.min(Math.floor(parsed), MAX_TIMEOUT_MS);
}

export function hasValidCitationReferences(answer: string, evidenceCount: number): boolean {
  const suppliedSources = Math.min(5, evidenceCount);
  let hasCitation = false;
  for (const match of answer.matchAll(/\[(\d+)\]/g)) {
    hasCitation = true;
    const citationNumber = Number(match[1]);
    if (!Number.isInteger(citationNumber) || citationNumber < 1 || citationNumber > suppliedSources) return false;
  }
  return hasCitation;
}

function normalizeQuote(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase("en")
    .replace(/[’‘]/g, "'")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** Verifies each cited answer source has a verbatim quote from that passage. */
export function hasVerifiedEvidenceQuotes(answer: string, quoteBlock: string, evidence: RetrievedEvidence[]): boolean {
  if (!hasValidCitationReferences(answer, evidence.length)) return false;
  const sourceNumbers = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1])));
  const verifiedNumbers = new Set<number>();
  const quoteLines = quoteBlock.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (const line of quoteLines) {
    const match = line.match(/^\[(\d+)\]\s+["“](.+?)["”]$/);
    if (!match) return false;
    const citationNumber = Number(match[1]);
    if (!sourceNumbers.has(citationNumber) || citationNumber > Math.min(5, evidence.length)) return false;
    const quote = normalizeQuote(match[2]);
    const passage = normalizeQuote(evidence[citationNumber - 1]?.evidence ?? "");
    if (quote.length < 24 || !passage.includes(quote)) return false;
    verifiedNumbers.add(citationNumber);
  }
  return sourceNumbers.size > 0 && [...sourceNumbers].every((number) => verifiedNumbers.has(number));
}

type StreamOptions = {
  apiKey?: string;
  fetcher?: typeof fetch;
  timeoutMs?: number;
  signal?: AbortSignal;
};

function requestContents(
  message: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  evidence: RetrievedEvidence[],
) {
  const evidenceText = evidence.slice(0, 5).map((item, index) =>
    `[${index + 1}] Department: ${item.citation.department}\n` +
    `Source: ${item.citation.label}\n` +
    `Date: ${item.citation.policy_date || "not supplied"}\n` +
    `Passage: ${item.evidence.slice(0, 1_200)}`,
  ).join("\n\n");

  const contents = history.slice(-6).map((turn) => ({
    role: turn.role === "assistant" ? "model" : "user",
    parts: [{ text: turn.content.slice(0, 800) }],
  }));
  contents.push({
    role: "user",
    parts: [{ text: `Question: ${message}\n\nRetrieved policy evidence:\n${evidenceText}` }],
  });

  return {
    systemInstruction: {
    parts: [{ text: "You are Yiti AI, a Bennett University policy assistant. Answer only from the supplied retrieved policy passages. Do not use general knowledge to fill gaps or invent policies. If a requested time, amount, date, deadline, or duration is absent, say that the passage does not state it. For multi-topic questions, answer each supported topic separately. Use concise natural English; reply in Hinglish when the student asks in Hinglish. Add inline citations like [1] that refer only to the supplied passage numbers. Never claim a passage says more than it does. After the answer, add a newline followed by the exact marker <evidence_quotes>. On following lines include one line for every citation used in your answer, in this exact format: [1] \"a verbatim quote of at least 24 characters copied exactly from passage 1\". Choose the shortest passage text that directly supports the answer. Do not place any other text after the quote lines." }],
    },
    contents,
    generationConfig: { temperature: 0, maxOutputTokens: 450 },
  };
}

function textParts(data: unknown): string[] {
  if (!data || typeof data !== "object") return [];
  const candidates = (data as { candidates?: Array<{ content?: { parts?: Array<{ text?: unknown }> } }> }).candidates;
  const parts = candidates?.[0]?.content?.parts ?? [];
  return parts.flatMap((part) => typeof part.text === "string" && part.text ? [part.text] : []);
}

function parseSseData(block: string): string | null {
  const data = block.split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  return data && data !== "[DONE]" ? data : null;
}

function extractSseBlock(buffer: string): { block: string; rest: string } | null {
  const match = /\r?\n\r?\n/.exec(buffer);
  if (!match || match.index === undefined) return null;
  return {
    block: buffer.slice(0, match.index),
    rest: buffer.slice(match.index + match[0].length),
  };
}

export async function* generateGroundedAnswerStream(
  message: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  evidence: RetrievedEvidence[],
  options: StreamOptions = {},
): AsyncGenerator<string> {
  const apiKey = options.apiKey ?? process.env.GEMINI_API_KEY;
  if (!apiKey) throw new GeminiError("gemini_not_configured");
  if (!evidence.length) throw new GeminiError("grounded_evidence_required");

  const fetcher = options.fetcher ?? fetch;
  const timeoutMs = options.timeoutMs ?? getGeminiTimeoutMs();
  const controller = new AbortController();
  const onParentAbort = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  else options.signal?.addEventListener("abort", onParentAbort, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetcher(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(MODEL)}:streamGenerateContent?alt=sse`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify(requestContents(message, history, evidence)),
        signal: controller.signal,
        cache: "no-store",
      },
    );

    if (!response.ok) throw new GeminiError("gemini_request_failed", response.status);
    if (!response.body) throw new GeminiError("gemini_stream_missing_body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawText = false;
    const consumeBlock = function* (block: string): Generator<string> {
      const data = parseSseData(block);
      if (!data) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        throw new GeminiError("gemini_stream_invalid_event");
      }
      for (const text of textParts(parsed)) {
        sawText = true;
        yield text;
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let extracted: { block: string; rest: string } | null;
      while ((extracted = extractSseBlock(buffer))) {
        buffer = extracted.rest;
        yield* consumeBlock(extracted.block);
      }
      if (done) break;
    }
    if (buffer.trim()) yield* consumeBlock(buffer);
    if (!sawText) throw new GeminiError("gemini_returned_no_text");
  } catch (error) {
    if (controller.signal.aborted) throw new GeminiError("gemini_timeout");
    if (error instanceof GeminiError) throw error;
    throw new GeminiError("gemini_network_error");
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", onParentAbort);
  }
}
