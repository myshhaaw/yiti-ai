import type { RetrievedEvidence } from "./policy-search";

const MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";
const TIMEOUT_MS = 7_000;

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

export async function generateGroundedAnswer(
  message: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  evidence: RetrievedEvidence[],
  apiKey = process.env.GEMINI_API_KEY,
  fetcher: typeof fetch = fetch,
): Promise<string> {
  if (!apiKey) throw new GeminiError("gemini_not_configured");
  if (!evidence.length) throw new GeminiError("grounded_evidence_required");

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

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetcher(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(MODEL)}:generateContent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          systemInstruction: {
            parts: [{ text: "You are Yiti AI, a Bennett University policy assistant. Answer only from the supplied retrieved policy passages. Do not use general knowledge to fill gaps or invent policies. If a requested time, amount, date, deadline, or duration is absent, say that the passage does not state it. For multi-topic questions, answer each supported topic separately. Use concise natural English and understand Hinglish. Add inline citations like [1] that match the supplied passage numbers. Never claim a passage says more than it does." }],
          },
          contents,
          generationConfig: { temperature: 0, maxOutputTokens: 450 },
        }),
        signal: controller.signal,
        cache: "no-store",
      },
    );

    if (!response.ok) throw new GeminiError("gemini_request_failed", response.status);
    const data = await response.json() as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    const answer = data.candidates?.[0]?.content?.parts?.map((part) => part.text || "").join("").trim();
    if (!answer) throw new GeminiError("gemini_returned_no_text");
    return answer;
  } catch (error) {
    if (error instanceof GeminiError) throw error;
    if (controller.signal.aborted) throw new GeminiError("gemini_timeout");
    throw new GeminiError("gemini_network_error");
  } finally {
    clearTimeout(timer);
  }
}
