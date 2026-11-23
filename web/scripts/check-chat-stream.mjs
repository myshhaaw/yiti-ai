import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const routePath = fileURLToPath(new URL("../.next/server/app/api/chat/route.js", import.meta.url));
const { routeModule } = require(routePath);
const { POST, maxDuration } = routeModule.userland;
assert.equal(maxDuration, 60, "chat route should fit the non-Fluid Hobby ceiling");

process.env.GEMINI_API_KEY = "test-only-no-network-key";
process.env.YITI_GEMINI_TIMEOUT_MS = "12000";

function eventPayloads(body) {
  return body.trim().split(/\r?\n\r?\n/).map((block) => {
    const event = block.match(/^event:\s*(.+)$/m)?.[1];
    const data = block.split(/\r?\n/).find((line) => line.startsWith("data:"))?.slice(5).trimStart();
    return { event, data: data ? JSON.parse(data) : null };
  });
}

function geminiChunk(text) {
  return `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text }] } }] })}\n\n`;
}

function streamedResponse(text, splitAt = 19) {
  const payload = geminiChunk(text);
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(payload.slice(0, splitAt)));
      controller.enqueue(encoder.encode(payload.slice(splitAt)));
      controller.close();
    },
  }), { headers: { "Content-Type": "text/event-stream" } });
}

async function ask(message) {
  return POST(new Request("http://localhost/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message }),
  }));
}

const originalFetch = globalThis.fetch;
try {
  // A response slower than the removed 7-second deadline must stream successfully.
  globalThis.fetch = async (url, init) => {
    assert.match(String(url), /:streamGenerateContent\?alt=sse$/);
    assert.equal(new Headers(init.headers).get("x-goog-api-key"), "test-only-no-network-key");
    await new Promise((resolve) => setTimeout(resolve, 7_200));
    return streamedResponse("A student below 70% receives an X grade [1].");
  };
  let start = Date.now();
  let response = await ask("What is the attendance requirement?");
  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error(`Expected Gemini stream, got ${response.status}: ${JSON.stringify(await response.json())}`);
  }
  let events = eventPayloads(await response.text());
  assert.ok(Date.now() - start > 7_000, "slow mock should exceed the old 7-second timeout");
  assert.equal(events[0].event, "start");
  assert.equal(events.at(-1).event, "done");
  assert.ok(events.some((item) => item.event === "delta" && item.data.text.includes("X grade")));
  assert.ok(events[0].data.citations.some((citation) => citation.source_url && citation.page));
  console.log("PASS slow streamed generation (>7s) and clickable citation metadata");

  // A stalled provider must time out predictably and finish with retrieved evidence.
  process.env.YITI_GEMINI_TIMEOUT_MS = "1000";
  globalThis.fetch = async (_url, init) => new Promise((_resolve, reject) => {
    if (init.signal.aborted) reject(new DOMException("Aborted", "AbortError"));
    else init.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  });
  response = await ask("What is the attendance requirement?");
  events = eventPayloads(await response.text());
  const timeoutFallback = events.find((item) => item.event === "fallback")?.data;
  assert.match(timeoutFallback?.generation_warning || "", /timed out/i);
  assert.ok(timeoutFallback?.evidence?.length > 0);
  assert.ok(timeoutFallback?.citations?.some((citation) => citation.source_url));
  console.log("PASS timeout returns retrieved passages and source citations");

  // Invalid source numbers are never emitted; the answer is replaced with fallback.
  process.env.YITI_GEMINI_TIMEOUT_MS = "12000";
  globalThis.fetch = async () => streamedResponse("This answer cites an unknown passage [99].", 40);
  response = await ask("What is the attendance requirement?");
  events = eventPayloads(await response.text());
  const invalidCitationFallback = events.find((item) => item.event === "fallback")?.data;
  assert.ok(invalidCitationFallback?.generation_warning?.includes("citation outside"));
  assert.ok(!events.filter((item) => item.event === "delta").some((item) => item.data.text.includes("[99]")));
  assert.ok(invalidCitationFallback.citations.length > 0);
  console.log("PASS invalid citation is suppressed and replaced with evidence fallback");

  // No-evidence questions stay on the honest retrieval path and do not call Gemini.
  let called = false;
  globalThis.fetch = async () => { called = true; throw new Error("unexpected model request"); };
  response = await ask("What is the climate on Mars?");
  const noEvidence = await response.json();
  assert.equal(noEvidence.evidence_status, "no_reliable_evidence");
  assert.equal(noEvidence.citations.length, 0);
  assert.equal(called, false);
  console.log("PASS missing-evidence handling avoids Gemini and citations");
} finally {
  globalThis.fetch = originalFetch;
}
