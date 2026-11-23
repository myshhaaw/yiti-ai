import { NextResponse } from "next/server";
import { loadPolicyIndex } from "@/lib/policy-search";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const index = await loadPolicyIndex();
    return NextResponse.json({
      status: index ? "ok" : "degraded",
      local_index: { available: Boolean(index), documents: index?.document_count ?? 0, chunks: index?.chunk_count ?? 0 },
      semantic_index: { available: false },
      gemini_configured: Boolean(process.env.GEMINI_API_KEY),
      answer_mode: process.env.GEMINI_API_KEY ? "gemini" : "retrieval",
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ status: "degraded", local_index: { available: false, documents: 0, chunks: 0 }, gemini_configured: Boolean(process.env.GEMINI_API_KEY) }, { status: 500, headers: { "Cache-Control": "no-store" } });
  }
}
