"use client";

import { FormEvent, useEffect, useState } from "react";

type Citation = { title: string; source_url: string | null; department: string; file_path: string; page: number | null; heading: string | null; policy_date: string | null; label: string; complete: boolean };
type Evidence = { evidence: string; citation: Citation; term_coverage: number; semantic_score?: number | null; retrieval_source?: string };
type ChatReply = { status: string; answer: string; departments: string[]; citations: Citation[]; evidence: Evidence[]; answer_mode?: "gemini" | "retrieval"; generation_warning?: string | null; evidence_status?: "exact_answer_supported" | "relevant_document_found" | "related_policy_found_detail_missing" | "no_reliable_evidence"; requested_detail?: string | null; retrieval_method?: string };
type Message = { role: "student" | "yiti"; text: string; reply?: ChatReply };
const examples = ["What are the rules for late fee payment?", "How do I connect to campus Wi-Fi?", "SCSET UG academic rules", "Hostel curfew timings"];

function evidenceStatusLabel(reply?: ChatReply) {
  if (!reply?.evidence_status) return "";
  if (reply.evidence_status === "exact_answer_supported") return "Requested detail appears in the retrieved policy";
  if (reply.evidence_status === "related_policy_found_detail_missing") return `Related policy found; ${reply.requested_detail?.replaceAll("_", " ") || "requested detail"} is not stated in the retrieved passages`;
  if (reply.evidence_status === "no_reliable_evidence") return "No reliable policy passage found";
  return "Relevant policy passages found";
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [api, setApi] = useState(process.env.NEXT_PUBLIC_YITI_API_URL || "http://127.0.0.1:8000");
  const [health, setHealth] = useState("checking");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  useEffect(() => {
    setApi(window.localStorage.getItem("yiti-api") || process.env.NEXT_PUBLIC_YITI_API_URL || "http://127.0.0.1:8000");
    try { const saved=window.localStorage.getItem("yiti-current-chat"); if(saved) setMessages(JSON.parse(saved)); } catch { window.localStorage.removeItem("yiti-current-chat"); }
    setHistoryLoaded(true);
  }, []);
  useEffect(() => { if(historyLoaded) window.localStorage.setItem("yiti-current-chat",JSON.stringify(messages)); },[messages,historyLoaded]);
  useEffect(() => { let alive=true; fetch(`${api}/health`).then(r => r.json()).then(h => { if(alive) setHealth(h.semantic_index?.available ? "Local hybrid policy library ready" : h.local_index?.available ? "Local keyword policy library ready" : `Backend online · ${h.error || "policy index unavailable"}`); }).catch(() => {if(alive)setHealth("Backend offline");}); return ()=>{alive=false}; }, [api]);

  async function send(text: string) {
    const question = text.trim(); if (!question || busy) return;
    setMessages(old => [...old, { role: "student", text: question }]); setDraft(""); setError(""); setBusy(true);
    try {
      const history=messages.slice(-6).map(m=>({role:m.role==="student"?"user":"assistant",content:m.text}));
      const res = await fetch(`${api}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: question, history }) });
      const body = await res.json(); if (!res.ok) throw new Error(body.detail || "The request failed. Check that the local Yiti API is running.");
      setMessages(old => [...old, { role: "yiti", text: body.answer, reply: body }]);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not reach Yiti AI."); }
    finally { setBusy(false); }
  }
  function submit(e: FormEvent) { e.preventDefault(); void send(draft); }
  return <main className="shell">
    <aside className="rail"><div className="brand"><span className="brand-icon">Y</span><span>yiti<span className="brand-light">.ai</span></span></div><div className="rail-label">YOUR CAMPUS, IN ONE PLACE</div><button className="nav active"><span>✦</span> Ask Yiti</button></aside>
    <section className="main"><header><div className="crumb">Campus guide <span>/</span> Ask Yiti</div><div className="header-right"><span className={`status ${health === "Backend offline" ? "offline" : ""}`}><i/> {health}</span><span className="help" title="Questions can be in English or Hinglish">English / Hinglish</span></div></header>
      <div className="content"><div className="intro"><div className="eyebrow"><span>✦</span> YOUR CAMPUS COMPANION</div><h1>Hi there. What can<br/>I help you <em>figure out?</em></h1><p>Ask about policies, academics, hostel life, fees, and more.<br/>Try English or Hinglish — I’ll help you find the right source.</p></div>
        <div className="chat">
          {messages.length === 0 ? <><div className="suggest-label">A FEW THINGS YOU CAN ASK</div><div className="suggestions">{examples.map((q,i)=><button key={q} onClick={() => void send(q)}><span>{["₹","⌁","▤","⌂"][i]}</span>{q}<b>↗</b></button>)}</div></> : <div className="thread">{messages.map((m,i)=><div key={i} className={`message ${m.role}`}><div className="message-icon">{m.role === "student" ? "S" : "Y"}</div><div className="bubble"><div className="role-label">{m.role === "student" ? "YOU" : m.reply?.answer_mode === "gemini" ? "YITI AI · GEMINI ANSWER" : "YITI AI · RETRIEVED POLICY EVIDENCE"}</div><p>{m.text}</p>{m.reply && <div className={`evidence-status evidence-${m.reply.evidence_status || "unknown"}`}>{evidenceStatusLabel(m.reply)}{m.reply.retrieval_method && <span> · {m.reply.retrieval_method.replaceAll("_", " ")}</span>}</div>}{m.reply?.answer_mode !== "gemini" && m.reply?.generation_warning && <div className="fallback-note">Retrieval only · {m.reply.generation_warning}</div>}{(m.reply?.citations?.length ?? 0) > 0 && <div className="inline-citations">Sources: {m.reply?.citations?.map((c,j)=><span key={`${c.file_path}-${c.page}-${j}`}>{j>0 ? ", " : ""}{c.source_url ? <a href={c.source_url} target="_blank" rel="noreferrer">[{j+1}]</a> : <span>[{j+1}]</span>}</span>)}</div>}{m.reply?.evidence.map((ev,j)=><article className="source" key={j}><div className="source-top"><span>{ev.citation.department.replaceAll("_"," ")}</span><span>{ev.retrieval_source?.replaceAll("+", " + ") || "keyword"} retrieval</span></div><h3>{ev.citation.title}</h3>{ev.citation.page && <small>Page {ev.citation.page}{ev.citation.heading ? ` · ${ev.citation.heading}` : ""}</small>}<blockquote>{ev.evidence}</blockquote>{ev.citation.source_url && <a href={ev.citation.source_url} target="_blank" rel="noreferrer">Open source ↗</a>}</article>)}</div></div>)}{busy && <div className="loading"><span/> Looking through campus policies…</div>}</div>}
          {error && <div className="error">{error}</div>}
          <form className="composer" onSubmit={submit}><textarea value={draft} onChange={e=>setDraft(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();void send(draft)}}} placeholder="Ask a campus question… (English or Hinglish)" rows={2}/><div className="composer-foot"><span>↵ Enter to send · Shift + ↵ for a new line</span><button disabled={busy||!draft.trim()} aria-label="Send message">↑</button></div></form>
          <div className="disclaimer">Yiti AI finds relevant source passages. Always check the linked policy for complete and current details.</div>
        </div>
      </div><footer><span>✳ Yiti AI</span><span>Campus knowledge, made easier.</span><button onClick={()=>{setMessages([]);setError("")}}>New conversation</button></footer>
    </section>
  </main>;
}
