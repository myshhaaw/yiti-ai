import { gunzipSync } from "node:zlib";
import { readFile } from "node:fs/promises";
import path from "node:path";

export type PolicyRecord = {
  chunk_id: number;
  document_id: string;
  department: string;
  title: string;
  heading: string | null;
  file_path: string;
  source_url: string | null;
  page: number | null;
  policy_date: string | null;
  course: string | null;
  content: string;
};

type PolicyIndex = {
  schema_version: number;
  retrieval: "bm25";
  document_count: number;
  chunk_count: number;
  records: PolicyRecord[];
};

export type PolicyCitation = {
  title: string;
  source_url: string | null;
  department: string;
  file_path: string;
  page: number | null;
  heading: string | null;
  policy_date: string | null;
  label: string;
  complete: boolean;
  document_id: string;
  chunk_id: number;
};

export type RetrievedEvidence = {
  evidence: string;
  citation: PolicyCitation;
  term_coverage: number;
  matched_terms: string[];
  retrieval_source: "keyword";
  score: number;
};

export type PolicySearchResult = {
  index: PolicyIndex;
  topics: string[];
  results: RetrievedEvidence[];
};

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "can", "do", "does", "for", "from", "how", "i", "in", "is",
  "it", "me", "my", "of", "on", "or", "please", "should", "the", "to", "what", "when", "where", "which", "who",
  "why", "with", "would", "you", "your", "get", "find", "tell", "show", "explain", "know", "check",
  "look", "search", "give", "provide", "need", "want", "about", "kya", "hai", "hain", "ka", "ki", "ke",
  "ko", "mein", "me", "mujhe", "mera", "meri", "kaise", "kese", "karu", "kare", "karna", "kab", "tak",
  "kaha", "kahan",
]);

const BROAD_TERMS = new Set([
  "policy", "policies", "rule", "rules", "campus", "university", "bennett", "hostel", "time", "times",
  "timing", "timings", "hour", "hours",
]);

const QUERY_NORMALIZATIONS: Array<[RegExp, string]> = [
  [/\bwi[\s-]?fi\b/gi, "wifi"],
  [/\bconnect(?:ing)?\s+(?:to\s+)?(?:the\s+)?(?:campus\s+)?internet\b/gi, "wifi connection"],
  [/\bin[\s-]+time\b/gi, "curfew"],
  [/\bfinancial aid\b/gi, "scholarship"],
  [/\bfee reimbursement\b/gi, "fee refund"],
  [/\bstudent complaints?\b/gi, "student grievance"],
  [/\b(?:tuition|fees?|money)\b.*\bback\b.*/gi, "tuition refund"],
  [/\bhostel\b.*\b(?:kab tak|wa?pas|va?apas)\b.*/gi, "hostel curfew return in-time"],
  [/\bfees?\s+(?:wa?pas|va?apas)\b.*/gi, "fee refund"],
  [/\b(?:wa?pas|va?apas)\s+fees?\b.*/gi, "fee refund"],
  [/\b(?:attendance\s+(?:kam|low|less)|miss(?:ed|ing)?\s+(?:too\s+many\s+)?(?:lectures?|classes?)|skip(?:ped|ping)?\s+(?:too\s+many\s+)?(?:lectures?|classes?))\b.*/gi, "attendance exam eligibility"],
  [/\b(?:come back late|return late|gate closing time)\b/gi, "hostel curfew return time"],
];

const TERM_ALIASES: Record<string, string[]> = {
  curfew: ["in-time", "in time", "return time", "night return", "stipulated time"],
  refund: ["reimbursement", "return of fees"],
  reimbursement: ["refund", "fee return"],
  scholarship: ["fellowship", "financial aid"],
  wifi: ["wi-fi", "wireless", "internet connection"],
  internet: ["wifi", "wi-fi", "wireless"],
  grievance: ["complaint", "complaints"],
  complaint: ["grievance", "grievances"],
  transport: ["bus", "shuttle"],
  bus: ["transport", "shuttle"],
  shuttle: ["transport", "bus"],
  wireless: ["wifi", "wi-fi", "internet"],
  return: ["curfew", "in-time", "stipulated time", "night return"],
  "kab": ["return time", "curfew"],
  "wapas": ["return", "refund"],
  "vaapas": ["return", "refund"],
  "baje": ["time", "hours"],
};

const DOMAIN_PATTERNS: Array<[string, RegExp]> = [
  ["FINANCE", /\b(fee|fees|refund|reimburse|scholarship|payment|tuition|receipt|installment|money back)\b/i],
  ["HOSTEL", /\b(hostel|warden|curfew|in[ -]?time|roommate|room partner|leave from hostel|come back late|return late|gate|kab tak wapas)\b/i],
  ["IT", /\b(it policy|information technology|wifi|wi-fi|wireless|internet|password|login|email|lms|erp|portal|account)\b/i],
  ["COURSE_SPECIFIC", /\b(scset|seas|sol|mba|bajmc|bba|course rules|academic rules)\b/i],
  ["ACADEMICS", /\b(attendance|exam|examination|semester|academic calendar|registrar|course registration|lecture|class|eligibility)\b/i],
  ["STUDENT_SUPPORT", /\b(anti-ragging|ragging|posh|harassment|wellbeing|counselling|grievance|complaint)\b/i],
  ["CAMPUS_SERVICES", /\b(mess|food|transport|bus|shuttle|library|sports|maintenance|cleaning|lost and found)\b/i],
  ["EMPLOYEES", /\b(employee|staff|faculty|payroll|hr manual)\b/i],
];

const DETAIL_REQUESTS: Array<[string, RegExp, RegExp]> = [
  ["clock_time", /\b(curfew|in[ -]?time|closing time|what time|at what time|gate.*time|timings?|kitne baje|kab tak|come back late|return late)\b/i,
    /\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b|\b\d{1,2}:[0-5]\d\b|\b(?:midnight|noon)\b/i],
  ["amount", /\b(how much|what amount|how many rupees|cost|price|amount|kitna|kitne paise)\b/i,
    /(?:₹|\b(?:rs\.?|inr|rupees?)\s*)[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:rupees?|inr|rs\.?|percent|%)\b/i],
  ["deadline", /\b(deadline|due date|last date|by what date|submit by|kab tak)\b/i,
    /\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?/i],
  ["duration", /\b(how long|how many days|within how many|kitne din|kitne time tak)\b/i,
    /\b\d+\s+(?:working\s+)?(?:days?|weeks?|months?|years?|hours?)\b/i],
];

let cachedIndex: PolicyIndex | null = null;
const tokenCache = new WeakMap<PolicyRecord, string[]>();

function tokens(text: string): string[] {
  return text.toLocaleLowerCase("en").match(/[\p{L}\p{N}]+/gu) ?? [];
}

export function normalizePolicyQuery(query: string): string {
  let normalized = query.trim();
  for (const [pattern, replacement] of QUERY_NORMALIZATIONS) normalized = normalized.replace(pattern, replacement);
  return normalized;
}

function queryTerms(query: string): string[] {
  let normalized = normalizePolicyQuery(query);
  normalized = normalized.replace(/\bin[\s-]+time\b/gi, "curfew")
    .replace(/\bfinancial aid\b/gi, "scholarship")
    .replace(/\bfee reimbursement\b/gi, "fee refund")
    .replace(/\bstudent complaints?\b/gi, "student grievance");
  return [...new Set(tokens(normalized).filter((term) => !STOP_WORDS.has(term)))];
}

function expandTerm(term: string): string[] {
  return [term, ...(TERM_ALIASES[term] ?? []).flatMap(tokens)];
}

function departmentName(value: string | null | undefined): string | null {
  if (!value?.trim()) return null;
  const normalized = value.trim().toUpperCase().replace(/[\s-]+/g, "_");
  const aliases: Record<string, string> = {
    FEES: "FINANCE", "CAMPUS_SERVICES": "CAMPUS_SERVICES", "STUDENT_SUPPORT": "STUDENT_SUPPORT",
  };
  const final = aliases[normalized] ?? normalized;
  return DOMAIN_PATTERNS.some(([name]) => name === final) ? final : null;
}

function identifyDepartments(query: string): string[] {
  return DOMAIN_PATTERNS.filter(([, pattern]) => pattern.test(query)).map(([name]) => name);
}

function departmentFamily(department: string | null): Array<string | null> {
  if (department === "ACADEMICS") return ["ACADEMICS", "COURSE_SPECIFIC"];
  if (department === "COURSE_SPECIFIC") return ["COURSE_SPECIFIC", "ACADEMICS"];
  return [department];
}

function splitTopics(query: string): Array<{ query: string; department: string | null }> {
  const clauses = query.split(/\b(?:and|aur|also)\b|[;]/i).map((part) => part.trim()).filter(Boolean);
  if (clauses.length > 1) {
    const clauseDepartments = clauses.map(identifyDepartments);
    const allExplicit = clauseDepartments.every((found) => found.length > 0);
    const differentDomains = new Set(clauseDepartments.flat()).size > 1;
    if (allExplicit && differentDomains) {
      return clauses.flatMap((part, index) => clauseDepartments[index].map((department) => ({ query: part, department })));
    }
  }
  const departments = identifyDepartments(query);
  return departments.length ? departments.map((department) => ({ query, department })) : [{ query, department: null }];
}

function recordTokens(record: PolicyRecord): string[] {
  const cached = tokenCache.get(record);
  if (cached) return cached;
  const value = tokens([record.title, record.heading ?? "", record.department, record.content].join(" "));
  tokenCache.set(record, value);
  return value;
}

function rankTopic(index: PolicyIndex, query: string, department: string | null, limit: number): RetrievedEvidence[] {
  const terms = queryTerms(query);
  if (!terms.length) return [];
  const broad = terms.filter((term) => !BROAD_TERMS.has(term));
  const relevanceTerms = broad.length ? broad : terms.filter((term) => ["hostel", "wifi", "policy", "rule"].includes(term));
  if (!relevanceTerms.length) return [];

  const expandedTerms = new Map(relevanceTerms.map((term) => [term, new Set(expandTerm(term))]));
  const corpus = index.records.map((record) => ({ record, words: recordTokens(record), frequency: new Map<string, number>() }));
  for (const item of corpus) for (const word of item.words) item.frequency.set(word, (item.frequency.get(word) ?? 0) + 1);
  const avgLength = corpus.reduce((sum, item) => sum + item.words.length, 0) / Math.max(1, corpus.length);
  const docFrequency = new Map<string, number>();
  for (const [, values] of expandedTerms) {
    for (const term of values) {
      const count = corpus.reduce((sum, item) => sum + (item.frequency.has(term) ? 1 : 0), 0);
      docFrequency.set(term, count);
    }
  }

  const candidates: RetrievedEvidence[] = [];
  for (const item of corpus) {
    if (department && !departmentFamily(department).includes(item.record.department)) continue;
    let score = 0;
    const matchedTerms: string[] = [];
    for (const [concept, alternatives] of expandedTerms) {
      let conceptScore = 0;
      let matched = false;
      for (const term of alternatives) {
        const tf = item.frequency.get(term) ?? 0;
        if (!tf) continue;
        matched = true;
        const df = docFrequency.get(term) ?? 0;
        const idf = Math.log(1 + (corpus.length - df + 0.5) / (df + 0.5));
        const k1 = 1.2;
        const b = 0.75;
        conceptScore = Math.max(conceptScore, idf * (tf * (k1 + 1)) /
          (tf + k1 * (1 - b + b * item.words.length / Math.max(avgLength, 1))));
      }
      if (matched) {
        matchedTerms.push(concept);
        score += conceptScore;
      }
    }
    const coverage = matchedTerms.length / relevanceTerms.length;
    const minimumCoverage = relevanceTerms.length > 1 ? 0.55 : 1;
    if (score <= 0 || coverage < minimumCoverage) continue;
    const record = item.record;
    const label = record.title + (record.page ? `, p. ${record.page}` : record.heading ? ` — ${record.heading}` : "");
    candidates.push({
      evidence: record.content.slice(0, 1200),
      citation: {
        title: record.title,
        source_url: record.source_url || null,
        department: record.department,
        file_path: record.file_path,
        page: record.page,
        heading: record.heading || null,
        policy_date: record.policy_date || null,
        label,
        complete: Boolean(record.source_url && (record.page !== null || record.heading)),
        document_id: record.document_id,
        chunk_id: record.chunk_id,
      },
      term_coverage: Number(coverage.toFixed(3)),
      matched_terms: matchedTerms,
      retrieval_source: "keyword",
      score,
    });
  }
  return candidates.sort((left, right) => right.score - left.score ||
    right.term_coverage - left.term_coverage || left.citation.chunk_id - right.citation.chunk_id).slice(0, limit);
}

export function searchPolicyIndex(index: PolicyIndex, query: string, department?: string | null): PolicySearchResult {
  const searchQuery = normalizePolicyQuery(query);
  const topics = department
    ? [{ query: searchQuery, department: departmentName(department) }]
    : splitTopics(searchQuery);
  const merged = new Map<string, RetrievedEvidence>();
  for (const topic of topics) {
    for (const result of rankTopic(index, topic.query, topic.department, 8)) {
      const key = `${result.citation.document_id}:${result.citation.chunk_id}`;
      const existing = merged.get(key);
      if (existing) existing.score += result.score;
      else merged.set(key, result);
    }
  }
  const results = [...merged.values()].sort((left, right) => right.score - left.score ||
    right.term_coverage - left.term_coverage).slice(0, 8);
  return { index, topics: topics.map((topic) => topic.query), results };
}

export async function loadPolicyIndex(): Promise<PolicyIndex | null> {
  if (cachedIndex) return cachedIndex;
  const indexPath = process.env.YITI_POLICY_INDEX_PATH || path.join(process.cwd(), ".data", "policy-index.json.gz");
  try {
    const compressed = await readFile(indexPath);
    const parsed = JSON.parse(gunzipSync(compressed).toString("utf8")) as PolicyIndex;
    if (parsed.schema_version !== 1 || parsed.retrieval !== "bm25" ||
        parsed.document_count !== 24 || parsed.chunk_count !== 1160 || parsed.records.length !== 1160) {
      throw new Error("policy_index_metadata_invalid");
    }
    cachedIndex = parsed;
    return parsed;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw new Error("policy_index_invalid");
  }
}

export function requestedDetail(query: string): { name: string; pattern: RegExp } | null {
  for (const [name, requestPattern, valuePattern] of DETAIL_REQUESTS) {
    if (requestPattern.test(query)) return { name, pattern: valuePattern };
  }
  return null;
}

export function evidenceStatus(query: string, results: RetrievedEvidence[]): { status: string; requested_detail: string | null } {
  if (!results.length) return { status: "no_reliable_evidence", requested_detail: null };
  const detail = requestedDetail(query);
  if (!detail) return { status: "relevant_document_found", requested_detail: null };
  if (results.some((item) => detail.pattern.test(item.evidence))) {
    return { status: "exact_answer_supported", requested_detail: detail.name };
  }
  return { status: "related_policy_found_detail_missing", requested_detail: detail.name };
}
