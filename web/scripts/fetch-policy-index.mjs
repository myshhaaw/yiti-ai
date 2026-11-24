import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const output = path.resolve(here, "..", ".data", "policy-index.json.gz");
const token = process.env.YITI_POLICY_ARTIFACT_TOKEN?.trim();
const repository = (process.env.YITI_POLICY_ARTIFACT_REPO || "myshhaaw/yiti-ai-policy-artifacts").trim();
const expectedHash = process.env.YITI_POLICY_INDEX_SHA256?.trim().toLowerCase();

function verify(buffer) {
  if (buffer.length < 3 || buffer[0] !== 0x1f || buffer[1] !== 0x8b) throw new Error("Policy artifact is not gzip data.");
  const actualHash = createHash("sha256").update(buffer).digest("hex");
  if (expectedHash && actualHash !== expectedHash) throw new Error("Policy artifact checksum mismatch.");
  const metadata = JSON.parse(gunzipSync(buffer).toString("utf8"));
  if (metadata.schema_version !== 1 || metadata.retrieval !== "bm25" ||
      metadata.document_count !== 24 || metadata.chunk_count !== 1160 || metadata.records?.length !== 1160) {
    throw new Error("Policy artifact counts or schema do not match the reviewed index.");
  }
  return actualHash;
}

if (process.env.VERCEL && !expectedHash) {
  throw new Error("Set YITI_POLICY_INDEX_SHA256 in Vercel to pin the reviewed policy artifact.");
}
if (expectedHash && !/^[a-f0-9]{64}$/.test(expectedHash)) {
  throw new Error("YITI_POLICY_INDEX_SHA256 must be a 64-character hexadecimal SHA-256 value.");
}

if (!token) {
  if (process.env.VERCEL) throw new Error("Set YITI_POLICY_ARTIFACT_TOKEN in Vercel for build-time private index provisioning.");
  try {
    const local = await readFile(output);
    const hash = verify(local);
    console.log(`Using local policy index (${local.length} bytes, SHA-256 ${hash}).`);
    process.exit(0);
  } catch (error) {
    if (error?.code === "ENOENT") throw new Error("Build the local index first or configure private artifact access.");
    throw error;
  }
}

if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) throw new Error("YITI_POLICY_ARTIFACT_REPO is invalid.");

const apiBase = `https://api.github.com/repos/${repository}`;
const headers = {
  Authorization: `Bearer ${token}`,
  Accept: "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
  "User-Agent": "yiti-ai-policy-index-build",
};

async function githubFetch(url, accept = headers.Accept) {
  try {
    return await fetch(url, {
      headers: { ...headers, Accept: accept },
      cache: "no-store",
    });
  } catch {
    throw new Error("GitHub API request failed due to network connectivity; check the build network and retry.");
  }
}

function repositoryAccessError(status) {
  if (status === 401) return "GitHub rejected YITI_POLICY_ARTIFACT_TOKEN (HTTP 401): the token is invalid or expired.";
  if (status === 403) return "GitHub denied repository access (HTTP 403): the token lacks repository access or an organization policy blocks it.";
  if (status === 404) return "GitHub returned HTTP 404 for repository metadata: verify YITI_POLICY_ARTIFACT_REPO and that the token can access this private repository.";
  return `GitHub repository access check failed with HTTP ${status}.`;
}

const repositoryResponse = await githubFetch(apiBase);
if (!repositoryResponse.ok) throw new Error(repositoryAccessError(repositoryResponse.status));
const repositoryInfo = await repositoryResponse.json();
if (repositoryInfo.private !== true) throw new Error("The policy artifact repository must be private.");

const artifactUrl = `${apiBase}/contents/policy-index.json.gz?ref=main`;
const response = await githubFetch(artifactUrl, "application/vnd.github.raw+json");
if (!response.ok) {
  if (response.status === 401) throw new Error("GitHub rejected YITI_POLICY_ARTIFACT_TOKEN (HTTP 401): the token is invalid or expired.");
  if (response.status === 403) throw new Error("GitHub denied policy file access (HTTP 403): grant this token Contents: Read-only access to the private artifact repository.");
  if (response.status === 404) throw new Error("GitHub returned HTTP 404 for main/policy-index.json.gz although repository metadata is accessible; verify the file path, main branch, and token Contents: Read-only permission.");
  throw new Error(`Private policy artifact fetch failed with HTTP ${response.status}.`);
}
const buffer = Buffer.from(await response.arrayBuffer());
if (buffer.length > 20 * 1024 * 1024) throw new Error("Policy artifact exceeds the 20 MB build limit.");
const hash = verify(buffer);
await mkdir(path.dirname(output), { recursive: true });
const temporary = `${output}.tmp`;
await writeFile(temporary, buffer, { mode: 0o600 });
await rename(temporary, output);
console.log(`Provisioned private policy index (${buffer.length} bytes, SHA-256 ${hash}).`);
