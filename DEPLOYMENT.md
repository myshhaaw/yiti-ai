# Yiti AI on Vercel

Yiti AI runs as one Next.js application. The same-origin `/api/chat` route uses a
local BM25 index exported from the reviewed 24-document, 1,160-chunk SQLite
index. Policy text remains out of the public source repository.

## Provision the private policy index

From the repository root, export the verified production index:

```powershell
python tools/build_web_policy_index.py
```

The output is `web/.data/policy-index.json.gz` and is ignored by Git. Confirm
the printed document/chunk counts and checksum. Upload only that compressed
file to the approved **private** repository
`myshhaaw/yiti-ai-policy-artifacts`, at its root as `policy-index.json.gz`.
Keep the repository private and create a fine-grained read-only token scoped to
that repository's Contents. The token is used only during Vercel's build and
must never be committed or placed in a `NEXT_PUBLIC_` variable. The build
downloads the file, verifies gzip/schema/counts and optional SHA-256, then
includes it only in the server-side API function trace. The artifact is not a
public static asset. Public chat answers intentionally show limited retrieved
passages and source citations.

## Vercel setup

Import `myshhaaw/yiti-ai` and set the Root Directory to `web`. Framework is
Next.js; Build Command `npm run build`; Install Command `npm ci`. Add these
server-side environment variables for Production (and Preview only if desired):

- `GEMINI_API_KEY`: your Gemini key.
- `GEMINI_MODEL`: `gemini-3.5-flash-lite`.
- `YITI_POLICY_ARTIFACT_REPO`: `myshhaaw/yiti-ai-policy-artifacts`.
- `YITI_POLICY_ARTIFACT_TOKEN`: fine-grained read-only token described above.
- `YITI_POLICY_INDEX_SHA256`: SHA-256 printed by the export script.

No `NEXT_PUBLIC_` variables are needed. Do not add Render services, persistent
disks, paid storage, or billing. Stay on Vercel Hobby. If the free plan or
Google API quota is unavailable for your account, stop rather than upgrading.

The `/api/health` route reports whether the index and Gemini key are configured.
It does not validate Gemini generation. `/api/chat` returns retrieval-only
answers if Gemini is unavailable and refuses to invent missing requested facts.
