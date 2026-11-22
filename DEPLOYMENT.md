# Deploy Yiti AI

## Render FastAPI backend

Create a Docker Web Service from the GitHub repository and use the root-level
render.yaml. It launches yiti_ai.api:app on 0.0.0.0:$PORT. Configure GEMINI_API_KEY
as a Render secret and WEB_ORIGIN as the exact Vercel origin.

The SQLite and vector files are excluded from Git and Docker. The blueprint does
not provision a disk or enable billing. Until storage is approved and configured,
policy retrieval will be unavailable. The backend expects:

- YITI_LOCAL_INDEX=/var/data/search.sqlite3
- YITI_SEMANTIC_INDEX=/var/data/policy-vectors.npz

## Vercel Next.js frontend

Import the same repository, set Root Directory to web/, and use the detected
Next.js build settings. Add NEXT_PUBLIC_YITI_API_URL with the Render service
origin (no trailing slash). This is public URL configuration, not a secret.
After Vercel assigns a URL, set Render WEB_ORIGIN to its exact https origin.

## Costs and data boundary

The Render blueprint requests the Free plan and creates no persistent disk.
No paid service has been enabled. The database and vector artifacts must not be
uploaded until their contents, destination, public/private visibility, and
storage cost have been reviewed and approved.