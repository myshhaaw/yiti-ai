import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/api/chat": ["./.data/policy-index.json.gz"],
    "/api/health": ["./.data/policy-index.json.gz"],
  },
};

export default nextConfig;
