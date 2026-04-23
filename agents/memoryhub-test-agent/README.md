# MemoryHub Test Agent

Test agent demonstrating MemoryHub integration with Kagenti ADK. Uses governed cross-session memory alongside the standard ContextStore for conversation replay.

## Environment Variables

- `MEMORYHUB_URL` — MemoryHub MCP server endpoint
- `MEMORYHUB_AUTH_URL` — MemoryHub OAuth 2.1 auth server (or use `MEMORYHUB_API_KEY`)
- `MEMORYHUB_CLIENT_ID` / `MEMORYHUB_CLIENT_SECRET` — OAuth credentials
- `LLM_URL` — OpenAI-compatible LLM endpoint
- `LLM_MODEL` — Model name (default: `gpt-oss-20b`)
